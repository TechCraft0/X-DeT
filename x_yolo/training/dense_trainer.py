"""Shared loop for point-based FCOS and RTMDet; their losses remain version-local."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader

from x_yolo.data.dataset import YoloTxtDetectionDataset, detection_collate, load_dataset_config
from x_yolo.evaluation.metrics import evaluate_model
from x_yolo.models.factory import build_model
from x_yolo.training.checkpoint import load_checkpoint, save_checkpoint
from x_yolo.training.trainer import (
    _code_revision,
    _optimizer_parameter_groups,
    _restore_optimizer_lr_multipliers,
    _seed_everything,
    _set_backbone_trainability,
    _should_save_last_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    optimizer_config: dict[str, Any],
    progress_epoch: float,
    planned_epochs: int,
) -> float:
    base_lr = float(optimizer_config["learning_rate"])
    warmup_epochs = float(optimizer_config.get("warmup_epochs", 0))
    if warmup_epochs and progress_epoch < warmup_epochs:
        start = float(optimizer_config.get("warmup_initial_learning_rate", base_lr * 0.01))
        learning_rate = start + (base_lr - start) * progress_epoch / warmup_epochs
    elif optimizer_config.get("schedule", "multistep") == "cosine":
        start_epoch = float(optimizer_config.get("cosine_start_epoch", planned_epochs * 0.5))
        if progress_epoch <= start_epoch:
            learning_rate = base_lr
        else:
            progress = min(1.0, (progress_epoch - start_epoch) / max(planned_epochs - start_epoch, 1e-6))
            eta_min = base_lr * float(optimizer_config.get("eta_min_ratio", 0.05))
            learning_rate = eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * progress)) / 2
    else:
        decay_count = sum(progress_epoch >= float(epoch) for epoch in optimizer_config.get("decay_epochs", []))
        learning_rate = base_lr * float(optimizer_config.get("decay_factor", 0.1)) ** decay_count
    learning_rate = float(learning_rate)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group.get("lr_multiplier", 1.0))
    return float(learning_rate)


def train_dense_detector(
    recipe_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    pretrained_checkpoint: str | Path | None = None,
    device_name: str = "auto",
    epochs_override: int | None = None,
    micro_batch_size_override: int | None = None,
    precision_override: str | None = None,
    max_steps: int | None = None,
    resume: str | Path | None = None,
) -> Path:
    recipe_path = Path(recipe_path).expanduser().resolve()
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError(f"Training recipe must be a mapping: {recipe_path}")
    model_config = dict(recipe.get("model", {}))
    architecture = str(model_config.get("architecture", ""))
    if architecture not in ("fcos", "rtmdet"):
        raise ValueError("dense_trainer supports only the distinct FCOS and RTMDet training paths")
    if micro_batch_size_override is not None:
        effective_batch = int(recipe["effective_batch_size"])
        if micro_batch_size_override <= 0 or effective_batch % micro_batch_size_override:
            raise ValueError("micro batch must be positive and divide effective_batch_size")
        recipe["micro_batch_size"] = micro_batch_size_override
    if precision_override is not None:
        recipe["precision"] = precision_override
    precision = str(recipe.get("precision", "fp32")).lower()
    if precision not in ("fp32", "bf16"):
        raise ValueError("Training precision must be fp32 or bf16")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 requires a CUDA device with BF16 support")

    project_root = Path(__file__).resolve().parents[2]
    dataset_config_path = Path(recipe["dataset_config"]).expanduser()
    if not dataset_config_path.is_absolute():
        relative_to_recipe = recipe_path.parent / dataset_config_path
        dataset_config_path = relative_to_recipe if relative_to_recipe.exists() else project_root / dataset_config_path
    dataset_config_path = dataset_config_path.resolve()
    dataset_config = load_dataset_config(dataset_config_path)
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(recipe.get("seed", 42)))

    train_data = YoloTxtDetectionDataset(
        dataset_config, root, "train", recipe["input_size"], True, recipe.get("training_augmentation")
    )
    val_data = YoloTxtDetectionDataset(dataset_config, root, "val", recipe["input_size"])
    workers = int(recipe.get("num_workers", 0))
    micro_batch = int(recipe["micro_batch_size"])
    train_loader = DataLoader(
        train_data, batch_size=micro_batch, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=device.type == "cuda", collate_fn=detection_collate,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_data, batch_size=int(recipe.get("validation_batch_size", micro_batch)), shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda", collate_fn=detection_collate,
        persistent_workers=workers > 0,
    )

    model = build_model(model_config, len(dataset_config["class_names"])).to(device)
    checkpoint_path = Path(pretrained_checkpoint).expanduser().resolve() if pretrained_checkpoint else None
    if checkpoint_path is not None:
        if resume:
            raise ValueError("Choose either a pretraining checkpoint or --resume, not both")
        if architecture == "fcos":
            loaded = model.load_backbone_checkpoint(checkpoint_path)
            print(f"Loaded {loaded} TorchVision ResNet-50 backbone tensors", flush=True)
        else:
            loaded = model.load_pretrained_checkpoint(checkpoint_path, dataset_config["class_names"])
            print(json.dumps(loaded, ensure_ascii=False), flush=True)
    elif not resume and bool(recipe.get("requires_pretrained_checkpoint", True)):
        raise ValueError(
            f"{architecture} recipe requires an explicit local pretrained checkpoint; "
            "weights are never downloaded by the training script"
        )

    if architecture == "fcos":
        from x_yolo.models.fcos.loss import FCOSLoss
        from x_yolo.models.fcos.postprocess import decode_predictions

        loss_function = FCOSLoss(len(dataset_config["class_names"]), model.strides)
        decoder = partial(decode_predictions, strides=model.strides)
    else:
        from x_yolo.models.rtmdet.loss import RTMDetLoss
        from x_yolo.models.rtmdet.postprocess import decode_predictions

        loss_function = RTMDetLoss(
            len(dataset_config["class_names"]), model.strides,
            topk=int(recipe.get("assigner", {}).get("topk", 13)),
            soft_center_radius=float(recipe.get("assigner", {}).get("soft_center_radius", 3.0)),
            iou_cost_weight=float(recipe.get("assigner", {}).get("iou_cost_weight", 3.0)),
            bbox_loss_weight=float(recipe.get("loss", {}).get("bbox_loss_weight", 2.0)),
        )
        decoder = partial(
            decode_predictions,
            strides=model.strides,
            nms_iou_threshold=float(recipe["evaluation"]["nms_iou_threshold"]),
        )

    optim_config = dict(recipe["optimizer"])
    backbone_lr_multiplier = float(optim_config.get("backbone_lr_multiplier", 1.0))
    if architecture == "fcos":
        pretrained_parameters = list(model.backbone.body.parameters())
        pretrained_ids = {id(parameter) for parameter in pretrained_parameters}
        detector_parameters = [parameter for parameter in model.parameters() if id(parameter) not in pretrained_ids]
        parameter_groups = [
            {"params": detector_parameters, "lr_multiplier": 1.0, "name": "detector"},
            {"params": pretrained_parameters, "lr_multiplier": backbone_lr_multiplier, "name": "backbone"},
        ]
    else:
        parameter_groups = _optimizer_parameter_groups(
            model, float(optim_config["learning_rate"]), backbone_lr_multiplier
        )
    optimizer_name = str(optim_config["name"]).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups, lr=float(optim_config["learning_rate"]),
            betas=tuple(optim_config.get("betas", (0.9, 0.999))),
            weight_decay=float(optim_config.get("weight_decay", 0.05)),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups, lr=float(optim_config["learning_rate"]),
            momentum=float(optim_config.get("momentum", 0.9)),
            weight_decay=float(optim_config.get("weight_decay", 0.0001)),
        )
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}")

    accumulation_steps = int(recipe["effective_batch_size"]) // micro_batch
    if accumulation_steps < 1:
        raise ValueError("effective_batch_size must be >= micro_batch_size")
    target_epochs = int(epochs_override or recipe["epochs"])
    start_epoch, best_metric = 0, -1.0
    configured_multipliers = [float(group.get("lr_multiplier", 1.0)) for group in optimizer.param_groups]
    if resume:
        state = load_checkpoint(resume, model, optimizer, map_location=device)
        _restore_optimizer_lr_multipliers(optimizer, configured_multipliers)
        start_epoch = int(state["epoch"])
        best_metric = float(state.get("best_metric", -1.0))

    run_config = {
        "recipe": recipe,
        "dataset_config": str(dataset_config_path),
        "dataset_root": str(root),
        "dataset_name": dataset_config.get("name", root.name),
        "class_names": dataset_config["class_names"],
        "splits": dataset_config["splits"],
        "train_images": len(train_data),
        "val_images": len(val_data),
        "device": str(device),
        "precision": precision,
        "seed": int(recipe.get("seed", 42)),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pillow": pillow_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_revision": _code_revision(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "planned_epochs": target_epochs,
        "micro_batch_size": micro_batch,
        "effective_batch_size": micro_batch * accumulation_steps,
        "gradient_accumulation_steps": accumulation_steps,
        "pretrained_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "pretrained_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "resume_from": str(resume) if resume else None,
        "max_steps": max_steps,
    }
    (output / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (output / "metrics.jsonl").touch(exist_ok=True)
    (output / "train_recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    shutil.copy2(dataset_config_path, output / "dataset_config.yaml")

    global_step = int(state.get("global_step", 0)) if resume else 0
    eval_config = recipe["evaluation"]
    for epoch in range(start_epoch, target_epochs):
        backbone_trainable = epoch >= int(recipe.get("freeze_backbone_epochs", 0))
        _set_backbone_trainability(model, backbone_trainable)
        model.train()
        if not backbone_trainable:
            model.backbone.eval()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        batches = 0
        accumulated = 0
        max_steps_hit = False
        for batch_index, (images, box_lists, _names, _sizes) in enumerate(train_loader):
            if max_steps is not None and global_step >= max_steps:
                max_steps_hit = True
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=precision == "bf16"):
                outputs = model(images)
            losses = loss_function(outputs, box_lists)
            (losses["total"] / accumulation_steps).backward()
            accumulated += 1
            at_boundary = (batch_index + 1) % accumulation_steps == 0
            at_final_batch = batch_index + 1 == len(train_loader)
            at_step_limit = max_steps is not None and global_step + 1 >= max_steps
            if at_boundary or at_final_batch or at_step_limit:
                if accumulated < accumulation_steps:
                    correction = accumulation_steps / accumulated
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                progress = epoch + (batch_index + 1) / max(len(train_loader), 1)
                _set_learning_rate(optimizer, optim_config, progress, target_epochs)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            batches += 1
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                max_steps_hit = True
                break
        if batches == 0:
            break
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "backbone_trainable": backbone_trainable,
            **{f"train_{key}": value / batches for key, value in totals.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.monotonic() - started,
        }
        if len(optimizer.param_groups) > 1 and optimizer.param_groups[1].get("name") == "backbone":
            row["backbone_learning_rate"] = float(optimizer.param_groups[1]["lr"])
        interval = int(eval_config["interval_epochs"])
        should_evaluate = (epoch + 1) % interval == 0 or epoch + 1 == target_epochs or max_steps_hit
        if should_evaluate:
            metrics = evaluate_model(
                model, val_loader, dataset_config["class_names"], device,
                confidence_threshold=float(eval_config["confidence_threshold"]),
                nms_iou_threshold=float(eval_config["nms_iou_threshold"]),
                max_detections=int(eval_config.get("max_detections", 100)), decoder=decoder,
            )
            row["validation"] = metrics
            current_metric = float(metrics["mAP50_95"])
            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint(output / "best.pt", model, optimizer, epoch + 1, best_metric, run_config, global_step)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if _should_save_last_checkpoint(
            epoch + 1, int(recipe.get("checkpoint_interval_epochs", 10)),
            epoch + 1 == target_epochs or max_steps_hit,
        ):
            save_checkpoint(output / "last.pt", model, optimizer, epoch + 1, best_metric, run_config, global_step)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if max_steps_hit:
            break
    return output
