"""VOC trainer for CenterNet and CornerNet's distinct keypoint objectives."""
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
    _should_save_last_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_learning_rate(optimizer: torch.optim.Optimizer, config: dict[str, Any],
                       progress_epoch: float, epochs: int) -> float:
    base = float(config["learning_rate"])
    warmup = float(config.get("warmup_epochs", 0))
    if warmup and progress_epoch < warmup:
        start = float(config.get("warmup_initial_learning_rate", base * 0.01))
        rate = start + (base - start) * progress_epoch / warmup
    elif config.get("schedule", "multistep") == "cosine":
        start_epoch = float(config.get("cosine_start_epoch", 0))
        if progress_epoch <= start_epoch:
            rate = base
        else:
            progress = min(1.0, (progress_epoch - start_epoch) / max(epochs - start_epoch, 1e-6))
            eta_min = base * float(config.get("eta_min_ratio", 0.01))
            rate = eta_min + (base - eta_min) * (1 + np.cos(np.pi * progress)) / 2
    else:
        decays = sum(progress_epoch >= float(epoch) for epoch in config.get("decay_epochs", []))
        rate = base * float(config.get("decay_factor", 0.1)) ** decays
    for group in optimizer.param_groups:
        group["lr"] = float(rate) * float(group.get("lr_multiplier", 1.0))
    return float(rate)


def _decoder(architecture: str):
    if architecture == "centernet":
        from x_yolo.models.centernet.postprocess import decode_predictions
        return decode_predictions
    from x_yolo.models.cornernet.postprocess import decode_predictions
    return decode_predictions


def train_keypoint_detector(recipe_path: str | Path, dataset_root: str | Path,
                            output_dir: str | Path, *,
                            pretrained_checkpoint: str | Path | None = None,
                            device_name: str = "auto", epochs_override: int | None = None,
                            micro_batch_size_override: int | None = None,
                            precision_override: str | None = None,
                            max_steps: int | None = None,
                            resume: str | Path | None = None) -> Path:
    """Train the configured keypoint model; no checkpoint downloads are implicit."""
    recipe_path = Path(recipe_path).expanduser().resolve()
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    model_config = dict(recipe.get("model", {}))
    architecture = str(model_config.get("architecture", ""))
    if architecture not in ("centernet", "cornernet"):
        raise ValueError("keypoint_trainer supports CenterNet and CornerNet")
    if micro_batch_size_override is not None:
        effective_batch = int(recipe["effective_batch_size"])
        if micro_batch_size_override <= 0 or effective_batch % micro_batch_size_override:
            raise ValueError("micro batch must be positive and divide effective_batch_size")
        recipe["micro_batch_size"] = micro_batch_size_override
    if precision_override is not None:
        recipe["precision"] = precision_override
    precision = str(recipe.get("precision", "fp32")).lower()
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 requires a CUDA device with BF16 support")

    root_dir = Path(__file__).resolve().parents[2]
    dataset_config_path = Path(recipe["dataset_config"]).expanduser()
    if not dataset_config_path.is_absolute():
        candidate = recipe_path.parent / dataset_config_path
        dataset_config_path = candidate if candidate.exists() else root_dir / dataset_config_path
    dataset_config_path = dataset_config_path.resolve()
    dataset_config = load_dataset_config(dataset_config_path)
    dataset_path = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(recipe.get("seed", 42)))

    train_data = YoloTxtDetectionDataset(
        dataset_config, dataset_path, "train", int(recipe["input_size"]), True,
        recipe.get("training_augmentation"),
    )
    val_data = YoloTxtDetectionDataset(dataset_config, dataset_path, "val", int(recipe["input_size"]))
    workers = int(recipe.get("num_workers", 0))
    micro_batch = int(recipe["micro_batch_size"])
    train_loader = DataLoader(train_data, batch_size=micro_batch, shuffle=True,
                              num_workers=workers, pin_memory=device.type == "cuda",
                              collate_fn=detection_collate, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=int(recipe.get("validation_batch_size", micro_batch)),
                            shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
                            collate_fn=detection_collate)

    model = build_model(model_config, len(dataset_config["class_names"])).to(device)
    checkpoint_path = Path(pretrained_checkpoint).expanduser().resolve() if pretrained_checkpoint else None
    if checkpoint_path and not resume:
        loaded = model.load_pretrained_checkpoint(checkpoint_path, dataset_config["class_names"])
        print(json.dumps({"pretrained": loaded}, ensure_ascii=False), flush=True)
    elif not resume and recipe.get("requires_pretrained_checkpoint", False):
        raise ValueError("This recipe requires --pretrained to initialize its detector")
    if architecture == "centernet":
        from x_yolo.models.centernet.loss import CenterNetLoss
        loss_function = CenterNetLoss(len(dataset_config["class_names"]), **recipe.get("loss", {}))
    else:
        from x_yolo.models.cornernet.loss import CornerNetLoss
        loss_function = CornerNetLoss(len(dataset_config["class_names"]), **recipe.get("loss", {}))
    decoder = _decoder(architecture)

    optim_config = recipe["optimizer"]
    lr = float(optim_config["learning_rate"])
    groups = _optimizer_parameter_groups(model, lr, float(optim_config.get("backbone_lr_multiplier", 1.0)))
    optimizer_name = str(optim_config["name"]).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(groups, lr=lr, weight_decay=float(optim_config.get("weight_decay", 0)))
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=float(optim_config.get("weight_decay", 0.05)))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(groups, lr=lr, momentum=float(optim_config.get("momentum", 0.9)),
                                    weight_decay=float(optim_config.get("weight_decay", 0.0001)))
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}")

    effective_batch = int(recipe["effective_batch_size"])
    if effective_batch < micro_batch or effective_batch % micro_batch:
        raise ValueError("effective_batch_size must be a multiple of micro_batch_size")
    accumulation_steps = effective_batch // micro_batch
    epochs = int(epochs_override or recipe["epochs"])
    start_epoch, best_metric, state = 0, -1.0, {}
    lr_multipliers = [float(group.get("lr_multiplier", 1.0)) for group in optimizer.param_groups]
    if resume:
        state = load_checkpoint(resume, model, optimizer, map_location=device)
        _restore_optimizer_lr_multipliers(optimizer, lr_multipliers)
        start_epoch = int(state["epoch"])
        best_metric = float(state.get("best_metric", -1.0))

    run_config = {
        "recipe": recipe, "dataset_config": str(dataset_config_path),
        "dataset_root": str(dataset_path), "dataset_name": dataset_config.get("name", dataset_path.name),
        "class_names": dataset_config["class_names"], "splits": dataset_config["splits"],
        "train_images": len(train_data), "val_images": len(val_data), "device": str(device),
        "precision": precision, "seed": int(recipe.get("seed", 42)),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "pillow": pillow_version, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_revision": _code_revision(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "planned_epochs": epochs, "micro_batch_size": micro_batch,
        "effective_batch_size": micro_batch * accumulation_steps,
        "gradient_accumulation_steps": accumulation_steps,
        "pretrained_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "pretrained_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "resume_from": str(resume) if resume else None, "max_steps": max_steps,
    }
    (output / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (output / "train_recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    (output / "metrics.jsonl").touch(exist_ok=True)
    shutil.copy2(dataset_config_path, output / "dataset_config.yaml")

    global_step = int(state.get("global_step", 0)) if resume else 0
    eval_config = recipe["evaluation"]
    freeze_epochs = int(recipe.get("freeze_backbone_epochs", 0))
    for epoch in range(start_epoch, epochs):
        backbone_trainable = epoch >= freeze_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad = backbone_trainable
        model.train()
        if not backbone_trainable:
            model.backbone.eval()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        batches = accumulated = 0
        max_steps_hit = False
        for batch_index, (images, boxes, _names, _sizes) in enumerate(train_loader):
            if max_steps is not None and global_step >= max_steps:
                max_steps_hit = True
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=precision == "bf16"):
                predictions = model(images)
            losses = loss_function(predictions, boxes)
            (losses["total"] / accumulation_steps).backward()
            accumulated += 1
            last_batch = batch_index + 1 == len(train_loader)
            at_limit = max_steps is not None and global_step + 1 >= max_steps
            if (batch_index + 1) % accumulation_steps == 0 or last_batch or at_limit:
                if accumulated < accumulation_steps:
                    correction = accumulation_steps / accumulated
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                progress = epoch + (batch_index + 1) / max(len(train_loader), 1)
                _set_learning_rate(optimizer, optim_config, progress, epochs)
                if float(optim_config.get("max_grad_norm", 0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(optim_config["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
            global_step += 1
            if at_limit:
                max_steps_hit = True
                break
        if not batches:
            break
        row: dict[str, Any] = {
            "epoch": epoch + 1, "global_step": global_step,
            "backbone_trainable": backbone_trainable,
            **{f"train_{name}": value / batches for name, value in totals.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.monotonic() - started,
        }
        interval = int(eval_config["interval_epochs"])
        if (epoch + 1) % interval == 0 or epoch + 1 == epochs or max_steps_hit:
            metrics = evaluate_model(
                model, val_loader, dataset_config["class_names"], device,
                confidence_threshold=float(eval_config["confidence_threshold"]),
                nms_iou_threshold=float(eval_config.get("nms_iou_threshold", 0.5)),
                max_detections=int(eval_config.get("max_detections", 100)),
                decoder=partial(decoder, **recipe.get("decoder", {})),
            )
            row["validation"] = metrics
            current = float(metrics["mAP50_95"])
            if current > best_metric:
                best_metric = current
                save_checkpoint(output / "best.pt", model, optimizer, epoch + 1,
                                best_metric, run_config, global_step)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if _should_save_last_checkpoint(epoch + 1, int(recipe.get("checkpoint_interval_epochs", 10)),
                                        epoch + 1 == epochs or max_steps_hit):
            save_checkpoint(output / "last.pt", model, optimizer, epoch + 1,
                            best_metric, run_config, global_step)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if max_steps_hit:
            break
    return output
