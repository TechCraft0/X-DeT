"""Readable YOLOv1 training loop with gradient accumulation and run records."""
from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import time
from functools import partial
from datetime import datetime, timezone
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
from x_yolo.models.yolov1.loss import YoloV1Loss, build_targets as build_yolov1_targets
from x_yolo.models.yolov1.postprocess import decode_predictions as decode_yolov1
from .checkpoint import load_checkpoint, save_checkpoint


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _optimizer_parameter_groups(
    model: torch.nn.Module,
    learning_rate: float,
    backbone_lr_multiplier: float,
) -> list[dict[str, Any]]:
    """Keep transfer-backbone parameters in the optimizer during its freeze phase."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return [{"params": list(model.parameters()), "lr_multiplier": 1.0, "name": "model"}]

    backbone_parameters = list(backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    detector_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    if not backbone_parameters or not detector_parameters:
        raise ValueError("A transfer model must have both backbone and detector parameters")
    return [
        {"params": detector_parameters, "lr": learning_rate, "lr_multiplier": 1.0, "name": "detector"},
        {
            "params": backbone_parameters,
            "lr": learning_rate * backbone_lr_multiplier,
            "lr_multiplier": backbone_lr_multiplier,
            "name": "backbone",
        },
    ]


def _set_backbone_trainability(model: torch.nn.Module, trainable: bool) -> None:
    setter = getattr(model, "set_backbone_trainable", None)
    if callable(setter):
        setter(trainable)


def _should_save_last_checkpoint(epoch_number: int, interval_epochs: int, is_final_epoch: bool) -> bool:
    """Save resumable state periodically and always at the end of a run."""
    interval = max(1, int(interval_epochs))
    return epoch_number % interval == 0 or is_final_epoch


def train_detector(
    recipe_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    epochs_override: int | None = None,
    micro_batch_size_override: int | None = None,
    precision_override: str | None = None,
    max_steps: int | None = None,
    resume: str | Path | None = None,
    backbone_checkpoint: str | Path | None = None,
) -> Path:
    """Train a configured YOLO detector and return its run directory."""
    recipe_path = Path(recipe_path).resolve()
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError(f"Training recipe must be a mapping: {recipe_path}")
    if micro_batch_size_override is not None:
        if micro_batch_size_override <= 0:
            raise ValueError("micro_batch_size_override must be positive")
        effective_batch_size = int(recipe["effective_batch_size"])
        if effective_batch_size % micro_batch_size_override:
            raise ValueError("micro batch size must divide recipe effective_batch_size exactly")
        recipe["micro_batch_size"] = micro_batch_size_override
    if precision_override is not None:
        recipe["precision"] = precision_override
    model_config = dict(recipe.get("model", {}))
    if backbone_checkpoint is not None:
        model_config["backbone_checkpoint"] = str(Path(backbone_checkpoint).expanduser().resolve())
    recipe["model"] = model_config
    project_root = Path(__file__).resolve().parents[2]
    dataset_config_path = Path(recipe["dataset_config"]).expanduser()
    if not dataset_config_path.is_absolute():
        recipe_relative_path = recipe_path.parent / dataset_config_path
        dataset_config_path = (
            recipe_relative_path if recipe_relative_path.exists() else project_root / dataset_config_path
        )
    dataset_config_path = dataset_config_path.resolve()
    dataset_config = load_dataset_config(dataset_config_path)
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = str(recipe.get("precision", "fp32")).lower()
    if precision not in ("fp32", "bf16"):
        raise ValueError("Training precision must be fp32 or bf16")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 training requires a CUDA device with BF16 support")
    seed = int(recipe.get("seed", 42))
    _seed_everything(seed)

    train_data = YoloTxtDetectionDataset(
        dataset_config, root, "train", recipe["input_size"], True, recipe.get("training_augmentation")
    )
    val_data = YoloTxtDetectionDataset(dataset_config, root, "val", recipe["input_size"])
    workers = int(recipe.get("num_workers", 0))
    loader_args = {
        "batch_size": int(recipe["micro_batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": detection_collate,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(
        val_data,
        batch_size=loader_args["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=loader_args["pin_memory"],
        collate_fn=detection_collate,
        persistent_workers=workers > 0,
    )
    model = build_model(model_config, len(dataset_config["class_names"])).to(device)
    if model_config.get("architecture", "yolov1") == "yolov2":
        from x_yolo.models.yolov2.loss import YoloV2Loss, build_targets as build_yolov2_targets
        from x_yolo.models.yolov2.postprocess import decode_predictions as decode_yolov2

        expected_grid_size = int(recipe["input_size"]) // 32
        if int(recipe["input_size"]) % 32 or int(recipe["grid_size"]) != expected_grid_size:
            raise ValueError("YOLOv2 input_size must be divisible by 32 and grid_size must equal input_size / 32")
        if int(recipe.get("boxes_per_cell", model.num_anchors)) != model.num_anchors:
            raise ValueError("YOLOv2 boxes_per_cell must match the number of configured anchors")
        target_builder = partial(
            build_yolov2_targets,
            num_classes=len(dataset_config["class_names"]),
            anchors=model.anchors.detach().cpu(),
        )
        decoder = partial(decode_yolov2, anchors=model.anchors)
        loss_function = YoloV2Loss(
            len(dataset_config["class_names"]), model.anchors, **recipe.get("loss", {})
        ).to(device)
    else:
        target_builder = partial(
            build_yolov1_targets,
            num_classes=len(dataset_config["class_names"]),
            grid_size=int(recipe["grid_size"]),
        )
        decoder = decode_yolov1
        loss_function = YoloV1Loss(
            len(dataset_config["class_names"]),
            grid_size=int(recipe["grid_size"]),
            boxes_per_cell=int(recipe["boxes_per_cell"]),
            **recipe.get("loss", {}),
        ).to(device)
    optimizer_config = recipe["optimizer"]
    parameter_groups = _optimizer_parameter_groups(
        model,
        float(optimizer_config["learning_rate"]),
        float(optimizer_config.get("backbone_lr_multiplier", 1.0)),
    )
    if not any(group["params"] for group in parameter_groups):
        raise ValueError("The selected model has no trainable parameters")
    optimizer_name = str(optimizer_config["name"]).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=float(optimizer_config["learning_rate"]),
            momentum=float(optimizer_config.get("momentum", 0.0)),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=float(optimizer_config["learning_rate"]),
            betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_config['name']!r}; use SGD or Adam")
    accumulation_steps = max(
        1, round(int(recipe["effective_batch_size"]) / int(recipe["micro_batch_size"]))
    )
    target_epochs = int(epochs_override or recipe["epochs"])
    start_epoch = 0
    best_map = -1.0
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, map_location=device)
        _move_optimizer(optimizer, device)
        start_epoch = int(checkpoint["epoch"])
        best_map = float(checkpoint.get("best_metric", -1.0))

    run_config: dict[str, Any] = {
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
        "seed": seed,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pillow": pillow_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_revision": _code_revision(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "planned_epochs": target_epochs,
        "micro_batch_size": int(recipe["micro_batch_size"]),
        "effective_batch_size": accumulation_steps * int(recipe["micro_batch_size"]),
        "gradient_accumulation_steps": accumulation_steps,
        "resume_from": str(resume) if resume else None,
        "max_steps": max_steps,
    }
    (output / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (output / "metrics.jsonl").touch(exist_ok=True)
    try:
        (output / "train_recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
        shutil.copy2(dataset_config_path, output / "dataset_config.yaml")
    except shutil.SameFileError:
        pass

    global_step = int(checkpoint.get("global_step", 0)) if resume else 0
    max_steps_hit = False
    for epoch in range(start_epoch, target_epochs):
        freeze_epochs = recipe.get("freeze_backbone_epochs")
        if freeze_epochs is None:
            backbone_trainable = not bool(model_config.get("freeze_backbone", False))
        else:
            backbone_trainable = epoch >= int(freeze_epochs)
        _set_backbone_trainability(model, backbone_trainable)
        model.train()
        epoch_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals: dict[str, float] = {}
        epoch_steps = 0
        epoch_ignored = 0
        for batch_index, (images, box_lists, _names, _sizes) in enumerate(train_loader):
            if max_steps is not None and global_step >= max_steps:
                max_steps_hit = True
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                predictions = model(images)
            if model_config.get("architecture", "yolov1") == "yolov2":
                targets = target_builder(box_lists, grid_size=int(predictions.shape[1]))
            else:
                targets = target_builder(box_lists)
            # Keep coordinate/IoU and squared-error arithmetic in FP32 even
            # when backbone/head convolutions run in BF16.
            losses = loss_function(predictions.float(), targets)
            (losses["total"] / accumulation_steps).backward()
            at_accumulation_boundary = (batch_index + 1) % accumulation_steps == 0
            at_last_batch = batch_index + 1 == len(train_loader)
            if at_accumulation_boundary or at_last_batch:
                progress_epoch = epoch + (batch_index + 1) / max(len(train_loader), 1)
                base_lr = float(optimizer_config["learning_rate"])
                warmup_epochs = float(optimizer_config.get("warmup_epochs", 0))
                if warmup_epochs > 0 and progress_epoch < warmup_epochs:
                    warmup_start = float(optimizer_config.get("warmup_initial_learning_rate", base_lr))
                    learning_rate = warmup_start + (base_lr - warmup_start) * progress_epoch / warmup_epochs
                else:
                    decay_epochs = list(optimizer_config.get("decay_epochs", []))
                    decay_count = sum(progress_epoch >= float(boundary) for boundary in decay_epochs)
                    learning_rate = base_lr * float(optimizer_config.get("decay_factor", 0.1)) ** decay_count
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * float(group.get("lr_multiplier", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for name, value in losses.items():
                epoch_totals[name] = epoch_totals.get(name, 0.0) + float(value.detach())
            epoch_ignored += targets.ignored_ground_truths
            epoch_steps += 1
            global_step += 1
            if epoch_steps % 100 == 0:
                elapsed = time.monotonic() - epoch_started
                print(
                    f"epoch={epoch + 1}/{target_epochs} batches={epoch_steps}/{len(train_loader)} "
                    f"global_step={global_step} loss={float(losses['total'].detach()):.4f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if epoch_steps == 0:
            if max_steps_hit:
                break
            raise RuntimeError("Training loader produced no batches")
        train_metrics = {f"train_{name}": value / epoch_steps for name, value in epoch_totals.items()}
        train_metrics["ignored_ground_truths"] = epoch_ignored
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "backbone_trainable": backbone_trainable,
            **train_metrics,
        }
        record["learning_rate"] = optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) > 1 and optimizer.param_groups[1].get("name") == "backbone":
            record["backbone_learning_rate"] = optimizer.param_groups[1]["lr"]
        eval_interval = int(recipe["evaluation"]["interval_epochs"])
        should_evaluate = (epoch + 1) % eval_interval == 0 or epoch + 1 == target_epochs or max_steps_hit
        if should_evaluate:
            metric_result = evaluate_model(
                model,
                val_loader,
                dataset_config["class_names"],
                device,
                confidence_threshold=float(recipe["evaluation"]["confidence_threshold"]),
                nms_iou_threshold=float(recipe["evaluation"]["nms_iou_threshold"]),
                max_detections=int(recipe["evaluation"].get("max_detections", 100)),
                decoder=decoder,
            )
            record["validation"] = metric_result
            current_map = float(metric_result["mAP50_95"])
            if current_map > best_map:
                best_map = current_map
                save_checkpoint(output / "best.pt", model, optimizer, epoch + 1, best_map, run_config, global_step)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint_interval = int(recipe.get("checkpoint_interval_epochs", 1))
        if _should_save_last_checkpoint(
            epoch_number=epoch + 1,
            interval_epochs=checkpoint_interval,
            is_final_epoch=(epoch + 1 == target_epochs or max_steps_hit),
        ):
            save_checkpoint(output / "last.pt", model, optimizer, epoch + 1, best_map, run_config, global_step)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if max_steps_hit:
            break
    return output


def train_yolov1(*args: Any, **kwargs: Any) -> Path:
    """Backward-compatible name for the original YOLOv1 training entrypoint."""
    return train_detector(*args, **kwargs)
