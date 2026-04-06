"""
training/train.py
─────────────────
Fine-tune a YOLOv8 model on a custom dataset.

Demonstrates:
  - GPU-accelerated training with PyTorch
  - Dataset loading and augmentation
  - Checkpoint management
  - Exporting trained weights for edge deployment

Usage:
    python training/train.py --config configs/train_config.yaml
    python training/train.py --config configs/train_config.yaml --epochs 25 --device cpu
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and validate the training configuration YAML."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def build_data_yaml(cfg: dict, output_path: str = "configs/data.yaml") -> str:
    """
    Write a YOLO-compatible data.yaml from our config.
    Ultralytics expects a specific format with nc, names, train, val paths.
    """
    data = {
        "nc": cfg["model"]["num_classes"],
        "names": cfg["dataset"]["classes"],
        "train": cfg["dataset"]["train"],
        "val": cfg["dataset"]["val"],
    }
    if "test" in cfg["dataset"]:
        data["test"] = cfg["dataset"]["test"]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Data YAML written to {output_path}")
    return output_path


def get_device(device_cfg) -> str:
    """
    Resolve device string for training.
    Checks CUDA availability and falls back gracefully.
    """
    try:
        import torch

        if str(device_cfg) == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available — {device_count} GPU(s) detected")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {name} ({vram:.1f} GB VRAM)")
            return str(device_cfg)
        else:
            logger.warning("CUDA not available — training on CPU (will be slow)")
            return "cpu"
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch torchvision")
        sys.exit(1)


def train(cfg: dict, overrides: dict) -> None:
    """
    Execute the training run using Ultralytics YOLO.

    Ultralytics wraps PyTorch training with sensible defaults
    while exposing all knobs we need for serious experimentation.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Apply CLI overrides on top of config file
    train_cfg = cfg["training"].copy()
    train_cfg.update(overrides)

    model_name = cfg["model"]["architecture"]
    if cfg["model"]["pretrained"]:
        model_name = f"{model_name}.pt"

    data_yaml = build_data_yaml(cfg)
    device = get_device(train_cfg.get("device", "0"))

    logger.info("=" * 60)
    logger.info("EdgeAI Sentinel — Training Run")
    logger.info("=" * 60)
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Epochs:     {train_cfg['epochs']}")
    logger.info(f"  Batch size: {train_cfg['batch_size']}")
    logger.info(f"  Image size: {train_cfg['image_size']}")
    logger.info(f"  Classes:    {cfg['dataset']['classes']}")
    logger.info("=" * 60)

    model = YOLO(model_name)
    start_time = time.time()

    results = model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["image_size"],
        batch=train_cfg["batch_size"],
        lr0=train_cfg["learning_rate"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        optimizer=train_cfg["optimizer"],
        patience=train_cfg["patience"],
        save_period=train_cfg["save_period"],
        device=device,
        project=cfg["logging"]["save_dir"],
        name=cfg["logging"]["name"],
        verbose=cfg["logging"]["verbose"],
        # Augmentation params from config
        hsv_h=cfg["augmentation"]["hsv_h"],
        hsv_s=cfg["augmentation"]["hsv_s"],
        hsv_v=cfg["augmentation"]["hsv_v"],
        degrees=cfg["augmentation"]["degrees"],
        translate=cfg["augmentation"]["translate"],
        scale=cfg["augmentation"]["scale"],
        flipud=cfg["augmentation"]["flipud"],
        fliplr=cfg["augmentation"]["fliplr"],
        mosaic=cfg["augmentation"]["mosaic"],
        mixup=cfg["augmentation"]["mixup"],
    )

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.1f} minutes")

    # Print final metrics
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info("Final metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Save path for best weights
    best_weights = Path(cfg["logging"]["save_dir"]) / cfg["logging"]["name"] / "weights/best.pt"
    if best_weights.exists():
        logger.info(f"Best weights saved to: {best_weights}")
        logger.info("Next step: python training/export.py --checkpoint " + str(best_weights))
    else:
        logger.warning("best.pt not found — check training output directory")


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Sentinel — YOLOv8 Training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        help="Override batch size")
    parser.add_argument("--device", type=str, help="Override device (0, 1, cpu)")
    parser.add_argument("--lr", type=float, dest="learning_rate",
                        help="Override learning rate")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Collect CLI overrides (non-None values only)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config",) and v is not None}

    train(cfg, overrides)


if __name__ == "__main__":
    main()
