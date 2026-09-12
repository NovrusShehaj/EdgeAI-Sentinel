# Dataset

Committed files are YOLO labels for a **synthetic** single-class smoke set (`object`). Images are not stored in Git.

```bash
python scripts/test_data.py
```

That command writes white rectangles under `data/train/images` and `data/val/images`. Image and label counts must match (50 train, 10 val). Do not report those runs as real-world mAP.

Replace this directory with a documented public or domain dataset before any production-style evaluation. Record provenance, splits, and a revision id in `results/model_manifest.json`.
