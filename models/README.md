# models/

This directory is the local mount/source for ONNX artifacts. Weights are not committed.

```bash
python scripts/generate_demo_onnx.py --output models/demo.onnx --manifest models/model_manifest.json
```

Copy `model_manifest.example.json` when exporting a real model and fill in the SHA-256 of that file.

Local `best.onnx` or `*.pt` files are gitignored. A COCO-pretrained export has 80 classes and will fail startup if `CLASS_NAMES=object`. Either export a one-class detector that matches `configs/data.yaml` or override `CLASS_NAMES` to the model's actual labels and record them in the manifest.
