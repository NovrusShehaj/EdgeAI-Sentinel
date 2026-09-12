# Model artifact strategy

The edge image does not bake weights. `/app/models` exists in the image and stays empty until an operator mounts or copies a versioned ONNX file.

## Runtime contract

| Item | Value |
|---|---|
| Default path | `MODEL_PATH=/app/models/best.onnx` |
| Manifest | `MODEL_MANIFEST_PATH=/app/models/model_manifest.json` |
| Classes | `CLASS_NAMES=object` unless a multi-class model exists |
| Integrity | SHA-256 in the manifest must match the file |
| Mount | read-only shared bind (`:ro,z`) |

## Manifest fields

See `models/model_manifest.example.json`. Required for verification:

- `filename`
- `class_names`
- `sha256`

Optional provenance: `version`, `source`, `dataset_revision`, `seed`, `input_size`.

## How to supply a model

1. Export from a trusted checkpoint: `python training/export.py --checkpoint path/to/best.pt --format onnx`
2. Write a manifest with the SHA-256 of the exported file
3. Mount the directory at `/app/models` read-only. Use `:ro,z` on the bind so Fedora/RHEL SELinux allows the container to read the files. Docker ignores `:z` when SELinux is not enabled.
4. Restart the process; the engine loads once at startup

Large weights should live in Git LFS or external artifact storage. This repository does not commit `.onnx`, `.pt`, or `.engine` files.

## CI / demo graph

`python scripts/generate_demo_onnx.py` writes a Constant YOLO-shaped graph. It is labeled `not_for_production` and must not be used as an accuracy or latency success signal for a trained detector.
