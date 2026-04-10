# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semiconductor (IC package) surface defect detection system. Three-stage pipeline:
1. **HALCON preprocessing** — extract ROI from raw camera images by detecting black borders
2. **Anomalib training** — train PaDiM (gray) / PatchCore (color) anomaly detection, export ONNX
3. **C++ DLL inference** — real-time Windows DLL combining HALCON preprocessing + ONNX Runtime inference

## Build & Run Commands

```powershell
# Train gray model (PaDiM, ResNet18)
python OK_Detection\anomalib_train_gray.py

# Train color model (PatchCore, wide_resnet50_2)
python OK_Detection\anomalib_train_color.py

# Build DLL (requires HALCON_ROOT and ONNXRUNTIME_ROOT env vars)
cmake -S dll -B dll\out\build\x64-Debug -G Ninja
cmake --build dll\out\build\x64-Debug --config Debug
```

HALCON `.hdev` scripts run interactively in HDevelop 25.11 — there is no CLI runner.

## Architecture

### Pipeline Flow

```
Camera image (uint8) → HALCON ROI extraction → crop/resize → ONNX inference → DefectResult
```

- **Gray pipeline**: Single image → detect black borders → crop middle ROI → split left/right workpieces → PaDiM inference on each half
- **Color pipeline**: Single image → crop color ROI → PatchCore inference (TODO: `Preprocess_Color` in `main.cpp` is a stub)

### Key Modules

- `Halcon_preprocess/` — `.hdev` scripts (XML format). `roi_surface_defect_demo.hdev` is the main demo using `dyn_threshold` + `connection` + `select_shape` for blob-based detection.
- `OK_Detection/` — Python anomalib training. `anomalib_train_gray.py` and `anomalib_train_color.py` produce ONNX models under `OK_Detection/results/{gray,color}/export/weights/onnx/model.onnx`.
- `dll/` — CMake project building `defect_dll`. Core files: `defect_dll.h` (API surface), `defect_dll.cpp` (ONNX Runtime wrapper + inference threading), `preprocess_with_roi.cpp` (HALCON ROI extraction for DLL).
- `main.cpp` (root) — HDevelop-generated C++ export of `Preprocess_Gray`. Compiled into DLL with `NO_EXPORT_MAIN` guard to suppress its `main()`.

### Cross-Stage Contracts

| Contract | Gray | Color |
|---|---|---|
| Image size (H x W) | 1144 x 611 | 448 x 960 |
| Input channels | 1 (replicated to 3) | 3 (RGB) |
| ONNX input shape | (N, 3, 1144, 611) | (N, 3, 448, 960) |
| Normalization | ImageNet mean/std | ImageNet mean/std |
| Model | PaDiM (ResNet18, layers 1-3) | PatchCore (wide_resnet50_2, layers 2-3) |

ONNX outputs: `score` tensor `[N]` (anomaly score 0-1) and `anomaly` map `[N, 1, H, W]` (resized back to ROI).

**If you change image size or normalization in training, the DLL's `halcon_to_tensor()` must match.**

### DLL API

Defined in `dll/defect_dll.h`:
- `Algo_Init(DefectConfig*)` — load ONNX models, start inference threads
- `Algo_ProcessImage(DefectImage*)` — preprocess + infer, returns `DefectResult` with left/right workpiece verdicts, confidence, anomaly map, timing
- `Algo_Release()` / `Algo_ReleaseResult()` — cleanup

### Gray ROI Detection Logic

Both `preprocess_with_roi.cpp` and `main.cpp` implement the same algorithm:
1. Threshold [0-16] to find black border pixels
2. Search bands: 15% left/right, 18% top/bottom of image
3. Largest connected component in each band → ROI boundary
4. Fallback if detection fails: rows 22%-78%, cols 10%-90%
5. Minimum ROI size: 150 rows x 250 cols
6. Split at horizontal midpoint for left/right workpiece crops

## Critical Rules

- **`.hdev` files are XML** — must be edited in HDevelop, not as text. Programmatic edits corrupt the format. Propose HALCON operator changes for the user to apply in HDevelop.
- **Chinese folder names in `image/` and `Defective_product_testing/`** are the ground-truth defect taxonomy. Preserve them exactly; downstream code depends on them. Always quote paths and use UTF-8.
- **Categories marked `（待理解）`** have unconfirmed semantics — treat cautiously.
- **C++17 required** per `dll/CMakeLists.txt`.
- **`main.cpp` encoding**: cp936 (local-8-bit). Non-ASCII strings are octal-escaped. Call `SetHcppInterfaceStringEncodingIsUtf8(false)` before using HALCON API.

## Validation Checklist

No automated test suite. Validate with smoke tests:
1. Open edited `.hdev` in HDevelop, confirm ROI/debug outputs on representative images
2. Re-run training script, verify ONNX export appears under `OK_Detection/results/`
3. Rebuild DLL after C++ or model-path changes, confirm it loads the exported model

## Dependencies

- HALCON 25.11 (MVTec)
- ONNX Runtime (GPU, Windows x64)
- Python: anomalib, torchvision, onnx, pytorch
- Build: CMake 3.16+, Ninja, MSVC (C++17)
