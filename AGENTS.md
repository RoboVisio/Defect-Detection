# Repository Guidelines

## Project Structure & Module Organization
`Halcon_preprocess/` contains the HALCON 25.11 `.hdev` preprocessing and ROI scripts. Treat these as the source of truth for crop geometry and debug output. `OK_Detection/` contains anomalib training and ONNX export scripts for grayscale and color pipelines, plus `datasets/` and `results/`. `dll/` builds the `defect_dll` shared library with HALCON and ONNX Runtime. Root-level `main.cpp` is the exported HALCON C++ workflow used both for local experiments and DLL builds. `image/` and `Defective_product_testing/` hold sample and labeled defect images.

## Build, Test, and Development Commands
Run HALCON logic interactively in HDevelop; do not hand-edit `.hdev` XML.

```powershell
python OK_Detection\anomalib_train_gray.py
python OK_Detection\anomalib_train_color.py
cmake -S dll -B dll\out\build\x64-Debug -G Ninja
cmake --build dll\out\build\x64-Debug --config Debug
```

The Python scripts train and export ONNX models under `OK_Detection\results\`. The CMake commands build the Windows DLL; set `HALCON_ROOT` and `ONNXRUNTIME_ROOT` first if they are not already in your environment.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and C++. Keep C++ compatible with C++17 as required by [`dll/CMakeLists.txt`](/D:/Project/半导体缺陷检测/dll/CMakeLists.txt). Follow the existing naming pattern: `Preprocess_Gray`, `IMAGE_SIZE`, and descriptive snake_case paths such as `datasets/gray/normal`. Preserve Chinese defect-category folder names exactly; downstream scripts rely on them. Quote paths in commands because the project path contains Chinese characters.

## Testing Guidelines
There is no formal automated test suite yet. Validate changes with pipeline smoke tests:

- Open the edited `.hdev` in HDevelop and confirm ROI/debug outputs on representative images.
- Re-run the relevant training script and verify ONNX export appears under `OK_Detection\results\gray\export\` or `OK_Detection\results\color\export\`.
- Rebuild the DLL after any C++ or model-path change and confirm it still loads the exported model.

## Commit & Pull Request Guidelines
This workspace does not include `.git`, so no repository-specific history is available. Use short imperative commit subjects such as `Add gray ROI fallback guard` or `Align color export path`. In pull requests, describe which stage changed (HALCON, training, or DLL), list any contract changes such as image size or channel count, and include screenshots for ROI/debug-output changes.

## Security & Configuration Tips
Do not commit large generated artifacts from `OK_Detection\results\` or `dll\out\build\`. Keep local machine paths configurable through environment variables or CMake cache entries instead of hard-coding them.

## Agent Collaboration Rules
Codex should use four fixed roles when delegation helps:

- `algorithm-expert`: owns algorithm direction, reference lookup, experiment strategy, and conflict resolution on preprocessing, training, export, and inference contracts.
- `algorithm-engineer`: owns implementation and debugging across HALCON, Python, ONNX export, and C++ integration. When algorithm advice conflicts, follow the expert.
- `code-reviewer`: owns correctness and regression review. Focus on image size, channel handling, normalization, paths, and cross-stage contract mismatches.
- `git-manager`: owns version-control hygiene only. Keep generated artifacts out of source changes and keep commit scope narrow.

Default collaboration order: `algorithm-expert` -> `algorithm-engineer` -> `code-reviewer` -> `git-manager`.

Role boundaries are strict:

- The expert gives direction but does not own implementation.
- The engineer implements but does not overrule algorithm decisions.
- The reviewer audits but does not take over coding.
- The Git manager handles change boundaries, not technical design.

## Agent Collaboration Rules
Codex should collaborate with four fixed roles when a task benefits from delegation.

- `algorithm-expert`: owns algorithm direction, literature or reference lookup, experiment strategy, and conflict resolution on preprocessing, training, export, or inference contracts.
- `algorithm-engineer`: owns implementation and debugging across HALCON, Python, ONNX export, and C++ integration. When algorithm advice conflicts, follow the expert.
- `code-reviewer`: owns correctness and regression review. Focus on image size, channel handling, normalization, paths, and cross-stage contract mismatches.
- `git-manager`: owns version-control hygiene only. Group changes cleanly, avoid mixing generated artifacts with source edits, and keep commit scope narrow.

Use this order by default: `algorithm-expert` -> `algorithm-engineer` -> `code-reviewer` -> `git-manager`.

Role boundaries are strict:

- The expert gives direction but does not own implementation.
- The engineer implements but does not overrule algorithm decisions.
- The reviewer audits but does not take over coding.
- The Git manager handles change boundaries, not technical design.
