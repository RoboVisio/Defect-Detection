# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Semiconductor (IC package) surface defect detection prototyping workspace. The actual detection logic lives in **HALCON HDevelop scripts** (`.hdev`, XML format, HALCON 25.11). `main.py` is an unused PyCharm stub — there is no Python pipeline yet.

## Files

- `roi_surface_defect_demo.hdev` — main demo: reads images, defines ROI, applies `dyn_threshold` + `connection` + `select_shape` for blob-based defect detection, writes annotated crops to `test_images_out/`.
- `ROI.hdev` — ROI extraction experiments (`gen_rectangle`, `reduce_domain`).
- `test.hdev` — scratch / test script.
- `image/` — labeled defect dataset organized by Chinese defect-type folder names (`01 X方向胶身偏移`, `04 小孔`, `05 毛边`, `22 露铜`, `断脚`, `裂胶身`, etc.). These category names are the ground-truth defect taxonomy — preserve them exactly when referencing or generating code.
- `test_images_out/` — generated crops from running the hdev scripts.

## Working with this repo

- `.hdev` files are XML and **must be edited in HDevelop**, not as plain text — programmatic edits will corrupt the file. When asked to modify detection logic, propose the HALCON operator changes and let the user apply them in HDevelop, or generate a new `.hdev` only if explicitly requested.
- There is no build, lint, or test command. Scripts are run interactively from HDevelop (HALCON 25.11).
- Defect category folder names contain Chinese characters and spaces; quote paths and use UTF-8.
- Several categories are marked `（待理解）` ("to be understood") — treat their semantics as unconfirmed.
