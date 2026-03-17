## Project Overview

Sudoku solver built with Python and Tkinter.  
It combines a backtracking solver with an OCR pipeline that can read Sudoku boards from images and populate the UI automatically.

This project evolved from an earlier version that solved Sudoku boards through backtracking only, with manual board entry through the UI:

- Original repository: https://github.com/Claudia-Rocha-H/Sudoku-Solver-GUI-Python.git

## Key Features

- **Improved GUI layout:** cleaner board, title, helper text, and status messages.
- **Multiple puzzle flow:** solve one Sudoku and immediately input a new one with **New Sudoku** (no restart needed).
- **Board actions:** **Solve**, **Clear**, and **New Sudoku** buttons.
- **Input validation:** each cell only accepts one digit from `1` to `9`.
- **Initial board validation:** detects conflicts in rows, columns, and 3x3 boxes before solving.
- **Result highlighting:** original values and solved values use different background colors.
- **Error feedback:** popups and status line for invalid or unsolvable puzzles.
- **Image upload (vision pipeline):** load a Sudoku image, detect board, split cells, classify digits, and fill the board.

## Requirements

- Python 3.10+
- Tkinter
- NumPy
- OpenCV
- TensorFlow

## Run

From the project folder:

```bash
python main.py
```

## Image Loading Setup

To use **Load Image**, install Python dependencies:

```bash
pip install numpy opencv-python tensorflow
```

If your Python installation does not include Tkinter, install it through your Python distribution or system package manager.

This repository already includes a pretrained model at `sudoku/vision/models/digit_cnn.keras`, so users can use OCR immediately after installing dependencies.

## Train the CNN (optional)

Retrain the digit classifier locally if you want to replace the bundled model:

```bash
python train_cnn.py --epochs 30 --batch-size 128
```

The model will be saved automatically at:

```text
sudoku/vision/models/digit_cnn.keras
```

Optional parameters:

```bash
python train_cnn.py --epochs 40 --batch-size 96 --validation-split 0.1 --learning-rate 0.001
```

The image pipeline auto-loads the model when you click **Load Image**.
For responsive UI, the app does not train during image loading.
If the bundled model is removed, train it again with `python train_cnn.py`.

Optional (advanced): use your own CNN model by replacing this path:

- Save a model at `sudoku/vision/models/digit_cnn.keras`
- Keep input shape as 28x28x1 and output classes as digits 1..9

## If TensorFlow is missing

Install TensorFlow:

```bash
pip install tensorflow
```

Then in the app:

1. Click **Load Image**.
2. Select a clear Sudoku image.
3. Review detected values (image quality can still require manual correction).
4. Click **Solve**.

## OCR Notes

- Best results come from square or near-square Sudoku images with visible grid lines.
- Decorative borders, low resolution, or stylized fonts can still require manual correction.
- The OCR pipeline includes board detection, cell candidate extraction, CNN digit classification, and Sudoku-consistency filtering.

## Debugging OCR

Use the debug script to inspect warped boards, extracted cells, and per-cell predictions:

```bash
python debug_ocr.py <image_path>
```

The script writes debug images into a `debug_cells/` folder next to the source image.

## Architecture (MVC-lite)


```text
Sudoku-Solver-GUI-Python/
├── main.py
├── train_cnn.py
├── debug_ocr.py
└── sudoku/
	├── core/
	│   └── solver.py
	├── vision/
	│   ├── board_detection.py
	│   ├── board_reader.py
	│   ├── cell_candidates.py
	│   ├── cnn_architecture.py
	│   ├── cnn_data.py
	│   ├── cnn_trainer.py
	│   ├── cv_env.py
	│   ├── digit_model.py
	│   ├── ocr_loader.py
	│   └── prediction_filter.py
	└── ui/
		└── sudoku_gui.py
```

- `main.py`: app entrypoint only (creates window and starts UI).
- `train_cnn.py`: CLI entrypoint to train and save the digit classifier.
- `debug_ocr.py`: OCR inspection tool for debugging image recognition.
- `sudoku/core/solver.py`: Sudoku rules, validation, and backtracking solver.
- `sudoku/vision/board_detection.py`: image normalization, board detection, perspective warp, and grid refinement.
- `sudoku/vision/cell_candidates.py`: cell thresholding and candidate digit extraction.
- `sudoku/vision/prediction_filter.py`: per-cell selection and Sudoku-consistency filtering.
- `sudoku/vision/cv_env.py`: lazy dependency loading for OpenCV and NumPy.
- `sudoku/vision/board_reader.py`: orchestrates the full OCR pipeline.
- `sudoku/vision/cnn_data.py`: MNIST 1-9 dataset prep + data augmentation pipeline.
- `sudoku/vision/cnn_architecture.py`: CNN definition and compile config.
- `sudoku/vision/cnn_trainer.py`: end-to-end training and model export.
- `sudoku/vision/digit_model.py`: model loading and runtime prediction API.
- `sudoku/vision/ocr_loader.py`: compatibility wrapper around the current OCR loader.
- `sudoku/ui/sudoku_gui.py`: Tkinter widgets, user interactions, and rendering.

This structure makes it easier to test logic, evolve UI, and add features without turning one file into a monolith.
