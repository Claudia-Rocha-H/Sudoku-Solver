from __future__ import annotations

import os

from .board_detection import find_sudoku_board, normalize_polarity
from .cell_candidates import cell_digit_candidates
from .cv_env import import_cv_np
from .digit_model import get_digit_classifier
from .prediction_filter import OCR_MIN_CONFIDENCE, apply_consistency_filter, choose_cell_prediction

_normalize_polarity = normalize_polarity
_find_sudoku_board = find_sudoku_board
_cell_digit_candidates = cell_digit_candidates
_choose_cell_prediction = choose_cell_prediction
_apply_consistency_filter = apply_consistency_filter


def load_sudoku_from_image(image_path: str, status_callback=None, debug_dir: str | None = None):
    cv2, np = import_cv_np()

    if status_callback:
        status_callback("Loading classifier...", "#2f4858")
    classifier = get_digit_classifier(status_callback, auto_train=False)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open the selected image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = normalize_polarity(gray, cv2, np)

    if status_callback:
        status_callback("Detecting board...", "#2f4858")
    board_img = find_sudoku_board(gray, cv2, np)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "_board_warped.png"), board_img)

    board = [[0] * 9 for _ in range(9)]
    cell_size = board_img.shape[0] // 9

    if status_callback:
        status_callback("Reading cells...", "#2f4858")

    candidates_by_cell = {}
    all_candidates = []

    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            cell = board_img[y1:y2, x1:x2]

            if debug_dir is not None:
                cv2.imwrite(os.path.join(debug_dir, f"cell_{row}_{col}.png"), cell)

            candidates = cell_digit_candidates(cell, cv2, np)
            if not candidates:
                continue

            candidates_by_cell[(row, col)] = []
            for idx, (img, ar) in enumerate(candidates):
                candidates_by_cell[(row, col)].append(len(all_candidates))
                all_candidates.append((img, ar))
                if debug_dir is not None:
                    cv2.imwrite(
                        os.path.join(debug_dir, f"cand_{row}_{col}_{idx}.png"),
                        img,
                    )

    all_images = [img for img, _ar in all_candidates]
    all_ars = [ar for _img, ar in all_candidates]
    predictions = classifier.predict_many(all_images)

    raw_predictions = []
    for (row, col), indices in candidates_by_cell.items():
        cell_predictions = []
        for index in indices:
            pred = predictions[index]
            cell_predictions.append((pred.digit, pred.confidence, pred.margin, all_ars[index]))

        best_digit, best_conf, support_count, margin = choose_cell_prediction(cell_predictions)
        raw_predictions.append((row, col, best_digit, best_conf, support_count, margin))

    board = apply_consistency_filter(raw_predictions, min_confidence=0.68)
    detected_count = sum(1 for row in board for value in row if value != 0)
    return board, detected_count
