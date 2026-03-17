from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import importlib


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_dir = os.path.join(os.path.dirname(image_path), "debug_cells")

    cv2 = importlib.import_module("cv2")
    np = importlib.import_module("numpy")

    from sudoku.vision.board_reader import (
        OCR_MIN_CONFIDENCE,
        _apply_consistency_filter,
        _cell_digit_candidates,
        _choose_cell_prediction,
        _find_sudoku_board,
        _normalize_polarity,
    )
    from sudoku.vision.digit_model import get_digit_classifier

    print("Loading classifier...")
    classifier = get_digit_classifier(auto_train=False)

    image = cv2.imread(image_path)
    if image is None:
        print("Could not open image:", image_path)
        sys.exit(1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _normalize_polarity(gray, cv2, np)
    board_img = _find_sudoku_board(gray, cv2, np)

    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "_board_warped.png"), board_img)
    print(f"Warped board saved to: {debug_dir}/_board_warped.png\n")

    cell_size = board_img.shape[0] // 9
    all_candidates = []
    cell_meta = []

    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            cell = board_img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(debug_dir, f"cell_{row}_{col}.png"), cell)

            candidates = _cell_digit_candidates(cell, cv2, np)
            start = len(all_candidates)
            for idx, (img, ar) in enumerate(candidates):
                all_candidates.append((img, ar))
                cv2.imwrite(
                    os.path.join(debug_dir, f"cand_{row}_{col}_{idx}.png"), img
                )

            cell_meta.append((row, col, start, len(candidates)))

    all_images = [img for img, _ar in all_candidates]
    all_ars = [ar for _img, ar in all_candidates]
    predictions = classifier.predict_many(all_images) if all_images else []

    results = [[None] * 9 for _ in range(9)]
    chosen = [[(0, 0.0)] * 9 for _ in range(9)]
    for row, col, start, count in cell_meta:
        if count == 0:
            results[row][col] = (0, 0.0)
            chosen[row][col] = (0, 0.0)
            continue

        cell_predictions = []
        for i, pred in enumerate(predictions[start : start + count]):
            ar = all_ars[start + i]
            cell_predictions.append((pred.digit, pred.confidence, pred.margin, ar))
        best_digit, best_conf, support_count, margin = _choose_cell_prediction(cell_predictions)
        results[row][col] = (best_digit, best_conf)

        chosen_digit, chosen_conf, _, _ = _choose_cell_prediction(cell_predictions)
        chosen[row][col] = (chosen_digit, chosen_conf)

    for threshold in (0.70, 0.75, 0.80, 0.85):
        detected = sum(
            1
            for r in results
            for d, c in r
            if c >= threshold and d != 0
        )
        print(f"  threshold={threshold:.2f}  ->  {detected} detected cells")

    print()
    print("Confidence map (digit @ conf):")
    print("-" * 72)
    for row_idx, row in enumerate(results):
        parts = []
        for digit, conf in row:
            if conf <= 0:
                parts.append("  ---  ")
            else:
                marker = "✓" if conf >= 0.75 else "?"
                parts.append(f"{digit}@{conf:.2f}{marker}")
        print(f"Row {row_idx}: " + "  ".join(parts))

    print()
    print("Final per-cell map (same selector as app):")
    print("-" * 72)
    for row_idx, row in enumerate(chosen):
        parts = []
        for digit, conf in row:
            if conf <= 0 or digit == 0:
                parts.append("  ---  ")
            else:
                marker = "✓" if conf >= 0.68 else "?"
                parts.append(f"{digit}@{conf:.2f}{marker}")
        print(f"Row {row_idx}: " + "  ".join(parts))

    print()
    raw_predictions = []
    for row_idx, row in enumerate(chosen):
        for col_idx, (digit, conf) in enumerate(row):
            raw_predictions.append((row_idx, col_idx, digit, conf, 1 if digit != 0 else 0, 0.0))

    final_board = _apply_consistency_filter(raw_predictions, min_confidence=0.68)
    print("Final board (after Sudoku consistency filter):")
    for row in final_board:
        print(row)

    print()
    print(f"Cell images saved to: {debug_dir}/")


if __name__ == "__main__":
    main()
