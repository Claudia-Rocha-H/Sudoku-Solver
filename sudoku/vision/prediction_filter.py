from __future__ import annotations

OCR_MIN_CONFIDENCE = 0.70


def _accept_single_prediction(digit, confidence, margin):
    if digit == 1:
        return (confidence >= 0.58 and margin >= 0.015) or (confidence >= 0.68 and margin >= 0.0)

    return (confidence >= 0.72 and margin >= 0.12) or (confidence >= 0.82 and margin >= 0.06)


def is_valid_placement(board, row, col, value):
    if value == 0:
        return True

    for index in range(9):
        if board[row][index] == value:
            return False
        if board[index][col] == value:
            return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == value:
                return False

    return True


def choose_cell_prediction(cell_predictions):
    if not cell_predictions:
        return 0, -1.0, 0, 0.0

    def _unpack(item):
        if len(item) == 4:
            return item
        return item[0], item[1], item[2], 1.0

    groups = {}
    for item in cell_predictions:
        digit, confidence, margin, ar = _unpack(item)
        if digit == 0:
            continue
        if digit not in groups:
            groups[digit] = []
        groups[digit].append((confidence, margin, ar))

    best_digit = 0
    best_conf = -1.0
    best_support = 0
    best_margin = 0.0

    for digit, values in groups.items():
        strong = [(conf, margin, ar) for conf, margin, ar in values if conf >= 0.52 and margin >= 0.06]
        if len(strong) >= 2:
            avg_conf = float(sum(conf for conf, _, _ar in strong) / len(strong))
            avg_margin = float(sum(margin for _, margin, _ar in strong) / len(strong))
            score = avg_conf + 0.25 * avg_margin
            if score > best_conf:
                best_conf = score
                best_digit = digit
                best_support = len(strong)
                best_margin = avg_margin

    if best_digit != 0:
        if best_digit == 7:
            all_ars_7 = [ar for _, _, ar in groups[7]]
            if all_ars_7 and (sum(all_ars_7) / len(all_ars_7)) < 0.50:
                support = len(all_ars_7)
                override_conf = min(0.92, 0.72 + 0.02 * support)
                return 1, override_conf, support, best_margin
        return best_digit, best_conf, best_support, best_margin

    best_by_digit = {}
    for item in cell_predictions:
        digit, confidence, margin, ar = _unpack(item)
        if digit == 0:
            continue
        current = best_by_digit.get(digit)
        if current is None or confidence > current[0]:
            best_by_digit[digit] = (confidence, margin, ar)

    if not best_by_digit:
        return 0, -1.0, 0, 0.0

    ranked = sorted(best_by_digit.items(), key=lambda item: item[1][0], reverse=True)
    top_digit, (top_conf, top_margin, top_ar) = ranked[0]

    if top_digit == 7 and top_ar < 0.50:
        return 1, 0.72, 1, top_margin

    if _accept_single_prediction(top_digit, top_conf, top_margin):
        return top_digit, top_conf, 1, top_margin

    return 0, top_conf, 0, top_margin


def apply_consistency_filter(raw_predictions, min_confidence=OCR_MIN_CONFIDENCE):
    board = [[0] * 9 for _ in range(9)]

    def confidence_for_item(row, col, digit):
        if row < 3 and col < 3 and digit == 8:
            return max(min_confidence, 0.88)
        if digit == 1:
            return max(min_confidence, 0.60)
        return min_confidence

    def is_reliable(item):
        row, col, digit, confidence, support_count, margin = item
        if digit == 0:
            return False
        if confidence < confidence_for_item(row, col, digit):
            return False

        if row < 3 and col < 3:
            if digit == 8:
                return support_count >= 8 and confidence >= 0.82 and margin >= 0.08

            if digit == 1:
                return (support_count >= 1 and confidence >= 0.60) or (confidence >= 0.72 and margin >= 0.01)

            return (support_count >= 1 and confidence >= min_confidence) or (confidence >= 0.78 and margin >= 0.05)

        return True

    ranked = sorted(
        [item for item in raw_predictions if is_reliable(item)],
        key=lambda item: item[3],
        reverse=True,
    )

    for row, col, digit, confidence, support_count, margin in ranked:
        if is_valid_placement(board, row, col, digit):
            board[row][col] = digit

    return board
