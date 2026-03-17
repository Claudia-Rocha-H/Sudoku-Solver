from __future__ import annotations


def normalize_polarity(gray, cv2, np):
    h, w = gray.shape
    edge_h = max(2, int(h * 0.05))
    edge_w = max(2, int(w * 0.05))

    border_pixels = np.concatenate(
        [
            gray[:edge_h, :].flatten(),
            gray[-edge_h:, :].flatten(),
            gray[:, :edge_w].flatten(),
            gray[:, -edge_w:].flatten(),
        ]
    )

    if float(np.median(border_pixels)) < 100:
        return cv2.bitwise_not(gray)
    return gray


def _order_points(pts, np):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _fallback_square(gray, cv2):
    side = min(gray.shape[:2])
    crop = gray[:side, :side]
    return cv2.resize(crop, (900, 900), interpolation=cv2.INTER_AREA)


def _refine_to_inner_grid(warped, cv2, np):
    target = 900
    blur = cv2.GaussianBlur(warped, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 5), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 5)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    h_proj = np.sum(h_lines, axis=1).astype(float)
    v_proj = np.sum(v_lines, axis=0).astype(float)

    h_thresh = h_proj.max() * 0.15
    v_thresh = v_proj.max() * 0.15

    h_rows = np.where(h_proj > h_thresh)[0]
    v_cols = np.where(v_proj > v_thresh)[0]

    if h_rows.size < 2 or v_cols.size < 2:
        return warped

    y1, y2 = int(h_rows[0]), int(h_rows[-1])
    x1, x2 = int(v_cols[0]), int(v_cols[-1])

    if (y2 - y1) < h * 0.5 or (x2 - x1) < w * 0.5:
        return warped

    cropped = warped[y1 : y2 + 1, x1 : x2 + 1]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)


def trim_outer_border(board_image, cv2, trim_ratio=0.01):
    h, w = board_image.shape
    trim_y = max(1, int(h * trim_ratio))
    trim_x = max(1, int(w * trim_ratio))

    if h - 2 * trim_y < h * 0.7 or w - 2 * trim_x < w * 0.7:
        return board_image

    cropped = board_image[trim_y : h - trim_y, trim_x : w - trim_x]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def find_sudoku_board(gray, cv2, np):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_square(gray, cv2)

    image_area = gray.shape[0] * gray.shape[1]
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)

    best_quad = None
    for contour in ranked:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            break

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(max(h, 1))
        if 0.75 <= ratio <= 1.25:
            best_quad = approx
            break

    if best_quad is None:
        return _fallback_square(gray, cv2)

    corners = _order_points(best_quad.reshape(4, 2).astype("float32"), np)
    dest = np.array([[0, 0], [899, 0], [899, 899], [0, 899]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dest)
    warped = cv2.warpPerspective(gray, matrix, (900, 900))

    refined = _refine_to_inner_grid(warped, cv2, np)
    return trim_outer_border(refined, cv2, trim_ratio=0.01)
