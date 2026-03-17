from __future__ import annotations


def _center_and_resize(component, cv2, np):
    target = 28
    side = max(component.shape) + 10
    canvas = np.zeros((side, side), dtype=np.uint8)

    y_offset = (side - component.shape[0]) // 2
    x_offset = (side - component.shape[1]) // 2
    canvas[y_offset : y_offset + component.shape[0], x_offset : x_offset + component.shape[1]] = component

    resized = cv2.resize(canvas, (target, target), interpolation=cv2.INTER_AREA)
    return cv2.bitwise_not(resized)


def cell_digit_candidates(cell_gray, cv2, np):
    h, w = cell_gray.shape

    min_fill = 0.018
    max_fill = 0.55
    min_contour_ratio = 0.020
    min_height_ratio = 0.16
    min_width_ratio = 0.03
    min_ar = 0.05
    max_ar = 2.2
    centre_margin = 0.14
    edge_margin_ratio = 0.04

    candidates = []
    seen_hashes = set()

    for margin_ratio in (0.12, 0.14, 0.16):
        margin_y = max(3, int(h * margin_ratio))
        margin_x = max(3, int(w * margin_ratio))
        inner = cell_gray[margin_y : h - margin_y, margin_x : w - margin_x]
        if inner.size == 0:
            continue

        ih, iw = inner.shape
        blur = cv2.GaussianBlur(inner, (3, 3), 0)

        adaptive_variants = []
        for block_size, c_value in ((11, 2), (15, 3), (11, 4)):
            adaptive_variants.append(
                cv2.adaptiveThreshold(
                    blur,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    block_size,
                    c_value,
                )
            )

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants = adaptive_variants + [otsu]

        kernel = np.ones((2, 2), dtype=np.uint8)
        processed_masks = []
        for variant in variants:
            processed_masks.append(variant)
            processed_masks.append(cv2.dilate(variant, kernel, iterations=1))
            processed_masks.append(cv2.erode(variant, kernel, iterations=1))

        for base in processed_masks:
            fill = cv2.countNonZero(base) / float(base.size)
            if fill < min_fill or fill > max_fill:
                continue

            min_px = ih * iw * min_contour_ratio
            cnts, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            contour = None
            contour_area = 0.0
            x = y = cw = ch = 0

            for candidate_contour in sorted(cnts, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(candidate_contour)
                if area < min_px:
                    continue

                bx, by, bw, bh = cv2.boundingRect(candidate_contour)
                if bw <= 2 or bh <= 2:
                    continue
                if bh < ih * min_height_ratio or bw < iw * min_width_ratio:
                    continue

                min_x = iw * edge_margin_ratio
                min_y = ih * edge_margin_ratio
                if bx <= min_x or by <= min_y:
                    continue

                ar = bw / float(max(bh, 1))
                if ar < min_ar or ar > max_ar:
                    continue

                M = cv2.moments(candidate_contour)
                if M["m00"] == 0:
                    continue
                true_cx = M["m10"] / M["m00"]
                true_cy = M["m01"] / M["m00"]
                cx_rel = true_cx / float(iw)
                cy_rel = true_cy / float(ih)
                if not (
                    centre_margin < cx_rel < 1 - centre_margin
                    and centre_margin < cy_rel < 1 - centre_margin
                ):
                    continue

                solidity = area / float(max(bw * bh, 1))
                if solidity < 0.18:
                    continue

                contour = candidate_contour
                contour_area = area
                x, y, cw, ch = bx, by, bw, bh
                break

            if contour is None:
                continue

            component = base[y : y + ch, x : x + cw]
            digit_28 = _center_and_resize(component, cv2, np)

            tiny = (digit_28 > 128).astype(np.uint8)
            candidate_hash = tiny.tobytes()
            if candidate_hash in seen_hashes:
                continue
            seen_hashes.add(candidate_hash)

            component_ar = cw / float(max(ch, 1))
            candidates.append((digit_28, component_ar))

    return candidates
