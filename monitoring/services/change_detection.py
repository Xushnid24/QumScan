import cv2
import numpy as np


def detect_changes(
    before_path,
    after_path,
    output_path
):
    """
    QumScan Change Detection v3

    Compares two already aligned images.

    Returns:
        change_percentage: float
        significant_objects: int
    """

    # ==========================================
    # 1. LOAD IMAGES
    # ==========================================

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    if before is None:
        raise ValueError(
            "Не удалось прочитать изображение ДО."
        )

    if after is None:
        raise ValueError(
            "Не удалось прочитать изображение ПОСЛЕ."
        )

    height, width = before.shape[:2]

    # Bring AFTER to same pixel size
    after = cv2.resize(
        after,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # ==========================================
    # 2. REDUCE SMALL IMAGE NOISE
    # ==========================================

    before_blur = cv2.GaussianBlur(
        before,
        (7, 7),
        0
    )

    after_blur = cv2.GaussianBlur(
        after,
        (7, 7),
        0
    )

    # ==========================================
    # 3. CONVERT TO LAB COLOR SPACE
    # ==========================================

    before_lab = cv2.cvtColor(
        before_blur,
        cv2.COLOR_BGR2LAB
    )

    after_lab = cv2.cvtColor(
        after_blur,
        cv2.COLOR_BGR2LAB
    )

    # Convert to signed type
    before_float = before_lab.astype(
        np.float32
    )

    after_float = after_lab.astype(
        np.float32
    )

    # ==========================================
    # 4. PIXEL COLOR DIFFERENCE
    # ==========================================

    delta = (
        after_float
        - before_float
    )

    difference_score = np.sqrt(
        np.sum(
            delta ** 2,
            axis=2
        )
    )

    # ==========================================
    # 5. CREATE CHANGE MASK
    # ==========================================

    # Higher value = less sensitive
    CHANGE_THRESHOLD = 38.0

    mask = (
        difference_score
        > CHANGE_THRESHOLD
    ).astype(np.uint8) * 255

    # ==========================================
    # 6. IGNORE OUTER IMAGE BORDER
    # ==========================================

    border_x = max(
        5,
        int(width * 0.01)
    )

    border_y = max(
        5,
        int(height * 0.01)
    )

    mask[:border_y, :] = 0
    mask[-border_y:, :] = 0
    mask[:, :border_x] = 0
    mask[:, -border_x:] = 0

    # ==========================================
    # 7. MORPHOLOGICAL CLEANUP
    # ==========================================

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (17, 17)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    # ==========================================
    # 8. FIND CHANGED REGIONS
    # ==========================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = width * height

    MIN_AREA = (
        image_area * 0.002
    )

    MAX_AREA = (
        image_area * 0.55
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Ignore thin artifacts
        if w < 20 or h < 20:
            continue

        valid_contours.append(
            contour
        )

    # ==========================================
    # 9. SORT LARGEST FIRST
    # ==========================================

    valid_contours = sorted(
        valid_contours,
        key=cv2.contourArea,
        reverse=True
    )

    # ==========================================
    # 10. CALCULATE REAL DETECTED MASK AREA
    # ==========================================

    clean_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.drawContours(
        clean_mask,
        valid_contours,
        -1,
        255,
        thickness=cv2.FILLED
    )

    changed_pixels = cv2.countNonZero(
        clean_mask
    )

    change_percentage = (
        changed_pixels
        / image_area
    ) * 100

    # ==========================================
    # 11. VISUAL RESULT
    # ==========================================

    result = after.copy()

    red_layer = np.zeros_like(
        result
    )

    red_layer[:, :] = (
        0,
        0,
        255
    )

    changed_area = cv2.bitwise_and(
        red_layer,
        red_layer,
        mask=clean_mask
    )

    # Blend red only over detected zones
    blended = cv2.addWeighted(
        result,
        1.0,
        changed_area,
        0.32,
        0
    )

    result[
        clean_mask > 0
    ] = blended[
        clean_mask > 0
    ]

    # ==========================================
    # 12. DRAW PRECISE CONTOURS
    # ==========================================

    cv2.drawContours(
        result,
        valid_contours,
        -1,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    # ==========================================
    # 13. NUMBER EACH ZONE
    # ==========================================

    for number, contour in enumerate(
        valid_contours,
        start=1
    ):

        moments = cv2.moments(
            contour
        )

        if moments["m00"] != 0:

            center_x = int(
                moments["m10"]
                / moments["m00"]
            )

            center_y = int(
                moments["m01"]
                / moments["m00"]
            )

        else:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            center_x = x + w // 2
            center_y = y + h // 2

        cv2.circle(
            result,
            (
                center_x,
                center_y
            ),
            19,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

        label = str(number)

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        text_width = text_size[0]
        text_height = text_size[1]

        cv2.putText(
            result,
            label,
            (
                center_x
                - text_width // 2,
                center_y
                + text_height // 2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ==========================================
    # 14. SAVE RESULT
    # ==========================================

    success = cv2.imwrite(
        output_path,
        result
    )

    if not success:
        raise ValueError(
            "Не удалось сохранить результат анализа."
        )

    significant_objects = len(
        valid_contours
    )

    return (
        round(
            change_percentage,
            2
        ),
        significant_objects
    )