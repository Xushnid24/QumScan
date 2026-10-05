from pathlib import Path

import cv2
import numpy as np
import rasterio

from rasterio.transform import xy
from rasterio.warp import transform


# ============================================================
# QUMSCAN SENTINEL SETTINGS
# ============================================================

# Рабочий порог для выбранного AOI
CHANGE_THRESHOLD = 53.42

# NDVI: фильтр сезонной растительности
VEGETATION_NDVI = 0.35
VEGETATION_DELTA = 0.20

# NDWI: фильтр воды
WATER_NDWI = 0.10

# Минимальный объект
MIN_PIXELS = 20

# Не принимать слишком большой единый контур
MAX_AREA_PERCENT = 0.15

# Максимум зон для отображения
TOP_N = 5

# Убираем артефакты по краям
EDGE_MARGIN = 12


# ============================================================
# HELPERS
# ============================================================

def read_band(path):
    """
    Read one Sentinel-2 spectral band as float32.
    """

    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)


def calculate_index(band_a, band_b):
    """
    Generic normalized difference index:

    (A - B) / (A + B)

    NDVI:
        A = NIR
        B = RED

    NDWI:
        A = GREEN
        B = NIR
    """

    denominator = band_a + band_b

    result = np.zeros_like(
        band_a,
        dtype=np.float32,
    )

    valid = denominator != 0

    result[valid] = (
        band_a[valid] - band_b[valid]
    ) / denominator[valid]

    return result


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_satellite_changes(
    before_path,
    after_path,
    geotiff_path,
    output_path,
):
    """
    Detect significant dry-surface changes between two
    aligned Sentinel-2 images.

    Pipeline:

    1. RGB LAB change detection
    2. border artifact filtering
    3. morphology
    4. NDVI vegetation filtering
    5. NDWI water filtering
    6. contour filtering
    7. area calculation
    8. coordinate conversion
    9. result visualization

    Returns:
        change_percentage
        total_area_m2
        zones
    """

    # ========================================================
    # 1. PATHS
    # ========================================================

    before_path = Path(before_path)
    after_path = Path(after_path)
    geotiff_path = Path(geotiff_path)
    output_path = Path(output_path)

    spectral_dir = geotiff_path.parent

    before_red_path = (
        spectral_dir
        / "before_red.tif"
    )

    before_green_path = (
        spectral_dir
        / "before_green.tif"
    )

    before_nir_path = (
        spectral_dir
        / "before_nir.tif"
    )

    after_red_path = (
        spectral_dir
        / "after_red.tif"
    )

    after_green_path = (
        spectral_dir
        / "after_green.tif"
    )

    after_nir_path = (
        spectral_dir
        / "after_nir.tif"
    )

    required_files = [
        before_path,
        after_path,
        geotiff_path,
        before_red_path,
        before_green_path,
        before_nir_path,
        after_red_path,
        after_green_path,
        after_nir_path,
    ]

    for path in required_files:

        if not path.exists():
            raise RuntimeError(
                f"Required Sentinel file not found: {path}"
            )


    # ========================================================
    # 2. LOAD RGB
    # ========================================================

    before = cv2.imread(
        str(before_path)
    )

    after = cv2.imread(
        str(after_path)
    )

    if before is None:
        raise RuntimeError(
            f"Could not open BEFORE: {before_path}"
        )

    if after is None:
        raise RuntimeError(
            f"Could not open AFTER: {after_path}"
        )

    if before.shape != after.shape:

        after = cv2.resize(
            after,
            (
                before.shape[1],
                before.shape[0],
            ),
            interpolation=cv2.INTER_AREA,
        )

    h, w = before.shape[:2]


    # ========================================================
    # 3. RGB CHANGE DETECTION
    # ========================================================

    before_blur = cv2.GaussianBlur(
        before,
        (5, 5),
        0,
    )

    after_blur = cv2.GaussianBlur(
        after,
        (5, 5),
        0,
    )

    before_lab = cv2.cvtColor(
        before_blur,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    after_lab = cv2.cvtColor(
        after_blur,
        cv2.COLOR_BGR2LAB,
    ).astype(
        np.float32
    )

    diff = (
        before_lab
        - after_lab
    )

    score = np.sqrt(
        np.sum(
            diff ** 2,
            axis=2,
        )
    )

    change_mask = (
        score
        > CHANGE_THRESHOLD
    ).astype(
        np.uint8
    ) * 255


    # ========================================================
    # 4. REMOVE BORDER ARTIFACTS
    # ========================================================

    change_mask[
        :EDGE_MARGIN,
        :
    ] = 0

    change_mask[
        -EDGE_MARGIN:,
        :
    ] = 0

    change_mask[
        :,
        :EDGE_MARGIN
    ] = 0

    change_mask[
        :,
        -EDGE_MARGIN:
    ] = 0


    # ========================================================
    # 5. MORPHOLOGY
    # ========================================================

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1,
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=1,
    )


    # ========================================================
    # 6. LOAD SPECTRAL BANDS
    # ========================================================

    before_red = read_band(
        before_red_path
    )

    before_green = read_band(
        before_green_path
    )

    before_nir = read_band(
        before_nir_path
    )

    after_red = read_band(
        after_red_path
    )

    after_green = read_band(
        after_green_path
    )

    after_nir = read_band(
        after_nir_path
    )

    expected_shape = (
        h,
        w
    )

    spectral_bands = {
        "before_red": before_red,
        "before_green": before_green,
        "before_nir": before_nir,
        "after_red": after_red,
        "after_green": after_green,
        "after_nir": after_nir,
    }

    for name, band in spectral_bands.items():

        if band.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape {band.shape} "
                f"does not match RGB shape {expected_shape}"
            )


    # ========================================================
    # 7. NDVI
    # ========================================================

    ndvi_before = calculate_index(
        before_nir,
        before_red,
    )

    ndvi_after = calculate_index(
        after_nir,
        after_red,
    )

    ndvi_delta = (
        ndvi_after
        - ndvi_before
    )


    # ========================================================
    # 8. NDWI
    # ========================================================

    ndwi_before = calculate_index(
        before_green,
        before_nir,
    )

    ndwi_after = calculate_index(
        after_green,
        after_nir,
    )


    # ========================================================
    # 9. FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        change_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = list(
        contours
    )

    contours.sort(
        key=cv2.contourArea,
        reverse=True,
    )

    image_area_pixels = (
        h * w
    )


    # ========================================================
    # 10. OPEN GEOTIFF
    # ========================================================

    geo = rasterio.open(
        str(geotiff_path)
    )

    geo_transform = (
        geo.transform
    )

    geo_crs = (
        geo.crs
    )

    pixel_area_m2 = abs(
        geo_transform.a
        * geo_transform.e
        -
        geo_transform.b
        * geo_transform.d
    )


    # ========================================================
    # 11. FILTER CANDIDATES
    # ========================================================

    accepted = []

    rejected_vegetation = 0
    rejected_water = 0


    for contour in contours:

        contour_area = cv2.contourArea(
            contour
        )

        if contour_area < MIN_PIXELS:
            continue

        if (
            contour_area
            > image_area_pixels
            * MAX_AREA_PERCENT
        ):
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < 4 or ch < 4:
            continue

        # ----------------------------------
        # Edge filter
        # ----------------------------------

        if (
            x <= EDGE_MARGIN
            or y <= EDGE_MARGIN
            or x + cw
            >= w - EDGE_MARGIN
            or y + ch
            >= h - EDGE_MARGIN
        ):
            continue


        # ----------------------------------
        # Zone mask
        # ----------------------------------

        zone_mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        cv2.drawContours(
            zone_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        pixels_mask = (
            zone_mask > 0
        )

        pixel_count = int(
            np.count_nonzero(
                pixels_mask
            )
        )

        if pixel_count == 0:
            continue


        # ----------------------------------
        # NDVI statistics
        # ----------------------------------

        mean_ndvi_before = float(
            np.mean(
                ndvi_before[
                    pixels_mask
                ]
            )
        )

        mean_ndvi_after = float(
            np.mean(
                ndvi_after[
                    pixels_mask
                ]
            )
        )

        mean_ndvi_delta = (
            mean_ndvi_after
            - mean_ndvi_before
        )


        # ----------------------------------
        # NDWI statistics
        # ----------------------------------

        mean_ndwi_before = float(
            np.mean(
                ndwi_before[
                    pixels_mask
                ]
            )
        )

        mean_ndwi_after = float(
            np.mean(
                ndwi_after[
                    pixels_mask
                ]
            )
        )


        # ----------------------------------
        # VEGETATION FILTER
        # ----------------------------------

        is_vegetation = (
            mean_ndvi_after
            > VEGETATION_NDVI
            and
            mean_ndvi_delta
            > VEGETATION_DELTA
        )

        if is_vegetation:

            rejected_vegetation += 1
            continue


        # ----------------------------------
        # WATER FILTER
        # ----------------------------------

        is_water = (
            mean_ndwi_before
            > WATER_NDWI
            or
            mean_ndwi_after
            > WATER_NDWI
        )

        if is_water:

            rejected_water += 1
            continue


        # ----------------------------------
        # ACCEPT DRY-SURFACE CANDIDATE
        # ----------------------------------

        accepted.append(
            {
                "contour": contour,
                "zone_mask": zone_mask,
                "pixel_count": pixel_count,

                "ndvi_before":
                    mean_ndvi_before,

                "ndvi_after":
                    mean_ndvi_after,

                "ndvi_delta":
                    mean_ndvi_delta,

                "ndwi_before":
                    mean_ndwi_before,

                "ndwi_after":
                    mean_ndwi_after,
            }
        )

        if len(accepted) >= TOP_N:
            break


    # ========================================================
    # 12. PREPARE RESULT
    # ========================================================

    result = after.copy()

    overlay = after.copy()

    zones = []

    zone_centers = []

    total_changed_pixels = 0


    # ========================================================
    # 13. CALCULATE ZONES
    # ========================================================

    for number, candidate in enumerate(
        accepted,
        start=1,
    ):

        contour = (
            candidate["contour"]
        )

        pixel_count = (
            candidate["pixel_count"]
        )

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = int(
            moments["m10"]
            / moments["m00"]
        )

        cy = int(
            moments["m01"]
            / moments["m00"]
        )


        # ----------------------------------
        # Area
        # ----------------------------------

        area_m2 = (
            pixel_count
            * pixel_area_m2
        )

        total_changed_pixels += (
            pixel_count
        )


        # ----------------------------------
        # Pixel -> projected CRS
        # ----------------------------------

        x_coord, y_coord = xy(
            geo_transform,
            cy,
            cx,
            offset="center",
        )


        # ----------------------------------
        # Projected CRS -> WGS84
        # ----------------------------------

        lon_list, lat_list = transform(
            geo_crs,
            "EPSG:4326",
            [x_coord],
            [y_coord],
        )

        lon = float(
            lon_list[0]
        )

        lat = float(
            lat_list[0]
        )


        # ----------------------------------
        # Django zone object
        # ----------------------------------

        zones.append(
            {
                "number": number,

                "area_m2": round(
                    area_m2,
                    2,
                ),

                "latitude": round(
                    lat,
                    6,
                ),

                "longitude": round(
                    lon,
                    6,
                ),

                "ndvi_before": round(
                    candidate[
                        "ndvi_before"
                    ],
                    3,
                ),

                "ndvi_after": round(
                    candidate[
                        "ndvi_after"
                    ],
                    3,
                ),

                "ndvi_delta": round(
                    candidate[
                        "ndvi_delta"
                    ],
                    3,
                ),

                "ndwi_before": round(
                    candidate[
                        "ndwi_before"
                    ],
                    3,
                ),

                "ndwi_after": round(
                    candidate[
                        "ndwi_after"
                    ],
                    3,
                ),

                "status":
                    "Требует проверки",
            }
        )


        zone_centers.append(
            (
                number,
                contour,
                cx,
                cy,
            )
        )


        # ----------------------------------
        # Red overlay
        # ----------------------------------

        cv2.drawContours(
            overlay,
            [contour],
            -1,
            (0, 0, 255),
            thickness=cv2.FILLED,
        )


    # ========================================================
    # 14. BLEND ONCE
    # ========================================================

    result = cv2.addWeighted(
        after,
        0.82,
        overlay,
        0.18,
        0,
    )


    # ========================================================
    # 15. DRAW CONTOURS + NUMBERS
    # ========================================================

    for (
        number,
        contour,
        cx,
        cy,
    ) in zone_centers:

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 0, 255),
            2,
        )

        cv2.circle(
            result,
            (cx, cy),
            9,
            (0, 0, 255),
            -1,
        )

        text = str(
            number
        )

        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1,
        )[0]

        text_x = (
            cx
            - text_size[0] // 2
        )

        text_y = (
            cy
            + text_size[1] // 2
        )

        cv2.putText(
            result,
            text,
            (
                text_x,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


    # ========================================================
    # 16. CLOSE GEOTIFF
    # ========================================================

    geo.close()


    # ========================================================
    # 17. SAVE RESULT
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        result,
    )

    if not success:
        raise RuntimeError(
            f"Could not save result: {output_path}"
        )


    # ========================================================
    # 18. STATISTICS
    # ========================================================

    change_percentage = (
        total_changed_pixels
        / image_area_pixels
        * 100
    )

    total_area_m2 = (
        total_changed_pixels
        * pixel_area_m2
    )


    return (
        round(
            change_percentage,
            2,
        ),

        round(
            total_area_m2,
            2,
        ),

        zones,
    )