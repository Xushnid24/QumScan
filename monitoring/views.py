import os
import shutil

from pathlib import Path

from django.conf import settings
from django.shortcuts import (
    render,
    redirect,
    get_object_or_404,
)

from .forms import AnalysisForm
from .models import Analysis

from .services.change_detection import (
    detect_changes,
)

from .services.satellite_change_detection import (
    detect_satellite_changes,
)


# ============================================================
# NORMAL USER IMAGE ANALYSIS
# ============================================================

def dashboard(request):

    if request.method == "POST":

        form = AnalysisForm(
            request.POST,
            request.FILES,
        )

        if form.is_valid():

            analysis = form.save(
                commit=False
            )

            analysis.status = "processing"

            analysis.save()

            try:

                before_path = (
                    analysis.before_image.path
                )

                after_path = (
                    analysis.after_image.path
                )

                output_directory = os.path.join(
                    settings.MEDIA_ROOT,
                    "analyses",
                    "results",
                )

                os.makedirs(
                    output_directory,
                    exist_ok=True,
                )

                output_filename = (
                    f"result_{analysis.id}.jpg"
                )

                output_path = os.path.join(
                    output_directory,
                    output_filename,
                )

                (
                    change_percentage,
                    object_count,
                ) = detect_changes(
                    before_path,
                    after_path,
                    output_path,
                )

                analysis.result_image = (
                    f"analyses/results/"
                    f"{output_filename}"
                )

                analysis.change_percentage = (
                    change_percentage
                )

                analysis.status = "completed"

                analysis.save()

            except Exception as error:

                print(
                    "QumScan analysis error:",
                    error,
                )

                analysis.status = "failed"

                analysis.save()

            return redirect(
                "analysis_result",
                pk=analysis.pk,
            )

    else:

        form = AnalysisForm()


    analyses = (
        Analysis.objects
        .order_by("-created_at")[:5]
    )


    context = {
        "form": form,
        "analyses": analyses,
    }


    return render(
        request,
        "monitoring/dashboard.html",
        context,
    )


# ============================================================
# NORMAL ANALYSIS RESULT
# ============================================================

def analysis_result(request, pk):

    analysis = get_object_or_404(
        Analysis,
        pk=pk,
    )

    context = {
        "analysis": analysis,
    }

    return render(
        request,
        "monitoring/result.html",
        context,
    )


# ============================================================
# REAL SENTINEL-2 MONITORING
# ============================================================

def satellite_demo(request):

    base_dir = Path(
        settings.BASE_DIR
    )


    sentinel_dir = (
        base_dir
        / "sentinel_output"
    )


    before_path = (
        sentinel_dir
        / "before_rgb.png"
    )

    after_path = (
        sentinel_dir
        / "after_rgb.png"
    )

    geotiff_path = (
        sentinel_dir
        / "after_rgb.tif"
    )


    spectral_files = [

        sentinel_dir
        / "before_red.tif",

        sentinel_dir
        / "before_green.tif",

        sentinel_dir
        / "before_nir.tif",

        sentinel_dir
        / "after_red.tif",

        sentinel_dir
        / "after_green.tif",

        sentinel_dir
        / "after_nir.tif",
    ]


    required_files = [
        before_path,
        after_path,
        geotiff_path,
        *spectral_files,
    ]


    for path in required_files:

        if not path.exists():

            raise RuntimeError(
                f"Sentinel file missing: {path}"
            )


    # ========================================================
    # COPY CURRENT RGB IMAGES TO STATIC
    # ========================================================

    static_demo_dir = (
        base_dir
        / "monitoring"
        / "static"
        / "monitoring"
        / "demo"
    )


    static_demo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    static_before = (
        static_demo_dir
        / "before_rgb.png"
    )

    static_after = (
        static_demo_dir
        / "after_rgb.png"
    )


    shutil.copy2(
        before_path,
        static_before,
    )

    shutil.copy2(
        after_path,
        static_after,
    )


    # ========================================================
    # RESULT OUTPUT
    # ========================================================

    output_dir = (
        Path(settings.MEDIA_ROOT)
        / "analyses"
        / "results"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path = (
        output_dir
        / "sentinel_demo_result.jpg"
    )


    # ========================================================
    # RUN SATELLITE DETECTOR
    # ========================================================

    (
        change_percentage,
        total_area_m2,
        zones,
    ) = detect_satellite_changes(
        before_path,
        after_path,
        geotiff_path,
        output_path,
    )


    # ========================================================
    # TEMPLATE CONTEXT
    # ========================================================

    context = {

        "before_image": (
            "/static/monitoring/demo/"
            "before_rgb.png"
        ),

        "after_image": (
            "/static/monitoring/demo/"
            "after_rgb.png"
        ),

        "result_image": (
            settings.MEDIA_URL
            + "analyses/results/"
            + "sentinel_demo_result.jpg"
        ),

        "change_percentage":
            change_percentage,

        "total_area_m2":
            total_area_m2,

        "zones":
            zones,

        "before_date":
            "29.05.2026",

        "after_date":
            "12.09.2026",

        "satellite":
            "Sentinel-2",

        "resolution":
            "10 м",

        "analysis_type":
            "RGB + NDVI + NDWI",

        "disclaimer": (
            "QumScan выявляет значимые изменения "
            "земной поверхности и формирует зоны "
            "для последующей проверки. "
            "Результат не является доказательством "
            "незаконной деятельности."
        ),
    }


    return render(
        request,
        "monitoring/satellite_demo.html",
        context,
    )