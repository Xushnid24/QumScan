from django.urls import path

from . import views


urlpatterns = [

    path(
        "",
        views.dashboard,
        name="dashboard"
    ),

    path(
        "analysis/<int:pk>/",
        views.analysis_result,
        name="analysis_result"
    ),

    path(
    "satellite-demo/",
    views.satellite_demo,
    name="satellite_demo",
    ),

]