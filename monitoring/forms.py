from django import forms

from .models import Analysis


class AnalysisForm(forms.ModelForm):

    class Meta:
        model = Analysis

        fields = [
            "before_image",
            "after_image",
        ]

        widgets = {
            "before_image": forms.FileInput(
                attrs={
                    "class": "file-input",
                    "accept": "image/*",
                }
            ),
            "after_image": forms.FileInput(
                attrs={
                    "class": "file-input",
                    "accept": "image/*",
                }
            ),
        }