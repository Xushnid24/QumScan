from django.db import models


class Analysis(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    before_image = models.ImageField(
        upload_to="analyses/before/"
    )

    after_image = models.ImageField(
        upload_to="analyses/after/"
    )

    result_image = models.ImageField(
        upload_to="analyses/results/",
        blank=True,
        null=True
    )

    change_percentage = models.FloatField(
        blank=True,
        null=True
    )

    change_area_m2 = models.FloatField(
        blank=True,
        null=True
    )

    latitude = models.FloatField(
        blank=True,
        null=True
    )

    longitude = models.FloatField(
        blank=True,
        null=True
    )

    ai_report = models.TextField(
        blank=True
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending"
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"QumScan Analysis #{self.id}"

class ChangeZone(models.Model):
    analysis = models.ForeignKey(
        Analysis,
        on_delete=models.CASCADE,
        related_name="zones"
    )

    zone_number = models.PositiveIntegerField()

    area_m2 = models.FloatField()

    latitude = models.FloatField()
    longitude = models.FloatField()

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["zone_number"]

    def __str__(self):
        return (
            f"Zone #{self.zone_number} "
            f"— {self.area_m2:.0f} m²"
        )