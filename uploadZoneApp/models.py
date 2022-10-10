from django.db import models

class bird(models.Model):
    beakShape = models.CharField(max_length = 180)
    eyeColor = models.CharField(max_length = 180)
    wingsColor = models.CharField(max_length = 180)
    Location = models.CharField(max_length = 180)
    birdColor = models.CharField(max_length = 180)

    def __str__(self):
        return self.Location
