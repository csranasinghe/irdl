from rest_framework import serializers
from .models import bird
class birdSerializer(serializers.ModelSerializer):
    class Meta:
        model = bird
        fields = ["beakShape", "eyeColor", "wingsColor", "Location", "birdColor"]

