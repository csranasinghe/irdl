# todo/todo_api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import birdSerializer
from .predict import main

class TodoListApiView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = birdSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            print(serializer.data)
            list_value =[serializer.data['beakShape'],
                        serializer.data['eyeColor'],
                        serializer.data['wingsColor'],
                        serializer.data['Location'],
                        serializer.data['birdColor']]
            return Response(main(list_value), status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # def get(self, request, *args, **kwargs):
    #     snippets = sensors.objects.all()
    #     serializer = sensorSerializer(snippets, many=True)
    #     return Response(serializer.data)