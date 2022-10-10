from django.urls import path
from uploadZoneApp import views
from django.urls import path, include

urlpatterns = [
    path('', views.TodoListApiView.as_view()),
]
 
# from device import views

# snippets/urls.py
# from django.urls import path
# from device import views

# urlpatterns = [
#     path('snippets/', views.SnippetList.as_view()),
#     path('snippets/<int:pk>/', views.SnippetDetail.as_view()),
# ]

