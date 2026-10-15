from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload, name='upload'),
    path('pieces/<int:piece_id>/', views.piece_detail, name='piece_detail'),
    path('pieces/<int:piece_id>/run/', views.run_group, name='run_group'),
    path('pieces/<int:piece_id>/status/', views.piece_status, name='piece_status'),
    path('settings/', views.system_settings, name='settings'),
]
