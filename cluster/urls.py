"""聚类模块的 URL。

只读页挂在 /cluster/，管理员操作全部收在 /cluster/admin/ 下。
"""
from django.urls import path

from . import views

app_name = 'cluster'

urlpatterns = [
    path('', views.overview, name='cluster_overview'),
    path('admin/', views.admin_panel, name='cluster_admin'),
    path('admin/rebuild/', views.admin_rebuild, name='cluster_rebuild'),
    path('admin/run/', views.admin_run, name='cluster_run'),
    path('admin/settings/', views.admin_settings, name='cluster_settings'),
    path('admin/publish/<int:run_id>/', views.admin_publish,
         name='cluster_publish'),
    path('admin/rename/<int:cluster_id>/', views.admin_rename,
         name='cluster_rename'),
    path('c/<int:cluster_id>/', views.cluster_detail, name='cluster_detail'),
    path('piece/<int:piece_id>/', views.similar, name='cluster_similar'),
]
