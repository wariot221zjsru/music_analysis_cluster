from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload, name='upload'),
    path('pieces/<int:piece_id>/', views.piece_detail, name='piece_detail'),
    path('pieces/<int:piece_id>/run/', views.run_group, name='run_group'),
    path('pieces/<int:piece_id>/rerun/', views.rerun_step, name='rerun_step'),
    path('pieces/<int:piece_id>/convert/', views.retry_convert, name='retry_convert'),
    path('pieces/<int:piece_id>/status/', views.piece_status, name='piece_status'),
    # —— 单步骤的历史运行记录（?step=xxx[&run=id]）——
    path('pieces/<int:piece_id>/history/', views.step_history, name='step_history'),
    path('pieces/<int:piece_id>/download/midi/', views.download_file,
         {'kind': 'midi'}, name='download_midi'),
    path('pieces/<int:piece_id>/download/musicxml/', views.download_file,
         {'kind': 'musicxml'}, name='download_musicxml'),
    # —— 结构 / 织体 / 风格（三卡片页面）——
    path('pieces/<int:piece_id>/sts/', views.sts_page, name='sts_page'),
    # —— 自定义问答 ——
    path('pieces/<int:piece_id>/ask/', views.qa_page, name='qa_page'),
    path('pieces/<int:piece_id>/ask/submit/', views.ask_question,
         name='ask_question'),
    path('pieces/<int:piece_id>/ask/status/', views.qa_status,
         name='qa_status'),
    path('pieces/<int:piece_id>/visibility/', views.toggle_visibility,
         name='toggle_visibility'),
    path('settings/', views.system_settings, name='settings'),
]
