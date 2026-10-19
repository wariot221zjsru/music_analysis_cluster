from django.contrib import admin

from .models import AnalysisResult, AnalysisRun, Piece, QAQuestion, SystemConfig

admin.site.site_header = '音乐分析和处理智能体 · 管理后台'


@admin.register(Piece)
class PieceAdmin(admin.ModelAdmin):
    list_display = ('title', 'owner', 'source_type', 'contains_voice',
                    'is_public', 'source_purged', 'created_at')
    list_filter = ('source_type', 'contains_voice', 'is_public',
                   'source_purged', 'created_at')
    search_fields = ('title', 'owner__username')


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ('piece', 'step', 'status', 'started_at', 'finished_at')
    list_filter = ('step', 'status')
    search_fields = ('piece__title',)


@admin.register(AnalysisRun)
class AnalysisRunAdmin(admin.ModelAdmin):
    list_display = ('piece', 'step', 'seq', 'status', 'trigger',
                    'started_at', 'finished_at', 'archived_at')
    list_filter = ('step', 'status')
    search_fields = ('piece__title',)
    readonly_fields = ('archived_at',)


@admin.register(QAQuestion)
class QAQuestionAdmin(admin.ModelAdmin):
    list_display = ('piece', 'user', 'status', 'created_at', 'finished_at')
    list_filter = ('status', 'created_at')
    search_fields = ('piece__title', 'question', 'answer')


@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    list_display = ('polyphony_threshold', 'exchange_min_switches', 'updated_at')
