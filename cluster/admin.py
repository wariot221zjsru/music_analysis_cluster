from django.contrib import admin

from .models import Cluster, ClusterMember, ClusterRun, ClusterSetting, PieceVector


@admin.register(ClusterRun)
class ClusterRunAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'status', 'published',
                    'piece_count', 'cluster_count')
    list_filter = ('status', 'published')
    readonly_fields = ('params',)


@admin.register(Cluster)
class ClusterAdmin(admin.ModelAdmin):
    list_display = ('id', 'run', 'index', 'display_label', 'size')
    list_filter = ('run',)
    search_fields = ('label', 'auto_label')


@admin.register(ClusterMember)
class ClusterMemberAdmin(admin.ModelAdmin):
    list_display = ('cluster', 'piece', 'mode_sim', 'harmony_sim',
                    'melody_sim', 'dist_to_medoid')
    list_filter = ('cluster',)


@admin.register(PieceVector)
class PieceVectorAdmin(admin.ModelAdmin):
    list_display = ('piece', 'status', 'mode_key', 'mode_tonic_pc',
                    'melody_phrase_count', 'computed_at')
    list_filter = ('status', 'mode_uncertain')
    search_fields = ('piece__title', 'mode_key')


@admin.register(ClusterSetting)
class ClusterSettingAdmin(admin.ModelAdmin):
    list_display = ('w_mode', 'w_harmony', 'w_melody', 'top_n',
                    'cluster_threshold', 'updated_at')
