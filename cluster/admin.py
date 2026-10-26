"""聚类模块在 Django 内置后台的注册。

**执行聚类不在这里** —— 后台只能查看 / 清理数据，真正跑聚类去
``/cluster/admin/`` 管理台（有「重建特征」「执行聚类」「发布」按钮）。

这里做的两处限制都是为了少踩坑：
  * ``ClusterRun`` 禁止手动新增 —— 手写一条既没有聚类成员、
    又会让总览页挑到一条假记录（"聚类需要几步"这个问题就是这么来的）；
  * 列表页顶部加了管理台入口按钮 + 发布/撤回动作。
"""
from django.contrib import admin
from django.utils.html import format_html

from .models import Cluster, ClusterMember, ClusterRun, ClusterSetting, PieceVector


@admin.register(ClusterRun)
class ClusterRunAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'status', 'published',
                    'piece_count', 'cluster_count', 'executor', 'open_console')
    list_filter = ('status', 'published')
    readonly_fields = ('params',)
    actions = ('make_published', 'make_unpublished')
    change_list_template = 'admin/cluster/clusterrun/change_list.html'

    def has_add_permission(self, request):
        """跑聚类必须走管理台；手动新增一条 run 没有聚类成员，只会误导。"""
        return False

    @admin.display(description='执行人')
    def executor(self, obj):
        return obj.created_by.username if obj.created_by_id else '—'

    @admin.display(description='')
    def open_console(self, obj):
        return format_html('<a class="button" href="/cluster/admin/">管理台</a>')

    @admin.action(description='发布所选聚类结果（普通用户可见）')
    def make_published(self, request, queryset):
        n = queryset.update(published=True)
        self.message_user(request, f'已发布 {n} 次聚类结果')

    @admin.action(description='撤回所选聚类结果（普通用户不可见）')
    def make_unpublished(self, request, queryset):
        n = queryset.update(published=False)
        self.message_user(request, f'已撤回 {n} 次聚类结果')


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
