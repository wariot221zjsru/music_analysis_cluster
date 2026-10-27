"""音乐聚类的数据模型：特征向量、聚类运行、类与成员。

与 core（单作品分析）的分工
---------------------------
core 回答"这一首曲子是什么"（调性 / 和弦 / 主旋律线…）；本 app 回答
"这些曲子彼此像不像、该分成几类"。**只读** core 已有的分析结果来算特征，
不回写、不修改 core 的任何模块（见 features.py）。

三轴向量
--------
  调式 mode     12 维音级分布，**先按主音对齐、再直接比余弦**
  和声 harmony  级数 7 + 性质 4 + 密度 1 = 12 维
  旋律 melody   乐句级（一乐句一向量），比对时取"最相似乐句对"的最大值

调式为什么"主音对齐"而不是"循环移位取最大"
------------------------------------------
多利亚其实就是自然大调的循环移位（D 多利亚与 C 自然大调的音级集合相同），
若按"12 次循环移位取最大余弦"比，多利亚会被判成和自然大调一类 —— 分不开。
改成先确定主音、把音级分布旋转到主音为 0、再直接比余弦（不再移位）：

  C 大调 / D 大调 / E 大调    对齐后向量相同 → 一类（同调式、不同调高）
  自然大调 / 多利亚 / 立底亚   对齐后向量不同 → 能分开（真正的调式差异）

一次对齐同时满足"不同调高归一"与"不同调式要能区分"这两条要求。

pgvector 加速层（可选）
----------------------
三轴向量**主存储仍是 JSONField**（PieceVector），保证任何环境都能跑。
若 PostgreSQL 装了 pgvector 扩展，index.py 会额外维护三张 raw-SQL 加速表
（见 index.py 的 PgBackend 与 0002 迁移）：查询走原生 <=>（余弦距离），
检测不到扩展则自动回退纯 Python 暴力检索，功能一致、结果一致。
"""
from django.conf import settings
from django.db import models

from core.models import Piece


class ClusterSetting(models.Model):
    """聚类参数（单例，管理员可调）。"""

    class Agg(models.TextChoices):
        MAX = 'max', '最相似乐句对（max）'
        MEAN_MAX = 'mean_max', '各乐句最佳匹配均值（更稳）'

    w_mode = models.FloatField('调式轴权重', default=0.30)
    w_harmony = models.FloatField('和声轴权重', default=0.30)
    w_melody = models.FloatField('旋律轴权重', default=0.40)
    top_n = models.PositiveIntegerField('各轴 Top-N', default=20)
    rrf_k = models.PositiveIntegerField('RRF 平滑常数 k', default=60)
    phrase_top_m = models.PositiveIntegerField(
        '旋律乐句粗排：每个乐句取几条', default=10,
        help_text='旋律轴先按乐句向量粗排，再对候选曲做乐句级精排')
    melody_agg = models.CharField(
        '旋律曲级聚合', max_length=8, choices=Agg.choices, default=Agg.MAX)
    cluster_threshold = models.FloatField(
        '聚类切分距离阈值', default=0.45,
        help_text='层次聚类在此距离处切一刀；调小 → 类更多更碎')
    feature_version = models.PositiveIntegerField('特征版本', default=1)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        verbose_name = '聚类参数'
        verbose_name_plural = '聚类参数'

    def __str__(self):
        return '聚类参数'

    @classmethod
    def get(cls) -> 'ClusterSetting':
        obj = cls.objects.first()
        if obj is None:
            obj = cls.objects.create()
        return obj


# ---------------------------------------------------------------- 参数展示

#: 参数的展示顺序与中文名（历史页 / 运行详情页共用）
PARAM_LABELS = [
    ('w_mode', '调式轴权重'),
    ('w_harmony', '和声轴权重'),
    ('w_melody', '旋律轴权重'),
    ('top_n', '各轴 Top-N'),
    ('rrf_k', 'RRF 常数 k'),
    ('phrase_top_m', '旋律乐句粗排（每乐句取几条）'),
    ('melody_agg', '旋律曲级聚合方式'),
    ('cluster_threshold', '聚类切分阈值'),
]

#: melody_agg 的中文含义
AGG_DISPLAY = {
    'max': '最相似乐句对（max）',
    'mean_max': '各乐句最佳匹配均值（更稳）',
}


def param_rows(params: dict) -> list:
    """把参数快照 dict 转成页面友好的 [(名称, 展示值)] 列表。"""
    params = params or {}
    rows = []
    for key, label in PARAM_LABELS:
        v = params.get(key)
        if key == 'melody_agg':
            v = AGG_DISPLAY.get(v, v or '—')
        if v is None:
            v = '—'
        rows.append((label, v))
    return rows


class ClusterSettingLog(models.Model):
    """一次聚类参数保存的记录（审计）。

    ClusterSetting 是会被覆盖的单例；本表把**每次**保存的完整参数留档，
    谁在什么时候把参数调成了什么，事后可查（历史页对所有用户可见）。
    """

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True,
        related_name='cluster_setting_logs', verbose_name='操作人')
    created_at = models.DateTimeField('保存时间', auto_now_add=True)
    w_mode = models.FloatField('调式轴权重')
    w_harmony = models.FloatField('和声轴权重')
    w_melody = models.FloatField('旋律轴权重')
    top_n = models.PositiveIntegerField('各轴 Top-N')
    rrf_k = models.PositiveIntegerField('RRF 平滑常数 k')
    phrase_top_m = models.PositiveIntegerField('旋律乐句粗排：每个乐句取几条')
    melody_agg = models.CharField('旋律曲级聚合', max_length=8)
    cluster_threshold = models.FloatField('聚类切分距离阈值')
    feature_version = models.PositiveIntegerField('特征版本', default=1)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '聚类参数记录'
        verbose_name_plural = '聚类参数记录'

    def __str__(self):
        return f'参数记录 #{self.pk} · {self.created_at:%Y-%m-%d %H:%M}'

    @property
    def params_dict(self) -> dict:
        return {k: getattr(self, k) for k, _ in PARAM_LABELS}

    @property
    def param_rows(self) -> list:
        return param_rows(self.params_dict)

    @property
    def melody_agg_display(self) -> str:
        return AGG_DISPLAY.get(self.melody_agg, self.melody_agg)


class PieceVector(models.Model):
    """一首作品的三轴特征（聚类 / 检索的最小单元）。"""

    class Status(models.TextChoices):
        PENDING = 'pending', '待提取'
        DONE = 'done', '已提取'
        PARTIAL = 'partial', '部分轴缺失'
        ERROR = 'error', '失败'

    piece = models.OneToOneField(
        Piece, on_delete=models.CASCADE, related_name='vector',
        verbose_name='作品')
    status = models.CharField(
        '状态', max_length=8, choices=Status.choices, default=Status.PENDING)
    error = models.TextField('错误信息', blank=True)

    # ---- 调式轴：12 维，已按主音对齐 ----
    mode_vec = models.JSONField('调式向量（12 维，主音对齐）', null=True, blank=True)
    mode_key = models.CharField('主调（原文）', max_length=64, blank=True)
    mode_tonic_pc = models.IntegerField(
        '主音音级', null=True, blank=True,
        help_text='0=C … 11=B；为空表示主音没解析出来，该轴降级为循环移位比对')
    mode_uncertain = models.BooleanField(
        '调式不确定（主音未知）', default=False)

    # ---- 和声轴：级数 7 + 性质 4 + 密度 1 ----
    harmony_vec = models.JSONField('和声向量（12 维）', null=True, blank=True)
    harmony_chord_count = models.IntegerField('和弦窗口数', null=True, blank=True)

    # ---- 旋律轴：乐句级 ----
    # [{ivs, rhythm, pc, vec, count}]，比对时取"最相似乐句对"的最大值
    melody_phrases = models.JSONField('乐句指纹列表', null=True, blank=True)
    melody_phrase_count = models.IntegerField('乐句数', default=0)
    melody_note_count = models.IntegerField('旋律音符数', default=0)

    feature_version = models.IntegerField('特征版本', default=1)
    computed_at = models.DateTimeField('提取时间', auto_now=True)

    class Meta:
        verbose_name = '作品特征'
        verbose_name_plural = '作品特征'

    def __str__(self):
        return f'{self.piece.title} · 特征 · {self.get_status_display()}'

    @property
    def has_mode(self) -> bool:
        return bool(self.mode_vec)

    @property
    def has_harmony(self) -> bool:
        return bool(self.harmony_vec)

    @property
    def has_melody(self) -> bool:
        return bool(self.melody_phrases)

    @property
    def axes_ready(self) -> list:
        """已就绪的轴名列表（缺轴时聚类按可用轴加权，见 clustering.py）。"""
        out = []
        if self.has_mode:
            out.append('mode')
        if self.has_harmony:
            out.append('harmony')
        if self.has_melody:
            out.append('melody')
        return out


class ClusterRun(models.Model):
    """一次聚类运行（管理员触发）。"""

    class Status(models.TextChoices):
        RUNNING = 'running', '运行中'
        DONE = 'done', '已完成'
        ERROR = 'error', '失败'

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True,
        related_name='cluster_runs', verbose_name='执行人')
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    finished_at = models.DateTimeField('结束时间', null=True, blank=True)
    status = models.CharField(
        '状态', max_length=8, choices=Status.choices, default=Status.RUNNING)
    # 参数快照：这次是用哪套权重 / 阈值跑的（改了参数也不影响历史结果）
    params = models.JSONField('参数快照', null=True, blank=True)
    piece_count = models.IntegerField('参与作品数', default=0)
    cluster_count = models.IntegerField('类数', default=0)
    published = models.BooleanField(
        '已发布', default=False,
        help_text='发布后普通用户才能在聚类总览里看到这次结果')
    note = models.CharField('备注', max_length=200, blank=True)
    error = models.TextField('错误信息', blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '聚类运行'
        verbose_name_plural = '聚类运行'

    def __str__(self):
        return f'聚类第 {self.pk} 次 · {self.get_status_display()}'

    @property
    def param_rows(self) -> list:
        """参数快照 → [(名称, 展示值)]，历史页 / 详情页直接渲染。"""
        return param_rows(self.params)

    @property
    def melody_agg_display(self) -> str:
        agg = (self.params or {}).get('melody_agg')
        return AGG_DISPLAY.get(agg, agg or '—')


class Cluster(models.Model):
    """一个类（同一次运行下）。"""

    run = models.ForeignKey(
        ClusterRun, on_delete=models.CASCADE, related_name='clusters',
        verbose_name='所属运行')
    index = models.IntegerField('类序号', default=0)
    label = models.CharField(
        '类名', max_length=120, blank=True,
        help_text='管理员命名；留空则显示自动标签')
    auto_label = models.CharField('自动标签', max_length=200, blank=True)
    size = models.IntegerField('成员数', default=0)
    # 类代表（最中心的那一首），页面上用它当封面
    medoid = models.ForeignKey(
        Piece, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='medoid_of', verbose_name='代表作品')

    class Meta:
        ordering = ['index']
        verbose_name = '聚类类'
        verbose_name_plural = '聚类类'

    def __str__(self):
        return f'{self.label or self.auto_label or f"第 {self.index} 类"}'

    @property
    def display_label(self) -> str:
        return self.label or self.auto_label or f'第 {self.index} 类'


class ClusterMember(models.Model):
    """一个类里的一个成员（含三轴相似度，页面要解释"为什么归在一起"）。"""

    cluster = models.ForeignKey(
        Cluster, on_delete=models.CASCADE, related_name='members',
        verbose_name='所属类')
    piece = models.ForeignKey(
        Piece, on_delete=models.CASCADE, related_name='cluster_memberships',
        verbose_name='作品')
    mode_sim = models.FloatField('调式相似度', null=True, blank=True)
    harmony_sim = models.FloatField('和声相似度', null=True, blank=True)
    melody_sim = models.FloatField('旋律相似度', null=True, blank=True)
    fused_score = models.FloatField('融合分', default=0.0)
    # 离类中心的远近（0 = 就是中心）
    dist_to_medoid = models.FloatField('到代表距离', default=0.0)

    class Meta:
        unique_together = [('cluster', 'piece')]
        ordering = ['dist_to_medoid', 'id']
        verbose_name = '聚类成员'
        verbose_name_plural = '聚类成员'

    def __str__(self):
        return f'{self.cluster} · {self.piece.title}'
