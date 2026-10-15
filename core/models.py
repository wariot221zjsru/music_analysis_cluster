"""核心数据模型：作品、分析结果、系统配置。

层级约定（与计划书一致的分析因果链）：
    rhythm_grid → rhythm_pattern
    instrumentation → voice_id → polyphony / exchange
    chords ← voice_id + polyphony
    tonality ← chords
    melody_extraction（可选、并列）← voice_id + polyphony + exchange
    melody_shape ← rhythm_grid + voice_id + polyphony + exchange + chords + tonality
"""
from django.conf import settings
from django.db import models


class Piece(models.Model):
    """一个待分析的音乐作品（音频 / MIDI / MusicXML 上传件）。"""

    class SourceType(models.TextChoices):
        AUDIO = 'audio', '音频（需 MuScriptor 转录）'
        MIDI = 'midi', 'MIDI'
        MUSICXML = 'musicxml', 'MusicXML'

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              related_name='pieces', verbose_name='上传者')
    title = models.CharField('标题', max_length=200)
    source_file = models.FileField('源文件', upload_to='pieces/%Y%m%d/')
    source_type = models.CharField('源类型', max_length=16,
                                   choices=SourceType.choices)
    contains_voice = models.BooleanField('是否含人声', default=False,
                                         help_text='含人声时 MuScriptor 走"器乐+人声"两遍转录管线')
    midi_file = models.FileField('转录 MIDI', upload_to='converted/%Y%m%d/', blank=True)
    musicxml_file = models.FileField('MusicXML', upload_to='converted/%Y%m%d/', blank=True)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '作品'
        verbose_name_plural = '作品'

    def __str__(self):
        return f'{self.title}（{self.get_source_type_display()}）'


class AnalysisResult(models.Model):
    """单步分析结果。每个 (piece, step) 只保留最新一条。"""

    class Step(models.TextChoices):
        CONVERT = 'convert', '格式转换'
        RHYTHM_GRID = 'rhythm_grid', '节奏网格与速度曲线'
        RHYTHM_PATTERN = 'rhythm_pattern', '节奏型与律动分析'
        INSTRUMENTATION = 'instrumentation', '配器提取（静态）'
        VOICE_ID = 'voice_id', '声部识别'
        POLYPHONY = 'polyphony', '复调检测'
        EXCHANGE = 'exchange', '主旋律交接检测'
        CHORDS = 'chords', '和弦进行分析'
        TONALITY = 'tonality', '调性布局分析'
        MELODY_EXTRACTION = 'melody_extraction', '主旋律提取（可选输出）'
        MELODY_SHAPE = 'melody_shape', '旋律形态分析'

    class Status(models.TextChoices):
        PENDING = 'pending', '等待中'
        RUNNING = 'running', '运行中'
        DONE = 'done', '已完成'
        ERROR = 'error', '失败'

    piece = models.ForeignKey(Piece, on_delete=models.CASCADE, related_name='results')
    step = models.CharField('步骤', max_length=32, choices=Step.choices)
    status = models.CharField('状态', max_length=16, choices=Status.choices,
                              default=Status.PENDING)
    output = models.JSONField('输出', null=True, blank=True)
    error = models.TextField('错误信息', blank=True)
    started_at = models.DateTimeField('开始时间', null=True, blank=True)
    finished_at = models.DateTimeField('结束时间', null=True, blank=True)

    class Meta:
        unique_together = [('piece', 'step')]
        ordering = ['id']
        verbose_name = '分析结果'
        verbose_name_plural = '分析结果'

    def __str__(self):
        return f'{self.piece.title} · {self.get_step_display()} · {self.get_status_display()}'


class SystemConfig(models.Model):
    """系统级配置（单例）。管理员可在前端 /settings/ 页面修改。"""

    # —— 复调检测 ——
    polyphony_threshold = models.FloatField(
        '复调判定阈值', default=0.30,
        help_text='同 Part 同时发声音数占所有发声事件的比例达到该值即判定为复调（计划书 4.5.2.1）')
    # —— 主旋律交接检测 ——
    exchange_min_switches = models.IntegerField(
        '交接最少切换次数', default=2,
        help_text='主旋律归属时间线上出现 ≥ 该次数的声部切换才可能判为交接段')
    exchange_similarity_threshold = models.FloatField(
        '交接动机相似度阈值', default=0.60,
        help_text='切换前后声部之间的音高/动机相似度达到该值才判为交接（fugato / 模仿）')
    # —— 分析窗口 ——
    analysis_window_bars = models.IntegerField(
        '显著性时间窗（小节）', default=4,
        help_text='主旋律显著性分数按该小节数的时间窗计算')
    chord_window = models.CharField(
        '和弦聚合窗口', max_length=16, default='measure',
        choices=[('measure', '按小节'), ('beat', '按拍')],
        help_text='和弦判定前音高集合的聚合粒度')
    # —— dsh（DeepSeek Harness 命令行）——
    dsh_profile = models.CharField(
        'dsh profile', max_length=64, default='headless',
        help_text='dsh --profile <name>，需已安装 dsh-musictheory 插件')
    dsh_timeout = models.IntegerField(
        'dsh 单次调用超时（秒）', default=600)
    max_chord_windows_per_call = models.IntegerField(
        '和弦单次 LLM 批量窗口数', default=24)

    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        verbose_name = '系统配置'
        verbose_name_plural = '系统配置'

    def save(self, *args, **kwargs):
        self.pk = 1  # 单例
        super().save(*args, **kwargs)

    @classmethod
    def get_config(cls):
        obj = cls.objects.filter(pk=1).first()
        if obj is None:
            obj = cls(pk=1)
            obj.save()
        return obj

    def __str__(self):
        return '系统配置'
