"""核心数据模型：作品、分析结果、系统配置。

层级约定（与计划书一致的分析因果链）：
    rhythm_grid → rhythm_pattern
    instrumentation → voice_id → polyphony / exchange
    chords ← voice_id + polyphony
    tonality ← chords
    melody_extraction（可选、并列）← voice_id + polyphony + exchange
    melody_shape ← rhythm_grid + voice_id + polyphony + exchange + chords + tonality

更高层三问（新页 /pieces/<id>/sts/）：
    structure ← rhythm_grid + chords + tonality
    texture   ← instrumentation + voice_id + polyphony + rhythm_grid
    style     ← structure + texture + tonality + rhythm_pattern + instrumentation
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
    source_file = models.FileField(
        '源文件（临时）', upload_to='tmp/%Y%m%d/', blank=True,
        help_text='原始音频仅作临时存放，转换成功后自动删除，不做长期保存')
    source_type = models.CharField('源类型', max_length=16,
                                   choices=SourceType.choices)
    contains_voice = models.BooleanField(
        '是否含人声', default=False,
        help_text='唯一事实来源是作品文件本身：MIDI / MusicXML 上传后按文件里的实际'
                  '声部自动回写（覆盖上传时的勾选）；音频上传按 MuScriptor 转录结果'
                  '回写。上传页的勾选只对音频转录管线的选择起作用。')
    is_public = models.BooleanField(
        '公开', default=False,
        help_text='公开后其他登录用户也可查看该作品及其分析结果（不可代为运行分析）')
    source_purged = models.BooleanField('临时源文件已清理', default=False)
    midi_file = models.FileField('MIDI（永久）', upload_to='midi/%Y%m%d/', blank=True,
                                 help_text='永久保存，独立于 MusicXML 目录')
    musicxml_file = models.FileField('MusicXML（永久）', upload_to='musicxml/%Y%m%d/',
                                     blank=True, help_text='永久保存，独立于 MIDI 目录')
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '作品'
        verbose_name_plural = '作品'

    def __str__(self):
        return f'{self.title}（{self.get_source_type_display()}）'

    @property
    def visibility_display(self) -> str:
        return '公开' if self.is_public else '私有'

    @property
    def audio_purged(self) -> bool:
        """原始音频是否已被清理（音频类作品才有意义）。"""
        return self.source_type == self.SourceType.AUDIO and self.source_purged


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
    STRUCTURE = 'structure', '音乐结构分析（乐句 / 乐段 / 曲式）'
    TEXTURE = 'texture', '织体分析'
    STYLE = 'style', '风格分析'

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


class AnalysisRun(models.Model):
    """一次分析运行的归档快照（历史记录）。

    AnalysisResult 每个 (piece, step) 只有一行、重跑即覆盖；
    本表把**每一次**执行（done / error）都留一份，供回看与前后对比。
    重跑、换参数、LLM 抽风重来之后，上一次的结果都能在这里找回来。
    """

    piece = models.ForeignKey(Piece, on_delete=models.CASCADE,
                              related_name='runs', verbose_name='作品')
    step = models.CharField('步骤', max_length=32,
                            choices=AnalysisResult.Step.choices)
    seq = models.PositiveIntegerField('第几次运行', default=1)
    status = models.CharField('状态', max_length=16,
                              choices=AnalysisResult.Status.choices)
    output = models.JSONField('输出', null=True, blank=True)
    error = models.TextField('错误信息', blank=True)
    trigger = models.CharField(
        '触发方式', max_length=32, blank=True,
        help_text='convert / group:melody / rerun:chords 等，用于区分是谁发起的')
    started_at = models.DateTimeField('开始时间', null=True, blank=True)
    finished_at = models.DateTimeField('结束时间', null=True, blank=True)
    archived_at = models.DateTimeField('归档时间', auto_now_add=True)

    class Meta:
        ordering = ['step', '-seq']
        unique_together = [('piece', 'step', 'seq')]
        verbose_name = '运行历史'
        verbose_name_plural = '运行历史'

    def __str__(self):
        return f'{self.piece.title} · {self.get_step_display()} · 第 {self.seq} 次'


class QAQuestion(models.Model):
    """用户对某部作品的自定义提问（异步交给 dsh 回答）。

    与分析结果分开存放：分析步骤是 (piece, step) 唯一、结果会被覆盖，
    而问答是历史流——同一作品可以反复追问，旧记录必须保留。
    """

    class Status(models.TextChoices):
        PENDING = 'pending', '等待中'
        RUNNING = 'running', '运行中'
        DONE = 'done', '已完成'
        ERROR = 'error', '失败'

    piece = models.ForeignKey(Piece, on_delete=models.CASCADE,
                              related_name='questions', verbose_name='作品')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='music_questions', verbose_name='提问人')
    question = models.TextField('问题')
    answer = models.TextField('回答', blank=True)
    details = models.JSONField('结构化回答', null=True, blank=True,
                               help_text='模型返回的 JSON（答案 / 依据 / 提醒）')
    status = models.CharField('状态', max_length=16, choices=Status.choices,
                              default=Status.PENDING)
    error = models.TextField('错误信息', blank=True)
    created_at = models.DateTimeField('提问时间', auto_now_add=True)
    finished_at = models.DateTimeField('完成时间', null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '自定义提问'
        verbose_name_plural = '自定义提问'

    def __str__(self):
        return f'{self.piece.title} · {self.question[:30]}'


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
