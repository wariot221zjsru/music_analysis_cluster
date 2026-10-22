"""核心数据模型：作品、分析结果、系统配置。

层级约定（与计划书一致的分析因果链）：
    rhythm_grid → rhythm_pattern
    instrumentation → voice_id → polyphony / exchange
    chords ← voice_id + polyphony
    tonality ← chords
    melody_extraction（主旋律声部识别 / 选轨，可选、并列）
                  ← voice_id + polyphony + exchange
    melody_line（主旋律线追踪 / 逐段，MIR 口径）← voice_id
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
        MELODY_EXTRACTION = 'melody_extraction', '主旋律声部识别（选轨）'
        MELODY_LINE = 'melody_line', '主旋律线追踪（逐段）'
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
    # —— 主旋律线追踪（逐段，MIR 口径）——
    # 与上面的「交接检测」互不相干：交接检测是曲式学描述（同动机传递），
    # 主旋律线追踪是 MIR 任务（每一段谁在听觉前景），不受 is_exchange 门控。
    melody_voice_priority = models.BooleanField(
        '人声优先', default=True,
        help_text='逐段追踪时，人声在唱就归人声，人声休止（前奏 / 间奏 / 尾奏）'
                  '才退回显著性最高的器乐声部。关掉则全程按显著性选轨。')
    melody_segment_min_bars = models.IntegerField(
        '分段最小长度（小节）', default=2,
        help_text='短于该长度的归属碎片并入相邻段，抑制窗间归属抖动。'
                  '前奏 / 间奏本身短于此值且两端归属不同时同样会被并入。')
    melody_confidence_high = models.FloatField(
        '置信度「高」下限', default=0.25,
        help_text='段内平均 margin（第一名 − 第二名显著性）≥ 该值标为「高」。')
    melody_confidence_mid = models.FloatField(
        '置信度「中」下限', default=0.10,
        help_text='margin ≥ 该值标为「中」，低于它标为「低」（归属接近、需人工复核）。')
    # —— 形态资格层（可选，默认关）——
    # 「形态决定有没有资格，显著性决定谁更强」：形态只做否决，不参与加分。
    melody_shape_gate = models.BooleanField(
        '形态（旋律轮廓）决定资格', default=False,
        help_text='开启后逐段追踪多一道资格筛选：声部的旋律轮廓形态'
                  '（级进比例 / 拱形度 / 轮廓平滑度）达不到下限就失去当主旋律的资格，'
                  '改由段内形态合格且显著性最高的声部顶上 —— 用来挡住分解和弦、'
                  '阿尔贝蒂低音、固定音型这类"周期音型"被误判成主旋律'
                  '（它们的动机重复度极高，单看显著性容易赢）。'
                  '关闭则只看显著性，与以前完全一致。')
    melody_shape_min_score = models.FloatField(
        '形态资格下限', default=0.35,
        help_text='仅在上面开启时生效。形态分（0–1）= 级进比×0.7 + 拱形度×0.15 '
                  '+ 轮廓平滑度×0.15，低于该值即失去资格。'
                  '段内音符太少不足以统计轮廓时不作判定，一律放行。')
    # —— 主旋律导出（节奏 / 音色）——
    # 两张主旋律卡（⑨ 选轨 / ⑩ 逐段）共用；每张卡上还能单作品临时改。
    melody_export_tempo_mode = models.CharField(
        '导出节奏', max_length=16, default='original',
        choices=[('original', '原始节奏'), ('changed', '统一 BPM')],
        help_text='原始节奏：按谱面速度标记写入（变速也保留）；'
                  '统一 BPM：整条旋律线用同一个速度（默认 120）。')
    melody_export_bpm = models.IntegerField(
        '统一 BPM', default=120,
        help_text='仅在「导出节奏 = 统一 BPM」时生效，默认 120。')
    melody_export_instrument_mode = models.CharField(
        '导出音色', max_length=16, default='original',
        choices=[('original', '各段原始音色'), ('changed', '统一音色')],
        help_text='各段原始音色：每段用它自己声部的 MIDI 音色号（多段接力能听出换乐器）；'
                  '统一音色：整条线用同一个音色（默认钢琴）。')
    melody_export_program = models.IntegerField(
        '统一音色的 GM 音色号', default=0,
        help_text='仅在「导出音色 = 统一音色」时生效，0 = 大钢琴，'
                  '可填 0–127 的任意 GM 音色号。')
    # —— 显著性：织体密度口径（可选，默认按发音数）——
    density_by_note_count = models.BooleanField(
        '织体密度按发音数计算', default=True,
        help_text='开（默认）：和弦按内部音数计入密度 —— 柱式和弦伴奏一个事件顶三个音，'
                  '只数事件会把它算成"稀疏≈旋律"，显著性反超真旋律。'
                  '关：按发声事件数计算（和弦算 1 个），与早期口径一致。'
                  '改动后请重跑声部识别，主旋律线会自动跟着重算。')
    # —— 主旋律判定：力度（谱面标了力度才作数，可选，默认开）——
    melody_dynamics = models.BooleanField(
        '主旋律判定考虑力度', default=True,
        help_text='开（默认）："听感上更像主旋律"这条也看力度 —— 明显弱于对手的'
                  '低音不当主旋律，明显更响的低音可放宽轮廓 / 实质门槛，显著性分'
                  '不出高下时偏响的那条赢。谱面没标力度时这项自动失效。'
                  '关：完全不读力度，退回纯形态 / 显著性的判据（早期口径）。'
                  '改动后请重跑声部识别，主旋律线会自动跟着重算。')
    # —— 主旋律线：双音 / 加厚旋律保不保留（可选，默认保留）——
    melody_keep_doubles = models.BooleanField(
        '主旋律线保留双音', default=True,
        help_text='开（默认）：谱面里一个发声事件有几个音，导出的主旋律线就'
                  '原样保留几个音——平行三度 / 六度、加八度、加和声音的'
                  '双音旋律整条留住，不会被压成一条骨瘦如柴的单音线。'
                  '关：每个多音事件只留「主音」那一个音，其余按和声填料丢掉；'
                  '主音由主音判定层挑选（和弦成员度 / 相对低音位置 / 候选层轮廓'
                  '三条证据加权，分不出高下时交给 DeepSeek 判定）。'
                  '改动后请重跑主旋律线追踪，导出文件名会带 doubles / single 标记。')
    # —— 主旋律线：困难判断交给 DeepSeek（可选，默认开）——
    melody_llm_judge = models.BooleanField(
        '主旋律线判定困难时交给 DeepSeek', default=True,
        help_text='开（默认）：本地判据咬得太紧、分不出高下时把这一处交给 '
                  'DeepSeek 裁决，并把它的判断与理由显示在作品页。'
                  '两类场合会用到它 ——'
                  '① 选轨：几个候选声部的显著性/形态/力度互相咬住，'
                  '第一名与第二名差距过小；'
                  '② 选音：单音模式下，一段双音里几条候选层难分主次。'
                  '关：完全靠本地判据，一次网络请求都不发，'
                  '结果与开启 DeepSeek 之前的版本一致。'
                  '模型不可用（未安装 dsh、超时、返回不可解析）时自动'
                  '退回本地最优，不会让整步分析失败。')
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
