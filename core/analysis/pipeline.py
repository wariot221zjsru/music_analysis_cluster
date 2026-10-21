"""分析流水线编排：依赖展开、后台线程执行、结果落库。

前端三个按钮对应三组步骤（详见 GROUPS）：
  「分析节奏」     → rhythm_grid + rhythm_pattern
  「分析旋律」     → instrumentation → voice_id → polyphony → exchange
                    → chords → tonality → melody_shape（依赖自动补齐）
  「主旋律声部识别」→ melody_extraction（选轨，旧口径；依赖自动补齐）
  「主旋律线追踪」 → melody_line（逐段，MIR 口径；只依赖 voice_id）

更高层三问（/pieces/<id>/sts/ 页面，三个卡片可分别单独点击）：
  「音乐结构」 → structure ← rhythm_grid + chords + tonality
  「织体」     → texture   ← instrumentation + voice_id + polyphony + rhythm_grid
  「风格」     → style     ← structure + texture + tonality + rhythm_pattern
                             + instrumentation + chords + rhythm_grid
"""
import logging
import threading
import traceback
from django.utils import timezone

from django.db import close_old_connections, connection

from ..models import AnalysisResult, AnalysisRun, Piece, QAQuestion, SystemConfig
from . import chords as chords_mod
from . import instrumentation as instr_mod
from . import melody as melody_mod
from . import melody_line as melody_line_mod
from . import qa as qa_mod
from . import rhythm as rhythm_mod
from . import style as style_mod
from . import structure as structure_mod
from . import texture as texture_mod
from . import tonality as tonality_mod
from . import voice_branches
from . import voice_id as voice_mod
from .scoredata import parse_score, windows_of_measures, build_measure_map

logger = logging.getLogger(__name__)

# 分析因果链的标准顺序（计划书 4.3.1）
CANONICAL_ORDER = [
    'rhythm_grid', 'rhythm_pattern',
    'instrumentation', 'voice_id', 'polyphony', 'exchange',
    'chords', 'tonality',
    'melody_extraction', 'melody_line', 'melody_shape',
    'structure', 'texture', 'style',
]

# 步骤依赖（缺失的依赖会自动补跑）
STEP_DEPS = {
    # —— 更高层三问（结构 / 织体 / 风格）——
    'structure': ['rhythm_grid', 'chords', 'tonality'],
    'texture': ['instrumentation', 'voice_id', 'polyphony', 'rhythm_grid'],
    'style': ['structure', 'texture', 'tonality', 'rhythm_pattern',
              'instrumentation', 'chords', 'rhythm_grid'],
    'rhythm_grid': [],
    'rhythm_pattern': ['rhythm_grid'],
    'instrumentation': [],
    'voice_id': ['instrumentation'],
    'polyphony': ['voice_id'],
    'exchange': ['voice_id'],
    'chords': ['voice_id', 'polyphony'],
    'tonality': ['chords'],
    'melody_extraction': ['voice_id', 'polyphony', 'exchange'],
    # 逐段追踪只看显著性矩阵与人声标记：不看 polyphony、更不看 exchange
    'melody_line': ['voice_id'],
    'melody_shape': ['rhythm_grid', 'voice_id', 'polyphony', 'exchange',
                     'chords', 'tonality'],
}

# 前端按钮 → 步骤组
GROUPS = {
    'rhythm': ['rhythm_grid', 'rhythm_pattern'],
    'melody': ['instrumentation', 'voice_id', 'polyphony', 'exchange',
               'chords', 'tonality', 'melody_shape'],
    'extraction': ['melody_extraction'],
    # 主旋律线追踪：与上面的选轨并列，独立触发
    'line': ['melody_line'],
    # 结构 / 织体 / 风格：一键全跑（页面默认让三者分开点击）
    'sts': ['structure', 'texture', 'style'],
}

STEP_LABELS = {value: label for value, label in AnalysisResult.Step.choices}


def expand_group(group: str, done_steps: set) -> list:
    """展开步骤组：补齐缺失依赖，按因果链标准顺序返回。"""
    targets = set(GROUPS[group])
    needed = set(targets)
    changed = True
    while changed:  # 依赖闭包
        changed = False
        for step in list(needed):
            for dep in STEP_DEPS.get(step, []):
                if dep not in needed:
                    needed.add(dep)
                    changed = True
    # 已完成的依赖不重跑（除非它也在目标组里）
    run = [s for s in needed if s not in done_steps or s in targets]
    return [s for s in CANONICAL_ORDER if s in run]


class _Ctx:
    """一次运行内的共享上下文：解析一次，全步骤复用。"""

    def __init__(self, piece: Piece, config: SystemConfig):
        self.piece = piece
        self.config = config
        self.score = None          # ScoreData（惰性解析）
        self.outputs = {}          # step -> output dict（本次或历史）

    def parsed(self):
        if self.score is None:
            if not self.piece.musicxml_file:
                raise RuntimeError('作品尚未转换为 MusicXML（请先完成格式转换）。')
            self.score = parse_score(self.piece.musicxml_file.path)
        return self.score


def _stale_deps(data, config, outputs) -> set:
    """已完成的依赖里，哪些是**按另一份谱面解析**算出来的（已过期）。

    起因（线上真实事故）：谱面解析改过之后（小节表不再只取第一个声部，
    一首曲子从 32 小节变回真实的 66 小节），声部识别的显著性矩阵还是
    8 个窗的旧值。单步重跑 melody_line 时 `ctx.outputs` 直接复用它，
    后半首曲子在矩阵里一个窗都查不到 —— 于是半首旋律凭空消失。

    这里按两个角度判断，过期的依赖跟着重跑，用户点一次 ⟲ 就能自愈，
    不必记得"先去重跑声部识别"：
      ① **窗数 / 窗长**是否还和当前谱面对得上（上面那次事故的直接原因）；
      ② 每个窗里"哪些声部在响"是否和当前谱面一致。这条是为了另一类事故：
         解析规则改了（例如不再把颤音琴 / 钟琴当无音高打击乐），声部识别
         是按旧口径算的，矩阵里根本没有那个声部 —— 窗数一个不少，
         光看 ① 完全查不出来，主旋律线却只能报出伴奏声部；
      ③ **织体密度口径**（按发音数 / 按事件数）是否还是设置里的那个；
      ④ **力度开关**（这一条证据算不算进主旋律判定）是否还是设置里的那个。
         ③④ 这类更隐蔽：谱面一模一样、窗数一模一样，只是设置变了开关，
         矩阵本身没错，但窗归属是按另一套判据挑出来的 —— 不重算就用回旧结论。
    """
    stale = set()
    vo = outputs.get('voice_id')
    if isinstance(vo, dict):
        sal = vo.get('salience') or {}
        matrix = sal.get('matrix') or []
        if matrix and (
                sal.get('window_bars') != config.analysis_window_bars
                or len(matrix) != len(windows_of_measures(
                    data, config.analysis_window_bars))
                or not _matrix_matches_score(data, matrix)
                # 老产物没有这两个字段 → 视为 True，正是当时算法的默认口径
                or bool(sal.get('density_by_note_count', True))
                != bool(getattr(config, 'density_by_note_count', True))
                or bool(sal.get('melody_dynamics', True))
                != bool(getattr(config, 'melody_dynamics', True))):
            stale.add('voice_id')
    return stale


def _matrix_matches_score(data, matrix) -> bool:
    """显著性矩阵里"每窗哪些声部在响"，是否和当前谱面逐窗一致。"""
    measure_offsets, _, _bar_ql = build_measure_map(data)
    for row in matrix:
        sm, em = row.get('start_measure'), row.get('end_measure')
        m_start = measure_offsets.get(sm)
        m_end = measure_offsets.get(int(em) + 1) if em is not None else None
        if m_start is None or m_end is None:
            return False
        got = {int(k) for k in (row.get('scores') or {})}
        want = {p.index for p in data.parts
                if any(m_start <= e.offset < m_end for e in p.pitched_events)}
        if got != want:
            return False
    return True


def _set_result(piece, step, status, output=None, error=None,
                trigger='', archive=True):
    """写入 / 更新单步结果。

    archive=True 时，每次以 done / error 收尾都会往 AnalysisRun 归档一份快照
    （seq 自增），这样重跑之后上一次的输出仍然查得到。
    """
    obj, _ = AnalysisResult.objects.get_or_create(piece=piece, step=step)
    obj.status = status
    obj.output = output
    obj.error = error or ''
    if status == 'running':
        obj.started_at = timezone.now()
        obj.finished_at = None
    elif status in ('done', 'error'):
        obj.finished_at = timezone.now()
    obj.save()
    if archive and status in ('done', 'error'):
        _archive_run(obj, trigger)


def _archive_run(obj: AnalysisResult, trigger: str = ''):
    """把一次运行的结果快照写入历史表（seq = 该步骤已有次数 + 1）。"""
    last = (AnalysisRun.objects.filter(piece=obj.piece, step=obj.step)
            .order_by('-seq').first())
    seq = (last.seq + 1) if last else 1
    AnalysisRun.objects.create(
        piece=obj.piece, step=obj.step, seq=seq, status=obj.status,
        output=obj.output, error=obj.error, trigger=trigger,
        started_at=obj.started_at, finished_at=obj.finished_at)


def _run_one_step(ctx: _Ctx, step: str):
    """执行单个步骤（已在 ORM 层面标记 running）。"""
    data = ctx.parsed()
    cfg = ctx.config

    if step == 'rhythm_grid':
        return rhythm_mod.rhythm_grid(data, cfg)
    if step == 'rhythm_pattern':
        return rhythm_mod.rhythm_pattern(data, cfg)
    if step == 'instrumentation':
        return instr_mod.instrumentation(data, cfg)
    if step == 'voice_id':
        return voice_mod.voice_id(data, cfg, ctx.outputs['instrumentation'])
    if step == 'polyphony':
        return voice_branches.polyphony(data, cfg, ctx.outputs['voice_id'])
    if step == 'exchange':
        return voice_branches.exchange(data, cfg, ctx.outputs['voice_id'])
    if step == 'chords':
        return chords_mod.chords(data, cfg, ctx.outputs['voice_id'])
    if step == 'tonality':
        return tonality_mod.tonality(data, cfg, ctx.outputs['chords'],
                                     ctx.outputs['voice_id'])
    if step == 'melody_extraction':
        return melody_mod.melody_extraction(
            data, cfg, ctx.piece, ctx.outputs['voice_id'],
            ctx.outputs['polyphony'], ctx.outputs['exchange'])
    if step == 'melody_line':
        return melody_line_mod.melody_line(
            data, cfg, ctx.piece, ctx.outputs['voice_id'])
    if step == 'melody_shape':
        return melody_mod.melody_shape(
            data, cfg, ctx.outputs['voice_id'], ctx.outputs['polyphony'],
            ctx.outputs['exchange'])
    if step == 'structure':
        return structure_mod.structure(data, cfg, ctx.outputs['rhythm_grid'],
                                       ctx.outputs['chords'],
                                       ctx.outputs['tonality'])
    if step == 'texture':
        return texture_mod.texture(data, cfg, ctx.outputs['instrumentation'],
                                   ctx.outputs['voice_id'],
                                   ctx.outputs['polyphony'],
                                   ctx.outputs['rhythm_grid'])
    if step == 'style':
        return style_mod.style(
            data, cfg, ctx.outputs['structure'], ctx.outputs['texture'],
            ctx.outputs['tonality'], ctx.outputs['rhythm_pattern'],
            ctx.outputs['instrumentation'], ctx.outputs['chords'],
            ctx.outputs['rhythm_grid'])
    raise ValueError(f'未知步骤：{step}')


def _run_steps(piece_id: int, steps: list, trigger: str = ''):
    """后台线程主体：顺序执行步骤，失败即停（后续步骤依赖前者）。"""
    close_old_connections()
    piece = Piece.objects.filter(pk=piece_id).first()
    if piece is None:
        return
    config = SystemConfig.get_config()

    # 载入已完成步骤的历史输出（供依赖读取）
    ctx = _Ctx(piece, config)
    for r in AnalysisResult.objects.filter(piece=piece, status='done'):
        ctx.outputs[r.step] = r.output

    # 依赖是否过期（按另一份谱面解析算出来的）→ 过期就跟着重跑，
    # 用户点一次 ⟲ 就能自愈，不必记得"先去重跑声部识别"。
    if steps:
        try:
            extra = _stale_deps(ctx.parsed(), config, ctx.outputs)
        except Exception:  # noqa: BLE001 —— 解析不了就让下面的步骤照常报错
            extra = set()
        if extra:
            done = set(ctx.outputs)
            changed = True
            while changed:          # 过期依赖自己的依赖也要补
                changed = False
                for s in list(extra):
                    for dep in STEP_DEPS.get(s, []):
                        if dep not in extra and dep not in done:
                            extra.add(dep)
                            changed = True
            steps = [s for s in CANONICAL_ORDER if s in (set(steps) | extra)]
            for s in steps:
                _set_result(piece, s, 'pending', trigger=trigger)
            logger.info('依赖已过期，自动补跑：%s', sorted(extra))

    try:
        executed = set()
        for step in steps:
            _set_result(piece, step, 'running', trigger=trigger)
            executed.add(step)
            try:
                output = _run_one_step(ctx, step)
                ctx.outputs[step] = output
                _set_result(piece, step, 'done', output=output, trigger=trigger)
            except Exception as exc:  # noqa: BLE001
                logger.exception('步骤 %s 失败', step)
                _set_result(piece, step, 'error', trigger=trigger,
                            error=f'{exc}\n---\n{traceback.format_exc()[-1500:]}')
                # 后续步骤不再执行：把残留的 pending 标记为明确失败
                # （没真正跑过，不进历史，避免污染记录）
                for rest in steps:
                    if rest not in executed:
                        _set_result(piece, rest, 'error', archive=False,
                                    trigger=trigger,
                                    error='未执行：前置步骤失败，请重跑本组。')
                break
    finally:
        if connection.connection is not None:
            connection.close()


def _run_convert(piece_id: int, trigger: str = 'convert'):
    """后台线程：上传后的格式转换。"""
    close_old_connections()
    piece = Piece.objects.filter(pk=piece_id).first()
    if piece is None:
        return
    from ..converters.musicxml import convert_piece
    _set_result(piece, 'convert', 'running', trigger=trigger)
    try:
        output = convert_piece(piece)
        _set_result(piece, 'convert', 'done', output=output, trigger=trigger)
    except NotImplementedError as exc:
        _set_result(piece, 'convert', 'error', error=str(exc), trigger=trigger)
    except Exception as exc:  # noqa: BLE001
        logger.exception('格式转换失败')
        _set_result(piece, 'convert', 'error', trigger=trigger,
                    error=f'{exc}\n---\n{traceback.format_exc()[-1500:]}')
    finally:
        if connection.connection is not None:
            connection.close()


def start_convert_async(piece: Piece, trigger: str = 'convert'):
    threading.Thread(target=_run_convert, args=(piece.id, trigger),
                     daemon=True, name=f'convert-{piece.id}').start()


def start_steps_async(piece: Piece, group: str, force: bool = False):
    """按组启动分析（后台线程）。返回实际将执行的步骤列表。"""
    done = set(AnalysisResult.objects.filter(piece=piece, status='done')
               .values_list('step', flat=True))
    if force:
        # 重跑：清掉目标组内已完成的结果
        AnalysisResult.objects.filter(piece=piece,
                                      step__in=GROUPS[group]).delete()
        done -= set(GROUPS[group])
    steps = expand_group(group, done)
    if not steps:
        return []
    # 占位 pending（前端立即能看到将执行的步骤）
    trigger = f'group:{group}'
    for step in steps:
        _set_result(piece, step, 'pending', trigger=trigger)
    threading.Thread(target=_run_steps, args=(piece.id, steps, trigger),
                     daemon=True, name=f'analyze-{piece.id}-{group}').start()
    return steps


def start_step_async(piece: Piece, step: str, with_deps: bool = True):
    """重新执行**单个步骤**（每模块右上角的 ⟲ 按钮）。

    与 start_steps_async 的区别：
      - 目标 step **必定重跑**（先清掉它的旧结果）；
      - 依赖若已完成则**复用**，未完成才自动补跑（with_deps=False 时完全不补）。

    价值：melody_shape 这类末端步骤失败时，不必连带重跑耗时 8 分钟的 chords。
    """
    if step not in STEP_DEPS:
        raise ValueError(f'未知步骤：{step}')

    done = set(AnalysisResult.objects.filter(piece=piece, status='done')
               .values_list('step', flat=True))
    # 目标步骤自身强制重跑
    AnalysisResult.objects.filter(piece=piece, step=step).delete()
    done.discard(step)

    needed = {step}
    if with_deps:
        changed = True
        while changed:      # 依赖闭包
            changed = False
            for s in list(needed):
                for dep in STEP_DEPS.get(s, []):
                    if dep not in needed:
                        needed.add(dep)
                        changed = True

    # 已完成的依赖不重跑；未完成的依赖必须补跑
    run = {s for s in needed if s == step or s not in done}
    steps = [s for s in CANONICAL_ORDER if s in run]
    if not steps:
        return []
    trigger = f'rerun:{step}'
    for s in steps:
        _set_result(piece, s, 'pending', trigger=trigger)
    threading.Thread(target=_run_steps, args=(piece.id, steps, trigger),
                     daemon=True, name=f'rerun-{piece.id}-{step}').start()
    return steps


# ---------------- 自定义问答（异步） ----------------

def _run_question(qa_id: int):
    """后台线程：基于已完成的分析回答一次用户提问。

    问答不重新分析：只把已完成步骤的结论压缩进上下文，
    没跑过的方面在上下文中记为"尚无数据"，模型不得臆断。
    """
    close_old_connections()
    item = QAQuestion.objects.filter(pk=qa_id).first()
    if item is None:
        return
    try:
        piece = item.piece
        config = SystemConfig.get_config()
        results = {r.step: r.output
                   for r in AnalysisResult.objects.filter(piece=piece,
                                                          status='done')}
        data = None
        if piece.musicxml_file:
            try:
                data = parse_score(piece.musicxml_file.path)
            except Exception:  # noqa: BLE001
                data = None      # 谱面读不出也照样能答，只是少了旋律片段
        ctx = qa_mod.build_context(piece, data, results)
        item.answer = qa_mod.answer(item.question, ctx, config)
        item.details = ctx
        item.status = 'done'
    except Exception as exc:  # noqa: BLE001
        logger.exception('问答失败')
        item.answer = ''
        item.status = 'error'
        item.error = str(exc)
    finally:
        item.finished_at = timezone.now()
        item.save()
        if connection.connection is not None:
            connection.close()


def start_question_async(piece: Piece, user, question: str) -> int:
    """登记一次提问并起后台线程回答。返回提问 id。"""
    item = QAQuestion.objects.create(piece=piece, user=user, question=question,
                                     status='running')
    threading.Thread(target=_run_question, args=(item.id,),
                     daemon=True, name=f'qa-{item.id}').start()
    return item.id
