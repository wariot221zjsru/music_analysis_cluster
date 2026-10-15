"""分析流水线编排：依赖展开、后台线程执行、结果落库。

前端三个按钮对应三组步骤（详见 GROUPS）：
  「分析节奏」     → rhythm_grid + rhythm_pattern
  「分析旋律」     → instrumentation → voice_id → polyphony → exchange
                    → chords → tonality → melody_shape（依赖自动补齐）
  「主旋律提取」   → melody_extraction（与旋律分析并列；依赖自动补齐）
"""
import logging
import threading
import traceback
from django.utils import timezone

from django.db import close_old_connections, connection

from ..models import AnalysisResult, Piece, SystemConfig
from . import chords as chords_mod
from . import instrumentation as instr_mod
from . import melody as melody_mod
from . import rhythm as rhythm_mod
from . import tonality as tonality_mod
from . import voice_branches
from . import voice_id as voice_mod
from .scoredata import parse_score

logger = logging.getLogger(__name__)

# 分析因果链的标准顺序（计划书 4.3.1）
CANONICAL_ORDER = [
    'rhythm_grid', 'rhythm_pattern',
    'instrumentation', 'voice_id', 'polyphony', 'exchange',
    'chords', 'tonality',
    'melody_extraction', 'melody_shape',
]

# 步骤依赖（缺失的依赖会自动补跑）
STEP_DEPS = {
    'rhythm_grid': [],
    'rhythm_pattern': ['rhythm_grid'],
    'instrumentation': [],
    'voice_id': ['instrumentation'],
    'polyphony': ['voice_id'],
    'exchange': ['voice_id'],
    'chords': ['voice_id', 'polyphony'],
    'tonality': ['chords'],
    'melody_extraction': ['voice_id', 'polyphony', 'exchange'],
    'melody_shape': ['rhythm_grid', 'voice_id', 'polyphony', 'exchange',
                     'chords', 'tonality'],
}

# 前端按钮 → 步骤组
GROUPS = {
    'rhythm': ['rhythm_grid', 'rhythm_pattern'],
    'melody': ['instrumentation', 'voice_id', 'polyphony', 'exchange',
               'chords', 'tonality', 'melody_shape'],
    'extraction': ['melody_extraction'],
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


def _set_result(piece, step, status, output=None, error=None):
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
    if step == 'melody_shape':
        return melody_mod.melody_shape(
            data, cfg, ctx.outputs['voice_id'], ctx.outputs['polyphony'],
            ctx.outputs['exchange'])
    raise ValueError(f'未知步骤：{step}')


def _run_steps(piece_id: int, steps: list):
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

    try:
        executed = set()
        for step in steps:
            _set_result(piece, step, 'running')
            executed.add(step)
            try:
                output = _run_one_step(ctx, step)
                ctx.outputs[step] = output
                _set_result(piece, step, 'done', output=output)
            except Exception as exc:  # noqa: BLE001
                logger.exception('步骤 %s 失败', step)
                _set_result(piece, step, 'error',
                            error=f'{exc}\n---\n{traceback.format_exc()[-1500:]}')
                # 后续步骤不再执行：把残留的 pending 标记为明确失败
                for rest in steps:
                    if rest not in executed:
                        _set_result(piece, rest, 'error',
                                    error='未执行：前置步骤失败，请重跑本组。')
                break
    finally:
        if connection.connection is not None:
            connection.close()


def _run_convert(piece_id: int):
    """后台线程：上传后的格式转换。"""
    close_old_connections()
    piece = Piece.objects.filter(pk=piece_id).first()
    if piece is None:
        return
    from ..converters.musicxml import convert_piece
    _set_result(piece, 'convert', 'running')
    try:
        output = convert_piece(piece)
        _set_result(piece, 'convert', 'done', output=output)
    except NotImplementedError as exc:
        _set_result(piece, 'convert', 'error', error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception('格式转换失败')
        _set_result(piece, 'convert', 'error',
                    error=f'{exc}\n---\n{traceback.format_exc()[-1500:]}')
    finally:
        if connection.connection is not None:
            connection.close()


def start_convert_async(piece: Piece):
    threading.Thread(target=_run_convert, args=(piece.id,),
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
    for step in steps:
        _set_result(piece, step, 'pending')
    threading.Thread(target=_run_steps, args=(piece.id, steps),
                     daemon=True, name=f'analyze-{piece.id}-{group}').start()
    return steps
