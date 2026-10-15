"""核心视图：上传、作品详情、分析触发、状态轮询、管理员设置。"""
import os
from pathlib import Path

from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .analysis.pipeline import GROUPS, STEP_LABELS, start_steps_async, start_convert_async
from .forms import UploadForm
from .models import AnalysisResult, Piece, SystemConfig

# 允许的上传扩展名
ALLOWED_EXT = {'.mid', '.midi', '.musicxml', '.xml', '.mxl',
               '.mp3', '.wav', '.flac', '.ogg', '.m4a'}

EXT_TO_TYPE = {
    '.mid': 'midi', '.midi': 'midi',
    '.musicxml': 'musicxml', '.xml': 'musicxml', '.mxl': 'musicxml',
    '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio',
    '.ogg': 'audio', '.m4a': 'audio',
}


def _staff_required(view):
    return user_passes_test(lambda u: u.is_staff, login_url='/accounts/login/')(view)


@login_required
def dashboard(request):
    """作品列表（管理员可看全部，普通用户只看自己的）。"""
    if request.user.is_staff:
        pieces = Piece.objects.select_related('owner').all()
    else:
        pieces = Piece.objects.filter(owner=request.user)
    steps_done = {
        p.id: {r.step for r in p.results.filter(status='done')}
        for p in pieces
    }
    return render(request, 'core/dashboard.html', {
        'pieces': pieces,
        'steps_done': steps_done,
    })


@login_required
def upload(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        f = request.FILES['source_file']
        ext = Path(f.name).suffix.lower()
        if ext not in ALLOWED_EXT:
            form.add_error('source_file', f'不支持的文件类型 {ext}。'
                                     '支持：MIDI / MusicXML / 常见音频格式。')
        else:
            piece = Piece.objects.create(
                owner=request.user,
                title=form.cleaned_data['title'] or Path(f.name).stem,
                source_file=f,
                source_type=EXT_TO_TYPE[ext],
                contains_voice=form.cleaned_data['contains_voice'],
            )
            start_convert_async(piece)
            messages.success(request, '上传成功，正在转换格式…')
            return redirect('core:piece_detail', piece_id=piece.id)
    return render(request, 'core/upload.html', {'form': form})


@login_required
def piece_detail(request, piece_id: int):
    piece = get_object_or_404(Piece, pk=piece_id)
    if piece.owner_id != request.user.id and not request.user.is_staff:
        piece = get_object_or_404(Piece, pk=piece_id, owner=request.user)
    return render(request, 'core/piece_detail.html', {
        'piece': piece,
        'step_labels': STEP_LABELS,
    })


@login_required
@require_POST
def run_group(request, piece_id: int):
    """触发一组分析（rhythm / melody / extraction）。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if piece.owner_id != request.user.id and not request.user.is_staff:
        return JsonResponse({'ok': False, 'error': '无权操作该作品。'}, status=403)

    group = request.POST.get('group', '')
    force = request.POST.get('force') == '1'
    if group not in GROUPS:
        return JsonResponse({'ok': False, 'error': f'未知分析组：{group}'}, status=400)
    if piece.results.filter(step='convert', status='done').exists() is False:
        return JsonResponse({'ok': False,
                             'error': '格式转换尚未完成（或失败），暂不能分析。'}, status=400)
    running = piece.results.filter(status__in=['running', 'pending']).exists()
    if running:
        return JsonResponse({'ok': False,
                             'error': '已有分析在运行中，请等待完成。'}, status=409)
    steps = start_steps_async(piece, group, force=force)
    if not steps:
        return JsonResponse({'ok': True, 'started': [],
                             'message': '该组所有步骤均已完成；如需重跑请使用"重跑"按钮。'})
    return JsonResponse({'ok': True, 'started': steps})


@login_required
def piece_status(request, piece_id: int):
    """状态轮询端点（前端每 2.5s 调用）。"""
    piece = get_object_or_404(Piece, pk=piece_id)
    if piece.owner_id != request.user.id and not request.user.is_staff:
        return JsonResponse({'ok': False, 'error': '无权访问该作品。'}, status=403)
    results = {}
    for r in piece.results.all():
        results[r.step] = {
            'status': r.status,
            'error': r.error[:600] if r.error else '',
            'output': r.output if r.status == 'done' else None,
            'started_at': r.started_at.isoformat() if r.started_at else None,
            'finished_at': r.finished_at.isoformat() if r.finished_at else None,
        }
    return JsonResponse({
        'ok': True,
        'results': results,
        'any_running': any(v['status'] in ('running', 'pending')
                           for v in results.values()),
    })


@_staff_required
def system_settings(request):
    """管理员配置页：复调阈值等参数前端可调。"""
    config = SystemConfig.get_config()
    if request.method == 'POST':
        try:
            config.polyphony_threshold = float(request.POST['polyphony_threshold'])
            config.exchange_min_switches = int(request.POST['exchange_min_switches'])
            config.exchange_similarity_threshold = float(
                request.POST['exchange_similarity_threshold'])
            config.analysis_window_bars = int(request.POST['analysis_window_bars'])
            config.chord_window = request.POST['chord_window']
            config.dsh_profile = request.POST['dsh_profile'].strip() or 'headless'
            config.dsh_timeout = int(request.POST['dsh_timeout'])
            config.max_chord_windows_per_call = int(
                request.POST['max_chord_windows_per_call'])
            config.full_clean()
            config.save()
            messages.success(request, '系统配置已保存。')
            return redirect('core:settings')
        except (KeyError, ValueError) as exc:
            messages.error(request, f'参数有误：{exc}')
    return render(request, 'core/settings.html', {'config': config})
