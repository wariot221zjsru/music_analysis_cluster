"""dsh（DeepSeek Harness 命令行版）封装。

后端通过 `dsh --profile <name> -`（stdin 传任务、stdout 收答案）调用
DeepSeek；dsh-musictheory 插件为模型提供 12-TET 音乐理论数学工具箱
（chord_identify / interval_info / scale_harmonize 等 9 个工具），
提示词中要求模型"先算后说、凡算必验"。

JSON 解析采用三级兜底（见 extract_json）：
  ① 括号配平扫描，取出第一个结构完整的对象
  ② 被 max_tokens 截断时，回退到最后一个完整成员再补齐闭合
  ③ 仍失败才抛错（调用方应用 run_llm_json_soft 降级为原文展示）
"""
import json
import re
import shutil
import subprocess
import sys


class LLMError(RuntimeError):
    """LLM 调用失败（超时 / 命令不可用 / 输出无法解析）。"""

    def __init__(self, message: str, raw_text: str = ''):
        super().__init__(message)
        # 解析失败时带上原文，供调用方降级展示，避免整步分析被拖垮
        self.raw_text = raw_text


def _dsh_command(profile: str) -> list:
    dsh = shutil.which('dsh')
    if dsh is None:
        raise LLMError(
            '找不到 dsh 命令。请先安装 DeepSeek Harness 命令行版 '
            '(npm i -g deepseek-harness 或按官方文档安装)，'
            '并确认 `dsh --version` 可用。')
    if sys.platform == 'win32' and dsh.lower().endswith(('.cmd', '.bat')):
        return ['cmd', '/c', dsh, '--profile', profile, '-']
    return [dsh, '--profile', profile, '-']


def run_llm(task: str, profile: str = 'headless', timeout: int = 240) -> str:
    """调用 dsh headless 执行一次任务，返回模型的最终回答文本。"""
    cmd = _dsh_command(profile)
    try:
        proc = subprocess.run(
            cmd, input=task, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        raise LLMError(f'dsh 调用超时（{timeout}s）。可在系统设置中调大超时。')
    if proc.returncode != 0:
        raise LLMError(
            f'dsh 退出码 {proc.returncode}。stderr 尾部：'
            f'{(proc.stderr or "")[-800:]}')
    return (proc.stdout or '').strip()


# ---------------- JSON 提取（三级兜底） ----------------

def _scan_balanced(text: str, start: int):
    """从 '{' / '[' 开始做括号配平扫描（跟踪字符串状态与转义），
    返回闭合字符的下标（含）；未闭合返回 None。"""
    open_ch = text[start]
    close_ch = '}' if open_ch == '{' else ']'
    depth, i, n = 0, start, len(text)
    in_str = esc = False
    while i < n:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in '{[':
            depth += 1
        elif c in '}]':
            depth -= 1
            if depth == 0:
                return i if c == close_ch else None
        i += 1
    return None


def _candidate_objects(text: str):
    """按出现顺序产出所有顶层 JSON 候选 ``(start, end)``。

    ``end`` 为 None 表示该候选**没有闭合**（输出被 max_tokens 截断）。
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] in '{[':
            end = _scan_balanced(text, i)
            if end is None:
                yield i, None
                return
            yield i, end + 1
            i = end + 1
        else:
            i += 1
    return


def _salvage_truncated(fragment: str):
    """救回被截断的 JSON：从后往前退到最后一个完整成员，再补齐闭合。

    返回解析出的对象（dict 会带 ``_truncated=True`` 标记），全失败返回 None。
    """
    cuts, depth = [], 0
    in_str = esc = False
    for i, c in enumerate(fragment):
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in '{[':
            depth += 1
        elif c in '}]':
            depth -= 1
        elif c == ',' and depth == 1:
            cuts.append(i)          # depth==1 层的成员分隔符

    close_ch = '}' if fragment[0] == '{' else ']'
    trials = [fragment.rstrip()] + [fragment[:c] for c in reversed(cuts)]
    for raw in trials:
        t = raw.rstrip().rstrip(',')
        if not t:
            continue
        if t.count('"') % 2:            # 字符串未闭合 → 补引号
            t += '"'
        t = t.rstrip(':').rstrip()      # 结尾停在冒号说明 value 缺失
        need = (t.count('{') - t.count('}')) + (t.count('[') - t.count(']'))
        if need > 0:
            t += close_ch * need
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj:
            obj['_truncated'] = True
            return obj
        if isinstance(obj, list) and obj:
            return obj
    return None


def extract_json(text: str):
    """从模型回答中提取 JSON（容忍代码块围栏、前后解释文字、输出被截断）。"""
    cleaned = re.sub(r'```(?:json|JSON)?', '', text or '').strip('` \n')
    truncated = None
    for start, end in _candidate_objects(cleaned):
        if end is None:
            truncated = cleaned[start:]
            continue
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            continue                      # 这一段不是 JSON（可能是解释文字）
    if truncated:
        salvaged = _salvage_truncated(truncated)
        if salvaged is not None:
            return salvaged
    raise LLMError(
        f'无法从模型回答中解析 JSON。回答前 800 字：{cleaned[:800]}',
        raw_text=cleaned[:4000])


def run_llm_json(task: str, profile: str = 'headless', timeout: int = 240):
    """调用 LLM 并解析为 JSON 对象 / 数组（失败仍会抛 LLMError）。"""
    return extract_json(run_llm(task, profile=profile, timeout=timeout))


COMPACT_HINT = (
    '\n（提醒：上一次你的回答超过了输出长度上限被截断，'
    '这次请把每个字段压缩到 60 字以内，先输出完整 JSON，务必以右花括号结尾。）'
)


def run_llm_json_soft(task: str, profile: str = 'headless', timeout: int = 240,
                      retries: int = 1):
    """调用 LLM 并解析 JSON；**任何失败都不抛异常**，降级为带诊断信息的 dict。

    这样单个分析段（如某个旋律片段）的 LLM 失败不会让整个分析步骤变为 error，
    已经跑完的其他段落结果得以保留。
    """
    err, raw = '', ''
    for attempt in range(max(retries, 0) + 1):
        try:
            return extract_json(
                run_llm(task + (COMPACT_HINT if attempt else ''),
                        profile=profile, timeout=timeout))
        except LLMError as exc:
            err = str(exc)
            raw = getattr(exc, 'raw_text', '') or ''
    return {
        '_error': err or 'LLM 未返回结果。',
        '_raw_text': raw,
        '_hint': '模型输出未能解析为 JSON，可在 /settings/ 调大超时或改用更强的 dsh 配置后重跑本步骤。',
    }


JSON_ONLY = (
    '\n\n【输出要求】只输出一个 JSON（不要 markdown 代码块、不要解释文字），'
    '字符串一律使用中文。每个字段不超过 150 字，JSON 总长度控制在 1200 字以内，'
    '必须在输出长度上限到达之前写完最后一个右花括号。'
)
