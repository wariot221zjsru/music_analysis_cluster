"""dsh（DeepSeek Harness 命令行版）封装。

后端通过 `dsh --profile <name> -`（stdin 传任务、stdout 收答案）调用
DeepSeek；dsh-musictheory 插件为模型提供 12-TET 音乐理论数学工具箱
（chord_identify / interval_info / scale_harmonize 等 9 个工具），
提示词中要求模型"先算后说、凡算必验"。
"""
import json
import re
import shutil
import subprocess
import sys


class LLMError(RuntimeError):
    """LLM 调用失败（超时 / 命令不可用 / 输出无法解析）。"""


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


def extract_json(text: str):
    """从模型回答中提取 JSON（容忍代码块围栏与前后说明文字）。"""
    text = re.sub(r'```(?:json)?', '', text).strip('` \n')
    for opener, closer in (('{', '}'), ('[', ']')):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise LLMError(f'无法从模型回答中解析 JSON。回答前 500 字：{text[:500]}')


def run_llm_json(task: str, profile: str = 'headless', timeout: int = 240):
    """调用 LLM 并解析为 JSON 对象 / 数组。"""
    return extract_json(run_llm(task, profile=profile, timeout=timeout))


JSON_ONLY = (
    '\n\n【输出要求】只输出一个 JSON（不要 markdown 代码块、不要解释文字），'
    '字符串一律使用中文。'
)
