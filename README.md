# 音乐分析和处理智能体

基于「音频 → 乐谱 → 大模型」三步走的音乐分析系统（Django + PostgreSQL + dsh/DeepSeek + dsh-musictheory）。

本次交付范围：**从输入（上传 / 格式转换）到「旋律形态分析」的完整链路**，
含分析因果链全部前置步骤；MuScriptor 音频转录为**预留接口**（见下文）。

## 一、快速启动

### 1. 环境要求

- Python 3.12+（含 Django / music21 / psycopg，`pip install -r requirements.txt`）
- PostgreSQL（本机 localhost:5432 已在监听）
- Node + dsh（DeepSeek Harness 命令行版），且 `headless` profile 已安装 dsh-musictheory 插件
  （本机已配置完成：`dsh --profile headless -` 可用）

### 2. 建库（psql 管理员执行一次）

```sql
CREATE DATABASE music_agent;
CREATE USER music_agent WITH PASSWORD 'music_agent';
GRANT ALL PRIVILEGES ON DATABASE music_agent TO music_agent;
-- PostgreSQL 15+ 还需：
\c music_agent
GRANT ALL ON SCHEMA public TO music_agent;
```

数据库名 / 用户 / 密码与 `.env`（从 `.env.example` 复制）保持一致即可。

### 3. 初始化并启动

```bash
cd 音乐分析和处理智能体
# 建议使用已有 venv：C:\Users\fzjwc\.workbuddy\binaries\python\envs\default
pip install -r requirements.txt          # 若未安装依赖
copy .env.example .env                    # Windows；Linux 用 cp
python manage.py makemigrations accounts core   #（仓库已带迁移文件，可跳过）
python manage.py migrate
python manage.py createsuperuser          # 创建第一个管理员
python manage.py runserver
```

访问 <http://127.0.0.1:8000/>。注册的是普通用户；
**管理员层级**：Django 后台 (/admin/) → 用户 → 勾选"员工状态"（is_staff），
即可在前端看到「系统设置」「后台管理」入口，并能在系统设置页调整复调阈值等参数。

## 二、使用流程

1. **上传作品**：MIDI / MusicXML 直接可分析；音频上传后自动走转换
   （音频需 MuScriptor，见第四节）。
2. **三个按钮（对应讨论定稿的分支策略）**：
   - 「分析节奏」→ 节奏网格与速度曲线 + 节奏型与律动分析
   - 「分析旋律」→ 配器（静态）→ 声部识别 → 复调检测 → 主旋律交接检测
     → 和弦进行 → 调性布局 → 旋律形态分析（依赖自动补齐，缺啥跑啥）
   - 「主旋律提取」→ 与旋律分析**并列**（可选输出；单声 → 唯一主旋律导出；
     复调 → 多候选 + 置信度；交接 → 归属时间线 + 拼接导出 MIDI/MusicXML）
3. **结果**：每个步骤一张状态卡，运行中自动轮询刷新；失败显示完整错误。
4. **系统设置（管理员）**：复调阈值、交接最少切换次数、动机相似度阈值、
   分析窗口、和弦聚合窗口、dsh profile / 超时，全部前端可调。

## 三、架构（与计划书一致）

```
上传 → [接入层] 三条路径在 MusicXML 汇合
        audio → MuScriptor(预留) → MIDI → music21 → MusicXML
        midi   → music21 → MusicXML
        musicxml → 直接使用
      → [分析引擎] 分析因果链（依赖自动展开、后台线程执行、结果落库）
        rhythm_grid → rhythm_pattern
        instrumentation(静态,零LLM) → voice_id(角色+显著性+归属时间线)
          → polyphony(阈值可调) / exchange(切换+动机相似度)
        chords(窗口聚合+chord_identify) → tonality(KS候选+LLM转调点)
        melody_extraction(并列/可选) / melody_shape(三分支:单声/per-voice/时间线)
      → [LLM] dsh --profile headless（dsh-musictheory 提供确定性乐理计算）
```

关键源码位置：

| 模块 | 文件 |
|---|---|
| MuScriptor 预留接口 | `core/converters/muscriptor.py` |
| MIDI→MusicXML | `core/converters/musicxml.py` |
| dsh 封装 | `core/analysis/llm.py` |
| 谱面解析 / KS / LCS 等 | `core/analysis/scoredata.py` |
| 节奏两步 | `core/analysis/rhythm.py` |
| 配器（静态） | `core/analysis/instrumentation.py` |
| 声部识别（显著性/时间线/密度） | `core/analysis/voice_id.py` |
| 复调 + 主旋律交接检测 | `core/analysis/voice_branches.py` |
| 和弦 / 调性 | `core/analysis/chords.py` / `tonality.py` |
| 主旋律提取 / 旋律形态 | `core/analysis/melody.py` |
| 流水线编排（依赖/线程） | `core/analysis/pipeline.py` |

## 四、MuScriptor 转录接口怎么接

`core/converters/muscriptor.py` 里有两个占位函数，**只需替换 raise 部分**：

```python
def transcribe_instrumental(audio_path, output_midi_path, progress_callback=None)
    # 纯器乐：model.transcribe_and_postprocess() 一步完成转录+节拍对齐

def transcribe_instrumental_with_voice(audio_path, output_midi_path, progress_callback=None)
    # 器乐+人声：两遍转录（第1遍整音频默认转录不加节奏；第2遍 instruments=["voice"]）
    # 合并坑：人声 NoteStartEvent.index 必须重新编号；detect_beat_grid_for() 两遍共用
```

模块 docstring 里写了**完整的实现要点**（来自两个实验脚本的结论）。
接好之后：上传音频 → 勾选"含人声"与否 → 系统自动选择对应管线 →
MIDI → MusicXML → 全部分析步骤自动可用。

## 五、dsh 说明

- 后端通过 `dsh --profile <name> -`（stdin 传任务）调用 DeepSeek；
- `headless` profile 已安装 dsh-musictheory（9 个 12-TET 工具），
  提示词要求模型"先算后说、凡算必验"（和弦识别用 chord_identify、
  音程核验用 interval_info、音阶核验用 scale_generate/scale_harmonize）；
- profile 名与超时可在系统设置页修改。
