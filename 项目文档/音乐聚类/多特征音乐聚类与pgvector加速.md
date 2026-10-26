# 基于调式 / 和声 / 旋律的多特征音乐聚类

用向量数据库方法对**已公开**的作品分类。全部代码在新增的 `cluster/` app 内，
不改动 `core/` 任何模块（只读 core 的分析结果算特征）。

## 一、三个索引与检索流程

| 轴 | 向量 | 存哪儿 | 检索 |
|---|---|---|---|
| 调式 mode | 12 维音级分布（按主音对齐） | `PieceVector.mode_vec` | 余弦 |
| 和声 harmony | 级数 7 + 性质 4 + 密度 1 = 12 维 | `PieceVector.harmony_vec` | 余弦 |
| 旋律 melody | 乐句级，一乐句一条 42 维指纹 | `PieceVector.melody_phrases` | 乐句 Top-M 粗排 → `theme.compare` 精排 |

查询时**各轴分别检索 Top-N，再用倒数排名融合（RRF）**合成最终排序：

```
score(d) = Σ_axis  w_axis / (k + rank_axis)      k=60，默认权重 调式 .3 / 和声 .3 / 旋律 .4
```

没进某轴 Top-N 的作品不从该轴拿分（RRF 语义：各轴只在自己候选里投票）。
聚类用**层次聚类 + 距离阈值**切分（类数由数据决定，不预设 k），距离为 `1 - 三轴加权相似度`。

## 二、调式为什么要"主音对齐"

调式向量的处理顺序是：**先定主音 → 把音级分布旋转到主音为 0 → 直接比余弦（不再移位）**。

这样一次同时满足两条要求（冒烟测试 A 组实测）：

| 对比 | 实测余弦 | 结论 |
|---|---|---|
| C大调 ↔ D大调 ↔ E大调 | **1.0000** | 不同调高、同一调式 → 归一成一类 ✓ |
| 自然大调 ↔ 多利亚 | 0.7143 | 能分开 ✓ |
| 自然大调 ↔ 立底亚 | 0.8571 | 能分开 ✓ |
| 循环移位降级时 自然大调 ↔ 多利亚 | **1.0000** | 会混淆 ✗（故必须主音对齐） |

最后一行是反证：D 多利亚的音级集合与 C 自然大调完全相同（都是七个白键），
所以"不按主音对齐、直接 12 次循环移位取最大"必然把两者判成一样。
端到端口径下（三轴融合）区分度也成立：大调↔大调 0.9320 > 大调↔多利亚 0.7610。

> 主音来自 `tonality.main_key`（LLM 判定）。若解析不出主音，会先用本地 KS 算法兜底；
> 实在定不了才降级为"循环移位比对"（此时只能比调式色彩，分不出具体调式）——
> 这时调式轴也会自动走纯 Python 路径（pgvector 的 `<=>` 做不了移位）。

## 三、pgvector 加速层（可选）

### 设计：为什么是"可选"而不是"强依赖"

pgvector 是 **PostgreSQL 服务端 C 扩展**。Python 的 `pgvector` 包只是客户端封装，
光 pip 装上不够，服务端必须真的装了扩展才能用 `vector` 类型。
为了让**任何环境都能跑**（含没装扩展的机器），架构是：

- **权威数据 = `PieceVector` 的 JSON 字段**，纯 Python 暴力余弦保证功能 100% 可用；
- 另建 raw-SQL 加速表（0002 迁移，缺扩展时自动跳过，绝不报错）：

  | 表 | 列 | 索引 |
  |---|---|---|
  | `cluster_pgvec_piece` | `mode_vec vector(12)`、`harmony_vec vector(12)` | 两列各一个 HNSW（余弦） |
  | `cluster_pgvec_phrase` | `vec vector(42)`、`payload jsonb` | HNSW（余弦）+ `(piece_id)` |

- `index.pg_available()` 探测（结果缓存）→ 可用则 SQL 走原生 `<=>`，
  否则静默回退纯 Python；两条路径结果一致（都是余弦）。

加速表是**纯缓存**，写入由 `sync_pg()` 在"重建特征"时自动完成，写失败也不影响主流程。

### 本机启用记录（2026-10-26，已跑通）

1. 放文件（PG 18.1，装在 `D:\Program Files\PostgreSQL\18`）：
   `lib\vector.dll`、`share\extension\vector.control`、`share\extension\vector--0.8.6.sql`
   —— 用的是社区预编译包 `andreiramani/pgvector_pgsql_windows` 的 `0.8.6_18`
   （官方只发源码；本机 VS2019 BuildTools + `postgres.lib` + `include\server` 齐全，也可自编译）。
2. **不需要重启服务**：放好文件后 `pg_available_extensions` 立即返回 `vector 0.8.6`。
3. `CREATE EXTENSION vector;` —— **必须用 postgres 超级用户**，`music_agent` 会报
   `permission denied to create extension`（pgvector 不是 trusted 扩展）。
   扩展是**按库**启用的，每个库都要执行一次。
4. 回滚重跑迁移（0002 之前已被标记为"已应用"，要回滚才会真正建表）：
   `python manage.py migrate cluster 0001` → `python manage.py migrate cluster`
5. 重建特征 → `sync_pg()` 把三轴灌进加速表。

以上 4–5 步已封装成**幂等一键脚本**：`python check_pgvector.py`
（加 `--skip-migrate` 保留现有表与数据，加 `--no-rebuild` 只做检查）。

实测（7 首公开作品 / 157 条乐句向量）：

| 检查 | 结果 |
|---|---|
| `pg_available()` | True |
| `cluster_pgvec_piece` / `_phrase` | 7 行 / 157 行 |
| HNSW 索引 | `pgvec_piece_mode`、`pgvec_piece_harmony`、`pgvec_phrase_vec` 三个 `vector_cosine_ops` |
| 三轴 pg 路径 vs 纯 Python | **逐首完全一致**（调式 7/7、和声 4/4、旋律 7/7 排序与分数相同） |
| `/cluster/`、`/cluster/piece/1/`、`/cluster/admin/` | 200 |

### 坑：IVFFlat 在小数据量上会漏召回（已改 HNSW）

0002 最初建的是 IVFFlat（`lists = 100`）。IVFFlat 查询只扫 `probes` 个聚类中心
（默认 1），而 7 行数据被分成 100 个 list、多数 list 为空 —— 实测**每首都少召回 1 条，
其中一首一条都查不到**，pg 路径与纯 Python 对不上。

处理：0003 迁移把三个索引换成 **HNSW**（`m=16, ef_construction=64`），
并在 `settings.py` 的 `DATABASES[...]['OPTIONS']` 加 `-c hnsw.ef_search=100`，
召回与精确余弦一致，曲库涨到几万条也不用调参。

## 四、使用

### 入口在哪（三处，任选其一）

| 位置 | 链接 | 谁可见 |
|---|---|---|
| 站点顶部导航栏 | 「音乐聚类」 `/cluster/` + 「**聚类管理**」 `/cluster/admin/` | 总览所有人；管理按钮仅 `is_staff` |
| Django 后台顶部（每页都有） | 站点标题旁的绿色条：聚类总览 \| **聚类管理 →** | 进得后台即可见 |
| `/admin/cluster/clusterrun/` 列表页右上角 | 绿色「执行聚类 / 重建特征 →」 | 同上 |

> **真正的执行入口是 `/cluster/admin/`（聚类管理台）**。
> `/admin/cluster/clusterrun/` 只能查看与发布/撤回，触发不了聚类，
> 也已禁止手动新增 ClusterRun（手写一条没有成员，只会误导）。

管理员（`is_staff`）在 `/cluster/` 管理台：

1. **重建特征**：算全部公开作品的三轴特征（分批，每批 50 首）；
2. **执行聚类**：按当前参数跑一次，可在"聚类参数"里调权重 / Top-N / k / 阈值；
3. **发布**：发布后普通用户才在总览看到结果；可给类命名（留空显示自动标签）。

普通登录用户只读：总览、类详情（含各成员三轴相似度）、某首曲子的相似作品（带 RRF 解释）。

## 五、回归

```
python smoke_cluster_test.py
```

独立 SQLite 库 + 独立 media，不污染真实数据。三组 23 项：

- **A 组**：调式轴算法（零外部依赖，不依赖 LLM / 数据库）—— 归一与可区分；
- **B 组**：端到端造 4 首曲 → core 流水线 → 三轴特征 → 聚类 → RRF 检索 → 区分度；
- **C 组**：pgvector 后端探测与降级安全性。
