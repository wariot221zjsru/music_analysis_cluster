"""聚类历史记录 + 参数留档的回归脚本。

覆盖用户要求：
  1. 已发布的聚类结果永久存档，/cluster/history/ 对**所有登录用户**可见；
  2. 未发布的运行普通用户看不到（404），管理员能看到；
  3. 已发布结果页面能看见当次完整参数：三轴权重 / 各轴 Top-N / RRF 常数 k /
     旋律乐句粗排 / 旋律曲级聚合方式 / 聚类切分阈值；
  4. 每次保存参数都留一条 ClusterSettingLog，历史页对所有人可见；
  5. 总览页 / 类详情页同样带参数面板。

用法：python smoke_cluster_history_test.py
"""
import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'smoke_cluster_settings'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django  # noqa: E402

django.setup()

from django.contrib.auth.models import User  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.test import Client  # noqa: E402

from cluster.models import Cluster, ClusterRun, ClusterSetting, ClusterSettingLog  # noqa: E402
from core.models import Piece  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


call_command('migrate', verbosity=0)

admin = User.objects.create_user('admin_u', password='x', is_staff=True)
normal = User.objects.create_user('normal_u', password='x')

# 一次"已发布"的运行（参数快照齐全）
params = {
    'w_mode': 0.25, 'w_harmony': 0.30, 'w_melody': 0.45,
    'top_n': 15, 'rrf_k': 42, 'phrase_top_m': 8,
    'melody_agg': 'mean_max', 'cluster_threshold': 0.38,
}
run_pub = ClusterRun.objects.create(
    created_by=admin, status=ClusterRun.Status.DONE, params=params,
    piece_count=5, cluster_count=2, published=True)
run_hidden = ClusterRun.objects.create(
    created_by=admin, status=ClusterRun.Status.DONE, params=params,
    piece_count=3, cluster_count=1, published=False)

p1 = Piece.objects.create(title='曲一', owner=normal, is_public=True)
p2 = Piece.objects.create(title='曲二', owner=normal, is_public=True)
c1 = Cluster.objects.create(run=run_pub, index=1, auto_label='大调 · 级进为主',
                            size=2, medoid=p1)
Cluster.objects.create(run=run_pub, index=2, auto_label='小调 · 跳进较多',
                       size=1, medoid=p2)

# 参数留档：模拟管理员两次保存参数
ClusterSettingLog.objects.create(created_by=admin, **params)
ClusterSettingLog.objects.create(
    created_by=admin, **{**params, 'cluster_threshold': 0.50})

print('1) 历史页对普通用户可见')
c = Client()
c.force_login(normal)
resp = c.get('/cluster/history/')
check(resp.status_code == 200, f'/cluster/history/ 普通用户 200（实际 {resp.status_code}）')
body = resp.content.decode()
check(str(run_pub.pk) in body, '已发布运行出现在历史页')
check(f'/cluster/run/{run_hidden.pk}/' not in body, '未发布运行对普通用户隐藏')
check(body.count('/cluster/run/') == 1, '普通用户历史页只有已发布那一条')
check('参数设置记录' in body, '历史页含参数设置记录区块')
check('参数设置记录' in body and ClusterSettingLog.objects.count() == 2,
      '参数留档两条都在')

print('2) 已发布运行详情页：普通用户可见 + 参数齐全')
resp = c.get(f'/cluster/run/{run_pub.pk}/')
check(resp.status_code == 200, f'已发布运行详情 200（实际 {resp.status_code}）')
body = resp.content.decode()
for label, val in [('调式轴权重', '0.25'), ('和声轴权重', '0.3'), ('旋律轴权重', '0.45'),
                   ('各轴 Top-N', '15'), ('RRF 常数 k', '42'),
                   ('旋律乐句粗排（每乐句取几条）', '8'),
                   ('旋律曲级聚合方式', '各乐句最佳匹配均值（更稳）'),
                   ('聚类切分阈值', '0.38')]:
    check(label in body, f'页面显示参数「{label}」')
    check(val in body, f'页面显示取值「{val}」')
check('大调 · 级进为主' in body, '详情页列出当次聚类')

print('3) 未发布运行详情页：普通用户 404')
check(c.get(f'/cluster/run/{run_hidden.pk}/').status_code == 404,
      '未发布运行详情对普通用户 404')

print('4) 总览页带参数面板 + 历史入口')
resp = c.get('/cluster/')
body = resp.content.decode()
check('本次聚类使用的参数' in body, '总览页有参数面板')
check('聚类历史记录' in body, '总览页有历史入口')
check('RRF 常数 k' in body and '0.38' in body, '总览页显示当次参数值')

print('5) 类详情页带参数面板')
resp = c.get(f'/cluster/c/{c1.id}/')
check(resp.status_code == 200, '类详情 200')
check('本次聚类使用的参数' in resp.content.decode(), '类详情页有参数面板')

print('6) 管理员：未发布也能看')
ca = Client()
ca.force_login(admin)
body = ca.get('/cluster/history/').content.decode()
check(str(run_hidden.pk) in body, '管理员在历史页能看到未发布运行')

# 清理
ClusterRun.objects.all().delete()
ClusterSettingLog.objects.all().delete()
Piece.objects.all().delete()
User.objects.all().delete()

print('\n结果：' + ('全部通过' if not FAILS else f'{len(FAILS)} 项失败'))
sys.exit(1 if FAILS else 0)
