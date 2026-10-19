"""把升级前已存在的分析结果补进历史表（作为第 1 次运行）。

这样老作品在页面上也能看到「历史 1」入口，而不是空空如也。
"""
from django.db import migrations


def backfill(apps, schema_editor):
    AnalysisResult = apps.get_model('core', 'AnalysisResult')
    AnalysisRun = apps.get_model('core', 'AnalysisRun')
    for r in AnalysisResult.objects.filter(status__in=['done', 'error']).iterator():
        AnalysisRun.objects.update_or_create(
            piece=r.piece, step=r.step, seq=1,
            defaults={
                'status': r.status,
                'output': r.output,
                'error': r.error,
                'trigger': 'legacy',
                'started_at': r.started_at,
                'finished_at': r.finished_at,
            },
        )


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [('core', '0004_analysisrun')]

    operations = [migrations.RunPython(backfill, noop)]
