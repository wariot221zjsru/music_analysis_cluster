/* 结构 / 织体 / 风格页：三个子窗口各自独立运行、独立重跑。
   卡片竖排、各占页面全宽，高度随内容自增长。 */
(function () {
  'use strict';

  var PIECE_ID = window.PIECE_ID;
  var CSRF = window.CSRF_TOKEN;
  var CAN_EDIT = window.CAN_EDIT;
  var STEPS = ['structure', 'texture', 'style'];

  var RERUN_SVG = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" '
    + 'stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    + '<path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v4h-4"/></svg>';

  var PILLS = {
    idle: ['未运行', ''],
    pending: ['等待中', 'pill-pending'],
    running: ['运行中…', 'pill-running'],
    done: ['已完成', 'pill-done'],
    error: ['失败', 'pill-error'],
  };

  var polling = null;

  function esc(s) {
    return (s === null || s === undefined ? '' : String(s)).replace(/[&<>"']/g,
      function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
      });
  }

  function pct(v) {
    return v === null || v === undefined ? '—' : (v * 100).toFixed(0) + '%';
  }

  function jpre(obj) {
    return '<pre class="json-pre">' + esc(JSON.stringify(obj, null, 2)) + '</pre>';
  }

  function tableFrom(rows, headers) {
    if (!rows.length) return '';
    var h = '<tr>' + headers.map(function (x) { return '<th>' + esc(x) + '</th>'; }).join('') + '</tr>';
    var b = rows.map(function (r) {
      return '<tr>' + r.map(function (c) { return '<td>' + esc(c) + '</td>'; }).join('') + '</tr>';
    }).join('');
    return '<table class="table small-table"><thead>' + h + '</thead><tbody>' + b + '</tbody></table>';
  }

  /* LLM 结果统一渲染：正常 / 截断修复 / 解析失败降级 */
  function renderLLM(obj, label) {
    if (!obj) return '';
    if (obj._error) {
      return '<div class="llm-box llm-failed"><b>' + esc(label) + '</b> '
        + '<span class="tag">解析失败</span>'
        + '<div class="kv small">' + esc(obj._error) + '</div>'
        + (obj._hint ? '<div class="kv small muted">' + esc(obj._hint) + '</div>' : '')
        + (obj._raw_text ? '<details><summary>查看原始回答</summary><pre class="json-pre">'
          + esc(obj._raw_text) + '</pre></details>' : '')
        + '<p class="muted small">点本卡右上角 ⟲ 只重跑本项，耗时长的依赖会直接复用。</p></div>';
    }
    return '<div class="llm-box"><b>' + esc(label) + '</b>'
      + (obj._truncated ? ' <span class="tag">截断修复：部分字段缺失</span>' : '')
      + jpre(obj) + '</div>';
  }

  var RENDERERS = {
    structure: function (o) {
      var f = o.form || {};
      var secs = (o.sections || []).map(function (s) {
        return [s.label, s.start_measure + '-' + s.end_measure, s.bars,
          s.cadence || '—', s.key || '—'];
      });
      var phs = (o.phrases || []).map(function (p) {
        return [p.index, p.start_measure + '-' + p.end_measure, p.bars,
          p.note_count, p.pitch_range, p.cadence || '—'];
      });
      return '<div class="kv"><b>曲式：</b><span class="highlight">' + esc(f.name)
        + '</span>（骨架 ' + esc(f.scheme) + '，置信度 ' + esc(f.confidence) + '）</div>'
        + '<div class="kv"><b>本地骨架推断：</b>' + esc(f.local_guess)
        + '　<b>乐句数：</b>' + esc(o.phrase_count)
        + '　<b>小节数：</b>' + esc(o.measure_count) + '</div>'
        + tableFrom(secs, ['乐段', '小节', '长度', '终止式', '调性'])
        + tableFrom(phs, ['#', '小节', '长度', '音符数', '音域', '终止式'])
        + renderLLM(o.llm_analysis, 'AI 结构叙述');
    },
    texture: function (o) {
      var d = o.density || {};
      var tl = (o.timeline || []).map(function (t) {
        return [t.start_measure + '-' + t.end_measure, t.texture, t.detail];
      });
      var ratios = o.ratios || {};
      var ratioText = Object.keys(ratios).map(function (k) {
        return k + ' ' + pct(ratios[k]);
      }).join('，');
      return '<div class="kv"><b>主织体：</b><span class="highlight">'
        + esc(o.main_texture) + '</span> ' + esc(o.main_detail) + '</div>'
        + '<div class="kv"><b>织体分布：</b>' + esc(ratioText || '—')
        + '　<b>复调判定：</b>' + (o.is_polyphonic ? '是' : '否') + '</div>'
        + '<div class="kv"><b>密度：</b>平均活跃声部 ' + esc(d.avg_active_parts)
        + '，每小节音符数 ' + esc(d.notes_per_bar) + '</div>'
        + tableFrom(tl, ['小节', '织体', '形态细节'])
        + renderLLM(o.llm_analysis, 'AI 织体叙述');
    },
    style: function (o) {
      var fp = o.fingerprint || {};
      var ev = o.evidence || [];
      var hints = o.local_hints || [];
      var html = '<div class="kv"><b>流派：</b><span class="highlight">'
        + esc(o.genre || '未判定') + '</span>　<b>年代：</b>' + esc(o.period || '—')
        + '　<b>地域：</b>' + esc(o.region || '—') + '</div>';
      if (o.description) {
        html += '<p class="llm-desc">' + esc(o.description) + '</p>';
      }
      if (ev.length) {
        html += '<ul class="ev-list">' + ev.map(function (e) {
          return '<li>' + esc(e) + '</li>';
        }).join('') + '</ul>';
      }
      html += '<div class="kv"><b>指纹：</b>主调 ' + esc(fp.main_key || '—')
        + '（' + esc(fp.mode) + '）　速度 ' + esc(fp.tempo)
        + '　曲式 ' + esc(fp.form || '—') + '　编制 ' + esc((fp.families || []).join('/')) + '</div>'
        + '<div class="kv"><b>材料：</b>和弦 ' + esc(fp.chord_count) + ' 个，高叠和弦 '
        + pct(fp.extended_ratio) + '；五声性 ' + pct(fp.pentatonic_ratio)
        + '；切分比例 ' + pct(fp.syncopation_ratio) + '</div>';
      if (hints.length) {
        html += '<div class="kv small muted">本地线索：' + esc(hints.join('；')) + '</div>';
      }
      return html + renderLLM(o.llm_analysis, 'AI 风格判断');
    },
  };

  /* ---------- 历史运行记录 ---------- */
  var HIST_OPEN = {};
  var RUN_COUNTS = {};

  function fmtTime(s) {
    return s ? String(s).slice(0, 19).replace('T', ' ') : '—';
  }

  function updateHistBtn(step) {
    var btn = document.querySelector('[data-histbtn="' + step + '"]');
    if (!btn) return;
    var n = RUN_COUNTS[step] || 0;
    btn.hidden = n === 0;
    btn.textContent = '🕘 历史 ' + n;
    btn.title = '查看历史运行记录（共 ' + n + ' 次）';
  }

  function renderHistoryList(step, items) {
    if (!items.length) return '<p class="muted small">暂无历史记录。</p>';
    var rows = items.map(function (it) {
      var cls = it.status === 'done' ? 'pill-done'
        : (it.status === 'error' ? 'pill-error' : '');
      var st = it.status === 'done' ? '已完成'
        : (it.status === 'error' ? '失败' : it.status);
      var dur = '';
      if (it.started_at && it.finished_at) {
        var sec = Math.round((new Date(it.finished_at) - new Date(it.started_at)) / 1000);
        if (sec >= 0) dur = '<span class="muted small">耗时 ' + sec + 's</span>';
      }
      return '<details class="hist-item" data-run="' + it.id + '" data-step="' + step + '">'
        + '<summary><span class="hist-seq">第 ' + it.seq + ' 次</span>'
        + '<span class="pill ' + cls + '">' + esc(st) + '</span>'
        + '<span class="muted small">' + esc(fmtTime(it.finished_at)) + '</span>'
        + '<span class="muted small">' + esc(it.trigger_label) + '</span>'
        + dur + '</summary>'
        + '<div class="hist-body"><p class="muted small">展开后加载…</p></div></details>';
    }).join('');
    return '<div class="hist-head">历史运行（共 ' + items.length + ' 次，最近在前）</div>' + rows;
  }

  function bindHistory(step) {
    document.querySelectorAll('.hist-item[data-step="' + step + '"]').forEach(function (d) {
      if (d.dataset.bound === '1') return;
      d.dataset.bound = '1';
      d.addEventListener('toggle', async function () {
        if (!d.open || d.dataset.loaded === '1') return;
        d.dataset.loaded = '1';
        var body = d.querySelector('.hist-body');
        try {
          var res = await fetch('/pieces/' + PIECE_ID + '/history/?step='
            + encodeURIComponent(step) + '&run=' + d.dataset.run);
          var data = await res.json();
          if (!data.ok) { body.innerHTML = '<p class="muted small">加载失败。</p>'; return; }
          if (data.item.status === 'error') {
            body.innerHTML = '<pre class="error-pre">' + esc(data.item.error) + '</pre>';
          } else if (data.item.output) {
            var fn = RENDERERS[step];
            body.innerHTML = fn ? fn(data.item.output) : jpre(data.item.output);
          } else {
            body.innerHTML = '<p class="muted small">该次运行没有留下输出。</p>';
          }
        } catch (e) {
          body.innerHTML = '<p class="muted small">加载失败。</p>';
        }
      });
    });
  }

  async function loadHistory(step, slot) {
    slot.innerHTML = '<p class="muted small">加载历史…</p>';
    try {
      var res = await fetch('/pieces/' + PIECE_ID + '/history/?step='
        + encodeURIComponent(step));
      var data = await res.json();
      if (!data.ok) { slot.innerHTML = '<p class="muted small">历史加载失败。</p>'; return; }
      slot.innerHTML = renderHistoryList(step, data.items);
      bindHistory(step);
    } catch (e) {
      slot.innerHTML = '<p class="muted small">历史加载失败。</p>';
    }
  }

  async function toggleHistory(step) {
    var slot = document.querySelector('[data-hist="' + step + '"]');
    if (!slot) return;
    if (HIST_OPEN[step]) {
      HIST_OPEN[step] = false;
      slot.innerHTML = '';
      return;
    }
    HIST_OPEN[step] = true;
    await loadHistory(step, slot);
  }

  function refreshOpenHistories() {
    STEPS.forEach(function (s) {
      if (!HIST_OPEN[s]) return;
      var slot = document.querySelector('[data-hist="' + s + '"]');
      if (slot) loadHistory(s, slot);
    });
  }

  function renderCard(step, r) {
    var pillEl = document.querySelector('[data-pill="' + step + '"]');
    var bodyEl = document.querySelector('[data-body="' + step + '"]');
    var labelEl = document.querySelector('[data-run-label="' + step + '"]');
    if (!pillEl || !bodyEl) return;

    if (!r) {
      pillEl.textContent = '未运行';
      pillEl.className = 'pill';
      bodyEl.innerHTML = '<p class="muted small">尚未运行。</p>';
      if (labelEl) labelEl.textContent = '运行';
      return;
    }
    var pair = PILLS[r.status] || [r.status, ''];
    pillEl.textContent = pair[0];
    pillEl.className = 'pill ' + pair[1];
    if (labelEl) labelEl.textContent = r.status === 'done' ? '重新运行' : '运行';

    if (r.status === 'pending' || r.status === 'running') {
      bodyEl.innerHTML = '<p class="muted small">运行中…缺失的依赖会自动补跑，'
        + '已完成的依赖直接复用（不重复耗时）。</p>';
      return;
    }
    if (r.status === 'error') {
      bodyEl.innerHTML = '<pre class="error-pre">' + esc(r.error) + '</pre>'
        + '<p class="muted small">修好后点右上角 ⟲ 只重跑本项。</p>';
      return;
    }
    if (r.status === 'done' && r.output) {
      var fn = RENDERERS[step];
      bodyEl.innerHTML = fn ? fn(r.output) : jpre(r.output);
      return;
    }
    bodyEl.innerHTML = '<p class="muted small">暂无结果。</p>';
  }

  function setButtonsDisabled(disabled) {
    document.querySelectorAll('.sts-card button').forEach(function (b) {
      b.disabled = disabled;
    });
  }

  async function post(url, body) {
    var res = await fetch(url, {
      method: 'POST',
      headers: Object.assign({ 'X-CSRFToken': CSRF },
        body ? { 'Content-Type': 'application/x-www-form-urlencoded' } : {}),
      body: body || null,
    });
    return res.json();
  }

  async function runStep(step) {
    var btn = document.querySelector('[data-run="' + step + '"]');
    if (btn) btn.disabled = true;
    try {
      var data = await post('/pieces/' + PIECE_ID + '/rerun/',
        new URLSearchParams({ step: step }).toString());
      if (!data.ok) {
        alert(data.error || '启动失败');
        if (btn) btn.disabled = false;
        return;
      }
      if (data.message) alert(data.message);
      startPolling();
    } catch (e) {
      if (btn) btn.disabled = false;
    }
  }

  async function poll() {
    try {
      var res = await fetch('/pieces/' + PIECE_ID + '/status/');
      var data = await res.json();
      var results = data.results || {};
      RUN_COUNTS = data.run_counts || {};
      STEPS.forEach(function (s) { renderCard(s, results[s]); updateHistBtn(s); });
      refreshOpenHistories();
      if (!data.any_running && polling) {
        clearInterval(polling);
        polling = null;
        setButtonsDisabled(false);
      }
    } catch (e) { /* 网络抖动忽略 */ }
  }

  function startPolling() {
    setButtonsDisabled(true);
    if (polling) return;
    polling = setInterval(poll, 2500);
  }

  /* 无操作权限时隐藏重跑按钮 */
  if (!CAN_EDIT) {
    document.querySelectorAll('[data-rerun]').forEach(function (b) {
      b.style.display = 'none';
    });
  }

  document.querySelectorAll('[data-rerun]').forEach(function (b) {
    b.innerHTML = RERUN_SVG;
    b.addEventListener('click', function () { runStep(b.dataset.rerun); });
  });
  document.querySelectorAll('[data-run]').forEach(function (b) {
    b.addEventListener('click', function () { runStep(b.dataset.run); });
  });
  document.querySelectorAll('[data-histbtn]').forEach(function (b) {
    b.addEventListener('click', function () { toggleHistory(b.dataset.histbtn); });
  });

  /* 首次加载：渲染一次，有任务在跑才起轮询 */
  fetch('/pieces/' + PIECE_ID + '/status/')
    .then(function (r) { return r.json(); })
    .then(function (d) {
      var results = d.results || {};
      RUN_COUNTS = d.run_counts || {};
      STEPS.forEach(function (s) { renderCard(s, results[s]); updateHistBtn(s); });
      if (d.any_running) startPolling();
    })
    .catch(function () { /* 忽略 */ });
})();
