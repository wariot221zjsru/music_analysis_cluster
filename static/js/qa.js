/* 自定义问答页：提问 → 后台回答 → 轮询呈现。
   答案按自然段落渲染（模型返回纯文本，非 JSON）。 */
(function () {
  'use strict';

  var PIECE_ID = window.PIECE_ID;
  var CSRF = window.CSRF_TOKEN;
  var CAN_EDIT = !!window.CAN_EDIT;
  var polling = null;

  function esc(s) {
    return (s === null || s === undefined ? '' : String(s)).replace(/[&<>"']/g,
      function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
      });
  }

  function nl2br(text) {
    return esc(text).replace(/\n{2,}/g, '\n').replace(/\n/g, '<br>');
  }

  function fmt(t) {
    return t ? String(t).slice(0, 16).replace('T', ' ') : '';
  }

  function renderList(items) {
    var box = document.getElementById('qa-list');
    if (!box) return;
    var totalEl = document.getElementById('qa-total');
    if (!items.length) {
      box.innerHTML = '<p class="muted small">还没有提问。'
        + '历史提问会一直保留在这里。</p>';
      if (totalEl) totalEl.textContent = '0';
      return;
    }
    box.innerHTML = items.map(function (q) {
      var reask = CAN_EDIT
        ? '<button class="qa-reask" type="button" data-reask="'
          + esc(q.question) + '" title="把这个问题填回输入框再问一次">↻ 再问一次</button>'
        : '';
      var head = '<div class="qa-item">'
        + '<div class="qa-question"><b>问：</b>' + nl2br(q.question)
        + '<span class="qa-meta muted small">'
        + esc(q.username) + ' · ' + esc(fmt(q.created_at))
        + (q.finished_at ? ' → 答于 ' + esc(fmt(q.finished_at)) : '')
        + '</span>' + reask + '</div>';
      if (q.status === 'pending' || q.status === 'running') {
        return head + '<div class="qa-answer qa-waiting">回答中…（模型需结合工具核验，请稍候）</div></div>';
      }
      if (q.status === 'error') {
        return head + '<div class="qa-answer qa-error">' + esc(q.error || '回答失败')
          + '</div><div class="muted small">可稍后重新提问；若反复失败，'
          + '到 /settings/ 调大 dsh 超时。</div></div>';
      }
      var based = (q.based_on || []).length
        ? '<div class="muted small">参考了：' + esc((q.based_on || []).join('、')) + '</div>'
        : '';
      var missing = (q.missing || []).length
        ? '<div class="muted small">未跑（模型不会臆断这些方面）：'
          + esc((q.missing || []).join('、')) + '</div>'
        : '';
      return head + '<div class="qa-answer">' + nl2br(q.answer) + '</div>'
        + based + missing + '</div>';
    }).join('');
    box.querySelectorAll('[data-reask]').forEach(function (b) {
      b.addEventListener('click', function () {
        var input = document.getElementById('q-input');
        if (!input) return;
        input.value = b.dataset.reask;
        input.focus();
        input.scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    });
  }

  async function refresh() {
    try {
      var res = await fetch('/pieces/' + PIECE_ID + '/ask/status/');
      var data = await res.json();
      if (!data.ok) return;
      renderList(data.items || []);
      var totalEl = document.getElementById('qa-total');
      if (totalEl && typeof data.total === 'number') {
        totalEl.textContent = String(data.total);
      }
      if (data.any_running) {
        startPolling();
      } else if (polling) {
        clearInterval(polling);
        polling = null;
      }
    } catch (e) { /* 网络抖动忽略 */ }
  }

  function startPolling() {
    if (polling) return;
    polling = setInterval(refresh, 3000);
  }

  var form = document.getElementById('qa-form');
  if (form) {
    form.addEventListener('submit', async function (e) {
      e.preventDefault();
      var input = document.getElementById('q-input');
      var tip = document.getElementById('qa-tip');
      var btn = document.getElementById('qa-submit');
      var question = (input.value || '').trim();
      if (!question) { if (tip) tip.textContent = '请先输入问题。'; return; }
      btn.disabled = true;
      if (tip) tip.textContent = '已提交，正在回答…';
      try {
        var res = await fetch('/pieces/' + PIECE_ID + '/ask/submit/', {
          method: 'POST',
          headers: {
            'X-CSRFToken': CSRF,
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: new URLSearchParams({ question: question }).toString(),
        });
        var data = await res.json();
        if (!data.ok) {
          if (tip) tip.textContent = data.error || '提交失败';
          btn.disabled = false;
          return;
        }
        input.value = '';
        if (tip) tip.textContent = '';
        refresh();
        startPolling();
      } catch (err) {
        if (tip) tip.textContent = '网络错误，请重试。';
      } finally {
        btn.disabled = false;
      }
    });
  }

  /* 首屏：直接用服务端带过来的历史，避免空窗；随后轮询接管 */
  var initialEl = document.getElementById('qa-initial');
  if (initialEl) {
    try {
      renderList(JSON.parse(initialEl.textContent) || []);
    } catch (e) { /* 解析失败就等轮询 */ }
  }

  refresh();
})();
