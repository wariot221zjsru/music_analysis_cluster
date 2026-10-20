/* 已转换 MIDI 的浏览器内试听：最小 MIDI 解析 + Web Audio 合成，零外部依赖。

   用法（模板里只加一行按钮即可）：
     <button class="btn btn-outline btn-sm midi-play" type="button"
             data-midi-url="{% url 'core:download_midi' piece.id %}">▶ 播放</button>
     <span class="midi-status" data-midi-status></span>

   绑定规则：扫描 .midi-play[data-midi-url]，状态文本写进同一父元素内的
   [data-midi-status]。音符用简单振荡器合成，不是真实音源，仅用于快速核对
   转换结果是否正确。 */
(function () {
  'use strict';

  var AHEAD = 0.25;        // 提前调度窗口（秒）
  var TICK = 25;           // 调度器轮询间隔（毫秒）
  var MASTER_GAIN = 0.55;  // 总线音量
  var FALLBACK_TPQ = 480;  // SMPTE 时间码的兜底分辨率

  /* ---------------- MIDI 解析 ---------------- */

  function ascii(view, off, len) {
    var s = '';
    for (var i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(off + i));
    return s;
  }

  function vlq(view, pos) {
    var value = 0, byte;
    do {
      byte = view.getUint8(pos++);
      value = (value << 7) | (byte & 0x7F);
    } while (byte & 0x80);
    return { value: value, next: pos };
  }

  function parseMidi(buffer) {
    var view = new DataView(buffer);
    if (view.byteLength < 14 || ascii(view, 0, 4) !== 'MThd') {
      throw new Error('不是合法的 MIDI 文件');
    }
    var headerLen = view.getUint32(4);
    var ntrks = view.getUint16(10);
    var division = view.getInt16(12);
    var ticksPerQn = division > 0 ? division : FALLBACK_TPQ;
    var pos = 8 + headerLen;

    var tempos = [];                 // {tick, usPerQn}
    var raw = [];                    // {tick, durTick, pitch, vel, program, channel}
    var pending = {};                // 'ch:pitch' -> 起始事件
    var program = {};                // channel -> program

    for (var t = 0; t < ntrks && pos + 8 <= view.byteLength; t++) {
      if (ascii(view, pos, 4) !== 'MTrk') {          // 非常规块：跳过
        pos += 8 + view.getUint32(pos + 4);
        continue;
      }
      var end = pos + 8 + view.getUint32(pos + 4);
      if (end > view.byteLength) end = view.byteLength;
      pos += 8;
      var tick = 0, status = 0;

      while (pos < end) {
        var v = vlq(view, pos);
        pos = v.next;
        tick += v.value;
        var b = view.getUint8(pos);
        if (b < 0x80) {                              // running status
          if (!status) throw new Error('轨道数据异常');
        } else {
          status = b;
          pos++;
        }
        if (status === 0xFF) {                       // meta
          var meta = view.getUint8(pos++);
          var lv = vlq(view, pos);
          pos = lv.next;
          if (meta === 0x51 && lv.value === 3) {
            tempos.push({
              tick: tick,
              usPerQn: (view.getUint8(pos) << 16) | (view.getUint8(pos + 1) << 8)
                | view.getUint8(pos + 2),
            });
          }
          pos += lv.value;
          if (meta === 0x2F) break;                  // end of track
          continue;
        }
        if (status === 0xF0 || status === 0xF7) {    // sysex
          var sv = vlq(view, pos);
          pos = sv.next + sv.value;
          continue;
        }
        var hi = status & 0xF0, ch = status & 0x0F;
        var d1 = view.getUint8(pos++);
        if (hi === 0xC0) { program[ch] = d1; continue; }
        if (hi === 0xD0) { continue; }
        var d2 = view.getUint8(pos++);
        if (hi === 0x90 && d2 > 0) {
          pending[ch + ':' + d1] = {
            tick: tick, vel: d2, channel: ch, program: program[ch] || 0,
          };
        } else if (hi === 0x80 || (hi === 0x90 && d2 === 0)) {
          var key = ch + ':' + d1, open = pending[key];
          if (open) {
            delete pending[key];
            raw.push({
              tick: open.tick,
              durTick: Math.max(tick - open.tick, 1),
              pitch: d1, vel: open.vel,
              program: open.program, channel: ch,
            });
          }
        }
      }
      pos = end;
    }
    if (!raw.length) throw new Error('MIDI 里没有可播放的音符');

    tempos.sort(function (a, b) { return a.tick - b.tick; });
    if (!tempos.length || tempos[0].tick !== 0) {
      tempos.unshift({ tick: 0, usPerQn: 500000 });   // 默认 120 BPM
    }
    var acc = [], sec = 0;
    for (var i = 0; i < tempos.length; i++) {
      if (i > 0) {
        sec += (tempos[i].tick - tempos[i - 1].tick) * tempos[i - 1].usPerQn
          / (ticksPerQn * 1e6);
      }
      acc.push({ tick: tempos[i].tick, sec: sec, usPerQn: tempos[i].usPerQn });
    }
    function toSec(tk) {
      var k = acc.length - 1;
      while (k > 0 && acc[k].tick > tk) k--;
      return acc[k].sec + (tk - acc[k].tick) * acc[k].usPerQn / (ticksPerQn * 1e6);
    }

    var notes = raw.map(function (n) {
      var start = toSec(n.tick);
      return {
        time: start,
        dur: Math.max(toSec(n.tick + n.durTick) - start, 0.04),
        pitch: n.pitch, vel: n.vel, program: n.program, channel: n.channel,
      };
    }).sort(function (a, b) { return a.time - b.time; });

    var total = 0;
    notes.forEach(function (n) { total = Math.max(total, n.time + n.dur); });
    return { notes: notes, duration: total };
  }

  /* ---------------- Web Audio 合成 ---------------- */

  function waveFor(program) {
    if (program <= 7) return 'triangle';      // 钢琴
    if (program <= 15) return 'sine';         // 色彩打击
    if (program <= 23) return 'sawtooth';     // 风琴
    if (program <= 31) return 'triangle';     // 吉他
    if (program <= 39) return 'triangle';     // 贝斯
    if (program <= 47) return 'sawtooth';     // 弦乐
    if (program <= 55) return 'triangle';     // 合奏 / 人声
    if (program <= 63) return 'sawtooth';     // 铜管
    if (program <= 71) return 'square';       // 簧片
    if (program <= 79) return 'sine';         // 管乐
    return 'square';                          // 合成与其它
  }

  function Player() {
    this.ctx = null;
    this.master = null;
    this.noise = null;
    this.notes = [];
    this.index = 0;
    this.t0 = 0;
    this.duration = 0;
    this.timer = null;
  }

  Player.prototype.ensureCtx = function () {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) throw new Error('当前浏览器不支持 Web Audio');
    if (!this.ctx) {
      this.ctx = new Ctx();
      this.master = this.ctx.createGain();
      this.master.gain.value = MASTER_GAIN;
      var comp = this.ctx.createDynamicsCompressor();
      this.master.connect(comp);
      comp.connect(this.ctx.destination);
      var len = Math.floor(this.ctx.sampleRate * 0.5);
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      var data = this.noise.getChannelData(0);
      for (var i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
  };

  Player.prototype.schedule = function (note) {
    var ctx = this.ctx;
    var at = Math.max(this.t0 + note.time, ctx.currentTime);
    var peak = Math.max(0.02, (note.vel / 127) * 0.22);
    var dur = Math.min(note.dur, 8);
    var g = ctx.createGain();

    if (note.channel === 9) {                 // 打击乐：白噪声短音
      var src = ctx.createBufferSource();
      src.buffer = this.noise;
      var bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = note.pitch < 60 ? 180 : 1400;
      g.gain.setValueAtTime(peak * 1.2, at);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 0.12);
      src.connect(bp); bp.connect(g); g.connect(this.master);
      src.start(at); src.stop(at + 0.14);
      src.onended = function () { try { src.disconnect(); bp.disconnect(); g.disconnect(); } catch (e) {} };
      return;
    }

    var osc = ctx.createOscillator();
    osc.type = waveFor(note.program);
    osc.frequency.value = 440 * Math.pow(2, (note.pitch - 69) / 12);
    g.gain.setValueAtTime(0.0001, at);
    g.gain.linearRampToValueAtTime(peak, at + 0.012);
    g.gain.linearRampToValueAtTime(peak * 0.6, at + Math.min(0.18, dur * 0.5));
    g.gain.setTargetAtTime(0.0001, at + dur, 0.05);
    osc.connect(g); g.connect(this.master);
    osc.start(at);
    osc.stop(at + dur + 0.3);
    osc.onended = function () { try { osc.disconnect(); g.disconnect(); } catch (e) {} };
  };

  Player.prototype.start = function (notes, duration, onEnd) {
    var self = this;
    this.ensureCtx();
    this.notes = notes;
    this.duration = duration;
    this.index = 0;
    this.t0 = this.ctx.currentTime + 0.12;
    this.onEnd = onEnd || function () {};
    this.timer = setInterval(function () { self.pump(); }, TICK);
    this.pump();
  };

  Player.prototype.pump = function () {
    var now = this.ctx.currentTime;
    while (this.index < this.notes.length
      && this.t0 + this.notes[this.index].time < now + AHEAD) {
      this.schedule(this.notes[this.index]);
      this.index++;
    }
    if (this.index >= this.notes.length && now - this.t0 > this.duration + 0.5) {
      this.stop();
      this.onEnd();
    }
  };

  Player.prototype.position = function () {
    if (!this.ctx) return 0;
    return Math.max(0, this.ctx.currentTime - this.t0);
  };

  Player.prototype.playing = function () { return this.timer !== null; };

  Player.prototype.stop = function () {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    if (this.master) {
      try { this.master.gain.value = 0; } catch (e) {}
      var old = this.master;
      this.master = null;
      try { old.disconnect(); } catch (e) {}
      var ctx = this.ctx;
      this.ctx = null;                        // 丢弃整条链路，静音立即生效
      if (ctx) { try { ctx.close(); } catch (e) {} }
    }
  };

  /* ---------------- 页面绑定 ---------------- */

  function fmt(sec) {
    var s = Math.max(0, Math.floor(sec));
    return Math.floor(s / 60) + ':' + ('0' + (s % 60)).slice(-2);
  }

  var cache = {};      // url -> 解析结果（同一首曲子只解析一次）

  function bind(btn) {
    var url = btn.getAttribute('data-midi-url');
    if (!url) return;
    var status = btn.parentNode.querySelector('[data-midi-status]');
    var player = new Player();
    var ticker = null;

    function setStatus(text, isError) {
      if (!status) return;
      status.textContent = text || '';
      status.className = 'midi-status' + (isError ? ' midi-status-error' : '');
    }

    function reset() {
      btn.textContent = '▶ 播放';
      btn.classList.remove('is-playing');
      btn.disabled = false;
      if (ticker) { clearInterval(ticker); ticker = null; }
    }

    function finish() {
      player.stop();
      setStatus('');
      reset();
    }

    btn.addEventListener('click', function () {
      if (player.playing()) { finish(); return; }

      btn.disabled = true;
      btn.textContent = '载入…';
      setStatus('');

      var task = cache[url];
      if (!task) {
        task = fetch(url, { credentials: 'same-origin' })
          .then(function (res) {
            if (!res.ok) throw new Error('下载失败（HTTP ' + res.status + '）');
            return res.arrayBuffer();
          })
          .then(function (buf) { return parseMidi(buf); });
        cache[url] = task;
      }
      task.then(function (song) {
        cache[url] = Promise.resolve(song);
        btn.disabled = false;
        btn.textContent = '⏹ 停止';
        btn.classList.add('is-playing');
        player.start(song.notes, song.duration, finish);
        if (status) {
          setStatus('0:00 / ' + fmt(song.duration));
          ticker = setInterval(function () {
            if (!player.playing()) return;
            setStatus(fmt(player.position()) + ' / ' + fmt(song.duration));
          }, 250);
        }
      }).catch(function (err) {
        cache[url] = null;
        setStatus('试听失败：' + (err && err.message ? err.message : '未知错误'), true);
        reset();
      });
    });
  }

  function init() {
    var btns = document.querySelectorAll('.midi-play[data-midi-url]');
    for (var i = 0; i < btns.length; i++) bind(btns[i]);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
