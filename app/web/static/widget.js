/*
 * Чат-виджет онлайн-консультанта.
 * Подключение на сайте:  <script src="https://ВАШ-СЕРВЕР/widget.js" defer></script>
 * Необязательные атрибуты: data-color="#a6ed00" (акцент), data-dark="#1c1d20", data-title="Онлайн-консультант",
 *   data-open="1"
 */
(function () {
  "use strict";
  if (window.__consultantWidget) return;
  window.__consultantWidget = true;

  var script = document.currentScript || document.querySelector('script[src*="widget.js"]');
  var BASE = new URL(script.src, location.href).origin;
  var ACCENT = script.getAttribute("data-color") || "#a6ed00";  // акцент: кнопка отправки, индикатор, ответы менеджера
  var DARK = script.getAttribute("data-dark") || "#1c1d20";     // шапка и сообщения клиента
  var TITLE = script.getAttribute("data-title") || "Онлайн-консультант";
  var AUTO_OPEN = script.getAttribute("data-open") === "1";
  var STORAGE_KEY = "consultant_dialog_" + BASE;

  var state = { dialogId: localStorage.getItem(STORAGE_KEY) || "", lastId: 0, open: false, started: false,
                busy: false, pollTimer: null, status: "bot", seen: {} };

  var host = document.createElement("div");
  host.style.cssText = "position:fixed;right:0;bottom:0;z-index:2147483000;";
  document.body.appendChild(host);
  var root = host.attachShadow({ mode: "open" });

  var SHADOW = "0 0 0 1px rgba(0,0,0,.05),0 1px 2px rgba(0,0,0,.04),0 12px 32px -8px rgba(0,0,0,.18)";
  root.innerHTML =
    '<style>' +
    ':host{all:initial}' +
    '*{box-sizing:border-box;font-family:Manrope,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}' +
    '.btn{position:fixed;right:22px;bottom:22px;width:64px;height:64px;border-radius:50%;border:0;cursor:pointer;background:#fff;color:' + DARK + ';box-shadow:' + SHADOW + ';display:flex;align-items:center;justify-content:center;transition:transform .15s}' +
    '.btn:hover{transform:scale(1.06)}' +
    '.btn svg{width:30px;height:30px}' +
    '.badge{position:absolute;top:0;right:0;min-width:22px;height:22px;border-radius:11px;background:' + ACCENT + ';color:' + DARK + ';font-size:12px;font-weight:700;display:none;align-items:center;justify-content:center;padding:0 6px;box-shadow:0 0 0 3px #fff}' +
    '.panel{position:fixed;right:22px;bottom:100px;width:384px;max-width:calc(100vw - 24px);height:600px;max-height:calc(100vh - 120px);background:#ededed;border-radius:28px;box-shadow:0 30px 60px -24px rgba(28,29,32,.45),' + SHADOW + ';display:none;flex-direction:column;overflow:hidden;color:#232427;font-size:14.5px}' +
    '.panel.open{display:flex}' +
    '.head{background:' + DARK + ';color:#fff;padding:18px 18px 16px 22px;display:flex;align-items:center;gap:10px}' +
    '.head .t{font-family:Unbounded,Manrope,-apple-system,"Segoe UI",Arial,sans-serif;font-weight:500;font-size:15px;letter-spacing:-.01em}' +
    '.head .s{font-size:12.5px;color:#b9bcc4;margin-top:4px;display:flex;align-items:center}' +
    '.head .x{margin-left:auto;background:rgba(255,255,255,.1);border:0;color:#fff;width:36px;height:36px;border-radius:50%;font-size:20px;cursor:pointer;line-height:1}' +
    '.head .x:hover,.head .new:hover{background:rgba(255,255,255,.2)}' +
    '.head .new{background:rgba(255,255,255,.1);border:0;color:#fff;height:36px;border-radius:18px;padding:0 12px;font-size:12.5px;font-weight:600;cursor:pointer;white-space:nowrap}' +
    '.head .new + .x{margin-left:6px}' +
    '.dot{width:8px;height:8px;border-radius:50%;background:' + ACCENT + ';display:inline-block;margin-right:7px}' +
    '.msgs{flex:1;overflow-y:auto;padding:16px 14px;display:flex;flex-direction:column;gap:10px}' +
    '.m{max-width:86%;padding:11px 15px;border-radius:20px;line-height:1.48;white-space:pre-wrap;word-wrap:break-word}' +
    '.m.client{align-self:flex-end;background:' + DARK + ';color:#fff;border-bottom-right-radius:6px}' +
    '.m.bot{align-self:flex-start;background:#fff;border-bottom-left-radius:6px;box-shadow:0 0 0 1px rgba(0,0,0,.04),0 1px 2px rgba(0,0,0,.04)}' +
    '.m.operator{align-self:flex-start;background:' + ACCENT + ';color:' + DARK + ';border-bottom-left-radius:6px}' +
    '.who{font-size:11.5px;font-weight:700;margin-bottom:3px;opacity:.7}' +
    '.m a{color:inherit;text-decoration:underline;text-underline-offset:3px}' +
    '.doc{display:flex;align-items:center;gap:8px;margin-top:9px;padding:9px 14px;border-radius:100px;background:' + DARK + ';color:#fff !important;text-decoration:none !important;font-weight:600;font-size:13px}' +
    '.m.operator .doc{background:' + DARK + ';color:#fff !important}' +
    '.typing{align-self:flex-start;color:#6e7178;font-size:13px;padding:4px 8px}' +
    '.foot{padding:10px 12px 8px;display:flex;gap:8px;align-items:flex-end;background:#ededed}' +
    'textarea{flex:1;resize:none;border:0;border-radius:24px;padding:12px 16px;font-size:14.5px;max-height:120px;min-height:46px;outline:2px solid transparent;background:#fff;color:#232427;box-shadow:0 0 0 1px rgba(0,0,0,.05)}' +
    'textarea:focus{outline-color:' + DARK + '}' +
    '.send{border:0;background:' + ACCENT + ';color:' + DARK + ';border-radius:50%;width:46px;height:46px;cursor:pointer;flex:0 0 auto;display:flex;align-items:center;justify-content:center}' +
    '.send svg{width:20px;height:20px}' +
    '.send:disabled{opacity:.5;cursor:default}' +
    '.note{font-size:10.5px;color:#8a8d94;padding:0 18px 12px;background:#ededed}' +
    '.quick{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 10px}' +
    '.quick button{border:0;color:#232427;background:#fff;border-radius:100px;padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer;box-shadow:0 0 0 1px rgba(0,0,0,.08)}' +
    '.quick button:hover{background:' + DARK + ';color:#fff}' +
    '</style>' +
    '<button class="btn" aria-label="Открыть чат"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12a8 8 0 0 1-11.6 7.1L4 20l.9-4.4A8 8 0 1 1 20 12z"/><circle cx="8.5" cy="12" r=".9" fill="currentColor"/><circle cx="12" cy="12" r=".9" fill="currentColor"/><circle cx="15.5" cy="12" r=".9" fill="currentColor"/></svg><span class="badge">1</span></button>' +
    '<div class="panel" role="dialog" aria-label="' + TITLE + '">' +
    '  <div class="head"><div><div class="t"></div><div class="s"><span class="dot"></span><span class="st">Отвечаем сразу, 24/7</span></div></div><button class="new" title="Начать новый диалог" style="margin-left:auto">Новый диалог</button><button class="x" aria-label="Закрыть">×</button></div>' +
    '  <div class="msgs"></div>' +
    '  <div class="quick"></div>' +
    '  <div class="foot"><textarea rows="1" placeholder="Напишите вопрос…"></textarea><button class="send" aria-label="Отправить"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button></div>' +
    '  <div class="note"></div>' +
    '</div>';

  var $ = function (s) { return root.querySelector(s); };
  var btn = $(".btn"), panel = $(".panel"), msgs = $(".msgs"), input = $("textarea"), send = $(".send"),
      badge = $(".badge"), quick = $(".quick");
  $(".t").textContent = TITLE;

  var QUICK = ["Нужен MDaemon Standard", "Не знаю, что выбрать", "Продлить антивирус", "Как получить ключ?"];

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function linkify(text) {
    return esc(text).replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  }

  function addMessage(m) {
    if (m.id && state.seen[m.id]) return;
    if (m.id) { state.seen[m.id] = true; state.lastId = Math.max(state.lastId, m.id); }
    var el = document.createElement("div");
    el.className = "m " + m.role;
    var who = m.role === "operator" ? '<div class="who">Менеджер</div>' : "";
    var html = who + linkify(m.text);
    (m.documents || []).forEach(function (d) {
      html += '<a class="doc" href="' + esc(d.url) + '" target="_blank" rel="noopener">' + esc(d.title) + " № " + esc(d.number) + "</a>";
    });
    el.innerHTML = html;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    if (!state.open && !state.quiet && m.role !== "client") {
      badge.style.display = "flex";
    }
  }

  function setTyping(on) {
    var t = root.querySelector(".typing");
    if (on && !t) {
      t = document.createElement("div");
      t.className = "typing";
      t.textContent = "Консультант печатает…";
      msgs.appendChild(t);
      msgs.scrollTop = msgs.scrollHeight;
    } else if (!on && t) {
      t.remove();
    }
  }

  function setStatus(status) {
    state.status = status;
    $(".st").textContent = status === "operator" ? "В диалоге менеджер" :
      status === "waiting_operator" ? "Подключаем менеджера…" : "Отвечаем сразу, 24/7";
  }

  function api(path, method, body) {
    return fetch(BASE + path, {
      method: method || "GET",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined
    }).then(function (r) {
      if (!r.ok) return r.json().catch(function () { return {}; }).then(function (j) {
        throw new Error(j.detail || ("Ошибка " + r.status));
      });
      return r.json();
    });
  }

  function start() {
    if (state.started) return Promise.resolve();
    state.started = true;
    return api("/api/chat/start", "POST", { dialog_id: state.dialogId, page_url: location.href }).then(function (d) {
      state.dialogId = d.dialog_id;
      localStorage.setItem(STORAGE_KEY, d.dialog_id);
      $(".note").textContent = d.privacy_notice || "";
      state.quiet = true;  // история и приветствие не считаются новыми сообщениями
      addMessage({ role: "bot", text: d.greeting });
      (d.messages || []).forEach(addMessage);
      state.quiet = false;
      setStatus(d.status);
      if (!(d.messages || []).length && !input.value) renderQuick();
    }).catch(function (e) {
      state.started = false;
      addMessage({ role: "bot", text: "Не удалось подключиться к чату. " + e.message });
    });
  }

  function renderQuick() {
    quick.innerHTML = "";
    QUICK.forEach(function (q) {
      var b = document.createElement("button");
      b.textContent = q;
      b.onclick = function () { input.value = q; sendMessage(); };
      quick.appendChild(b);
    });
  }

  function sendMessage() {
    var text = input.value.trim();
    if (!text || state.busy) return;
    quick.innerHTML = "";
    input.value = "";
    autoresize();
    addMessage({ role: "client", text: text });
    state.busy = true;
    send.disabled = true;
    setTyping(true);
    start().then(function () {
      return api("/api/chat/" + state.dialogId + "/messages", "POST", { text: text });
    }).then(function (d) {
      setTyping(false);
      d.messages.forEach(addMessage);
      setStatus(d.status);
    }).catch(function (e) {
      setTyping(false);
      addMessage({ role: "bot", text: "Сообщение не отправлено: " + e.message + ". Попробуйте ещё раз." });
    }).then(function () {
      state.busy = false;
      send.disabled = false;
      input.focus();
    });
  }

  function poll() {
    if (!state.dialogId || state.busy) return;
    api("/api/chat/" + state.dialogId + "/messages?after=" + state.lastId).then(function (d) {
      d.messages.forEach(addMessage);
      setStatus(d.status);
    }).catch(function () {});
  }

  function toggle(open) {
    state.open = open === undefined ? !state.open : open;
    panel.classList.toggle("open", state.open);
    if (state.open) {
      badge.style.display = "none";
      start();
      input.focus();
    }
  }

  function autoresize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 110) + "px";
  }

  btn.onclick = function () { toggle(); };
  $(".x").onclick = function () { toggle(false); };
  $(".new").onclick = function () {
    // новый диалог: прежний останется у менеджера в панели, а в окне чата — чистый лист
    localStorage.removeItem(STORAGE_KEY);
    state.dialogId = ""; state.lastId = 0; state.seen = {}; state.started = false; state.status = "bot";
    msgs.innerHTML = "";
    start().then(function () { input.focus(); });
  };
  send.onclick = sendMessage;
  input.addEventListener("input", autoresize);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  // Новые ответы менеджера приходят даже при свёрнутом чате
  state.pollTimer = setInterval(function () {
    if (state.started || state.dialogId) {
      if (!state.started && state.dialogId) { start(); }
      poll();
    }
  }, 4000);

  if (AUTO_OPEN) toggle(true);
  window.ConsultantWidget = {
    open: function () { toggle(true); },
    close: function () { toggle(false); },
    prefill: function (text) { quick.innerHTML = ""; input.value = text; autoresize(); input.focus(); },
    send: function (text) { input.value = text; sendMessage(); }
  };
})();
