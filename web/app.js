/* AI Lab 前端：对话（SSE 流式） + 目标检测（canvas 画框）。 */

const $ = (id) => document.getElementById(id);

const els = {
  provider: $("provider"),
  model: $("model"),
  tools: $("tools"),
  transcript: $("transcript"),
  composer: $("composer"),
  prompt: $("prompt"),
  send: $("send"),
  stop: $("stop"),
  clearChat: $("clear-chat"),
  chatHint: $("chat-hint"),
  llmPill: $("llm-pill-text"),
  llmDot: $("llm-dot"),
  visionPill: $("vision-pill-text"),
  visionDot: $("vision-dot"),
  engineBadge: $("engine-badge"),
  stage: $("stage"),
  canvasWrap: $("canvas-wrap"),
  dropzone: $("dropzone"),
  pick: $("pick"),
  file: $("file"),
  detect: $("detect"),
  sample: $("sample"),
  clearImage: $("clear-image"),
  visionStatus: $("vision-status"),
  detList: $("det-list"),
  detCount: $("det-count"),
};

const ctx = els.stage.getContext("2d");
const MAX_STAGE_HEIGHT = 380;
const BOX_COLORS = ["#0f766e", "#b45309", "#1d4ed8", "#b42318", "#7c3aed", "#0e7490"];

const state = {
  messages: [],
  sending: false,
  controller: null,
  config: null,
  image: null, // {img, blob, name, width, height}
  boxes: [],
  fitScale: 1,
  sampleList: [],
  sampleIndex: 0,
};

/* ---------------- 通用工具 ---------------- */

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMd(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(source) {
  const segments = String(source).split("```");
  let html = "";
  segments.forEach((segment, index) => {
    if (index % 2 === 1) {
      const code = segment.replace(/^[A-Za-z0-9_+-]*\n/, "");
      html += `<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
    } else {
      html += renderBlocks(segment);
    }
  });
  return html;
}

function renderBlocks(source) {
  const lines = source.split("\n");
  let html = "";
  let list = null;
  const closeList = () => {
    if (list) {
      html += `</${list}>`;
      list = null;
    }
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) {
      closeList();
      continue;
    }
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (bullet) {
      if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
      html += `<li>${inlineMd(bullet[1])}</li>`;
      continue;
    }
    if (numbered) {
      if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
      html += `<li>${inlineMd(numbered[1])}</li>`;
      continue;
    }
    closeList();
    const heading = /^#{1,4}\s+(.*)$/.exec(line);
    if (heading) {
      html += `<p><strong>${inlineMd(heading[1])}</strong></p>`;
      continue;
    }
    html += `<p>${inlineMd(line)}</p>`;
  }
  closeList();
  return html;
}

function setPill(textEl, dotEl, text, level) {
  textEl.textContent = text;
  dotEl.className = "dot" + (level ? " " + level : "");
}

/* ---------------- 对话 ---------------- */

function renderEmptyState() {
  els.transcript.innerHTML = `
    <div class="empty">
      <div id="empty-actions" class="chips"></div>
    </div>`;
  const chips = $("empty-actions");
  const samples = ["你好，介绍一下你自己", "12*(3+4) 帮我算一下", "现在是几点"];
  for (const text of samples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-ghost";
    button.textContent = text;
    button.addEventListener("click", () => {
      els.prompt.value = text;
      sendMessage();
    });
    chips.appendChild(button);
  }
}

function appendMessage(role, content) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role}`;
  const roleEl = document.createElement("div");
  roleEl.className = "msg-role";
  roleEl.textContent = role === "user" ? "我" : "AI";
  const body = document.createElement("div");
  body.className = "msg-body";
  body.innerHTML = renderMarkdown(content);
  wrap.append(roleEl, body);
  els.transcript.appendChild(wrap);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return body;
}

function appendStep(text, isError) {
  const el = document.createElement("div");
  el.className = "step" + (isError ? " error" : "");
  el.textContent = text;
  els.transcript.appendChild(el);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return el;
}

function setComposerBusy(busy) {
  state.sending = busy;
  els.send.disabled = busy;
  els.stop.hidden = !busy;
}

function chatPayload() {
  return {
    messages: state.messages,
    provider: els.provider.value,
    model: els.model.value.trim() || undefined,
    use_tools: els.tools.checked,
  };
}

async function sendMessage() {
  const text = els.prompt.value.trim();
  if (!text || state.sending) return;

  if (state.messages.length === 0) els.transcript.innerHTML = "";
  state.messages.push({ role: "user", content: text });
  appendMessage("user", text);
  els.prompt.value = "";
  autoSizePrompt();

  const body = appendMessage("assistant", "");
  body.classList.add("cursor");
  let buffer = "";
  let toolSteps = [];

  setComposerBusy(true);
  els.chatHint.textContent = "生成中…";
  state.controller = new AbortController();
  const started = performance.now();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatPayload()),
      signal: state.controller.signal,
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({ error: "请求失败" }));
      throw new Error(detail.error || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let finished = false;

    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      let boundary = pending.indexOf("\n\n");
      while (boundary >= 0) {
        const chunk = pending.slice(0, boundary);
        pending = pending.slice(boundary + 2);
        const parsed = parseSse(chunk);
        if (parsed) {
          if (parsed.event === "delta") {
            buffer += parsed.data.text || "";
            body.innerHTML = renderMarkdown(buffer);
            els.transcript.scrollTop = els.transcript.scrollHeight;
          } else if (parsed.event === "step") {
            toolSteps.push(parsed.data.message);
          } else if (parsed.event === "tool") {
            appendStep(
              `工具 ${parsed.data.name} → ${JSON.stringify(parsed.data.result)}`
            );
          } else if (parsed.event === "error") {
            appendStep(parsed.data.message, true);
            finished = true;
          } else if (parsed.event === "meta") {
            setPill(
              els.llmPill,
              els.llmDot,
              `${parsed.data.model}`,
              parsed.data.provider === "demo" ? "warn" : "ok"
            );
          }
        }
        boundary = pending.indexOf("\n\n");
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      appendStep("已停止生成");
    } else {
      appendStep(error.message, true);
    }
  } finally {
    body.classList.remove("cursor");
    if (!buffer) body.innerHTML = "";
    state.messages.push({ role: "assistant", content: buffer });
    setComposerBusy(false);
    state.controller = null;
    const seconds = ((performance.now() - started) / 1000).toFixed(2);
    els.chatHint.textContent = `${els.model.value || "默认模型"} · ${seconds}s`;
  }
}

function parseSse(chunk) {
  let event = "message";
  const dataLines = [];
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return { event, data: { text: dataLines.join("\n") } };
  }
}

function autoSizePrompt() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = Math.min(els.prompt.scrollHeight, 160) + "px";
}

/* ---------------- 图像与检测 ---------------- */

function setStatus(text, level) {
  els.visionStatus.textContent = text || "";
  els.visionStatus.className = "status-line" + (level ? " " + level : "");
}

function setBadge(text, level) {
  els.engineBadge.textContent = text;
  els.engineBadge.className = "badge" + (level ? " " + level : "");
}

function loadImage(blob, name) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("无法解析这张图片"));
    };
    img.src = url;
  });
}

async function acceptFile(blob, name) {
  try {
    const img = await loadImage(blob, name);
    state.image = { img, blob, name, width: img.naturalWidth, height: img.naturalHeight };
    state.boxes = [];
    els.dropzone.hidden = true;
    els.canvasWrap.hidden = false;
    renderResults([]);
    setStatus(`已载入 ${name} · ${img.naturalWidth}×${img.naturalHeight}`);
    draw();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function draw() {
  if (!state.image) return;
  const { img, width, height } = state.image;
  const maxWidth = Math.max(220, els.canvasWrap.clientWidth || 400);
  const scale = Math.min(maxWidth / width, MAX_STAGE_HEIGHT / height);
  const cssWidth = Math.round(width * scale);
  const cssHeight = Math.round(height * scale);
  const dpr = window.devicePixelRatio || 1;

  els.stage.style.width = `${cssWidth}px`;
  els.stage.style.height = `${cssHeight}px`;
  els.stage.width = Math.round(cssWidth * dpr);
  els.stage.height = Math.round(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.drawImage(img, 0, 0, cssWidth, cssHeight);
  state.fitScale = scale;

  state.boxes.forEach((box, index) => {
    const color = BOX_COLORS[index % BOX_COLORS.length];
    const x = box.x1 * scale;
    const y = box.y1 * scale;
    const w = (box.x2 - box.x1) * scale;
    const h = (box.y2 - box.y1) * scale;
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    const label = `${box.label} ${(box.score * 100).toFixed(0)}%`;
    ctx.font = "600 12px 'Segoe UI', 'Microsoft YaHei', sans-serif";
    const textWidth = ctx.measureText(label).width;
    const labelY = y > 20 ? y - 18 : y + 2;
    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, textWidth + 12, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 6, labelY + 13);
  });
}

function renderResults(boxes, engine) {
  els.detList.innerHTML = "";
  els.detCount.textContent = String(boxes.length);
  if (!boxes.length) {
    const li = document.createElement("li");
    li.className = "det-empty";
    li.textContent = state.image ? "暂无检测结果" : "还没有载入图片";
    els.detList.appendChild(li);
    return;
  }
  boxes.forEach((box, index) => {
    const li = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = BOX_COLORS[index % BOX_COLORS.length];
    const label = document.createElement("span");
    label.className = "det-label";
    label.textContent =
      `${box.label}  [${box.x1.toFixed(0)}, ${box.y1.toFixed(0)}, ` +
      `${box.x2.toFixed(0)}, ${box.y2.toFixed(0)}]`;
    const score = document.createElement("span");
    score.className = "det-score";
    score.textContent = `${(box.score * 100).toFixed(1)}%`;
    li.append(swatch, label, score);
    li.title = engine ? `来源：${engine}` : "";
    els.detList.appendChild(li);
  });
}

async function detect() {
  if (!state.image) {
    setStatus("先选择一张图片", "warn");
    return;
  }
  setStatus("正在推理…");
  setBadge("推理中", "warn");
  els.detect.disabled = true;
  try {
    const form = new FormData();
    form.append("file", state.image.blob, state.image.name);
    const response = await fetch("/api/detect", { method: "POST", body: form });
    const data = await response.json().catch(() => ({ ok: false, error: "响应解析失败" }));
    if (!response.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    if (data.engine === "none") {
      const boxes = fallbackDetect();
      state.boxes = boxes;
      draw();
      renderResults(boxes, "浏览器启发式");
      setBadge("启发式演示", "warn");
      setStatus(
        `${data.note} 当前结果由浏览器端显著性算法生成，仅用于验证链路，不等于 YOLO 结果。`,
        "warn"
      );
      return;
    }
    state.boxes = data.boxes || [];
    draw();
    renderResults(state.boxes, data.engine);
    const names = new Set(state.boxes.map((b) => b.label));
    setBadge(data.engine, "ok");
    setStatus(
      `${data.model || "模型"} · ${data.elapsed_ms} ms · ${state.boxes.length} 个目标` +
        (names.size ? ` · 类别：${[...names].join("、")}` : "")
    );
  } catch (error) {
    setBadge("失败", "warn");
    setStatus(error.message, "error");
  } finally {
    els.detect.disabled = false;
  }
}

/* ---------------- 浏览器端降级检测 ---------------- */

function boxBlur(src, width, height, radius) {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const window = radius * 2 + 1;
  for (let y = 0; y < height; y++) {
    let sum = 0;
    const row = y * width;
    for (let x = -radius; x <= radius; x++) {
      sum += src[row + Math.min(width - 1, Math.max(0, x))];
    }
    for (let x = 0; x < width; x++) {
      tmp[row + x] = sum / window;
      const add = src[row + Math.min(width - 1, x + radius + 1)];
      const drop = src[row + Math.max(0, x - radius)];
      sum += add - drop;
    }
  }
  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let y = -radius; y <= radius; y++) {
      sum += tmp[Math.min(height - 1, Math.max(0, y)) * width + x];
    }
    for (let y = 0; y < height; y++) {
      out[y * width + x] = sum / window;
      const add = tmp[Math.min(height - 1, y + radius + 1) * width + x];
      const drop = tmp[Math.max(0, y - radius) * width + x];
      sum += add - drop;
    }
  }
  return out;
}

function fallbackDetect() {
  const { img } = state.image;
  const maxSide = 200;
  const ratio = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  const width = Math.max(8, Math.round(img.naturalWidth * ratio));
  const height = Math.max(8, Math.round(img.naturalHeight * ratio));

  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const sctx = scratch.getContext("2d", { willReadFrequently: true });
  sctx.drawImage(img, 0, 0, width, height);
  const pixels = sctx.getImageData(0, 0, width, height).data;

  const lum = new Float32Array(width * height);
  for (let i = 0, p = 0; i < lum.length; i++, p += 4) {
    lum[i] = 0.299 * pixels[p] + 0.587 * pixels[p + 1] + 0.114 * pixels[p + 2];
  }

  const blur = boxBlur(lum, width, height, 4);
  const contrast = new Float32Array(lum.length);
  let total = 0;
  for (let i = 0; i < lum.length; i++) {
    contrast[i] = Math.abs(lum[i] - blur[i]);
    total += contrast[i];
  }
  const mean = total / lum.length;
  let variance = 0;
  for (let i = 0; i < contrast.length; i++) {
    const d = contrast[i] - mean;
    variance += d * d;
  }
  const std = Math.sqrt(variance / contrast.length) || 1;
  const threshold = Math.max(mean + 1.05 * std, 12);

  const mask = new Uint8Array(lum.length);
  for (let i = 0; i < contrast.length; i++) mask[i] = contrast[i] > threshold ? 1 : 0;

  const seen = new Uint8Array(lum.length);
  const components = [];
  const queue = new Int32Array(lum.length);
  const minArea = Math.max(8, Math.round(lum.length * 0.0012));
  const maxArea = Math.round(lum.length * 0.35);
  const upscale = 1 / ratio;

  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    seen[start] = 1;
    let minX = width, minY = height, maxX = 0, maxY = 0, area = 0, sum = 0;
    while (head < tail) {
      const index = queue[head++];
      const x = index % width;
      const y = (index - x) / width;
      area++;
      sum += contrast[index];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const ni = ny * width + nx;
          if (mask[ni] && !seen[ni]) {
            seen[ni] = 1;
            queue[tail++] = ni;
          }
        }
      }
    }
    if (area < minArea || area > maxArea) continue;
    const meanContrast = sum / area;
    const relative = (meanContrast - threshold) / (4 * std + 1e-6);
    const score = Math.min(0.99, Math.max(0.4, 0.45 + 0.55 * relative));
    components.push({
      x1: Math.max(0, (minX - 1) * upscale),
      y1: Math.max(0, (minY - 1) * upscale),
      x2: Math.min(img.naturalWidth, (maxX + 2) * upscale),
      y2: Math.min(img.naturalHeight, (maxY + 2) * upscale),
      score,
      label: "显著区域",
      cls: 0,
    });
  }

  return suppressOverlaps(components, 0.35).slice(0, 8);
}

function intersectionOverUnion(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function suppressOverlaps(boxes, threshold) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of sorted) {
    if (kept.some((other) => intersectionOverUnion(other, box) > threshold)) continue;
    kept.push(box);
  }
  return kept;
}

/* ---------------- 初始化 ---------------- */

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    state.config = config;

    els.provider.innerHTML = "";
    for (const item of config.providers) {
      const option = document.createElement("option");
      option.value = item.name;
      option.textContent = item.label;
      option.title = item.hint || "";
      els.provider.appendChild(option);
    }
    els.provider.value = config.llm.provider;
    els.model.value = config.llm.model;

    const vision = config.vision;
    if (vision.active_backend === "none") {
      setPill(els.visionPill, els.visionDot, "检测后端未就绪", "warn");
      setBadge("未加载", "warn");
      setStatus(vision.note || "YOLO 权重尚未加载。", "warn");
    } else {
      setPill(
        els.visionPill,
        els.visionDot,
        `${vision.active_backend} · ${vision.weights}`,
        "ok"
      );
      setBadge(vision.active_backend, "ok");
      setStatus(`权重 ${vision.weights} · conf ${vision.conf} · imgsz ${vision.imgsz}`);
    }
  } catch (error) {
    setStatus(`配置读取失败：${error.message}`, "error");
  }
}

async function loadSamples() {
  try {
    const response = await fetch("/api/samples");
    const data = await response.json();
    state.sampleList = data.samples || [];
  } catch {
    state.sampleList = [];
  }
}

async function useSample() {
  if (!state.sampleList.length) {
    setStatus("没有可用的示例图片", "warn");
    return;
  }
  const item = state.sampleList[state.sampleIndex % state.sampleList.length];
  state.sampleIndex += 1;
  try {
    const response = await fetch(item.url);
    const blob = await response.blob();
    await acceptFile(blob, item.name);
  } catch (error) {
    setStatus(`示例图载入失败：${error.message}`, "error");
  }
}

function clearImage() {
  state.image = null;
  state.boxes = [];
  els.canvasWrap.hidden = true;
  els.dropzone.hidden = false;
  els.detList.innerHTML = "";
  els.detCount.textContent = "0";
  setStatus("");
  const vision = state.config && state.config.vision;
  setBadge(vision && vision.active_backend !== "none" ? vision.active_backend : "未加载",
    vision && vision.active_backend !== "none" ? "ok" : "warn");
}

function wireEvents() {
  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  els.prompt.addEventListener("input", autoSizePrompt);
  els.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  els.stop.addEventListener("click", () => {
    if (state.controller) state.controller.abort();
  });
  els.clearChat.addEventListener("click", () => {
    state.messages = [];
    renderEmptyState();
    els.chatHint.textContent = "";
  });

  els.pick.addEventListener("click", () => els.file.click());
  els.file.addEventListener("change", () => {
    if (els.file.files && els.file.files[0]) {
      acceptFile(els.file.files[0], els.file.files[0].name);
      els.file.value = "";
    }
  });
  els.detect.addEventListener("click", detect);
  els.sample.addEventListener("click", useSample);
  els.clearImage.addEventListener("click", clearImage);

  els.dropzone.addEventListener("click", () => els.file.click());
  els.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      els.file.click();
    }
  });
  ["dragenter", "dragover"].forEach((name) =>
    els.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    els.dropzone.addEventListener(name, () => els.dropzone.classList.remove("over"))
  );
  els.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) acceptFile(file, file.name);
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  window.addEventListener("resize", () => {
    if (state.image) draw();
  });
}

async function init() {
  renderEmptyState();
  wireEvents();
  autoSizePrompt();
  await Promise.all([loadConfig(), loadSamples()]);
  try {
    const health = await (await fetch("/api/health")).json();
    const live = health.llm.live;
    setPill(
      els.llmPill,
      els.llmDot,
      `${health.llm.label} · ${health.llm.model}`,
      live ? "ok" : "warn"
    );
  } catch {
    setPill(els.llmPill, els.llmDot, "后端未响应", "err");
  }

  // 自动演示模式：/ ?demo=1，用于快速截图或验收时一键跑完整链路
  if (new URLSearchParams(location.search).get("demo") === "1") {
    await useSample();
    if (state.image) await detect();
    if (state.messages.length === 0) {
      els.prompt.value = "帮我算一下 12*(3+4)";
      await sendMessage();
    }
  }
}

init();
