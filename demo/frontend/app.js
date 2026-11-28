import {initHistory, refreshHistory} from './history.js?v=20261128-history-layout';

const $ = (selector) => document.querySelector(selector);
const MODEL_INFO = {
  vit: {name: 'ViT', title: 'Vision Transformer', description: 'Xem dự đoán của ViT đối chiếu với ground truth trên cùng một ảnh.'},
  efficientvit: {name: 'EfficientViT', title: 'EfficientViT', description: 'Xem dự đoán của EfficientViT đối chiếu với ground truth trên cùng một ảnh.'},
  statistics: {name: 'SO SÁNH', title: 'So sánh mô hình', description: 'Hai mô hình phân vùng cùng một ảnh và được đo theo cùng cách.'},
};
const METRICS = [
  {key: 'dice', label: 'Dice', desc: 'Độ trùng khớp với ground truth · càng cao càng tốt', suffix: ''},
  {key: 'iou', label: 'IoU', desc: 'Tỷ lệ giao trên hợp · càng cao càng tốt', suffix: ''},
  {key: 'latency_ms', label: 'Thời gian suy luận', desc: 'Thời gian một lần chạy trên ảnh này · càng thấp càng tốt', suffix: ' ms'},
  {key: 'params_m', label: 'Tham số', desc: 'Kích thước mô hình · triệu tham số', suffix: ' M'},
  {key: 'gflops', label: 'GFLOPs', desc: 'Ước tính một lượt forward · 1 ảnh 224 × 224 · càng thấp càng tốt', suffix: ''},
];
const state = {tab: 'vit', samples: [], tumorSamples: [], tumorOnly: false, sample: null,
  models: {}, results: {}, api: false, generation: 0, waiting: false,
  batchActive: false, followingBatch: false, batchPreview: null, batchSignature: ''};
const frameKeys = new WeakMap();
let inferenceTimer = null;
let pendingInference = null;
let inferenceInFlight = false;

async function getJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok || !(response.headers.get('content-type') || '').includes('application/json')) throw new Error('API unavailable');
  return response.json();
}
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function format(value, metric) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(metric.key === 'dice' || metric.key === 'iou' ? 3 : 1)}${metric.suffix}` : '—';
}
function safeMask(value) {
  return typeof value === 'string' && /^data:image\/png;base64,[a-z0-9+/=]+$/i.test(value);
}
function setTab(tab) {
  const leavingBatchPreview = !state.batchActive && state.followingBatch;
  state.tab = tab;
  document.querySelectorAll('.nav-item').forEach((button) => {
    const selected = button.dataset.tab === tab;
    button.classList.toggle('active', selected);
    if (selected) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  const info = MODEL_INFO[tab];
  $('#breadcrumb').textContent = info.name.toUpperCase();
  $('#page-title').replaceChildren(document.createTextNode(info.title), element('span', 'title-period', '.'));
  $('#page-description').textContent = info.description;
  $('#model-view').hidden = tab === 'statistics';
  $('#stats-view').hidden = tab !== 'statistics';
  if (tab === 'statistics') refreshHistory();
  if (tab !== 'statistics') $('#result-title').textContent = `Kết quả ${info.name}`;
  if (leavingBatchPreview && state.sample && browsingSamples().length) {
    showSample(Math.max(0, browsingSamples().findIndex((sample) => sample.index === state.sample.index)));
  } else {
    runCurrent();
  }
}
function browsingSamples() {
  return state.tumorOnly ? state.tumorSamples : state.samples;
}
function updateSampleControls() {
  $('#previous').disabled = $('#next').disabled = state.batchActive || browsingSamples().length < 2;
  const filter = $('#sample-tumor-only');
  filter.disabled = state.batchActive || !state.tumorSamples.length;
  filter.setAttribute('aria-pressed', String(state.tumorOnly));
}
function showSample(position) {
  const samples = browsingSamples();
  if (!samples.length || state.batchActive) return;
  state.followingBatch = false;
  displaySample(samples[(position + samples.length) % samples.length]);
  runCurrent();
}
function moveSample(direction) {
  const position = browsingSamples().findIndex((sample) => sample.index === state.sample?.index);
  showSample(position < 0 ? (direction > 0 ? 0 : -1) : position + direction);
}
function toggleTumorFilter() {
  if (state.batchActive || !state.tumorSamples.length) return;
  state.tumorOnly = !state.tumorOnly;
  const position = browsingSamples().findIndex((sample) => sample.index === state.sample?.index);
  showSample(Math.max(0, position));
}
function displaySample(sample, batch = false) {
  state.sample = sample;
  $('#sample-name').textContent = sample.name;
  $('#sample-label').textContent = sample.label === 'normal' ? 'Bình thường' : sample.category || 'Có u';
  const samples = browsingSamples();
  const position = batch ? sample.index : samples.findIndex((item) => item.index === sample.index);
  const total = batch ? (state.samples.length || sample.total) : samples.length;
  const scope = batch && state.batchActive ? ' · Đang chạy hàng loạt' : !batch && state.tumorOnly ? ' · Có u' : '';
  $('#sample-position').textContent = `Ảnh ${position + 1} / ${total || '—'}${scope}`;
  updateSampleControls();
  document.querySelectorAll('#single-input-image, #compare-input-image').forEach((image) => {
    if (image.getAttribute('src') !== sample.image_url) image.src = sample.image_url;
    image.hidden = false;
  });
  document.querySelectorAll('.truth-image').forEach((image) => {
    if (image.getAttribute('src') !== sample.ground_truth_url) image.src = sample.ground_truth_url;
    image.hidden = false;
  });
  document.querySelectorAll('.stage-placeholder').forEach((node) => { node.hidden = true; });
  $('#compare-input-caption').textContent = sample.name;
  $('#compare-truth-caption').textContent = sample.label === 'normal' ? 'Không có vùng u' : 'Vùng u được gán nhãn';
}
function modelMessage(id, waiting = false) {
  if (waiting && state.models[id]?.ready) return 'Đang chạy phân vùng...';
  return state.models[id]?.message || 'Mô hình chưa được kết nối.';
}
function paintModel(id, result, waiting = false, imageUrl = state.sample?.image_url) {
  const layer = $(id === 'single' ? '#model-layer' : `#compare-${id}-layer`);
  const mask = $(id === 'single' ? '#model-mask' : `#compare-${id}-mask`);
  const message = $(id === 'single' ? '#model-message' : `#compare-${id}-message`);
  const modelId = id === 'single' ? state.tab : id;
  const hasResult = result?.ready === true && safeMask(result.mask_url);
  if (!hasResult) {
    frameKeys.delete(layer);
    layer.hidden = true; message.hidden = false;
    mask.removeAttribute('src'); message.textContent = result?.error || modelMessage(modelId, waiting);
    return;
  }
  const key = `${imageUrl}|${result.mask_url}`;
  if (frameKeys.get(layer) === key) return;
  frameKeys.set(layer, key);
  layer.hidden = true; message.hidden = false; message.textContent = 'Đang tải kết quả...';
  const background = new Image(), overlay = new Image();
  background.src = imageUrl; overlay.src = result.mask_url;
  Promise.all([background.decode(), overlay.decode()]).then(() => {
    if (frameKeys.get(layer) !== key) return;
    layer.querySelector('.sample-image').src = imageUrl;
    mask.src = result.mask_url;
    layer.hidden = false; message.hidden = true;
  }).catch(() => {
    if (frameKeys.get(layer) !== key) return;
    frameKeys.delete(layer);
    message.textContent = 'Không tải được ảnh kết quả.';
  });
}
function render() {
  if (state.followingBatch && state.batchPreview) { renderBatchPreview(); return; }
  if (state.tab !== 'statistics') {
    const id = state.tab;
    const result = state.results[id];
    const ready = state.models[id]?.ready;
    paintModel('single', result, state.waiting);
    $('#model-result-status').textContent = result?.ready ? 'ĐÃ PHÂN VÙNG' : state.waiting && ready ? 'ĐANG CHẠY' : 'CHƯA SẴN SÀNG';
    $('#model-status').textContent = state.waiting && ready ? 'Đang phân tích' : ready ? 'Sẵn sàng' : 'Chưa kết nối';
    $('#model-dice').textContent = format(result?.dice, METRICS[0]);
    $('#model-iou').textContent = format(result?.iou, METRICS[1]);
    $('#model-time').textContent = format(result?.latency_ms, METRICS[2]);
  } else {
    for (const id of ['vit', 'efficientvit']) {
      paintModel(id, state.results[id], state.waiting);
      $(`#compare-${id}-caption`).textContent = state.results[id]?.ready ? state.sample?.name || '' : '';
      $(`#compare-${id}-status`).textContent = state.results[id]?.ready ? 'Xong' : state.waiting ? 'Đang chạy' : '';
    }
    const readyCount = ['vit', 'efficientvit'].filter((id) => state.results[id]?.ready).length;
    $('#comparison-status').textContent = state.waiting ? 'Đang phân tích' : readyCount === 2 ? 'Đã có hai kết quả' : readyCount === 1 ? 'Đã có một kết quả' : 'Chưa có kết quả';
    renderCharts();
  }
}
function renderCharts() {
  const charts = $('#chart-grid');
  const table = $('#metrics-table');
  charts.replaceChildren(); table.replaceChildren();
  METRICS.forEach((metric) => {
    const values = ['vit', 'efficientvit'].map((id) => state.results[id]?.[metric.key]);
    const numeric = values.filter((value) => typeof value === 'number' && Number.isFinite(value) && value >= 0);
    const max = metric.key === 'dice' || metric.key === 'iou' ? 1 : Math.max(...numeric, 0);
    const chart = element('article', 'chart-card');
    chart.append(element('h3', '', metric.label));
    ['vit', 'efficientvit'].forEach((id, index) => {
      const row = element('div', 'chart-row');
      const label = element('span', 'chart-label', MODEL_INFO[id].name);
      const track = element('div', 'chart-track');
      const bar = element('div', `chart-bar ${id}`);
      bar.style.width = max > 0 && typeof values[index] === 'number' && Number.isFinite(values[index]) ? `${Math.min(100, Math.max(0, values[index] / max * 100))}%` : '0%';
      track.append(bar);
      row.append(label, track, element('strong', 'chart-value', format(values[index], metric)));
      chart.append(row);
    });
    charts.append(chart);
    const row = element('tr');
    row.append(element('td', '', metric.label), element('td', '', format(values[0], metric)), element('td', '', format(values[1], metric)), element('td', '', metric.desc));
    table.append(row);
  });
}
function sampleUrls(sample) {
  return {...sample, image_url: `/api/image?index=${sample.index}`, ground_truth_url: `/api/ground-truth?index=${sample.index}`};
}
function renderBatchPreview() {
  const preview = state.batchPreview;
  if (!preview?.sample) return;
  displaySample(sampleUrls(preview.sample), true);
  state.results = Object.fromEntries(Object.entries(preview.models || {}).map(([id, entry]) => [id, entry.result]));
  for (const id of ['vit', 'efficientvit']) {
    const entry = preview.models?.[id];
    const waiting = state.batchActive && preview.processing_model === id;
    paintModel(id, entry?.result, waiting, entry ? sampleUrls(entry.sample).image_url : undefined);
    const current = entry?.sample.index === preview.sample.index;
    $(`#compare-${id}-caption`).textContent = entry ? `${entry.sample.name}${current ? '' : ' · trước đó'}` : 'Chờ kết quả';
    $(`#compare-${id}-status`).textContent = waiting ? 'Đang chạy' : current && entry?.result.ready ? 'Xong' : '';
  }
  $('#comparison-status').textContent = state.batchActive ? `Đang chạy ${MODEL_INFO[preview.processing_model]?.name || 'hàng loạt'}` : preview.state === 'complete' ? 'Hoàn tất' : preview.state === 'error' ? 'Gặp lỗi' : 'Đã dừng';
  $('#comparison-metric-note').textContent = 'Kết quả mới nhất của từng model; tên ảnh ở dưới mỗi khung.';
  renderCharts();
  if (state.tab !== 'statistics') {
    const entry = preview.models?.[state.tab];
    const waiting = state.batchActive && preview.processing_model === state.tab;
    paintModel('single', entry?.result, waiting, entry ? sampleUrls(entry.sample).image_url : undefined);
    $('#model-result-status').textContent = waiting ? 'ĐANG CHẠY' : entry?.result.ready ? 'ĐÃ PHÂN VÙNG' : 'CHỜ KẾT QUẢ';
    $('#model-status').textContent = waiting ? 'Đang chạy' : 'Sẵn sàng';
    $('#model-dice').textContent = format(entry?.result.dice, METRICS[0]);
    $('#model-iou').textContent = format(entry?.result.iou, METRICS[1]);
    $('#model-time').textContent = format(entry?.result.latency_ms, METRICS[2]);
    $('#single-output-caption').textContent = entry?.sample.name || 'Chờ kết quả';
  }
}
function handleBatchProgress(data) {
  const active = data.jobs.find((job) => ['running', 'stopping'].includes(job.state));
  if (active && !state.batchActive) {
    state.generation++;
    clearInferenceTimer();
    state.waiting = false;
  }
  state.batchActive = Boolean(active);
  updateSampleControls();
  const preview = data.preview;
  if (active) {
    state.followingBatch = true;
  }
  if (!preview || !state.followingBatch || (active && preview.job_id !== active.id)) return;
  const signature = `${preview.job_id}:${preview.revision}:${Boolean(active)}:${state.tab}`;
  if (signature === state.batchSignature) return;
  state.batchSignature = signature;
  state.batchPreview = preview;
  renderBatchPreview();
}
function clearInferenceTimer() {
  clearTimeout(inferenceTimer);
  inferenceTimer = null;
  pendingInference = null;
}
function runCurrent() {
  if (state.batchActive) { render(); return; }
  $('#comparison-metric-note').textContent = 'Dice/IoU trên ảnh đang xem. Hai mask rỗng được quy ước bằng 1.';
  $('#single-output-caption').textContent = 'Vùng màu là mask dự đoán.';
  const generation = ++state.generation;
  clearInferenceTimer();
  state.results = {};
  const tab = state.tab;
  const ids = tab === 'statistics' ? ['vit', 'efficientvit'] : [tab];
  state.waiting = Boolean(state.sample && state.api && ids.some((id) => state.models[id]?.ready));
  render();
  if (!state.waiting) return;
  pendingInference = {generation, tab, ids, index: state.sample.index};
  inferenceTimer = setTimeout(() => {
    inferenceTimer = null;
    flushInference();
  }, 160);
}
async function flushInference() {
  // Cancelling fetch does not cancel the model forward pass in the server.
  if (inferenceInFlight || inferenceTimer !== null || !pendingInference) return;
  const {generation, tab, ids, index} = pendingInference;
  pendingInference = null;
  inferenceInFlight = true;
  try {
    const endpoint = tab === 'statistics' ? `/api/compare?index=${index}` : `/api/predict?index=${index}&model=${tab}`;
    const data = await getJson(endpoint);
    if (generation !== state.generation || data.index !== index) return;
    state.results = tab === 'statistics' ? data.models : {[tab]: data.result};
  } catch (error) {
    if (generation !== state.generation) return;
    ids.forEach((id) => { state.results[id] = {error: 'Không thể chạy model. Kiểm tra backend và checkpoint.'}; });
  } finally {
    inferenceInFlight = false;
    if (generation === state.generation) {
      state.waiting = false;
      render();
      if (tab === 'statistics') refreshHistory();
    }
    flushInference();
  }
}
async function init() {
  setTab('vit');
  await batchReady;
  try {
    const [models, samples] = await Promise.all([getJson('/api/models'), getJson('/api/samples')]);
    state.models = models.models || {};
    state.samples = Array.isArray(samples.samples) ? samples.samples : [];
    state.api = true;
  } catch {
    try {
      const samples = await getJson('./samples/index.json');
      state.samples = Array.isArray(samples.samples) ? samples.samples : [];
      $('#notice').textContent = 'Ảnh và ground truth sẵn sàng. Để chạy model, mở bản chạy cục bộ đã nối backend.';
    } catch {
      $('#notice').textContent = 'Chưa tìm thấy dữ liệu BTXRD. Hãy chạy backend cục bộ và kiểm tra dataset.';
    }
  }
  state.tumorSamples = state.samples.filter((sample) => sample.label === 'tumor');
  if (state.samples.length) showSample(Math.max(0, state.samples.findIndex((sample) => sample.label === 'tumor')));
  else { $('#sample-name').textContent = 'Chưa có ảnh'; render(); }
}
document.querySelectorAll('.nav-item').forEach((button) => button.addEventListener('click', () => setTab(button.dataset.tab)));
$('#previous').addEventListener('click', () => moveSample(-1));
$('#next').addEventListener('click', () => moveSample(1));
$('#sample-tumor-only').addEventListener('click', toggleTumorFilter);
const batchReady = initHistory(handleBatchProgress);
init();
