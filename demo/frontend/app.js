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
  {key: 'gflops', label: 'GFLOPs', desc: 'Chi phí tính toán · chỉ hiển thị khi đã đo', suffix: ''},
];
const state = {tab: 'vit', samples: [], index: 0, models: {}, results: {}, api: false, generation: 0, controller: null};

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
  if (tab !== 'statistics') $('#result-title').textContent = `Kết quả ${info.name}`;
  runCurrent();
}
function showSample(index) {
  if (!state.samples.length) return;
  state.index = (index + state.samples.length) % state.samples.length;
  const sample = state.samples[state.index];
  $('#sample-name').textContent = sample.name;
  $('#sample-label').textContent = sample.label === 'normal' ? 'Bình thường' : sample.category;
  $('#sample-position').textContent = `Ảnh ${state.index + 1} / ${state.samples.length} · Tập test BTXRD`;
  $('#previous').disabled = $('#next').disabled = false;
  document.querySelectorAll('.sample-image').forEach((image) => { image.src = sample.image_url; image.hidden = false; });
  document.querySelectorAll('.truth-image').forEach((image) => { image.src = sample.ground_truth_url; image.hidden = false; });
  document.querySelectorAll('.stage-placeholder').forEach((node) => { node.hidden = true; });
  runCurrent();
}
function modelMessage(id, waiting = false) {
  if (waiting && state.models[id]?.ready) return 'Đang chạy phân vùng...';
  return state.models[id]?.message || 'Mô hình chưa được kết nối.';
}
function paintModel(id, result, waiting = false) {
  const layer = $(id === 'single' ? '#model-layer' : `#compare-${id}-layer`);
  const mask = $(id === 'single' ? '#model-mask' : `#compare-${id}-mask`);
  const message = $(id === 'single' ? '#model-message' : `#compare-${id}-message`);
  const modelId = id === 'single' ? state.tab : id;
  const hasResult = result?.ready === true && safeMask(result.mask_url);
  layer.hidden = !hasResult;
  message.hidden = hasResult;
  if (hasResult) mask.src = result.mask_url;
  else { mask.removeAttribute('src'); message.textContent = result?.error || modelMessage(modelId, waiting); }
}
function render() {
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
    for (const id of ['vit', 'efficientvit']) paintModel(id, state.results[id], state.waiting);
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
    chart.append(element('h3', '', metric.label), element('p', '', metric.desc));
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
async function runCurrent() {
  const generation = ++state.generation;
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  state.results = {};
  const ids = state.tab === 'statistics' ? ['vit', 'efficientvit'] : [state.tab];
  state.waiting = Boolean(state.samples.length && state.api && ids.some((id) => state.models[id]?.ready));
  render();
  if (!state.waiting) return;
  try {
    const endpoint = state.tab === 'statistics' ? `/api/compare?index=${state.index}` : `/api/predict?index=${state.index}&model=${state.tab}`;
    const data = await getJson(endpoint, {signal: state.controller.signal});
    if (generation !== state.generation || data.index !== state.index) return;
    state.results = state.tab === 'statistics' ? data.models : {[state.tab]: data.result};
  } catch (error) {
    if (error.name === 'AbortError' || generation !== state.generation) return;
    ids.forEach((id) => { state.results[id] = {error: 'Không thể chạy model. Kiểm tra backend và checkpoint.'}; });
  } finally {
    if (generation === state.generation) { state.waiting = false; render(); }
  }
}
async function init() {
  setTab('vit');
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
  if (state.samples.length) showSample(Math.max(0, state.samples.findIndex((sample) => sample.label === 'tumor')));
  else { $('#sample-name').textContent = 'Chưa có ảnh'; render(); }
}
document.querySelectorAll('.nav-item').forEach((button) => button.addEventListener('click', () => setTab(button.dataset.tab)));
$('#previous').addEventListener('click', () => showSample(state.index - 1));
$('#next').addEventListener('click', () => showSample(state.index + 1));
init();
