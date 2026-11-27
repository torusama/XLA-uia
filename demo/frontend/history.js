import {chartTooltip, dismissChartTooltip, initChartTooltips} from './chart-tooltip.js?v=20261127-tooltip';

const $ = (selector) => document.querySelector(selector);
const MODELS = {vit: {name: 'ViT', color: '#137e75'}, efficientvit: {name: 'EfficientViT', color: '#c8773e'}};
const METRICS = [
  {key: 'latency_ms', title: 'Thời gian suy luận', unit: 'ms', note: 'Thời gian forward trên từng ảnh. Giá trị thấp hơn tương ứng với suy luận nhanh hơn.', goal: 'THẤP HƠN TỐT HƠN'},
  {key: 'throughput_ips', title: 'Tốc độ xử lý', unit: 'ảnh/s', note: 'Quy đổi từ thời gian forward: 1.000 / thời gian tính bằng ms.', goal: 'CAO HƠN TỐT HƠN'},
  {key: 'dice', title: 'Dice', unit: '0–1', note: 'Mức độ khớp với ground truth trên từng ảnh. Giá trị trung bình theo ảnh khác Dice pixel gộp.', goal: 'CAO HƠN TỐT HƠN', digits: 3},
  {key: 'iou', title: 'Intersection over Union', unit: '0–1', note: 'Diện tích giao trên hợp giữa mask dự đoán và ground truth.', goal: 'CAO HƠN TỐT HƠN', digits: 3},
  {key: 'rss_peak_mb', title: 'RAM của server', unit: 'MiB', note: 'Đỉnh RSS lấy mẫu mỗi 10 ms. Tiến trình server chứa cả hai model.', goal: 'BỘ NHỚ TIẾN TRÌNH'},
  {key: 'rss_delta_mb', title: 'RAM tăng thêm', unit: 'MiB', note: 'Phần bộ nhớ tiến trình tăng so với trước forward; là số đo ước tính.', goal: 'BỘ NHỚ TĂNG THÊM'},
  {key: 'cpu_cores', title: 'Mức sử dụng CPU', unit: 'lõi', note: 'CPU time chia wall time. Có thể lớn hơn 1 khi model sử dụng nhiều lõi.', goal: 'TÀI NGUYÊN CPU'},
  {key: 'gpu_delta_mb', title: 'VRAM tăng thêm', unit: 'MiB', note: 'Bộ nhớ tensor CUDA tăng trong forward. Chỉ có dữ liệu khi dùng GPU.', goal: 'TÀI NGUYÊN GPU'},
  {key: 'params_m', title: 'Số tham số', unit: 'triệu', note: 'Thông số cố định của model đang nạp, không thay đổi theo nội dung ảnh.', goal: 'KÍCH THƯỚC MODEL', kind: 'bar'},
  {key: 'gflops', title: 'Chi phí tính toán', unit: 'GFLOPs', note: 'Ước tính một forward, ảnh 224 × 224; nhân cộng tính hai FLOPs, chỉ đếm toán tử được hỗ trợ.', goal: 'ĐỘ PHỨC TẠP', kind: 'bar'},
];
let busy = false;
let refreshPending = false;
let lastData = {rows: [], summaries: [], sessions: [], total: 0, shown: 0};
let loaded = false;
let openMetric = null;
let batchBusy = false;
let activeJob = null;
let detailSignature = '';
let chartSignature = '';
const numeric = (value) => typeof value === 'number' && Number.isFinite(value);
const number = (value, digits = 2) => numeric(value) ? value.toLocaleString('vi-VN', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
const node = (tag, text, cls) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; };
function filters() {
  return new URLSearchParams({session: $('#history-session').value, mode: 'compare', hours: $('#history-hours').value, limit: $('#history-limit').value, batch_id: $('#history-batch').value});
}
function svgNode(tag, attrs = {}, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (text !== undefined) el.textContent = text;
  return el;
}
function drawChart(rows, metric, expandable = true) {
  const card = node('article', undefined, 'metric-panel');
  card.dataset.metric = metric.key;
  const header = node('header', undefined, 'metric-panel-header'), heading = node('div');
  heading.append(node('span', metric.goal, 'metric-eyebrow'), node('h3', metric.title));
  const actions = node('div', undefined, 'metric-panel-actions');
  actions.append(node('span', metric.unit, 'metric-unit'));
  if (expandable) {
    const expand = node('button', 'Xem lớn', 'chart-expand');
    expand.type = 'button'; expand.setAttribute('aria-label', `Xem lớn biểu đồ ${metric.title}`);
    expand.addEventListener('click', () => openDetail(metric.key));
    actions.append(expand);
  }
  header.append(heading, actions); card.append(header);
  const valid = rows.filter((r) => numeric(r[metric.key])).sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at) || a.id - b.id);
  const latest = Object.fromEntries(Object.keys(MODELS).map((id) => [id, valid.filter((r) => r.model === id).at(-1)]));
  const values = node('div', undefined, 'metric-panel-values');
  for (const [id, model] of Object.entries(MODELS)) {
    const value = node('div', undefined, `metric-panel-value ${id}`);
    value.append(node('span', model.name, 'model-name'), node('strong', number(latest[id]?.[metric.key], metric.digits ?? 2)), node('small', metric.kind === 'bar' ? 'Model gần nhất' : 'Lượt gần nhất')); values.append(value);
  }
  card.append(values);
  const wrap = node('div', undefined, 'metric-plot');
  if (!valid.length) {
    const empty = node('div', undefined, 'metric-empty');
    const gpuUnavailable = metric.key === 'gpu_delta_mb' && rows.length;
    empty.append(node('strong', gpuUnavailable ? 'Chưa có lượt chạy CUDA' : loaded ? 'Chưa có dữ liệu trong bộ lọc' : 'Đang tải số đo'), node('p', gpuUnavailable ? 'Máy đang chạy CPU. Số đo VRAM sẽ xuất hiện khi sử dụng GPU.' : 'Mỗi ảnh được phân vùng sẽ thêm một điểm cho cả hai model.')); wrap.append(empty);
  } else if (metric.kind === 'bar') {
    const max = Math.max(...valid.map((r) => r[metric.key]), 0.001), bars = node('div', undefined, 'model-bars');
    for (const [id, model] of Object.entries(MODELS)) {
      const value = latest[id]?.[metric.key], row = node('div', undefined, 'model-bar-row'), label = node('div', undefined, 'model-bar-label');
      label.append(node('span', model.name), node('strong', `${number(value)} ${metric.unit}`));
      const track = node('div', undefined, 'model-bar-track'), bar = node('div', undefined, `model-bar ${id}`);
      bar.style.width = numeric(value) ? `${Math.max(0, value) / max * 100}%` : '0%';
      track.append(bar); row.append(label, track); bars.append(row);
    }
    wrap.append(bars);
  } else {
    const byTime = $('#history-axis').value === 'time';
    const allRows = [...rows].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at) || a.id - b.id);
    const groups = [...new Set(allRows.map((r) => r.group_id))], positions = new Map(groups.map((id, index) => [id, index]));
    const times = allRows.map((r) => Date.parse(r.created_at)), first = Math.min(...times), last = Math.max(...times);
    const max = ['dice', 'iou'].includes(metric.key) ? 1 : Math.max(...valid.map((r) => r[metric.key]), 0.001) * 1.12;
    const left = 58, right = 610, top = 22, bottom = 190;
    const x = (r) => left + (byTime ? (last === first ? 0.5 : (Date.parse(r.created_at) - first) / (last - first)) : groups.length === 1 ? 0.5 : positions.get(r.group_id) / (groups.length - 1)) * (right - left);
    const y = (value) => bottom - value / max * (bottom - top);
    const svg = svgNode('svg', {viewBox: '0 0 640 240', role: 'group', 'aria-label': `${metric.title}: ViT và EfficientViT`});
    for (let tick = 0; tick <= 4; tick++) {
      const value = max * tick / 4;
      svg.append(svgNode('line', {x1: left, x2: right, y1: y(value), y2: y(value), class: 'plot-grid'}), svgNode('text', {x: left - 12, y: y(value) + 4, 'text-anchor': 'end', class: 'plot-axis'}, number(value, max <= 2 ? 2 : 0)));
    }
    const tooltip = chartTooltip(wrap, metric, `${expandable ? 'panel' : 'detail'}:${metric.key}`, number);
    const series = new Map();
    valid.forEach((r) => { if (!series.has(r.config_id)) series.set(r.config_id, []); series.get(r.config_id).push(r); });
    let seriesIndex = 0;
    for (const values of series.values()) {
      const model = MODELS[values[0].model];
      svg.append(svgNode('polyline', {points: values.map((r) => `${x(r)},${y(r[metric.key])}`).join(' '), fill: 'none', stroke: model.color, 'stroke-width': 2.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': seriesIndex++ > 1 ? '6 4' : 'none'}));
      values.forEach((r) => {
        const point = svgNode('g', {class: 'plot-point', 'data-run-id': r.id});
        point.append(svgNode('circle', {cx: x(r), cy: y(r[metric.key]), r: 10, fill: 'transparent', class: 'plot-point-hit'}),
          svgNode('circle', {cx: x(r), cy: y(r[metric.key]), r: 4, fill: model.color, stroke: '#fff', 'stroke-width': 1.5, class: 'plot-point-dot'}));
        tooltip.bind(point, r, model, positions.get(r.group_id) + 1);
        svg.append(point);
      });
    }
    const timeLabel = (time) => new Date(time).toLocaleTimeString('vi-VN', {hour12: false});
    svg.append(svgNode('text', {x: left, y: 220, class: 'plot-axis'}, byTime ? timeLabel(first) : 'Lượt 1'), svgNode('text', {x: right, y: 220, class: 'plot-axis', 'text-anchor': 'end'}, byTime ? timeLabel(last) : `Lượt ${groups.length}`));
    wrap.append(svg);
    tooltip.restore();
  }
  card.append(wrap);
  const footer = node('footer', undefined, 'metric-panel-footer');
  footer.append(node('p', metric.note), node('span', metric.kind === 'bar' ? 'Cấu hình model' : `${valid.length} số đo`, 'metric-samples')); card.append(footer);
  return card;
}
function historyRows() {
  return lastData.rows.map((r) => ({...r, throughput_ips: r.latency_ms > 0 ? 1000 / r.latency_ms : null}));
}
function renderDetail(force = false) {
  if (!openMetric || !$('#chart-dialog').open) return;
  const metric = METRICS.find((m) => m.key === openMetric);
  const signature = [openMetric, $('#detail-window').value, $('#history-axis').value, lastData.rows.at(-1)?.id, lastData.shown, filters().toString()].join('|');
  if (!force && signature === detailSignature) return;
  detailSignature = signature;
  let rows = historyRows();
  const count = Number($('#detail-window').value);
  if (count) {
    const groups = [...new Set(rows.map((r) => r.group_id))].slice(-count);
    rows = rows.filter((r) => groups.includes(r.group_id));
  }
  $('#chart-dialog-title').textContent = metric.title;
  $('#detail-chart').replaceChildren(drawChart(rows, metric, false));
  const table = $('#detail-rows'); table.replaceChildren();
  for (const r of [...rows].reverse()) {
    const row = node('tr');
    [new Date(r.created_at).toLocaleString('vi-VN'), r.sample_name, MODELS[r.model].name,
      number(r[metric.key], metric.digits ?? 3), metric.unit, r.config_id.slice(0, 6)]
      .forEach((value) => row.append(node('td', value)));
    table.append(row);
  }
  $('#detail-count').textContent = `${rows.length} số đo · theo bộ lọc đang chọn · ${$('#history-axis').selectedOptions[0].textContent}`;
}
function openDetail(key) {
  dismissChartTooltip();
  openMetric = key; detailSignature = '';
  $('#detail-window').value = '0';
  $('#chart-dialog').showModal();
  document.body.classList.add('dialog-open');
  renderDetail(true);
}
function render(data) {
  const rows = data.rows.map((r) => ({...r, throughput_ips: r.latency_ms > 0 ? 1000 / r.latency_ms : null}));
  $('#history-count').textContent = `${data.shown} lượt đã lưu`;
  $('#history-range').textContent = `Đang hiển thị ${data.shown} / ${data.total} lượt phù hợp bộ lọc`;
  $('#history-updated').textContent = loaded ? `Cập nhật ${new Date().toLocaleTimeString('vi-VN')}` : 'Đang kết nối backend';
  $('#history-pairs').textContent = new Set(rows.map((r) => r.group_id)).size;
  $('#history-images').textContent = new Set(rows.map((r) => r.sample_index)).size;
  const select = $('#history-session'), selected = select.value;
  select.replaceChildren(new Option('Tất cả phiên', 'all'), new Option('Phiên hiện tại', 'current'));
  data.sessions.forEach((s) => select.add(new Option(`${new Date(s.started_at).toLocaleString('vi-VN')} · ${s.count} lượt`, s.session)));
  select.value = [...select.options].some((o) => o.value === selected) ? selected : 'all';
  const signature = [rows[0]?.id, rows.at(-1)?.id, rows.length, filters().toString(), $('#history-axis').value, loaded].join('|');
  if (signature !== chartSignature) {
    chartSignature = signature;
    $('#history-charts').replaceChildren(...METRICS.map((metric) => drawChart(rows, metric)));
  }
  const table = $('#history-summary'); table.replaceChildren();
  data.summaries.forEach((s) => {
    const row = node('tr'), identity = node('td', `${MODELS[s.model].name} · ${s.config_id.slice(0, 6)}`);
    identity.append(node('small', `${s.config.device || '—'} · ngưỡng ${s.config.threshold ?? '—'} · checkpoint ${(s.config.checkpoint_sha256 || 'không rõ').slice(0, 8)}`)); row.append(identity);
    [s.count, s.unique_samples, number(s.mean_latency_ms), number(s.median_ms), number(s.p95_ms), number(s.throughput_ips), number(s.mean_dice, 3), number(s.mean_iou, 3), number(s.micro_dice, 3), number(s.micro_iou, 3), number(s.mean_rss_peak_mb, 1), number(s.mean_rss_delta_mb, 1), number(s.mean_cpu_cores), number(s.mean_gpu_delta_mb, 1), number(s.mean_params_m), number(s.mean_gflops)].forEach((value) => row.append(node('td', String(value)))); table.append(row);
  });
  if (!data.summaries.length) { const row = node('tr'), cell = node('td', 'Chưa có số đo. Chọn ảnh trong tab So sánh để bắt đầu.'); cell.colSpan = 17; row.append(cell); table.append(row); }
  $('#history-csv').href = `/api/history.csv?${filters()}`;
  renderDetail();
}
const JOB_STATES = {running: 'Đang so sánh', stopping: 'Đang dừng sau ảnh hiện tại', complete: 'Đã hoàn tất', stopped: 'Đã dừng', interrupted: 'Bị ngắt do backend khởi động lại', error: 'Gặp lỗi'};
async function refreshBatch() {
  if (batchBusy) return;
  batchBusy = true;
  try {
    const response = await fetch('/api/batch', {cache: 'no-store'});
    if (!response.ok) throw new Error('Batch unavailable');
    const data = await response.json();
    activeJob = data.jobs.find((job) => ['running', 'stopping'].includes(job.state));
    const latest = activeJob || data.jobs[0];
    $('#batch-start').disabled = !data.ready || Boolean(activeJob);
    $('#batch-start').textContent = `So sánh toàn bộ ${data.total_samples} ảnh`;
    $('#batch-description').textContent = `Chạy ViT và EfficientViT qua ${data.total_samples} ảnh test đang có trên web. Kết quả tự lưu và cập nhật vào biểu đồ bên dưới.`;
    $('#batch-stop').hidden = !activeJob;
    $('#batch-stop').disabled = activeJob?.state === 'stopping';
    $('#batch-view').hidden = !latest;
    if (latest) {
      $('#batch-progress').max = latest.total; $('#batch-progress').value = latest.completed;
      $('#batch-status').textContent = `${JOB_STATES[latest.state] || latest.state} · ${latest.completed} / ${latest.total} ảnh`;
      $('#batch-current').textContent = latest.error || latest.current_name || (latest.state === 'complete' ? 'Cả hai model đã chạy xong toàn bộ tập ảnh.' : 'Số đo của các ảnh đã hoàn tất được giữ trong backend.');
    }
    const select = $('#history-batch'), selected = select.value;
    select.replaceChildren(new Option('Tất cả lượt so sánh', ''));
    data.jobs.forEach((job) => select.add(new Option(`${new Date(job.created_at).toLocaleString('vi-VN')} · ${job.completed}/${job.total} · ${JOB_STATES[job.state]}`, job.id)));
    if ([...select.options].some((o) => o.value === selected)) select.value = selected;
  } catch {
    $('#batch-start').disabled = true;
    $('#batch-description').textContent = 'Chưa kết nối tác vụ hàng loạt. Kiểm tra backend rồi tải lại trang.';
  } finally { batchBusy = false; }
}
async function batchAction(action) {
  $('#batch-start').disabled = true;
  $('#batch-error').hidden = true;
  try {
    const response = await fetch(`/api/batch/${action}`, {method: 'POST', headers: {'X-Requested-With': 'BoneScope'}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Không thể chạy tác vụ');
    if (action === 'start') {
      $('#history-batch').add(new Option('Tác vụ đang chạy', data.job_id));
      $('#history-batch').value = data.job_id;
      $('#history-session').value = 'all'; $('#history-hours').value = '0'; $('#history-limit').value = '2000';
    }
    await refreshBatch();
    await refreshHistory();
  } catch (error) {
    $('#batch-error').textContent = error.message;
    $('#batch-error').hidden = false;
    await refreshBatch();
  }
}
export async function refreshHistory() {
  if (busy) { refreshPending = true; return; }
  busy = true;
  const query = filters().toString();
  try {
    const response = await fetch(`/api/history?${query}`, {cache: 'no-store'});
    if (!response.ok) throw new Error('History unavailable');
    const data = await response.json();
    if (!Array.isArray(data.rows) || !Array.isArray(data.summaries) || !Array.isArray(data.sessions)) throw new Error('Invalid history response');
    if (filters().toString() === query) { loaded = true; lastData = data; render(data); $('#history-connection').hidden = true; }
    else refreshPending = true;
  } catch {
    $('#history-connection').hidden = false;
    $('#history-connection').textContent = 'Chưa tải được thống kê. Kiểm tra backend hoặc bấm Làm mới; các panel vẫn giữ số đo gần nhất.';
    $('#history-updated').textContent = 'Chưa kết nối';
  } finally {
    busy = false;
    if (refreshPending) { refreshPending = false; queueMicrotask(refreshHistory); }
  }
}
export function initHistory() {
  initChartTooltips();
  render(lastData);
  for (const id of ['#history-session', '#history-hours', '#history-limit', '#history-batch']) $(id).addEventListener('change', refreshHistory);
  $('#history-axis').addEventListener('change', () => { render(lastData); renderDetail(true); });
  $('#history-refresh').addEventListener('click', refreshHistory);
  $('#chart-dialog-close').addEventListener('click', () => $('#chart-dialog').close());
  $('#chart-dialog').addEventListener('click', (event) => { if (event.target === $('#chart-dialog')) $('#chart-dialog').close(); });
  $('#chart-dialog').addEventListener('close', () => {
    dismissChartTooltip();
    const key = openMetric; openMetric = null; document.body.classList.remove('dialog-open');
    document.querySelector(`[data-metric="${key}"] .chart-expand`)?.focus({preventScroll: true});
  });
  $('#detail-window').addEventListener('change', () => renderDetail(true));
  $('#batch-start').addEventListener('click', () => batchAction('start'));
  $('#batch-stop').addEventListener('click', () => batchAction('stop'));
  refreshBatch();
  setInterval(() => { if (!$('#stats-view').hidden) { refreshHistory(); refreshBatch(); } }, 2500);
}
