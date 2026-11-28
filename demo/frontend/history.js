import {chartTooltip, dismissChartTooltip, initChartTooltips} from './chart-tooltip.js?v=20261128-bars';
import {isOverlapMetric, maskStatus, overlapRows} from './overlap-metrics.js';
import {drawOverlapBars, recentRunRows} from './overlap-bars.js?v=20261128-history';

const $ = (selector) => document.querySelector(selector);
const MODELS = {vit: {name: 'ViT', color: '#137e75'}, efficientvit: {name: 'EfficientViT', color: '#c8773e'}};
const METRICS = [
  {key: 'latency_ms', title: 'Thời gian suy luận', unit: 'ms', note: 'Thời gian forward trên từng ảnh. Giá trị thấp hơn tương ứng với suy luận nhanh hơn.', goal: 'THẤP HƠN TỐT HƠN'},
  {key: 'throughput_ips', title: 'Tốc độ xử lý', unit: 'ảnh/s', note: 'Quy đổi từ thời gian forward: 1.000 / thời gian tính bằng ms.', goal: 'CAO HƠN TỐT HƠN'},
  {key: 'dice', title: 'Dice', unit: '0–1', note: 'Mỗi cột là Dice của đúng ảnh trong một lượt chạy: 2TP / (2TP + FP + FN). GT và dự đoán cùng rỗng được quy ước = 1.', goal: 'CAO HƠN TỐT HƠN', digits: 6},
  {key: 'iou', title: 'Intersection over Union', unit: '0–1', note: 'Mỗi cột là IoU của đúng ảnh trong một lượt chạy: TP / (TP + FP + FN). GT và dự đoán cùng rỗng được quy ước = 1.', goal: 'CAO HƠN TỐT HƠN', digits: 6},
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
let batchActionPending = false;
let activeJob = null;
let reportJob = null;
let detailSignature = '';
let chartSignature = '';
let onBatchProgress = () => {};
let detailCache = null;
let detailScope = '';
let detailRequest = 0;
let detailEndGroup = null;
let detailRange = {groups: [], start: 0, end: 0};
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
function metricRows(rows, metric) {
  return isOverlapMetric(metric.key) ? overlapRows(rows, $('#history-mask').value) : rows;
}
function drawChart(rows, metric, expandable = true, referenceRows = rows) {
  const allRows = [...rows].sort((a, b) => a.id - b.id);
  rows = metricRows(allRows, metric);
  const overlap = isOverlapMetric(metric.key);
  if (overlap && expandable) rows = recentRunRows(rows, 20);
  const card = node('article', undefined, 'metric-panel');
  card.dataset.metric = metric.key;
  if (overlap) card.classList.add('overlap-panel');
  const header = node('header', undefined, 'metric-panel-header'), heading = node('div');
  heading.append(node('h3', metric.title));
  const actions = node('div', undefined, 'metric-panel-actions');
  actions.append(node('span', metric.unit, 'metric-unit'));
  const info = node('span', 'i', 'metric-info');
  info.title = metric.note; info.setAttribute('aria-label', metric.note); info.tabIndex = 0;
  actions.append(info);
  if (expandable) {
    const expand = node('button', undefined, 'chart-expand');
    expand.type = 'button'; expand.title = `Phóng to ${metric.title}`;
    expand.setAttribute('aria-label', `Phóng to biểu đồ ${metric.title}`);
    const icon = svgNode('svg', {viewBox: '0 0 24 24', width: 18, height: 18, 'aria-hidden': true});
    icon.append(svgNode('path', {d: 'M8 3H3v5M3 3l6 6M16 3h5v5m0-5-6 6M3 16v5h5m-5 0 6-6m12 1v5h-5m5 0-6-6',
      fill: 'none', stroke: 'currentColor', 'stroke-width': 1.7, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}));
    expand.append(icon);
    expand.addEventListener('click', () => openDetail(metric.key));
    actions.append(expand);
  }
  header.append(heading, actions); card.append(header);
  const valid = rows.filter((r) => numeric(r[metric.key]));
  const latest = Object.fromEntries(Object.keys(MODELS).map((id) => [id, valid.filter((r) => r.model === id).at(-1)]));
  const values = node('div', undefined, 'metric-panel-values');
  values.title = 'Số đo mới nhất';
  for (const [id, model] of Object.entries(MODELS)) {
    const value = node('div', undefined, `metric-panel-value ${id}`);
    const row = latest[id];
    value.append(node('span', model.name, 'model-name'), node('strong', number(row?.[metric.key], metric.digits ?? 2)));
    if (overlap && row) {
      value.append(node('small', row.sample_name, 'metric-run'));
    }
    values.append(value);
  }
  card.append(values);
  const wrap = node('div', undefined, 'metric-plot');
  if (!valid.length) {
    const empty = node('div', undefined, 'metric-empty');
    const gpuUnavailable = metric.key === 'gpu_delta_mb' && rows.length;
    const foregroundOnly = overlap && loaded && $('#history-mask').value === 'foreground';
    const title = gpuUnavailable ? 'Chưa có lượt chạy CUDA' : foregroundOnly ? 'Chưa có lượt ảnh có vùng u trong phạm vi đang chọn' : loaded ? 'Chưa có dữ liệu trong bộ lọc' : 'Đang tải số đo';
    const note = gpuUnavailable ? 'Máy đang chạy CPU. Số đo VRAM sẽ xuất hiện khi sử dụng GPU.' : foregroundOnly ? 'Chạy ảnh có vùng u, tăng số lượt lịch sử hoặc chọn “Tất cả ảnh” để xem thêm GT rỗng.' : 'Mỗi ảnh được phân vùng sẽ thêm một điểm cho cả hai model.';
    empty.append(node('strong', title), node('p', note)); wrap.append(empty);
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
  } else if (overlap) {
    drawOverlapBars(wrap, valid, referenceRows, metric, MODELS, number,
      `${expandable ? 'panel' : 'detail'}:${metric.key}`, $('#history-axis').value === 'time', $('#history-mask').value);
  } else {
    const byTime = $('#history-axis').value === 'time';
    const groups = [...new Set(allRows.map((r) => r.group_id))], positions = new Map(groups.map((id, index) => [id, index]));
    const ordinals = new Map([...new Set(referenceRows.map((row) => row.group_id))].map((id, index) => [id, index + 1]));
    const times = allRows.map((r) => Date.parse(r.created_at)), first = Math.min(...times), last = Math.max(...times);
    const max = overlap ? 1 : Math.max(...valid.map((r) => r[metric.key]), 0.001) * 1.12;
    const expanded = !expandable;
    const width = expanded ? Math.max(280, $('#chart-dialog').clientWidth - 80) : 640;
    const height = expanded ? Math.max(260, Math.min(320, Math.round(innerHeight * 0.35))) : 240;
    const left = 58, right = width - 30, top = 22, bottom = height - 50;
    const x = (r) => left + (byTime ? (last === first ? 0.5 : (Date.parse(r.created_at) - first) / (last - first)) : groups.length === 1 ? 0.5 : positions.get(r.group_id) / (groups.length - 1)) * (right - left);
    const y = (value) => bottom - value / max * (bottom - top);
    const svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': `${metric.title}: ViT và EfficientViT`});
    if (expanded) {
      svg.style.height = `${height}px`;
      svg.style.width = '100%';
    }
    for (let tick = 0; tick <= 4; tick++) {
      const value = max * tick / 4;
      svg.append(svgNode('line', {x1: left, x2: right, y1: y(value), y2: y(value), class: 'plot-grid'}), svgNode('text', {x: left - 12, y: y(value) + 4, 'text-anchor': 'end', class: 'plot-axis'}, number(value, max <= 2 ? 2 : 0)));
    }
    const tooltip = chartTooltip(wrap, metric, `${expandable ? 'panel' : 'detail'}:${metric.key}`, number);
    const series = new Map();
    valid.forEach((r) => {
      const key = `${r.session}:${r.batch_id || 'manual'}:${r.config_id}`;
      if (!series.has(key)) series.set(key, []);
      series.get(key).push(r);
    });
    let seriesIndex = 0;
    for (const values of series.values()) {
      const model = MODELS[values[0].model];
      svg.append(svgNode('polyline', {points: values.map((r) => `${x(r)},${y(r[metric.key])}`).join(' '), fill: 'none', stroke: model.color, 'stroke-width': expanded ? 1.7 : 2.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke', 'stroke-dasharray': seriesIndex++ > 1 ? '6 4' : 'none'}));
      values.forEach((r) => {
        const point = svgNode('g', {class: `plot-point${overlap ? ' overlap-point' : ''}`, 'data-run-id': r.id, 'data-value': r[metric.key]});
        point.append(svgNode('circle', {cx: x(r), cy: y(r[metric.key]), r: 10, fill: 'transparent', class: 'plot-point-hit'}),
          svgNode('circle', {cx: x(r), cy: y(r[metric.key]), r: overlap && r.model === 'efficientvit' ? 5 : 3, fill: overlap && r.model === 'efficientvit' ? 'none' : model.color, stroke: overlap ? model.color : '#fff', 'stroke-width': 1.5, class: 'plot-point-dot'}));
        tooltip.bind(point, r, model, ordinals.get(r.group_id));
        svg.append(point);
      });
    }
    const timeLabel = (time) => new Date(time).toLocaleTimeString('vi-VN', {hour12: false});
    const firstTurn = ordinals.get(groups[0]), lastTurn = ordinals.get(groups.at(-1));
    svg.append(svgNode('text', {x: left, y: height - 20, class: 'plot-axis'}, byTime ? timeLabel(first) : `Lượt ${firstTurn}`), svgNode('text', {x: right, y: height - 20, class: 'plot-axis', 'text-anchor': 'end'}, byTime ? timeLabel(last) : `Lượt ${lastTurn}`));
    wrap.append(svg);
    tooltip.restore();
  }
  card.append(wrap);
  const footer = node('footer', undefined, 'metric-panel-footer');
  const emptyCount = overlap ? valid.filter((row) => maskStatus(row).kind === 'empty').length : 0;
  footer.append(node('span', metric.kind === 'bar' ? 'Cấu hình model' : `${valid.length} số đo${emptyCount ? ` · ${emptyCount} GT rỗng` : ''}`, 'metric-samples')); card.append(footer);
  return card;
}
function detailScopeKey() {
  const query = filters(); query.delete('limit');
  return query.toString();
}
function detailRows() {
  return (detailCache || lastData.rows).map((row) => ({...row,
    throughput_ips: row.latency_ms > 0 ? 1000 / row.latency_ms : null}));
}
async function loadDetailHistory() {
  const request = ++detailRequest;
  const scope = detailScopeKey();
  detailScope = scope;
  $('#detail-message').textContent = 'Đang tải lịch sử...';
  try {
    const query = new URLSearchParams(scope); query.set('limit', '2000');
    const response = await fetch(`/api/history?${query}`, {cache: 'no-store'});
    if (!response.ok) throw new Error('History unavailable');
    const data = await response.json();
    if (!Array.isArray(data.rows)) throw new Error('Invalid history');
    if (request !== detailRequest || !$('#chart-dialog').open || scope !== detailScopeKey()) return;
    const merged = new Map([...data.rows, ...lastData.rows].map((row) => [row.id, row]));
    detailCache = [...merged.values()].sort((a, b) => a.id - b.id).slice(-2000);
    $('#detail-message').textContent = data.total > data.rows.length ? `${data.rows.length} số đo gần nhất; lọc phiên hoặc tác vụ để xem xa hơn.` : '';
    renderDetail(true);
  } catch {
    if (request === detailRequest && $('#chart-dialog').open) $('#detail-message').textContent = 'Chưa tải thêm được lịch sử. Đang giữ các số đo đã có.';
  }
}
function syncDetailHistory() {
  if (!openMetric || !$('#chart-dialog').open) return;
  if (detailScope !== detailScopeKey()) {
    detailCache = null; detailEndGroup = null;
    loadDetailHistory();
  } else if (detailCache) {
    const merged = new Map([...detailCache, ...lastData.rows].map((row) => [row.id, row]));
    detailCache = [...merged.values()].sort((a, b) => a.id - b.id).slice(-2000);
  }
  renderDetail();
}
function renderDetail(force = false) {
  if (!openMetric || !$('#chart-dialog').open) return;
  const metric = METRICS.find((m) => m.key === openMetric);
  const sourceRows = detailRows();
  const signature = [openMetric, $('#detail-window').value, $('#history-axis').value, $('#history-mask').value, sourceRows[0]?.id, sourceRows.at(-1)?.id, sourceRows.length, detailEndGroup, filters().toString()].join('|');
  if (!force && signature === detailSignature) return;
  detailSignature = signature;
  let rows = metricRows(sourceRows, metric);
  const groups = [...new Set(rows.map((row) => row.group_id))];
  const count = Number($('#detail-window').value);
  const anchor = detailEndGroup === null ? -1 : groups.indexOf(detailEndGroup);
  const end = !count || detailEndGroup === null ? groups.length : anchor >= 0 ? anchor + 1 : Math.min(count, groups.length);
  const start = count ? Math.max(0, end - count) : 0;
  detailRange = {groups, start, end};
  if (count) {
    const selectedGroups = new Set(groups.slice(start, end));
    rows = rows.filter((row) => selectedGroups.has(row.group_id));
  }
  $('#chart-dialog-title').textContent = metric.title;
  $('#detail-chart').replaceChildren(drawChart(rows, metric, false, sourceRows));
  const table = $('#detail-rows'); table.replaceChildren();
  const selected = metricRows(rows, metric);
  for (const r of [...selected].reverse()) {
    const row = node('tr');
    [new Date(r.created_at).toLocaleString('vi-VN'), r.sample_name, MODELS[r.model].name,
      number(r[metric.key], metric.digits ?? 3), metric.unit, r.config_id.slice(0, 6), `#${r.id}`,
      isOverlapMetric(metric.key) ? maskStatus(r).text : '—',
      ...['tp', 'fp', 'fn'].map((key) => numeric(r[key]) ? String(r[key]) : '—')]
      .forEach((value) => row.append(node('td', value)));
    table.append(row);
  }
  $('#detail-count').textContent = groups.length ? `Lượt ${start + 1}–${end} / ${groups.length} · ${selected.length} số đo` : 'Chưa có số đo';
  $('#detail-older').disabled = !count || start === 0;
  $('#detail-newer').disabled = !count || end === groups.length;
  $('#detail-latest').disabled = detailEndGroup === null || !count;
}
function moveDetail(direction) {
  dismissChartTooltip();
  const {groups, start, end} = detailRange;
  const count = Number($('#detail-window').value);
  const nextEnd = direction < 0 ? start : Math.min(groups.length, end + count);
  detailEndGroup = nextEnd >= groups.length ? null : groups[Math.max(0, nextEnd - 1)];
  renderDetail(true);
}
function openDetail(key) {
  dismissChartTooltip();
  openMetric = key; detailSignature = '';
  detailCache = null; detailEndGroup = null; detailScope = detailScopeKey();
  $('#detail-window').value = '100';
  $('#chart-dialog').showModal();
  document.body.classList.add('dialog-open');
  renderDetail(true);
  loadDetailHistory();
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
  const signature = [rows[0]?.id, rows.at(-1)?.id, rows.length, filters().toString(), $('#history-axis').value, $('#history-mask').value, loaded].join('|');
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
  syncDetailHistory();
}
const JOB_STATES = {running: 'Đang so sánh', stopping: 'Đang dừng sau ảnh hiện tại', complete: 'Đã hoàn tất', stopped: 'Đã dừng', interrupted: 'Bị ngắt do backend khởi động lại', error: 'Gặp lỗi'};
const REPORT_CHART_LABELS = {
  'accuracy.png': 'Độ chính xác',
  'paired-dice.png': 'Dice theo ảnh',
  'latency.png': 'Tốc độ',
  'complexity.png': 'Độ phức tạp',
  'resources.png': 'Tài nguyên',
};
function renderReportCharts(charts, url) {
  const entries = Object.entries(charts);
  if (!entries.length) return;
  const image = $('#batch-chart-image');
  const buttons = entries.map(([file, title], index) => {
    const button = node('button', REPORT_CHART_LABELS[file] || title);
    button.type = 'button';
    button.setAttribute('aria-controls', image.id);
    button.addEventListener('click', () => selectChart(index));
    return button;
  });
  function selectChart(index) {
    const [file, title] = entries[index];
    const href = url + '&file=' + encodeURIComponent(file);
    buttons.forEach((button, position) => button.setAttribute('aria-pressed', String(position === index)));
    image.alt = title;
    image.src = href;
    $('#batch-chart-caption').textContent = title;
    $('#batch-chart-image-link').href = href;
    $('#batch-chart-original').href = href;
  }
  $('#batch-chart-buttons').replaceChildren(...buttons);
  selectChart(0);
  $('#batch-report-charts').hidden = false;
}

async function showBatchReport(job) {
  const id = job?.id ?? null;
  if (id === reportJob) return;
  reportJob = id;
  $('#batch-report').hidden = !id;
  const summary = $('#batch-report-summary'), highlights = $('#batch-report-highlights');
  summary.hidden = true;
  summary.open = false;
  const download = $('#batch-report-download'), status = $('#batch-report-status');
  download.hidden = true;
  highlights.replaceChildren();
  $('#batch-report-charts').hidden = true;
  $('#batch-chart-image').removeAttribute('src');
  if (!id) return;
  status.textContent = 'Đang tạo biểu đồ và file tải xuống…';
  const url = '/api/batch/report?job_id=' + encodeURIComponent(id);
  try {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Chưa thể tạo báo cáo.');
    if (id !== reportJob) return;
    const metrics = [
      ['Dice pixel gộp', 'micro_dice', 4], ['IoU pixel gộp', 'micro_iou', 4],
      ['Dice trung bình ảnh có u', 'tumor_mean_dice', 4],
      ['Tỷ lệ ảnh thường tô nhầm (%)', 'normal_false_positive_rate', 2, 100],
      ['Độ trễ trung vị (ms)', 'median_ms', 2], ['Độ trễ p95 (ms)', 'p95_ms', 2],
      ['Thông lượng (ảnh/s)', 'throughput_ips', 2],
    ];
    const values = metrics.map(([label, key, digits, scale = 1]) => ({
      label,
      key,
      models: Object.entries(MODELS).map(([model, info]) => {
        const value = data.summaries[model][key];
        return {name: info.name, text: number(numeric(value) ? value * scale : null, digits)};
      }),
    }));
    $('#batch-report-metrics').replaceChildren(...values.map(({label, models}) => {
      const row = node('tr');
      const heading = node('th', label);
      heading.scope = 'row';
      row.append(heading, ...models.map(({text}) => node('td', text)));
      return row;
    }));
    highlights.replaceChildren(...values.filter(({key}) => ['micro_dice', 'micro_iou', 'median_ms'].includes(key))
      .map(({label, models}) => {
        const card = node('div', undefined, 'batch-highlight');
        const modelValues = node('div', undefined, 'batch-highlight-values');
        modelValues.append(...models.map(({name, text}) => {
          const value = node('div');
          value.append(node('small', name), node('strong', text));
          return value;
        }));
        card.append(node('h4', label), modelValues);
        return card;
    }));
    renderReportCharts(data.charts, url);
    status.textContent = data.image_count + ' ảnh · ' + new Date(data.completed_at).toLocaleString('vi-VN');
    download.href = url + '&file=report.zip';
    download.hidden = false;
    summary.hidden = false;
  } catch (error) {
    if (id === reportJob) status.textContent = error.message + ' Tải lại trang để thử lại.';
  }
}
async function refreshBatch() {
  if (batchBusy) return;
  batchBusy = true;
  try {
    const response = await fetch('/api/batch', {cache: 'no-store'});
    if (!response.ok) throw new Error('Batch unavailable');
    const data = await response.json();
    activeJob = data.jobs.find((job) => ['running', 'stopping'].includes(job.state));
    const latest = activeJob || data.jobs[0];
    const selection = data.selection;
    $('#batch-start').disabled = batchActionPending || !data.ready || !selection || Boolean(activeJob);
    const actionLabel = latest && !activeJob ? 'Chạy lại từ đầu' : 'Chạy';
    $('#batch-start').textContent = selection ? `${actionLabel} · ${data.total_samples} ảnh` : 'Chạy so sánh';
    $('#batch-description').textContent = selection
      ? selection.included_count + ' / ' + selection.source_count + ' ảnh đạt · ' +
        (selection.included_by_label.tumor ?? 0) + ' có u · ' +
        (selection.included_by_label.normal ?? 0) + ' bình thường'
      : 'Server này chưa có danh sách ảnh đã kiểm tra. Khởi động lại BoneScope với start-web.ps1.';
    if (!('preview' in data)) $('#batch-description').textContent = 'Khởi động lại server để bật preview trực tiếp.';
    $('#batch-stop').hidden = !activeJob;
    $('#batch-stop').disabled = batchActionPending || activeJob?.state === 'stopping';
    $('#batch-view').hidden = !latest;
    if (latest) {
      $('#batch-progress').max = latest.total; $('#batch-progress').value = latest.completed;
      $('#batch-status').textContent = `${JOB_STATES[latest.state] || latest.state} · ${latest.completed} / ${latest.total} ảnh`;
      $('#batch-current').textContent = latest.error || '';
      $('#batch-current').hidden = !latest.error;
    }
    const select = $('#history-batch'), selected = select.value;
    select.replaceChildren(new Option('Tất cả lượt so sánh', ''));
    data.jobs.forEach((job) => select.add(new Option(`${new Date(job.created_at).toLocaleString('vi-VN')} · ${job.completed}/${job.total} · ${JOB_STATES[job.state]}`, job.id)));
    if ([...select.options].some((o) => o.value === selected)) select.value = selected;
    const report = select.value ? data.jobs.find((job) => job.id === select.value)
      : activeJob || data.jobs.find((job) => job.report_ready);
    showBatchReport(report?.report_ready ? report : null);
    onBatchProgress(data);
  } catch {
    $('#batch-start').disabled = true;
    $('#batch-description').textContent = 'Chưa kết nối tác vụ hàng loạt. Kiểm tra backend rồi tải lại trang.';
  } finally { batchBusy = false; }
}
async function batchAction(action) {
  if (batchActionPending) return;
  batchActionPending = true;
  $('#batch-start').disabled = true;
  $('#batch-stop').disabled = true;
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
  } catch (error) {
    $('#batch-error').textContent = error.message;
    $('#batch-error').hidden = false;
  } finally {
    batchActionPending = false;
    await refreshBatch();
    await refreshHistory();
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
export function initHistory(batchListener = () => {}) {
  onBatchProgress = batchListener;
  initChartTooltips();
  render(lastData);
  for (const id of ['#history-session', '#history-hours', '#history-limit', '#history-batch']) $(id).addEventListener('change', refreshHistory);
  $('#history-axis').addEventListener('change', () => { render(lastData); renderDetail(true); });
  $('#history-mask').addEventListener('change', () => { dismissChartTooltip(); detailEndGroup = null; render(lastData); renderDetail(true); });
  $('#history-refresh').addEventListener('click', refreshHistory);
  $('#chart-dialog-close').addEventListener('click', () => $('#chart-dialog').close());
  $('#chart-dialog').addEventListener('click', (event) => { if (event.target === $('#chart-dialog')) $('#chart-dialog').close(); });
  $('#chart-dialog').addEventListener('close', () => {
    dismissChartTooltip();
    detailRequest++; detailCache = null;
    const key = openMetric; openMetric = null; document.body.classList.remove('dialog-open');
    document.querySelector(`[data-metric="${key}"] .chart-expand`)?.focus({preventScroll: true});
  });
  $('#detail-window').addEventListener('change', () => { detailEndGroup = null; renderDetail(true); });
  $('#detail-older').addEventListener('click', () => moveDetail(-1));
  $('#detail-newer').addEventListener('click', () => moveDetail(1));
  $('#detail-latest').addEventListener('click', () => { detailEndGroup = null; renderDetail(true); });
  let resizeFrame;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => renderDetail(true));
  });
  $('#batch-start').addEventListener('click', () => batchAction('start'));
  $('#batch-stop').addEventListener('click', () => batchAction('stop'));
  $('#history-batch').addEventListener('change', refreshBatch);
  const batchReady = refreshBatch();
  setInterval(() => { if (!$('#stats-view').hidden) refreshHistory(); }, 2500);
  setInterval(() => { if (!$('#stats-view').hidden || activeJob) refreshBatch(); }, 750);
  return batchReady;
}
