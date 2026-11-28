import {chartTooltip} from './chart-tooltip.js?v=20261128-bars';
import {overlapRows} from './overlap-metrics.js';

const svgNode = (tag, attrs = {}, text) => {
  const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => result.setAttribute(key, value));
  if (text !== undefined) result.textContent = text;
  return result;
};

export function recentRunRows(rows, count) {
  if (!count) return rows;
  const groups = new Set([...new Set(rows.map((row) => row.group_id))].slice(-count));
  return rows.filter((row) => groups.has(row.group_id));
}

export function drawOverlapBars(wrap, rows, allRows, metric, models, format, context, byTime, selection = 'all') {
  const groups = new Map();
  rows.forEach((row) => {
    if (!groups.has(row.group_id)) groups.set(row.group_id, []);
    groups.get(row.group_id).push(row);
  });
  const ordinals = new Map([...new Set(allRows.map((row) => row.group_id))]
    .map((id, index) => [id, index + 1]));
  const legend = document.createElement('div');
  legend.className = 'overlap-bars-legend';
  const scope = document.createElement('strong');
  scope.style.cssText = 'flex-basis:100%;color:#3f606f;font-size:12px';
  scope.textContent = selection === 'foreground' ? 'Ảnh có u' : selection === 'empty' ? 'GT rỗng · quy ước 0/1' : 'Tất cả ảnh';
  legend.append(scope);
  for (const [id, model] of Object.entries(models)) {
    const label = document.createElement('span');
    label.className = `legend-${id}`;
    label.textContent = model.name;
    legend.append(label);
  }
  const range = document.createElement('small');
  const eligible = overlapRows(allRows, selection).filter((row) => Number.isFinite(row[metric.key]));
  range.textContent = `${groups.size} / ${new Set(eligible.map((row) => row.group_id)).size} lượt`;
  legend.append(range);
  const scroll = document.createElement('div');
  scroll.className = 'grouped-bars-scroll';
  scroll.tabIndex = 0;
  scroll.setAttribute('role', 'region');
  scroll.setAttribute('aria-label', `${metric.title}: cuộn ngang để xem các lượt chạy`);
  const expanded = context.startsWith('detail:');
  const width = Math.max(640, expanded ? document.querySelector('#chart-dialog').clientWidth - 80 : 0, 80 + groups.size * (expanded ? 24 : 32));
  const height = expanded ? Math.max(260, Math.min(320, Math.round(innerHeight * 0.35))) : 260;
  const left = 54, right = width - 20, top = 18, bottom = height - 62;
  const slot = (right - left) / groups.size;
  const barWidth = Math.min(12, slot * 0.34);
  const y = (value) => bottom - value * (bottom - top);
  const svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, role: 'group',
    'data-plot-top': top, 'data-plot-bottom': bottom,
    'aria-label': `${metric.title}: cột ViT và EfficientViT cạnh nhau theo từng lượt`});
  svg.style.width = `${width}px`;
  svg.style.height = `${height}px`;
  for (let tick = 0; tick <= 4; tick++) {
    const value = tick / 4;
    svg.append(svgNode('line', {x1: left, x2: right, y1: y(value), y2: y(value), class: 'plot-grid'}),
      svgNode('text', {x: left - 10, y: y(value) + 4, 'text-anchor': 'end', class: 'plot-axis'}, format(value, 2)));
  }
  const tooltip = chartTooltip(wrap, metric, context, format);
  let index = 0;
  for (const [groupId, values] of groups) {
    const center = left + (index + 0.5) * slot;
    Object.entries(models).forEach(([id, model], modelIndex) => {
      const row = values.find((value) => value.model === id);
      const x = center + (modelIndex === 0 ? -barWidth - 1 : 1);
      if (!row) {
        svg.append(svgNode('text', {x: x + barWidth / 2, y: bottom - 3,
          'text-anchor': 'middle', class: 'plot-axis'}, '—'));
        return;
      }
      const value = row[metric.key];
      const point = svgNode('g', {class: 'plot-point plot-bar', 'data-run-id': row.id,
        'data-value': value, 'data-model': id, 'data-group-id': groupId});
      point.append(svgNode('rect', {x, y: top, width: barWidth, height: bottom - top + 5,
        fill: 'transparent', class: 'plot-bar-hit'}));
      if (value === 0) {
        point.append(svgNode('line', {x1: x, x2: x + barWidth, y1: bottom, y2: bottom,
          stroke: model.color, 'stroke-width': 3, class: 'plot-bar-visual plot-zero-mark'}));
      } else {
        point.append(svgNode('rect', {x, y: y(value), width: barWidth,
          height: value * (bottom - top), fill: model.color, rx: 2, class: 'plot-bar-visual plot-bar-fill'}));
      }
      // Retain a coordinate anchor for the shared accessible tooltip.
      point.append(svgNode('circle', {cx: x + barWidth / 2, cy: y(value), r: 0,
        class: 'plot-point-hit', 'pointer-events': 'none'}));
      tooltip.bind(point, row, model, ordinals.get(groupId));
      svg.append(point);
    });
    if (index % Math.ceil(groups.size / 10) === 0 || index === groups.size - 1) {
      const label = byTime ? new Date(values[0].created_at).toLocaleTimeString('vi-VN',
        {hour: '2-digit', minute: '2-digit', hour12: false}) : `#${ordinals.get(groupId)}`;
      svg.append(svgNode('text', {x: center, y: bottom + 23, 'text-anchor': 'middle', class: 'plot-axis'}, label));
    }
    index++;
  }
  svg.append(svgNode('text', {x: left, y: height - 13, class: 'plot-axis'}, byTime ? 'Thời điểm' : 'Lượt chạy'));
  scroll.append(svg);
  wrap.append(legend, scroll);
  tooltip.restore();
}
