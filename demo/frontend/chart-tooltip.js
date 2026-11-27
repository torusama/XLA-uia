let active = null;
let pinned = null;
let hideTimer;
let nextId = 0;

const element = (tag, text, className) => {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
};

export function dismissChartTooltip() {
  clearTimeout(hideTimer);
  if (active) {
    active.tooltip.hidden = true;
    active.point.classList.remove('is-active');
    active.point.setAttribute('aria-pressed', 'false');
  }
  active = null;
  pinned = null;
}

function positionTooltip() {
  if (!active || !active.point.isConnected) return;
  const {point, tooltip} = active;
  const anchor = point.getBoundingClientRect();
  const dialog = point.closest('dialog');
  const boundary = dialog ? dialog.getBoundingClientRect() : {left: 0, top: 0, right: innerWidth, bottom: innerHeight};
  const left = Math.max(8, boundary.left + 8), right = Math.min(innerWidth - 8, boundary.right - 8);
  const top = Math.max(8, boundary.top + 8), bottom = Math.min(innerHeight - 8, boundary.bottom - 8);
  tooltip.style.width = `${Math.min(284, right - left)}px`;
  const box = tooltip.getBoundingClientRect();
  let x = anchor.right + 12;
  if (x + box.width > right) x = anchor.left - box.width - 12;
  let y = anchor.top - box.height - 12;
  if (y < top) y = anchor.bottom + 12;
  tooltip.style.left = `${Math.max(left, Math.min(x, right - box.width))}px`;
  tooltip.style.top = `${Math.max(top, Math.min(y, bottom - box.height))}px`;
}

function hideSoon() {
  clearTimeout(hideTimer);
  if (!pinned) hideTimer = setTimeout(dismissChartTooltip, 160);
}

export function initChartTooltips() {
  document.addEventListener('pointerdown', (event) => {
    if (!event.target.closest('.plot-point, .plot-tooltip')) dismissChartTooltip();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && active) {
      dismissChartTooltip();
      event.preventDefault();
      event.stopPropagation();
    }
  }, true);
  window.addEventListener('resize', positionTooltip);
  document.addEventListener('scroll', positionTooltip, {capture: true, passive: true});
}

export function chartTooltip(wrap, metric, context, format) {
  const tooltip = element('div', undefined, 'plot-tooltip');
  tooltip.id = `plot-tooltip-${++nextId}`;
  tooltip.hidden = true;
  tooltip.setAttribute('role', 'region');
  tooltip.setAttribute('aria-label', 'Thông tin điểm dữ liệu');
  tooltip.addEventListener('pointerenter', () => clearTimeout(hideTimer));
  tooltip.addEventListener('pointerleave', hideSoon);
  wrap.append(tooltip);
  const points = new Map();

  function show(point, row, model, turn, persist) {
    clearTimeout(hideTimer);
    if (!persist && pinned && active?.point.isConnected) return;
    if (!persist && active?.point === point) return;
    dismissChartTooltip();
    if (persist) pinned = {context, id: row.id};
    const header = element('div', undefined, 'plot-tooltip-header');
    const name = element('strong', model.name, 'plot-tooltip-model');
    name.style.color = model.color;
    const close = element('button', 'Đóng', 'plot-tooltip-close');
    close.type = 'button';
    close.setAttribute('aria-label', 'Đóng thông tin điểm dữ liệu');
    close.addEventListener('click', () => {
      point.focus({preventScroll: true});
      dismissChartTooltip();
    });
    header.append(name, close);
    const value = element('div', undefined, 'plot-tooltip-value');
    value.append(element('strong', format(row[metric.key], metric.digits ?? 2)), element('span', metric.unit));
    const details = element('dl', undefined, 'plot-tooltip-details');
    [['Ảnh', row.sample_name], ['Thời điểm', new Date(row.created_at).toLocaleString('vi-VN')],
      ['Lượt so sánh', String(turn)], ['Thiết bị', row.config?.device || '—']]
      .forEach(([label, text]) => details.append(element('dt', label), element('dd', text)));
    tooltip.replaceChildren(header, element('p', metric.title, 'plot-tooltip-title'), value, details,
      element('p', persist ? 'Đã ghim · bấm ngoài để đóng' : 'Bấm điểm để giữ thông tin', 'plot-tooltip-hint'));
    tooltip.hidden = false;
    point.classList.add('is-active');
    point.setAttribute('aria-pressed', String(persist));
    active = {point, tooltip};
    positionTooltip();
  }

  function nearest(event, fallback) {
    if (!event.clientX && !event.clientY) return fallback;
    const matrix = fallback.point.closest('svg').getScreenCTM();
    if (!matrix) return fallback;
    const location = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    let result = fallback, distance = Infinity;
    for (const candidate of points.values()) {
      const squared = (candidate.x - location.x) ** 2 + (candidate.y - location.y) ** 2;
      if (squared < distance) { distance = squared; result = candidate; }
    }
    return result;
  }

  return {
    bind(point, row, model, turn) {
      const hit = point.querySelector('.plot-point-hit');
      const binding = {point, row, model, turn, x: hit.cx.baseVal.value, y: hit.cy.baseVal.value};
      const toggle = (selected) => {
        if (pinned?.context === context && pinned.id === selected.row.id) dismissChartTooltip();
        else show(selected.point, selected.row, selected.model, selected.turn, true);
      };
      const hover = (event) => {
        const selected = nearest(event, binding);
        show(selected.point, selected.row, selected.model, selected.turn, false);
      };
      points.set(row.id, binding);
      point.setAttribute('role', 'button');
      point.setAttribute('tabindex', '0');
      point.setAttribute('aria-pressed', 'false');
      point.setAttribute('aria-controls', tooltip.id);
      point.setAttribute('aria-label', `${model.name}, ${row.sample_name}, ${metric.title}: ${format(row[metric.key], metric.digits ?? 2)} ${metric.unit}. Bấm để xem thông tin.`);
      point.addEventListener('pointerenter', hover);
      point.addEventListener('pointermove', hover);
      point.addEventListener('pointerleave', hideSoon);
      point.addEventListener('focus', () => show(point, row, model, turn, false));
      point.addEventListener('blur', hideSoon);
      point.addEventListener('click', (event) => toggle(nearest(event, binding)));
      point.addEventListener('keydown', (event) => {
        if (['Enter', ' '].includes(event.key)) { event.preventDefault(); toggle(binding); }
      });
    },
    restore() {
      if (pinned?.context !== context) return;
      const selected = points.get(pinned.id);
      if (!selected) { dismissChartTooltip(); return; }
      requestAnimationFrame(() => {
        if (selected.point.isConnected && pinned?.context === context && pinned.id === selected.row.id)
          show(selected.point, selected.row, selected.model, selected.turn, true);
      });
    },
  };
}
