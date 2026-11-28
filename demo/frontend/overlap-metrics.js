export const isOverlapMetric = (key) => ['dice', 'iou'].includes(key);

export function maskStatus(row) {
  const counted = ['tp', 'fp', 'fn'].every((key) => Number.isFinite(row[key]) && row[key] >= 0);
  if (!counted) return {kind: 'unknown', text: 'Chưa lưu bộ đếm mask'};
  if (row.tp + row.fn > 0) return {kind: 'foreground', text: 'GT có vùng u'};
  return {kind: 'empty', text: row.fp === 0 ? 'GT và dự đoán đều rỗng · quy ước = 1' : 'GT rỗng · dự đoán dư vùng · điểm = 0'};
}

export function overlapRows(rows, selection = 'all') {
  return selection === 'all' ? rows : rows.filter((row) => maskStatus(row).kind === selection);
}
