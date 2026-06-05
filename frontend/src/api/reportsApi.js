import api from './axiosClient';

export const getSummaryReport = (period = 'weekly') =>
  api.get('/reports/summary', { params: { period } }).then((res) => res.data);

export const getAnalystPerformance = () =>
  api.get('/reports/analyst-performance').then((res) => res.data);

export const getMissedFlags = (threshold = 0.6) =>
  api.get('/reports/missed-flags', { params: { threshold } }).then((res) => res.data);

// ── SAR Reports (Bug #4) ────────────────────────────────────────────────────

export const getSARReports = (params = {}) =>
  api.get('/reports/sar', { params }).then((res) => res.data);

export const getSARDetail = (alertId) =>
  api.get(`/reports/sar/${alertId}`).then((res) => res.data);

export const updateSARStatus = (alertId, sarStatus) =>
  api.patch(`/reports/sar/${alertId}`, { sar_status: sarStatus }).then((res) => res.data);

export const downloadSARPdf = async (alertId) => {
  const res = await api.get(`/reports/sar/${alertId}/pdf`, { responseType: 'blob' });
  const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `SAR-${alertId}.pdf`);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
};
