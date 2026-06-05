import api from './axiosClient';

export const getAlerts = (params = {}) => api.get('/alerts/', { params }).then((res) => res.data);

export const updateAlert = (id, payload) =>
  api.patch(`/alerts/${id}`, payload).then((res) => res.data);
