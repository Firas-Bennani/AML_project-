import api from './axiosClient';

export const getTransactions = (params = {}) =>
  api.get('/transactions/', { params }).then((res) => res.data);

export const createTransaction = (payload) =>
  api.post('/transactions/', payload).then((res) => res.data);
