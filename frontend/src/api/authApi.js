import api from './axiosClient';

export const loginUser = async (email, password) => {
  const response = await api.post('/auth/login', {
    email: email.trim(),
    password: password.trim(),
  });
  return response.data;
};

export const getCurrentUser = () => api.get('/users/me').then((res) => res.data);
