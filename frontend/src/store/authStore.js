import { create } from 'zustand';

const storedToken = localStorage.getItem('aml_token');
const storedUser = localStorage.getItem('aml_user');

export const useAuthStore = create((set) => ({
  token: storedToken || null,
  user: storedUser ? JSON.parse(storedUser) : null,

  login: (token, user) => {
    localStorage.setItem('aml_token', token);
    localStorage.setItem('aml_user', JSON.stringify(user));
    set({ token, user });
  },

  logout: () => {
    localStorage.removeItem('aml_token');
    localStorage.removeItem('aml_user');
    set({ token: null, user: null });
  },
}));
