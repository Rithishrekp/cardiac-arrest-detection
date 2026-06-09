/**
 * API client for the Cardiac Risk Prediction backend.
 * Base URL is configurable via VITE_API_URL (defaults to /api for Vite proxy).
 */

const BASE_URL = import.meta.env.VITE_API_URL || '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch {
      /* ignore parse errors */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  return res.json();
}

export const api = {
  health: () => request('/health'),

  predict: (payload) =>
    request('/predict', { method: 'POST', body: JSON.stringify(payload) }),

  history: (search = '') => {
    const params = search ? `?search=${encodeURIComponent(search)}` : '';
    return request(`/history${params}`);
  },

  stats: () => request('/history/stats'),

  getRecord: (id) => request(`/history/${id}`),

  correlations: () => request('/history/correlations'),
};

