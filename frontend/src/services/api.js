const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

// Helper for fetch with credentials
const fetchWithAuth = async (url, options = {}) => {
  const response = await fetch(`${BACKEND_URL}${url}`, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || 'Request failed');
  }
  
  return response.json();
};

// Family API
export const familyApi = {
  getAll: () => fetchWithAuth('/api/family'),
  
  create: (data) => fetchWithAuth('/api/family', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  
  update: (id, data) => fetchWithAuth(`/api/family/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  
  delete: (id) => fetchWithAuth(`/api/family/${id}`, {
    method: 'DELETE',
  }),
};

// Memories API
export const memoriesApi = {
  getAll: () => fetchWithAuth('/api/memories'),
  
  create: (data) => fetchWithAuth('/api/memories', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  
  update: (id, data) => fetchWithAuth(`/api/memories/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  
  delete: (id) => fetchWithAuth(`/api/memories/${id}`, {
    method: 'DELETE',
  }),
};

// Reminders API
export const remindersApi = {
  getAll: () => fetchWithAuth('/api/reminders'),
  
  create: (data) => fetchWithAuth('/api/reminders', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  
  toggle: (id) => fetchWithAuth(`/api/reminders/${id}/toggle`, {
    method: 'PUT',
  }),
  
  delete: (id) => fetchWithAuth(`/api/reminders/${id}`, {
    method: 'DELETE',
  }),
  
  reset: () => fetchWithAuth('/api/reminders/reset', {
    method: 'POST',
  }),
};

// Chat API
export const chatApi = {
  send: (message, sessionId) => fetchWithAuth('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId }),
  }),
  
  getHistory: (sessionId) => fetchWithAuth(`/api/chat/history/${sessionId}`),
};

// Upload API (files stored in MongoDB GridFS)
export const uploadApi = {
  uploadFile: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${BACKEND_URL}/api/upload`, {
      method: 'POST',
      credentials: 'include',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error('Upload failed');
    }
    
    const data = await response.json();
    // Return full URL - files are now served from /api/files/
    return `${BACKEND_URL}${data.url}`;
  },
  
  uploadMultiple: async (files) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    const response = await fetch(`${BACKEND_URL}/api/upload/multiple`, {
      method: 'POST',
      credentials: 'include',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error('Upload failed');
    }
    
    const data = await response.json();
    return data.urls.map(url => `${BACKEND_URL}${url}`);
  },
};

export default {
  familyApi,
  memoriesApi,
  remindersApi,
  chatApi,
  uploadApi,
};
