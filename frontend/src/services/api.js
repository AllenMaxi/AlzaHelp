const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

// Helper for fetch with credentials and auto-refresh on 401
const fetchWithAuth = async (url, options = {}, _retried = false) => {
  const response = await fetch(`${BACKEND_URL}${url}`, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (response.status === 401 && !_retried) {
    const refreshRes = await fetch(`${BACKEND_URL}/api/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
    });
    if (refreshRes.ok) return fetchWithAuth(url, options, true);
    window.location.href = '/login';
    throw new Error('Session expired');
  }

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

// Destinations API (GPS route guidance)
export const destinationsApi = {
  getAll: () => fetchWithAuth('/api/destinations'),

  create: (data) => fetchWithAuth('/api/destinations', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  update: (id, data) => fetchWithAuth(`/api/destinations/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),

  delete: (id) => fetchWithAuth(`/api/destinations/${id}`, {
    method: 'DELETE',
  }),
};

// Navigation guidance API
export const navigationApi = {
  getGuide: (destinationId, currentLatitude, currentLongitude) =>
    fetchWithAuth('/api/navigation/guide', {
      method: 'POST',
      body: JSON.stringify({
        destination_id: destinationId,
        current_latitude: currentLatitude,
        current_longitude: currentLongitude,
      }),
    }),
};

// Safety API (geofencing + alerts)
export const safetyApi = {
  getZones: (targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/zones${suffix}`);
  },

  createZone: (data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/zones${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  deleteZone: (zoneId, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/zones/${zoneId}${suffix}`, {
      method: 'DELETE',
    });
  },

  pingLocation: (latitude, longitude) =>
    fetchWithAuth('/api/safety/location-ping', {
      method: 'POST',
      body: JSON.stringify({ latitude, longitude }),
    }),

  getAlerts: (targetUserId, onlyUnacknowledged = false) => {
    const params = new URLSearchParams();
    if (targetUserId) params.set('target_user_id', targetUserId);
    if (onlyUnacknowledged) params.set('only_unacknowledged', 'true');
    const query = params.toString();
    return fetchWithAuth(`/api/safety/alerts${query ? `?${query}` : ''}`);
  },

  acknowledgeAlert: (alertId, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/alerts/${alertId}/ack${suffix}`, {
      method: 'PATCH',
    });
  },

  getEmergencyContacts: (targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/emergency-contacts${suffix}`);
  },

  createEmergencyContact: (data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/emergency-contacts${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  updateEmergencyContact: (contactId, data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/emergency-contacts/${contactId}${suffix}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  deleteEmergencyContact: (contactId, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/emergency-contacts/${contactId}${suffix}`, {
      method: 'DELETE',
    });
  },

  triggerSOS: (data = {}, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/sos${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  shareLocation: (latitude, longitude, reason = 'manual_share', targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/share-location${suffix}`, {
      method: 'POST',
      body: JSON.stringify({ latitude, longitude, reason }),
    });
  },

  getFallEvents: (targetUserId, limit = 20) => {
    const params = new URLSearchParams();
    params.set('limit', String(limit));
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/safety/fall-events?${params.toString()}`);
  },

  reportFallEvent: (data = {}, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/fall-events${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  getEscalationRules: (targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/escalation-rules${suffix}`);
  },

  upsertEscalationRule: (data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/escalation-rules${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  updateEscalationRule: (ruleId, data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/safety/escalation-rules/${ruleId}${suffix}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  runEscalations: (targetUserId, maxAlerts = 20) => {
    const params = new URLSearchParams();
    params.set('max_alerts', String(maxAlerts));
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/safety/escalations/run?${params.toString()}`, {
      method: 'POST',
    });
  },

  getEscalationHistory: (targetUserId, limit = 50) => {
    const params = new URLSearchParams();
    params.set('limit', String(limit));
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/safety/escalations/history?${params.toString()}`);
  },
};

// Medications API
export const medicationsApi = {
  getAll: (targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/medications${suffix}`);
  },

  create: (data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/medications${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  update: (id, data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/medications/${id}${suffix}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  delete: (id, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/medications/${id}${suffix}`, {
      method: 'DELETE',
    });
  },

  logIntake: (id, status = 'taken', targetUserId, options = {}) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/medications/${id}/intake${suffix}`, {
      method: 'POST',
      body: JSON.stringify({ status, ...options }),
    });
  },

  getMAR: (onDate = null, targetUserId, days = 1) => {
    const params = new URLSearchParams();
    params.set('days', String(days));
    if (onDate) params.set('on_date', onDate);
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/medications/mar?${params.toString()}`);
  },

  getAdherence: (days = 7, targetUserId) => {
    const params = new URLSearchParams();
    params.set('days', String(days));
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/medications/adherence?${params.toString()}`);
  },

  getMissed: (hoursOverdue = 2, targetUserId) => {
    const params = new URLSearchParams();
    params.set('hours_overdue', String(hoursOverdue));
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/medications/missed?${params.toString()}`);
  },

  getInteractions: (targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/medications/interactions${suffix}`);
  },
};

// Mood + BPSD API
export const moodApi = {
  getTaxonomy: () => fetchWithAuth('/api/bpsd/taxonomy'),

  getCheckins: (days = 30, targetUserId, limit = 100) => {
    const params = new URLSearchParams();
    params.set('days', String(days));
    params.set('limit', String(limit));
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/mood/checkins?${params.toString()}`);
  },

  createCheckin: (data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/mood/checkins${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  getObservations: (days = 30, targetUserId, limit = 200, symptom = null) => {
    const params = new URLSearchParams();
    params.set('days', String(days));
    params.set('limit', String(limit));
    if (targetUserId) params.set('target_user_id', targetUserId);
    if (symptom) params.set('symptom', symptom);
    return fetchWithAuth(`/api/bpsd/observations?${params.toString()}`);
  },

  createObservation: (data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/bpsd/observations${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  getAnalytics: (days = 30, targetUserId) => {
    const params = new URLSearchParams();
    params.set('days', String(days));
    if (targetUserId) params.set('target_user_id', targetUserId);
    return fetchWithAuth(`/api/bpsd/analytics?${params.toString()}`);
  },
};

// Caregiver portal API
export const caregiverApi = {
  createInvite: (data) => fetchWithAuth('/api/care/invites', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  acceptInvite: (code) => fetchWithAuth('/api/care/invites/accept', {
    method: 'POST',
    body: JSON.stringify({ code }),
  }),

  getLinks: () => fetchWithAuth('/api/care/links'),

  getPatientDashboard: (patientId) => fetchWithAuth(`/api/care/patients/${patientId}/dashboard`),

  createPatientReminder: (patientId, data) => fetchWithAuth(`/api/care/patients/${patientId}/reminders`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  createPatientFamilyMember: (patientId, data) => fetchWithAuth(`/api/care/patients/${patientId}/family`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  updatePatientFamilyMember: (patientId, memberId, data) => fetchWithAuth(`/api/care/patients/${patientId}/family/${memberId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),

  createReadOnlyShare: (patientId, expiresInDays = 14) => fetchWithAuth(`/api/care/patients/${patientId}/share-readonly`, {
    method: 'POST',
    body: JSON.stringify({ expires_in_days: expiresInDays }),
  }),
};

// External doctor bot API (Telegram/WhatsApp link + query)
export const externalBotApi = {
  getPatients: () => fetchWithAuth('/api/external-bot/patients'),

  createLinkCode: (data) => fetchWithAuth('/api/external-bot/link-codes', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  getLinks: () => fetchWithAuth('/api/external-bot/links'),

  updateLinkPatient: (linkId, patientUserId = null) => fetchWithAuth(`/api/external-bot/links/${encodeURIComponent(linkId)}/patient`, {
    method: 'PUT',
    body: JSON.stringify({ patient_user_id: patientUserId }),
  }),

  revokeLink: (linkId) => fetchWithAuth(`/api/external-bot/links/${encodeURIComponent(linkId)}`, {
    method: 'DELETE',
  }),

  query: (data) => fetchWithAuth('/api/external-bot/query', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
};

// Admin governance API
export const adminApi = {
  getPendingClinicians: () => fetchWithAuth('/api/admin/clinicians/pending'),

  getClinicians: (status = 'all') => fetchWithAuth(`/api/admin/clinicians?status=${encodeURIComponent(status)}`),

  approveClinician: (userId, notes = '') => fetchWithAuth(`/api/admin/clinicians/${userId}/approve`, {
    method: 'POST',
    body: JSON.stringify({ notes }),
  }),

  rejectClinician: (userId, notes = '') => fetchWithAuth(`/api/admin/clinicians/${userId}/reject`, {
    method: 'POST',
    body: JSON.stringify({ notes }),
  }),

  suspendClinician: (userId, notes = '') => fetchWithAuth(`/api/admin/clinicians/${userId}/suspend`, {
    method: 'POST',
    body: JSON.stringify({ notes }),
  }),

  reactivateClinician: (userId, notes = '') => fetchWithAuth(`/api/admin/clinicians/${userId}/reactivate`, {
    method: 'POST',
    body: JSON.stringify({ notes }),
  }),

  getAudit: (limit = 200) => fetchWithAuth(`/api/admin/audit?limit=${encodeURIComponent(String(limit))}`),
};

// Care instructions API (doctor/caregiver procedures for agentic RAG)
export const careInstructionsApi = {
  getAll: (
    targetUserId,
    onlyActive = true,
    frequency = null,
    { policyType = null, effectiveOn = null, includeDrafts = false } = {}
  ) => {
    const params = new URLSearchParams();
    if (targetUserId) params.set('target_user_id', targetUserId);
    if (onlyActive) params.set('only_active', 'true');
    if (frequency) params.set('frequency', frequency);
    if (policyType) params.set('policy_type', policyType);
    if (effectiveOn) params.set('effective_on', effectiveOn);
    if (includeDrafts) params.set('include_drafts', 'true');
    const query = params.toString();
    return fetchWithAuth(`/api/care/instructions${query ? `?${query}` : ''}`);
  },

  getTodayPlan: (targetUserId = null, onDate = null) => {
    const params = new URLSearchParams();
    if (targetUserId) params.set('target_user_id', targetUserId);
    if (onDate) params.set('on_date', onDate);
    const query = params.toString();
    return fetchWithAuth(`/api/care/today-plan${query ? `?${query}` : ''}`);
  },

  create: (data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/care/instructions${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  upload: async (
    {
      file,
      title,
      instructionText = '',
      summary = '',
      frequency = 'daily',
      dayOfWeek = '',
      timeOfDay = '',
      policyType = 'general',
      regimenKey = '',
      effectiveStartDate = '',
      effectiveEndDate = '',
      signoffRequired = null,
      tags = [],
    },
    targetUserId = null
  ) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    if (instructionText) formData.append('instruction_text', instructionText);
    if (summary) formData.append('summary', summary);
    if (frequency) formData.append('frequency', frequency);
    if (dayOfWeek) formData.append('day_of_week', dayOfWeek);
    if (timeOfDay) formData.append('time_of_day', timeOfDay);
    if (policyType) formData.append('policy_type', policyType);
    if (regimenKey) formData.append('regimen_key', regimenKey);
    if (effectiveStartDate) formData.append('effective_start_date', effectiveStartDate);
    if (effectiveEndDate) formData.append('effective_end_date', effectiveEndDate);
    if (typeof signoffRequired === 'boolean') formData.append('signoff_required', String(signoffRequired));
    if (tags?.length) formData.append('tags_csv', tags.join(','));
    if (targetUserId) formData.append('target_user_id', targetUserId);

    const response = await fetch(`${BACKEND_URL}/api/care/instructions/upload`, {
      method: 'POST',
      credentials: 'include',
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || 'Request failed');
    }
    return response.json();
  },

  update: (instructionId, data, targetUserId, { createNewVersion = false } = {}) => {
    const params = new URLSearchParams();
    if (targetUserId) params.set('target_user_id', targetUserId);
    if (createNewVersion) params.set('create_new_version', 'true');
    const suffix = params.toString() ? `?${params.toString()}` : '';
    return fetchWithAuth(`/api/care/instructions/${instructionId}${suffix}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  signoff: (instructionId, data, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/care/instructions/${instructionId}/signoff${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  delete: (instructionId, targetUserId) => {
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    return fetchWithAuth(`/api/care/instructions/${instructionId}${suffix}`, {
      method: 'DELETE',
    });
  },
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
  uploadFile: async (file, targetUserId = null) => {
    const formData = new FormData();
    formData.append('file', file);
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    
    const response = await fetch(`${BACKEND_URL}/api/upload${suffix}`, {
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
  
  uploadMultiple: async (files, targetUserId = null) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    const suffix = targetUserId ? `?target_user_id=${encodeURIComponent(targetUserId)}` : '';
    
    const response = await fetch(`${BACKEND_URL}/api/upload/multiple${suffix}`, {
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

// Account management API (GDPR)
export const accountApi = {
  exportData: () => fetchWithAuth('/api/auth/export'),
  deleteAccount: () => fetchWithAuth('/api/auth/account', { method: 'DELETE' }),
};

// Push notification subscription API
export const pushApi = {
  subscribe: (subscription) => fetchWithAuth('/api/push/subscribe', {
    method: 'POST',
    body: JSON.stringify(subscription),
  }),
  unsubscribe: (endpoint) => fetchWithAuth('/api/push/unsubscribe', {
    method: 'DELETE',
    body: JSON.stringify({ endpoint }),
  }),
};

// Billing / subscription API
export const billingApi = {
  getStatus: () => fetchWithAuth('/api/billing/status'),
  createCheckout: () => fetchWithAuth('/api/billing/create-checkout', { method: 'POST' }),
  createPortal: () => fetchWithAuth('/api/billing/create-portal', { method: 'POST' }),
};

export const authApi = {
  startDemo: () => fetchWithAuth('/api/auth/demo', { method: 'POST' }),
};

export const referralApi = {
  generate: () => fetchWithAuth('/api/referral/generate', { method: 'POST' }),
  stats: () => fetchWithAuth('/api/referral/stats'),
};

export const careReportApi = {
  getDailyDigest: (patientId) => fetchWithAuth(`/api/care/patients/${patientId}/daily-digest`),
  downloadReport: (patientId, days = 30) =>
    fetch(`${BACKEND_URL}/api/care/patients/${patientId}/report?days=${days}`, { credentials: 'include' })
      .then(r => r.blob()),
};

export default {
  familyApi,
  memoriesApi,
  remindersApi,
  destinationsApi,
  navigationApi,
  safetyApi,
  medicationsApi,
  moodApi,
  caregiverApi,
  externalBotApi,
  adminApi,
  careInstructionsApi,
  chatApi,
  uploadApi,
  accountApi,
  pushApi,
  billingApi,
  authApi,
  referralApi,
  careReportApi,
};
