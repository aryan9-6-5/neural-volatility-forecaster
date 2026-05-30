/**
 * API Client — thin wrappers for all backend endpoints.
 * 
 * Endpoints:
 *   GET  /health         → model status
 *   POST /predict        → run inference, returns plots + drift
 *   GET  /monitor/drift  → drift metrics
 *   POST /retrain        → trigger re-calibration
 */
var Api = {

  health: function() {
    return fetch('/health').then(function(r) { return r.json(); });
  },

  predict: function(payload) {
    return fetch('/predict?include_plots=true', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function(r) {
      if (!r.ok) {
        return r.json().then(function(e) {
          throw new Error(e.detail || 'HTTP ' + r.status);
        });
      }
      return r.json();
    });
  },

  drift: function() {
    return fetch('/monitor/drift').then(function(r) { return r.json(); });
  },

  retrain: function() {
    return fetch('/retrain', { method: 'POST' }).then(function(r) { return r.json(); });
  }
};
