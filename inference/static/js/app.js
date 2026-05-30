/**
 * VolForecaster — Main Application Logic
 *
 * Pages (4):
 *   dashboard    → GET /health + GET /monitor/drift
 *   forecast     → POST /predict
 *   drift        → GET /monitor/drift
 *   recalibrate  → POST /retrain
 */
(function() {
  'use strict';

  // ===================================================================
  // SAMPLE CONTRACT CHAIN
  // ===================================================================
  var SAMPLE = [
    {"strike":700,"expiry":"2026-06-30","option_type":"put","bid":4.1,"ask":4.3,"volume":50},
    {"strike":720,"expiry":"2026-06-30","option_type":"put","bid":8.5,"ask":8.8,"volume":60},
    {"strike":740,"expiry":"2026-06-30","option_type":"put","bid":16.2,"ask":16.6,"volume":100},
    {"strike":750,"expiry":"2026-06-30","option_type":"call","bid":18.1,"ask":18.5,"volume":120},
    {"strike":760,"expiry":"2026-06-30","option_type":"call","bid":13.2,"ask":13.6,"volume":80},
    {"strike":780,"expiry":"2026-06-30","option_type":"call","bid":6.1,"ask":6.4,"volume":40},
    {"strike":700,"expiry":"2026-09-30","option_type":"put","bid":12.5,"ask":13.0,"volume":30},
    {"strike":720,"expiry":"2026-09-30","option_type":"put","bid":19.2,"ask":19.8,"volume":45},
    {"strike":740,"expiry":"2026-09-30","option_type":"put","bid":29.5,"ask":30.2,"volume":75},
    {"strike":750,"expiry":"2026-09-30","option_type":"call","bid":38.5,"ask":39.5,"volume":90},
    {"strike":760,"expiry":"2026-09-30","option_type":"call","bid":32.8,"ask":33.6,"volume":65},
    {"strike":780,"expiry":"2026-09-30","option_type":"call","bid":23.5,"ask":24.2,"volume":35}
  ];

  // ===================================================================
  // STATE
  // ===================================================================
  var lastInferenceTime = null;
  var inferCount = 0;
  var eventLog = [];
  var activePage = 'dashboard';

  // ===================================================================
  // HELPERS
  // ===================================================================
  function $(id) { return document.getElementById(id); }

  function setText(id, text) {
    var e = $(id);
    if (e) e.textContent = text;
  }

  function setColor(id, color) {
    var e = $(id);
    if (e) e.style.color = color;
  }

  function setWidth(id, pct) {
    var e = $(id);
    if (e) e.style.width = Math.min(100, Math.max(0, pct)) + '%';
  }

  function setClass(id, cls) {
    var e = $(id);
    if (e) e.className = cls;
  }

  // ===================================================================
  // EVENT LOG
  // ===================================================================
  function logEvent(level, msg) {
    var now = new Date();
    var time = now.toTimeString().split(' ')[0];
    eventLog.unshift({ time: time, level: level, msg: msg });
    if (eventLog.length > 50) eventLog.pop();
    renderLogTo('dashboard-log');
    renderLogTo('recal-log');
  }

  function renderLogTo(containerId) {
    var container = $(containerId);
    if (!container) return;
    container.innerHTML = '';
    if (eventLog.length === 0) {
      container.innerHTML = '<div class="log-stream-empty">No events yet</div>';
      return;
    }
    var items = eventLog.slice(0, 25);
    for (var i = 0; i < items.length; i++) {
      var entry = items[i];
      var line = document.createElement('div');
      line.className = 'log-line';
      line.innerHTML =
        '<span class="log-time">' + entry.time + '</span>' +
        '<span class="log-tag ' + entry.level + '">[' + entry.level.toUpperCase() + ']</span>' +
        '<span class="log-msg">' + entry.msg + '</span>';
      container.appendChild(line);
    }
  }

  // ===================================================================
  // NAVIGATION
  // ===================================================================
  var PAGES = ['dashboard', 'forecast', 'drift', 'recalibrate'];
  var PAGE_TITLES = {
    dashboard:   'Dashboard',
    forecast:    'Forecast',
    drift:       'Drift Monitor',
    recalibrate: 'Re-calibrate'
  };
  var PAGE_SUBS = {
    dashboard:   'System overview',
    forecast:    '7x7 surface grid · arbitrage filter',
    drift:       'Rolling-window drift analysis',
    recalibrate: 'Model re-training pipeline'
  };

  function showPage(pageId) {
    if (PAGES.indexOf(pageId) === -1) return;
    activePage = pageId;

    for (var i = 0; i < PAGES.length; i++) {
      var p = PAGES[i];
      var pageEl = $('page-' + p);
      if (pageEl) pageEl.style.display = (p === pageId) ? 'flex' : 'none';
    }

    document.querySelectorAll('.nav-item[data-page]').forEach(function(item) {
      if (item.dataset.page === pageId) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    setText('topbar-title', PAGE_TITLES[pageId] || '');
    updateSubtitle();

    // Refresh data on page visit
    if (pageId === 'drift') fetchDrift();
    if (pageId === 'dashboard') {
      if (Charts.hasData('plot-dash-surface')) {
        setTimeout(function() { Charts.renderCached('plot-dash-surface'); }, 100);
      }
    }
    if (pageId === 'forecast') {
      var activeTab = document.querySelector('#forecast-tabs .tab-bar button.active');
      if (activeTab) {
        var tabId = activeTab.dataset.tab;
        var plotMap = { 'tab-3d': 'plot-3d', 'tab-heat': 'plot-heat', 'tab-cross': 'plot-cross' };
        if (plotMap[tabId]) {
          setTimeout(function() { Charts.renderCached(plotMap[tabId]); }, 100);
        }
      }
    }
  }

  function updateSubtitle() {
    var sub = PAGE_SUBS[activePage] || '';
    if (lastInferenceTime) {
      var diff = Math.round((new Date() - lastInferenceTime) / 1000);
      var ago = diff < 60 ? diff + 's ago' : Math.round(diff / 60) + 'm ago';
      sub += ' · last inference: ' + ago;
    }
    if (inferCount > 0) {
      sub += ' · ' + inferCount + ' request' + (inferCount !== 1 ? 's' : '');
    }
    setText('topbar-sub', sub);
  }

  // ===================================================================
  // DRIFT BAR UPDATER (shared between dashboard & drift page)
  // ===================================================================
  function updateDriftBars(prefix, kl, klThr, rmse, rmseThr) {
    var klPct = klThr > 0 ? Math.min(100, (kl / klThr) * 100) : 0;
    var rmsePct = rmseThr > 0 ? Math.min(100, (rmse / rmseThr) * 100) : 0;
    var klStatus  = kl > klThr ? 'danger' : 'success';
    var rmseStatus = rmse > rmseThr ? 'warning' : 'success';

    // KL row
    setWidth(prefix + '-bar-kl', klPct);
    setClass(prefix + '-bar-kl', 'drift-bar-fill ' + klStatus);
    setClass(prefix + '-dot-kl', 'indicator-dot ' + klStatus);
    setText(prefix + '-val-kl', kl.toFixed(4));
    setText(prefix + '-badge-kl', kl > klThr ? 'CRIT' : 'OK');
    setClass(prefix + '-badge-kl', 'drift-badge-text ' + klStatus);

    // RMSE row
    setWidth(prefix + '-bar-rmse', rmsePct);
    setClass(prefix + '-bar-rmse', 'drift-bar-fill ' + rmseStatus);
    setClass(prefix + '-dot-rmse', 'indicator-dot ' + rmseStatus);
    setText(prefix + '-val-rmse', rmse.toFixed(4));
    setText(prefix + '-badge-rmse', rmse > rmseThr ? 'WARN' : 'OK');
    setClass(prefix + '-badge-rmse', 'drift-badge-text ' + rmseStatus);
  }

  // ===================================================================
  // DASHBOARD METRICS UPDATE
  // ===================================================================
  function updateDashboard(drift) {
    var kl = drift.rolling_kl_divergence || 0;
    var klThr = drift.kl_threshold || 0.5;
    var rmse = drift.rolling_rmse || 0;
    var baseRmse = drift.baseline_rmse || 0.02;
    var rmseFactor = drift.rmse_threshold_factor || 1.5;
    var rmseThr = baseRmse * rmseFactor;

    // Metric cards
    setText('card-kl', kl.toFixed(4));
    setText('card-kl-sub', 'threshold: ' + klThr.toFixed(2));
    setColor('card-kl', kl > klThr ? 'var(--danger)' : 'var(--success)');

    setText('card-rmse', rmse.toFixed(4));
    setText('card-rmse-sub', 'limit: ' + rmseThr.toFixed(4));

    // Update Hero KPI Badge
    var klBadge = $('card-kl-badge');
    if (klBadge) {
      if (kl > klThr) {
        klBadge.textContent = 'CRIT';
        klBadge.className = 'hero-kpi-badge crit';
      } else if (rmse > rmseThr) {
        klBadge.textContent = 'WARN';
        klBadge.className = 'hero-kpi-badge warn';
      } else {
        klBadge.textContent = 'OK';
        klBadge.className = 'hero-kpi-badge';
      }
    }

    // Dashboard drift bars
    updateDriftBars('dash', kl, klThr, rmse, rmseThr);

    // Alert banner
    var banner = $('drift-alert-banner');
    if (banner) {
      if (kl > klThr || rmse > rmseThr) {
        banner.classList.add('visible');
        setText('drift-alert-text',
          'Drift detected — KL: ' + kl.toFixed(4) + ' (threshold: ' + klThr.toFixed(2) +
          '), RMSE: ' + rmse.toFixed(4) + ' (limit: ' + rmseThr.toFixed(4) + ')');
      } else {
        banner.classList.remove('visible');
      }
    }

    // Sidebar badges
    var driftBadge = $('drift-badge');
    var recalBadge = $('recal-badge');
    if (kl > klThr || rmse > rmseThr) {
      if (driftBadge) { driftBadge.textContent = 'alert'; driftBadge.className = 'badge danger'; }
      if (recalBadge) { recalBadge.textContent = 'due'; recalBadge.className = 'badge warning'; }
    } else {
      if (driftBadge) driftBadge.className = 'badge';
      if (recalBadge) recalBadge.className = 'badge';
    }
  }

  // ===================================================================
  // DRIFT MONITOR PAGE UPDATE
  // ===================================================================
  function updateDriftPage(data) {
    var kl = data.rolling_kl_divergence || 0;
    var klThr = data.kl_threshold || 0.5;
    var rmse = data.rolling_rmse || 0;
    var baseRmse = data.baseline_rmse || 0.02;
    var rmseFactor = data.rmse_threshold_factor || 1.5;
    var rmseThr = baseRmse * rmseFactor;

    // Drift page bars
    updateDriftBars('drift', kl, klThr, rmse, rmseThr);

    // Thresholds table
    setText('thr-kl', klThr.toFixed(2));
    setText('thr-base-rmse', baseRmse.toFixed(4));
    setText('thr-rmse-factor', rmseFactor + 'x');

    var retrain = data.retrain_recommended || false;
    var retrainEl = $('thr-retrain');
    if (retrainEl) {
      retrainEl.textContent = retrain ? 'Yes' : 'No';
      retrainEl.style.color = retrain ? 'var(--danger)' : 'var(--success)';
    }

    // Rolling history lists
    renderHistoryList('kl-history', data.recent_kls || [], klThr, 'danger');
    renderHistoryList('rmse-history', data.recent_rmses || [], rmseThr, 'warning');
  }

  function renderHistoryList(containerId, values, threshold, overClass) {
    var container = $(containerId);
    if (!container) return;
    container.innerHTML = '';
    if (values.length === 0) {
      container.innerHTML = '<div class="log-stream-empty">No data yet — run an inference first</div>';
      return;
    }
    for (var i = 0; i < values.length; i++) {
      var item = document.createElement('div');
      item.className = 'history-item';
      var color = values[i] > threshold ? 'var(--' + overClass + ')' : 'var(--success)';
      item.innerHTML =
        '<span>Entry ' + (i + 1) + '</span>' +
        '<span style="color:' + color + '">' + values[i].toFixed(6) + '</span>';
      container.appendChild(item);
    }
  }

  // ===================================================================
  // API: FETCH HEALTH
  // ===================================================================
  function fetchHealth() {
    Api.health().then(function(data) {
      var model = data.model_type || 'Unknown';
      var status = data.status || 'unknown';

      setText('card-model', model);
      setText('card-model-sub', status === 'healthy' ? 'API healthy' : 'API: ' + status);
      setText('footer-status', model + ' · online');

      var dot = $('status-dot');
      if (dot) dot.className = 'status-dot ' + (status === 'healthy' ? 'online' : 'offline');
    }).catch(function() {
      setText('footer-status', 'offline');
      setText('card-model', '--');
      setText('card-model-sub', 'API unreachable');
      var dot = $('status-dot');
      if (dot) dot.className = 'status-dot offline';
    });
  }

  // ===================================================================
  // API: FETCH DRIFT
  // ===================================================================
  function fetchDrift() {
    Api.drift().then(function(data) {
      updateDashboard(data);
      updateDriftPage(data);
    }).catch(function() {
      // silently ignore — dashboard shows last known values
    });
  }

  // ===================================================================
  // FORECAST: RUN INFERENCE
  // ===================================================================
  function runInference() {
    var btn = $('predict-btn');
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = 'Running...';

    // Parse contracts
    var contractsField = $('contracts_json');
    var contracts;
    try {
      contracts = JSON.parse(contractsField.value);
    } catch(e) {
      alert('Invalid JSON in options chain field.');
      btn.disabled = false;
      btn.textContent = 'Run Inference';
      return;
    }

    var payload = {
      timestamp: new Date().toISOString(),
      spot_price:     parseFloat($('spot_price').value),
      risk_free_rate: parseFloat($('risk_free_rate').value),
      dividend_yield: parseFloat($('dividend_yield').value),
      contracts: contracts
    };

    // Show loading state in chart placeholders
    ['status-3d', 'status-heat', 'status-cross', 'status-dash-surface'].forEach(function(id) {
      var s = $(id);
      if (s) { s.style.display = 'block'; s.innerHTML = '<div class="spinner"></div>Computing...'; }
    });

    var startTime = performance.now();

    Api.predict(payload).then(function(data) {
      var latency = Math.round(performance.now() - startTime);

      // Update latency card
      setText('card-latency', latency + 'ms');

      inferCount++;
      lastInferenceTime = new Date();
      updateSubtitle();
      logEvent('info', 'Inference complete, latency=' + latency + 'ms');

      // Parse and render plots
      var PLOT_MAP = {
        'surface_3d':        'plot-3d',
        'evolution_heatmap': 'plot-heat',
        'cross_sections':    'plot-cross'
      };
      var plotKeys = Object.keys(PLOT_MAP);
      for (var i = 0; i < plotKeys.length; i++) {
        var serverKey = plotKeys[i];
        var divId = PLOT_MAP[serverKey];
        if (data.plots && data.plots[serverKey]) {
          try {
            var plotData = JSON.parse(data.plots[serverKey]);
            Charts.render(divId, plotData);
            // Also render 3D surface on dashboard!
            if (serverKey === 'surface_3d') {
              Charts.render('plot-dash-surface', plotData);
            }
          } catch(e) {
            console.error('Plot parse error for ' + serverKey + ':', e);
          }
        }
      }

      // Refresh drift after inference
      fetchDrift();

    }).catch(function(err) {
      alert('Inference error: ' + err.message);
      logEvent('crit', 'Inference failed: ' + err.message);
      // Reset placeholders
      ['status-3d', 'status-heat', 'status-cross'].forEach(function(id) {
        var s = $(id);
        if (s) { s.style.display = 'block'; s.textContent = 'Error — try again'; }
      });
    }).finally(function() {
      btn.disabled = false;
      btn.textContent = 'Run Inference';
    });
  }

  // ===================================================================
  // RECALIBRATE: TRIGGER PIPELINE
  // ===================================================================
  function startRecalibration() {
    var btn = $('trigger-recal-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }

    logEvent('warn', 'Re-calibration triggered.');

    // Reset timeline UI
    setClass('step-collect',    'timeline-item active');
    setClass('step-preprocess', 'timeline-item');
    setClass('step-train',      'timeline-item');
    setClass('step-swap',       'timeline-item');
    setText('collect-status', 'Running...');
    setText('preprocess-status', 'Awaiting...');
    setText('train-status', 'Awaiting...');
    setText('swap-status', 'Awaiting...');
    
    setText('progress-label', 'Collecting snapshots...');
    setText('progress-pct', '10%');
    setWidth('progress-fill', 10);
    setText('recal-epoch', 'Starting...');
    setText('recal-eta', 'ETA: ~5s');

    Api.retrain().then(function() {
      // Step 2: Preprocess
      setTimeout(function() {
        setClass('step-collect',    'timeline-item complete');
        setClass('step-preprocess', 'timeline-item active');
        setText('collect-status', 'Complete');
        setText('preprocess-status', 'Running...');
        setText('progress-label', 'Preprocessing data...');
        setText('progress-pct', '35%');
        setWidth('progress-fill', 35);
        setText('recal-eta', 'ETA: ~4s');
        logEvent('info', 'Snapshot collection complete, preprocessing...');
      }, 1200);

      // Step 3: Train
      setTimeout(function() {
        setClass('step-preprocess', 'timeline-item complete');
        setClass('step-train',      'timeline-item active');
        setText('preprocess-status', 'Complete');
        setText('train-status', 'Running...');
        setText('progress-label', 'Training model...');
        setText('progress-pct', '70%');
        setWidth('progress-fill', 70);
        setText('recal-epoch', 'epoch 45/100');
        setText('recal-eta', 'ETA: ~2s');
        logEvent('info', 'Training in progress...');
      }, 2600);

      // Step 4: Hot-swap
      setTimeout(function() {
        setClass('step-train', 'timeline-item complete');
        setClass('step-swap',  'timeline-item active');
        setText('train-status', 'Complete');
        setText('swap-status', 'Running...');
        setText('progress-label', 'Hot-swapping checkpoint...');
        setText('progress-pct', '95%');
        setWidth('progress-fill', 95);
        setText('recal-epoch', 'epoch 100/100');
        setText('recal-eta', 'ETA: ~1s');
      }, 4200);

      // Complete
      setTimeout(function() {
        setClass('step-swap', 'timeline-item complete');
        setText('swap-status', 'Complete');
        setText('progress-label', 'Complete');
        setText('progress-pct', '100%');
        setWidth('progress-fill', 100);
        setText('recal-epoch', 'Done');
        setText('recal-eta', '');
        logEvent('info', 'Re-calibration complete. New checkpoint loaded.');

        fetchHealth();
        fetchDrift();

        if (btn) { btn.disabled = false; btn.textContent = 'Start Re-calibration'; }
      }, 5500);

    }).catch(function(err) {
      setText('progress-label', 'Failed');
      setText('progress-pct', 'Error');
      setWidth('progress-fill', 0);
      logEvent('crit', 'Re-calibration failed: ' + (err.message || 'Unknown error'));
      if (btn) { btn.disabled = false; btn.textContent = 'Start Re-calibration'; }
    });
  }

  // ===================================================================
  // TABS
  // ===================================================================
  function initTabs() {
    document.querySelectorAll('.tab-bar').forEach(function(tabBar) {
      tabBar.querySelectorAll('button[data-tab]').forEach(function(btn) {
        btn.addEventListener('click', function() {
          var tab = this.dataset.tab;

          // Update active button
          tabBar.querySelectorAll('button').forEach(function(b) { b.classList.remove('active'); });
          this.classList.add('active');

          // Update active page
          var content = tabBar.nextElementSibling;
          if (content) {
            content.querySelectorAll('.tab-page').forEach(function(p) { p.classList.remove('active'); });
            var page = document.getElementById(tab);
            if (page) page.classList.add('active');
          }

          // Re-render cached chart
          var plotMap = { 'tab-3d': 'plot-3d', 'tab-heat': 'plot-heat', 'tab-cross': 'plot-cross' };
          if (plotMap[tab]) {
            setTimeout(function() { Charts.renderCached(plotMap[tab]); }, 50);
          }
        });
      });
    });
  }

  // ===================================================================
  // INITIALIZATION
  // ===================================================================
  function init() {
    // 1. Navigation binding
    document.querySelectorAll('.nav-item[data-page]').forEach(function(item) {
      item.addEventListener('click', function() {
        showPage(this.dataset.page);
      });
    });

    // 2. Tab binding
    initTabs();

    // 3. Forecast form
    var predictBtn = $('predict-btn');
    if (predictBtn) predictBtn.addEventListener('click', runInference);

    var loadSampleBtn = $('load-sample-btn');
    if (loadSampleBtn) {
      loadSampleBtn.addEventListener('click', function() {
        $('contracts_json').value = JSON.stringify(SAMPLE, null, 2);
        logEvent('info', 'Sample contract data loaded.');
      });
    }

    // Pre-fill sample data
    var contractsField = $('contracts_json');
    if (contractsField) contractsField.value = JSON.stringify(SAMPLE, null, 2);

    // 4. Recalibrate
    var recalBtn = $('trigger-recal-btn');
    if (recalBtn) recalBtn.addEventListener('click', startRecalibration);

    var bannerRecalBtn = $('banner-recal-btn');
    if (bannerRecalBtn) {
      bannerRecalBtn.addEventListener('click', function() {
        showPage('recalibrate');
        startRecalibration();
      });
    }

    // 5. Drift page refresh
    var refreshDriftBtn = $('refresh-drift-btn');
    if (refreshDriftBtn) refreshDriftBtn.addEventListener('click', fetchDrift);

    // 6. Window resize — re-render visible chart (debounced for performance)
    var resizeTimeout;
    window.addEventListener('resize', function() {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(function() {
        // Active forecast tab
        var activeTab = document.querySelector('#forecast-tabs .tab-bar button.active');
        if (activeTab) {
          var tabId = activeTab.dataset.tab;
          var plotMap = { 'tab-3d': 'plot-3d', 'tab-heat': 'plot-heat', 'tab-cross': 'plot-cross' };
          if (plotMap[tabId]) Charts.renderCached(plotMap[tabId]);
        }
        // Active dashboard surface chart
        if (activePage === 'dashboard') {
          Charts.renderCached('plot-dash-surface');
        }
      }, 150);
    });

    // 7. Subtitle timer
    setInterval(updateSubtitle, 1000);

    // 8. Initial data load
    fetchHealth();
    fetchDrift();
    logEvent('info', 'Dashboard initialized.');

    // 9. Periodic refresh (every 60s)
    setInterval(fetchHealth, 60000);
    setInterval(fetchDrift, 60000);
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
