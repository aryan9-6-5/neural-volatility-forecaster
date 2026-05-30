/**
 * Charts module — Plotly rendering with terminal-dark theme.
 * Caches plot data so charts can re-render on tab switch or resize.
 */
var Charts = (function() {

  var cache = {};

  /**
   * Apply the dark terminal theme to a Plotly layout object.
   */
  function applyTheme(plotObj, containerEl) {
    var layout = {};
    // Deep-copy layout keys from plotObj
    var src = plotObj.layout || {};
    var keys = Object.keys(src);
    for (var i = 0; i < keys.length; i++) {
      layout[keys[i]] = src[keys[i]];
    }

    var rect = containerEl.getBoundingClientRect();
    layout.width  = rect.width  || 800;
    layout.height = rect.height || 480;
    layout.autosize = false;
    layout.paper_bgcolor = 'transparent';
    layout.plot_bgcolor  = 'transparent';
    layout.font = {
      color: '#94a3b8',
      family: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      size: 10
    };

    // Style 2D axes
    var axStyle = { 
      gridcolor: 'rgba(255, 255, 255, 0.04)', 
      linecolor: 'rgba(255, 255, 255, 0.05)', 
      zerolinecolor: 'rgba(255, 255, 255, 0.05)',
      tickfont: {
        family: "'JetBrains Mono', monospace",
        size: 9
      }
    };
    var lk = Object.keys(layout);
    for (var j = 0; j < lk.length; j++) {
      if (/^[xy]axis/.test(lk[j]) && layout[lk[j]] && typeof layout[lk[j]] === 'object') {
        Object.assign(layout[lk[j]], axStyle);
      }
    }

    // Style 3D scene
    if (layout.scene && typeof layout.scene === 'object') {
      layout.scene.bgcolor = 'transparent';
      var sceneProp = { 
        backgroundcolor: 'transparent', 
        gridcolor: 'rgba(255, 255, 255, 0.04)', 
        linecolor: 'rgba(255, 255, 255, 0.05)',
        tickfont: {
          family: "'JetBrains Mono', monospace",
          size: 9
        }
      };
      ['xaxis', 'yaxis', 'zaxis'].forEach(function(ax) {
        layout.scene[ax] = Object.assign(layout.scene[ax] || {}, sceneProp);
      });
    }

    return layout;
  }

  /**
   * Render a plot into a DOM element by ID.
   */
  function render(divId, plotData) {
    var el = document.getElementById(divId);
    if (!el || !plotData) return;

    // Apply Slate-Blue-Cyan palette overrides to Plotly trace data
    if (plotData.data) {
      plotData.data.forEach(function(trace) {
        // 1. Surface Overrides
        if (trace.type === 'surface') {
          trace.colorscale = [
            [0.0, '#0b0d12'],   // Warmer primary dark
            [0.2, '#1e3a8a'],   // Deep slate blue
            [0.6, '#3b82f6'],   // Royal blue
            [1.0, '#22d3ee']    // Cool cyan
          ];
          if (trace.colorbar) {
            trace.colorbar.thickness = 12;
            trace.colorbar.tickfont = { family: "'JetBrains Mono', monospace", size: 9 };
          }
        }
        // 2. Heatmap Overrides
        if (trace.type === 'heatmap') {
          trace.colorscale = [
            [0.0, '#f87171'],   // Muted soft red
            [0.5, '#1e293b'],   // Muted slate gray
            [1.0, '#60a5fa']    // Muted soft blue
          ];
          if (trace.colorbar) {
            trace.colorbar.thickness = 12;
            trace.colorbar.tickfont = { family: "'JetBrains Mono', monospace", size: 9 };
          }
        }
        // 3. Scatter Cross Sections Overrides
        if (trace.type === 'scatter') {
          if (trace.line && trace.line.color) {
            var c = trace.line.color;
            var colorMap = {
              '#FF4136': '#f87171', // Red -> soft coral
              '#2ECC40': '#34d399', // Green -> soft emerald
              '#0074D9': '#60a5fa', // Blue -> soft blue
              '#B10DC9': '#c084fc', // Purple -> soft lavender
              '#FF851B': '#fb923c', // Orange -> soft amber
              '#39CCCC': '#2dd4bf'  // Teal -> soft teal
            };
            if (colorMap[c]) {
              trace.line.color = colorMap[c];
            }
          }
          if (trace.marker && trace.marker.color) {
            var mc = trace.marker.color;
            var markerColorMap = {
              '#FF4136': '#f87171',
              '#2ECC40': '#34d399',
              '#0074D9': '#60a5fa',
              '#B10DC9': '#c084fc',
              '#FF851B': '#fb923c',
              '#39CCCC': '#2dd4bf'
            };
            if (markerColorMap[mc]) {
              trace.marker.color = markerColorMap[mc];
            }
          }
        }
      });
    }

    cache[divId] = plotData;
    var layout = applyTheme(plotData, el);

    Plotly.newPlot(el, plotData.data, layout, { responsive: false, displaylogo: false })
      .then(function() {
        var placeholder = el.parentElement.querySelector('.chart-placeholder');
        if (placeholder) placeholder.style.display = 'none';
      })
      .catch(function(err) {
        console.error('Chart render error [' + divId + ']:', err);
      });
  }

  /**
   * Re-render a cached plot (used on tab switch or window resize).
   */
  function renderCached(divId) {
    if (cache[divId]) {
      render(divId, cache[divId]);
    }
  }

  /**
   * Check if data exists in cache for a given div.
   */
  function hasData(divId) {
    return !!cache[divId];
  }

  return {
    render: render,
    renderCached: renderCached,
    hasData: hasData,
    cache: cache
  };

})();
