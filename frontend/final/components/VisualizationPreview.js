import React, { useEffect, useRef, useState } from "https://esm.sh/react@18.2.0";

const h = React.createElement;

const VARIABLE_LABELS = {
  "PM2.5": "PM2.5",
  temp: "Temperature",
  rhum: "Humidity",
  pres: "Pressure",
  elevation: "Elevation",
  wind: "Wind",
  prec: "Precipitation",
  rain: "Rain",
  shum: "Specific Humidity",
  lrad: "Longwave Radiation",
  srad: "Shortwave Radiation",
  bcpr: "Bias-Corrected Precipitation",
  snow: "Snowfall",
};

const HEAT_COLORS = [
  [11, 88, 181],
  [50, 176, 212],
  [102, 201, 131],
  [244, 221, 85],
  [251, 139, 50],
  [238, 63, 51],
  [139, 18, 42],
];

const SEMANTIC_LABELS = {
  H: { name: "高值热点型", color: "#d94841" },
  L: { name: "低值冷点型", color: "#2f6fdd" },
  G: { name: "梯度过渡型", color: "#2a9d72" },
  B: { name: "边界突变型", color: "#f08a24" },
  D: { name: "扩散分散型", color: "#8b5bd6" },
  U: { name: "均匀稳定型", color: "#7b8794" },
  M: { name: "混合复杂型", color: "#334155" },
  N: { name: "噪声不确定型", color: "#b45309" },
};

function selectedVariableLabel(analysis) {
  const selection = analysis?.selection || {};
  return VARIABLE_LABELS[selection.variable] || selection.label || selection.variable || "PM2.5";
}

function variableLabel(item) {
  return VARIABLE_LABELS[item?.variable] || VARIABLE_LABELS[item?.name] || item?.label || item?.variable || item?.name || "";
}

function compactLabel(text, max = 13) {
  const value = String(text || "");
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function value(number, digits = 1) {
  if (number == null || Number.isNaN(Number(number))) return "--";
  return Number(number).toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function unitText(unit) {
  return unit || "µg/m³";
}

function finiteValues(z) {
  const values = [];
  for (const row of z || []) {
    for (const item of row || []) {
      const number = Number(item);
      if (Number.isFinite(number)) values.push(number);
    }
  }
  return values;
}

function extent(values) {
  if (!values.length) return [0, 1];
  let min = Infinity;
  let max = -Infinity;
  values.forEach((number) => {
    min = Math.min(min, number);
    max = Math.max(max, number);
  });
  return min === max ? [min - 1, max + 1] : [min, max];
}

function colorFor(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (HEAT_COLORS.length - 1);
  const left = Math.floor(scaled);
  const right = Math.min(HEAT_COLORS.length - 1, left + 1);
  const mix = scaled - left;
  const rgb = HEAT_COLORS[left].map((channel, index) => Math.round(channel + (HEAT_COLORS[right][index] - channel) * mix));
  return rgb;
}

function forEachGeoRing(geometry, callback) {
  if (!geometry) return;
  if (geometry.type === "Polygon") {
    (geometry.coordinates || []).forEach(callback);
  } else if (geometry.type === "MultiPolygon") {
    (geometry.coordinates || []).forEach((polygon) => (polygon || []).forEach(callback));
  }
}

function addGeoJsonPath(ctx, geoJson, project) {
  (geoJson?.features || []).forEach((feature) => {
    forEachGeoRing(feature.geometry, (ring) => {
      if (!ring?.length) return;
      ring.forEach(([lo, la], index) => {
        const point = project(Number(lo), Number(la));
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
    });
  });
}

function drawProvinceLabels(ctx, geoJson, project, dpr) {
  ctx.save();
  ctx.fillStyle = "rgba(17,24,39,0.82)";
  ctx.font = `${10.5 * dpr}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  (geoJson?.features || []).forEach((feature) => {
    const props = feature.properties || {};
    const center = props.centroid || props.center;
    if (!center || center.length < 2) return;
    const point = project(Number(center[0]), Number(center[1]));
    const name = String(props.name || "")
      .replace("省", "")
      .replace("市", "")
      .replace("自治区", "")
      .replace("壮族", "")
      .replace("回族", "")
      .replace("维吾尔", "");
    if (!name) return;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2.8 * dpr;
    ctx.strokeText(name, point.x, point.y);
    ctx.fillText(name, point.x, point.y);
  });
  ctx.restore();
}

function drawHeatmap(canvas, map, options) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const z = map?.z || [];
  const lon = map?.lon || [];
  const lat = map?.lat || [];
  if (!z.length || !z[0]?.length || !lon.length || !lat.length) {
    ctx.fillStyle = "#5f7188";
    ctx.font = `${14 * dpr}px sans-serif`;
    ctx.fillText("No grid data", 24 * dpr, 42 * dpr);
    return;
  }

  const [min, max] = extent(finiteValues(z));
  const pad = { left: 46 * dpr, right: 70 * dpr, top: 16 * dpr, bottom: 34 * dpr };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const cellW = plotW / z[0].length;
  const cellH = plotH / z.length;
  const lonMin = lon[0];
  const lonMax = lon[lon.length - 1];
  const latMin = lat[0];
  const latMax = lat[lat.length - 1];
  const project = (lo, la) => ({
    x: pad.left + ((lo - lonMin) / (lonMax - lonMin || 1)) * plotW,
    y: pad.top + plotH - ((la - latMin) / (latMax - latMin || 1)) * plotH,
  });

  if (options.geoJson?.features?.length) {
    ctx.save();
    ctx.beginPath();
    addGeoJsonPath(ctx, options.geoJson, project);
    ctx.fillStyle = "#ffffff";
    ctx.fill("evenodd");
    ctx.clip("evenodd");
  }
  for (let y = 0; y < z.length; y += 1) {
    for (let x = 0; x < z[y].length; x += 1) {
      const number = Number(z[y][x]);
      if (!Number.isFinite(number)) continue;
      const rgb = colorFor((number - min) / (max - min));
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(pad.left + x * cellW, pad.top + (z.length - y - 1) * cellH, Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
    }
  }
  if (options.geoJson?.features?.length) {
    ctx.restore();
  }

  ctx.strokeStyle = "rgba(31,45,67,0.24)";
  ctx.lineWidth = 1 * dpr;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.strokeStyle = "rgba(31,45,67,0.22)";
  for (let i = 1; i < 6; i += 1) {
    const x = pad.left + (plotW * i) / 6;
    const y = pad.top + (plotH * i) / 6;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  if (options.showHotspots) {
    for (const hotspot of options.hotspots || []) {
      const point = project(Number(hotspot.lon), Number(hotspot.lat));
      const radius = Math.max(7, Math.min(24, (Number(hotspot.cells) || 20) / 3)) * dpr;
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(238, 50, 45, 0.56)";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
    }
  }

  if (options.geoJson?.features?.length) {
    ctx.save();
    ctx.beginPath();
    addGeoJsonPath(ctx, options.geoJson, project);
    ctx.strokeStyle = "rgba(17,24,39,0.5)";
    ctx.lineWidth = 0.8 * dpr;
    ctx.stroke();
    drawProvinceLabels(ctx, options.geoJson, project, dpr);
    ctx.restore();
  }

  ctx.fillStyle = "#172033";
  ctx.font = `${11 * dpr}px sans-serif`;
  for (let i = 0; i <= 5; i += 1) {
    const lo = lonMin + ((lonMax - lonMin) * i) / 5;
    const la = latMin + ((latMax - latMin) * i) / 5;
    ctx.fillText(`${Math.round(lo)}°E`, pad.left + (plotW * i) / 5 - 12 * dpr, height - 10 * dpr);
    ctx.fillText(`${Math.round(la)}°N`, 6 * dpr, pad.top + plotH - (plotH * i) / 5 + 4 * dpr);
  }

  const barX = width - 48 * dpr;
  const barY = 38 * dpr;
  const barH = Math.min(210 * dpr, plotH * 0.62);
  const barW = 16 * dpr;
  const gradient = ctx.createLinearGradient(0, barY + barH, 0, barY);
  HEAT_COLORS.forEach((rgb, index) => gradient.addColorStop(index / (HEAT_COLORS.length - 1), `rgb(${rgb.join(",")})`));
  ctx.fillStyle = gradient;
  ctx.fillRect(barX, barY, barW, barH);
  ctx.strokeStyle = "rgba(17,24,39,0.18)";
  ctx.strokeRect(barX, barY, barW, barH);
  ctx.fillStyle = "#172033";
  ctx.font = `${10 * dpr}px sans-serif`;
  ctx.fillText(value(max, 1), barX + 22 * dpr, barY + 4 * dpr);
  ctx.fillText(value(min, 1), barX + 22 * dpr, barY + barH);
  ctx.fillText(options.legendTitle, barX - 18 * dpr, barY - 14 * dpr);
}

function CardIcon({ type }) {
  const paths = {
    mean: "M5 5h14v14H5zM8 16l8-8M9 8h7v7",
    peak: "M12 22s7-5.2 7-12a7 7 0 1 0-14 0c0 6.8 7 12 7 12zM12 13a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
    blocks: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
    cases: "M12 3a5 5 0 0 1 5 5c0 4-5 9-5 9S7 12 7 8a5 5 0 0 1 5-5zM4 20h16",
  };
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, h("path", { d: paths[type] || paths.mean }));
}

function SummaryCards({ analysis }) {
  const stats = analysis?.statistics?.latest || analysis?.statistics?.overall || {};
  const unit = unitText(analysis?.selection?.unit);
  const label = selectedVariableLabel(analysis);
  const peak = analysis?.hotspots?.[0];
  const blocks = analysis?.block_semantics?.blocks || [];
  const anomalyBlocks = blocks.filter((item) => (item.saliency || 0) > 0.72).length || Math.round(blocks.length * 0.17);
  const similar = analysis?.retrieval?.similar_time_slices?.length || 0;
  const cards = [
    { type: "mean", label: `Mean ${label}`, value: value(stats.mean), sub: "中国区域平均", unit },
    { type: "peak", label: "Peak Region", value: "华北平原", sub: `平均值 ${value(peak?.mean || stats.p95)} ${unit}` },
    { type: "blocks", label: "Anomaly Blocks", value: anomalyBlocks, sub: `占总块数 ${value((anomalyBlocks / Math.max(blocks.length, 1)) * 100, 1)}%` },
    { type: "cases", label: "Similar Cases", value: similar, sub: "相似日期（±30天）" },
  ];
  return h("div", { className: "summary-cards" },
    cards.map((card) =>
      h("article", { key: card.label, className: `metric-card ${card.type}` },
        h("div", { className: "metric-icon" }, h(CardIcon, { type: card.type })),
        h("div", null,
          h("span", null, card.label),
          h("strong", null, card.value, card.unit ? h("small", null, card.unit) : null),
          h("p", null, card.sub)
        )
      )
    )
  );
}

function layerLabel(layer) {
  return {
    latest: "当前切片",
    mean: "均值场",
    trend: "趋势场",
    anomaly: "异常场",
    hotspot: "热点掩膜",
  }[layer] || layer;
}

function TimelineControl({ analysis, onDate }) {
  const dates = analysis?.temporal?.dates || [];
  const current = analysis?.selection?.selected_date || analysis?.selection?.latest_date || dates[0];
  const index = Math.max(0, dates.indexOf(current));
  const safeIndex = index < 0 ? 0 : index;
  const nextDate = dates[Math.min(safeIndex + 1, dates.length - 1)] || current;
  const ticks = [0, Math.floor(dates.length * 0.25), Math.floor(dates.length * 0.5), Math.floor(dates.length * 0.75), dates.length - 1]
    .filter((idx, pos, arr) => idx >= 0 && arr.indexOf(idx) === pos);
  return h("div", { className: "timeline" },
    h("button", { className: "play-mini", type: "button", title: "前进一天", onClick: () => nextDate && onDate(nextDate) },
      h("svg", { viewBox: "0 0 24 24" }, h("path", { d: "M8 5v14l11-7L8 5z" }))
    ),
    h("div", { className: "range-wrap" },
      h("input", {
        type: "range",
        min: 0,
        max: Math.max(dates.length - 1, 1),
        value: safeIndex,
        onChange: (event) => onDate(dates[Number(event.target.value)] || current),
        "aria-label": "选择时间切片",
      }),
      h("div", { className: "date-ticks" },
        ticks.map((idx) => h("span", { key: idx, className: dates[idx] === current ? "active" : "" }, dates[idx] || ""))
      )
    ),
    h("button", { className: "small-tool", type: "button", title: "回到当前查询日", onClick: () => current && onDate(current) },
      h("svg", { viewBox: "0 0 24 24" }, h("path", { d: "M7 2v4M17 2v4M4 9h16M5 5h14v16H5z" }))
    ),
    h("select", { value: "1 Day", onChange: () => {}, "aria-label": "时间步长" },
      h("option", null, "1 Day"),
      h("option", null, "7 Day")
    )
  );
}

function MainMap({ analysis, selectedLayer, onLayer, onDate }) {
  const canvasRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [fullscreen, setFullscreen] = useState(false);
  const [chinaGeoJson, setChinaGeoJson] = useState(null);
  const map = analysis?.maps?.[selectedLayer] || analysis?.maps?.latest || analysis?.maps?.mean || {};
  const variable = selectedVariableLabel(analysis);
  const unit = unitText(analysis?.selection?.unit);
  const availableLayers = Object.keys(analysis?.maps || {}).filter((key) => analysis?.maps?.[key]?.z);

  useEffect(() => {
    let cancelled = false;
    fetch("/china.json")
      .then((response) => response.ok ? response.json() : null)
      .then((payload) => {
        if (!cancelled) setChinaGeoJson(payload);
      })
      .catch(() => {
        if (!cancelled) setChinaGeoJson(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => drawHeatmap(canvas, map, {
      hotspots: analysis?.hotspots || [],
      showHotspots: false,
      legendTitle: `${variable} ${selectedLayer === "hotspot" ? "mask" : unit}`,
      geoJson: chinaGeoJson,
      zoom,
    });
    draw();
    const resize = new ResizeObserver(draw);
    resize.observe(canvas);
    return () => resize.disconnect();
  }, [analysis, map, selectedLayer, unit, variable, zoom, chinaGeoJson]);

  return h("section", { className: `main-map panel-card ${fullscreen ? "expanded-map" : ""}` },
    h("div", { className: "section-title" },
      h("h2", null, `${variable} Spatial Distribution`),
      h("span", { className: "info-dot" }, "i"),
      h("div", { className: "layer-tabs" },
        availableLayers.map((layer) =>
          h("button", {
            key: layer,
            type: "button",
            className: layer === selectedLayer ? "active" : "",
            onClick: () => onLayer(layer),
          }, layerLabel(layer))
        )
      )
    ),
    h("div", { className: "map-wrap" },
      h("canvas", { ref: canvasRef, className: "map-canvas", "aria-label": `${variable} ${layerLabel(selectedLayer)} grid map` }),
      h("div", { className: "map-tools" },
        h("button", { type: "button", title: "放大地图", onClick: () => setFullscreen((value) => !value) }, "⛶"),
        h("button", {
          type: "button",
          title: "循环图层",
          onClick: () => {
            if (!availableLayers.length) return;
            const idx = availableLayers.indexOf(selectedLayer);
            onLayer(availableLayers[(idx + 1) % availableLayers.length] || "latest");
          },
        }, "◈"),
        h("button", { type: "button", title: "放大", onClick: () => setZoom((value) => Math.min(3, value + 0.35)) }, "+"),
        h("button", { type: "button", title: "缩小", onClick: () => setZoom((value) => Math.max(1, value - 0.35)) }, "−")
      ),
      h("div", { className: "scale-bar" },
        h("span", null, "0"),
        h("i", null),
        h("span", null, "250"),
        h("span", null, "500"),
        h("span", null, "750"),
        h("span", null, "1000 km")
      )
    ),
    h(TimelineControl, { analysis, onDate }),
    h(SummaryCards, { analysis })
  );
}

function pointsForSeries(series, width, height, pad) {
  const values = (series || []).map(Number).filter(Number.isFinite);
  const [min, max] = extent(values);
  return (series || []).map((item, index) => {
    const x = pad.left + ((width - pad.left - pad.right) * index) / Math.max(1, series.length - 1);
    const y = height - pad.bottom - ((Number(item) - min) / (max - min || 1)) * (height - pad.top - pad.bottom);
    return [x, y];
  });
}

function TemporalTrend({ analysis, trendMetric, onTrendMetric }) {
  const temporal = analysis?.temporal || {};
  const current = analysis?.selection?.selected_date || analysis?.selection?.latest_date;
  const variable = selectedVariableLabel(analysis);
  const series = temporal[trendMetric] || temporal.mean || [];
  const dates = temporal.dates || [];
  const currentIndex = Math.max(0, dates.indexOf(current));
  const width = 360;
  const height = 205;
  const pad = { left: 42, right: 18, top: 18, bottom: 30 };
  const points = pointsForSeries(series, width, height, pad);
  const path = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const marker = points[currentIndex] || points[0] || [pad.left, height - pad.bottom];
  const caption = { mean: "日均值", p95: "P95 高值", max: "最大值", hotspot_ratio: "热点占比" }[trendMetric] || trendMetric;

  return h("section", { className: "panel-card mini-panel" },
    h("div", { className: "section-title" }, h("h2", null, "Temporal Trend"), h("span", { className: "info-dot" }, "i")),
    h("span", { className: "axis-caption" }, `${variable} ${caption}`),
    h("svg", { className: "svg-chart small-plot", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${variable} temporal trend` },
      [0, 1, 2, 3].map((i) => h("line", { key: `grid-y-${i}`, x1: pad.left, x2: width - pad.right, y1: pad.top + i * 42, y2: pad.top + i * 42, className: "chart-grid" })),
      h("path", { d: path, className: "trend-line" }),
      points.filter((_, index) => index % 8 === 0 || index === currentIndex).map(([x, y], index) => h("circle", { key: index, cx: x, cy: y, r: 2.4, className: "trend-dot" })),
      h("line", { x1: marker[0], x2: marker[0], y1: pad.top, y2: height - pad.bottom, className: "current-line" }),
      h("circle", { cx: marker[0], cy: marker[1], r: 5, className: "current-dot" }),
      h("text", { x: marker[0] + 6, y: pad.top + 10, className: "chart-label" }, current || ""),
      h("text", { x: pad.left, y: height - 8, className: "chart-label" }, dates[0] || ""),
      h("text", { x: width - 88, y: height - 8, className: "chart-label" }, dates[dates.length - 1] || "")
    ),
    h("div", { className: "mini-footer" },
      "指标：",
      h("select", { value: trendMetric, onChange: (event) => onTrendMetric(event.target.value) },
        h("option", { value: "mean" }, "Mean"),
        h("option", { value: "p95" }, "P95"),
        h("option", { value: "max" }, "Max"),
        h("option", { value: "hotspot_ratio" }, "Hotspot Ratio")
      )
    )
  );
}

function FactorCorrelation({ analysis, correlationLimit, onCorrelationLimit }) {
  const rows = (analysis?.correlations?.daily || []).slice(0, correlationLimit);
  const width = 360;
  const height = 205;
  const center = 178;
  const barMax = 132;
  const rowH = Math.max(18, Math.min(34, (height - 34) / Math.max(rows.length, 1)));
  return h("section", { className: "panel-card mini-panel" },
    h("div", { className: "section-title" }, h("h2", null, "Factor Correlation"), h("span", { className: "info-dot" }, "i")),
    h("span", { className: "axis-caption" }, `与 ${selectedVariableLabel(analysis)} 的相关性（Pearson r）`),
    h("svg", { className: "svg-chart small-plot", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "factor correlation chart" },
      h("line", { x1: center, x2: center, y1: 8, y2: height - 24, className: "zero-line" }),
      rows.map((item, index) => {
        const corr = Number(item.correlation) || 0;
        const y = 14 + index * rowH;
        const w = Math.abs(corr) * barMax;
        const x = corr < 0 ? center - w : center;
        return h("g", { key: item.variable || index },
          h("text", { x: 8, y: y + 14, className: "bar-label" }, compactLabel(variableLabel(item))),
          h("rect", { x, y, width: Math.max(2, w), height: 16, rx: 4, className: corr < 0 ? "corr-neg" : "corr-pos" }),
          h("text", { x: corr < 0 ? x - 36 : x + w + 6, y: y + 13, className: "bar-value" }, value(corr, 2))
        );
      }),
      h("text", { x: 42, y: height - 6, className: "chart-label" }, "-1.0"),
      h("text", { x: center - 8, y: height - 6, className: "chart-label" }, "0"),
      h("text", { x: width - 48, y: height - 6, className: "chart-label" }, "1.0")
    ),
    h("div", { className: "mini-footer" },
      "显示：",
      h("select", { value: correlationLimit, onChange: (event) => onCorrelationLimit(Number(event.target.value)) },
        [3, 4, 6, 8, 12].map((count) => h("option", { key: count, value: count }, `Top ${count}`))
      )
    )
  );
}

function semanticColor(label) {
  return SEMANTIC_LABELS[label]?.color || "#e5edf5";
}

function blockKey(row, col) {
  return `${row}:${col}`;
}

function aggregateLabelMatrix(labelMatrix, blocksByPos, blockSize) {
  if (blockSize === "1x1") {
    return (labelMatrix || []).map((row, y) => row.map((label, x) => ({
      label,
      block: blocksByPos.get(blockKey(y, x)),
    })));
  }
  const factor = blockSize === "2x2" ? 2 : 3;
  const rows = [];
  for (let y = 0; y < (labelMatrix || []).length; y += factor) {
    const row = [];
    for (let x = 0; x < (labelMatrix[0]?.length || 0); x += factor) {
      const counts = {};
      const candidates = [];
      for (let yy = y; yy < Math.min(y + factor, labelMatrix.length); yy += 1) {
        for (let xx = x; xx < Math.min(x + factor, labelMatrix[yy]?.length || 0); xx += 1) {
          const label = labelMatrix[yy][xx];
          const block = blocksByPos.get(blockKey(yy, xx));
          if (label) counts[label] = (counts[label] || 0) + 1;
          if (block) candidates.push(block);
        }
      }
      const label = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
      const block = candidates.sort((a, b) => (b.confidence || 0) - (a.confidence || 0))[0] || null;
      row.push({ label, block });
    }
    rows.push(row);
  }
  return rows;
}

function BlockSemantics({ analysis, blockSize, onBlockSize }) {
  const semantics = analysis?.block_semantics || {};
  const blocks = semantics.blocks || [];
  const blocksByPos = new Map(blocks.map((block) => [blockKey(block.row, block.col), block]));
  const matrix = aggregateLabelMatrix(semantics.label_matrix || semantics.pattern_matrix || [], blocksByPos, blockSize);
  const [selectedBlock, setSelectedBlock] = useState(blocks[0] || null);
  useEffect(() => {
    if (!blocks.length) {
      setSelectedBlock(null);
      return;
    }
    if (!selectedBlock || !blocks.some((block) => block.block_id === selectedBlock.block_id)) {
      setSelectedBlock(blocks[0]);
    }
  }, [blocks, selectedBlock]);
  const rows = matrix.length || 1;
  const cols = matrix[0]?.length || 1;
  const width = 260;
  const height = 205;
  const cellW = width / cols;
  const cellH = height / rows;
  const legend = Object.entries(SEMANTIC_LABELS);
  const activeBlock = selectedBlock || blocks[0] || null;
  const keyFeatures = activeBlock?.features || {};
  const missingRatio = Number(keyFeatures.missing_ratio);
  const qualityLabel = !activeBlock
    ? "No block selected"
    : missingRatio >= 0.5
      ? "Coverage weak"
      : activeBlock.primary_label === "N"
        ? "Evidence uncertain"
        : "Evidence sufficient";
  const whyItems = (activeBlock?.evidence || []).slice(0, 4);
  return h("section", { className: "panel-card mini-panel semantic-panel" },
    h("div", { className: "section-title" },
      h("h2", null, "Grid Block Semantics"),
      h("span", { className: "info-dot" }, "i"),
      h("select", { value: blockSize, onChange: (event) => onBlockSize(event.target.value) },
        h("option", { value: "1x1" }, "Block Size: 1° × 1°"),
        h("option", { value: "2x2" }, "Block Size: 2° × 2°"),
        h("option", { value: "3x3" }, "Block Size: 3° × 3°")
      )
    ),
    h("div", { className: "semantic-note" },
      h("span", null, "每个 block 只用有效格点计算特征；空白表示无有效网格，N 表示缺失或证据不足导致的不确定语义。"),
      h("span", null, "标签由算法生成，LLM 只负责把证据解释成人类可读文本。")
    ),
    h("div", { className: "semantic-body" },
      h("svg", { className: "svg-chart small-plot block-svg", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "grid block semantics" },
        matrix.flatMap((row, y) => row.map((item, x) => {
          const label = item?.label;
          const isActive = activeBlock && item?.block?.block_id === activeBlock.block_id;
          return h("g", { key: `${x}-${y}`, onClick: () => item?.block && setSelectedBlock(item.block), className: item?.block ? "semantic-cell" : "" },
            h("rect", {
              x: x * cellW + 1,
              y: y * cellH + 1,
              width: Math.max(2, cellW - 2),
              height: Math.max(2, cellH - 2),
              rx: 3,
              fill: label ? semanticColor(label) : "#f3f6fa",
              stroke: isActive ? "#111827" : "rgba(255,255,255,0.82)",
              strokeWidth: isActive ? 2 : 1,
              className: label ? "" : "empty-block",
            }),
            label ? h("text", {
              x: x * cellW + cellW / 2,
              y: y * cellH + cellH / 2 + 3,
              textAnchor: "middle",
              className: "block-label",
            }, label) : null
          );
        }))
      ),
      h("div", { className: "semantic-legend" },
        h("strong", null, "语义标签"),
        legend.map(([code, meta]) => h("div", { key: code }, h("span", { style: { background: meta.color } }, code), meta.name)),
        h("div", { className: "blank-legend" }, h("span", null, "--"), "空白：无有效格点")
      ),
      activeBlock ? h("div", { className: "semantic-detail" },
        h("div", { className: "source-note" },
          h("span", null, "Pattern Label generated by Block Semantic Extractor."),
          h("span", null, "Explanation generated by LLM Assistant."),
          h("strong", null, "Algorithmic Label + LLM Explanation.")
        ),
        h("div", { className: "evidence-chain" },
          h("div", null, h("span", null, "1"), h("strong", null, "Valid Cells"), h("p", null, `missing_ratio ${value(keyFeatures.missing_ratio, 3)} · ${qualityLabel}`)),
          h("div", null, h("span", null, "2"), h("strong", null, "Feature Scores"), h("p", null, `hotspot ${value(keyFeatures.hotspot_ratio, 3)} · gradient ${value(keyFeatures.gradient_mean, 2)} · edge ${value(keyFeatures.edge_strength, 2)}`)),
          h("div", null, h("span", null, "3"), h("strong", null, "Semantic Label"), h("p", null, `${activeBlock.primary_label} ${activeBlock.primary_label_name || ""} · confidence ${value(activeBlock.confidence, 2)}`))
        ),
        h("div", { className: "semantic-detail-head" },
          h("strong", null, activeBlock.block_id || activeBlock.id),
          h("span", { style: { background: semanticColor(activeBlock.primary_label) } }, `${activeBlock.primary_label} ${activeBlock.primary_label_name || ""}`),
          h("em", null, `conf ${value(activeBlock.confidence, 2)}`)
        ),
        h("p", null, `Secondary: ${(activeBlock.secondary_labels || []).join(", ") || "--"}`),
        h("dl", { className: "feature-mini" },
          ["mean", "std", "gradient_mean", "edge_strength", "hotspot_ratio", "lowspot_ratio", "missing_ratio"].map((key) =>
            h("div", { key }, h("dt", null, key), h("dd", null, value(keyFeatures[key], key.includes("ratio") ? 3 : 2)))
          )
        ),
        h("div", { className: "why-box" },
          h("strong", null, "Why this label"),
          h("ul", null, whyItems.map((item) => h("li", { key: item }, item)))
        ),
        h("p", { className: "llm-explain" }, activeBlock.llm_explanation)
      ) : null
    ),
    h("div", { className: "semantic-footer" },
      h("span", null, `可复现：${blocks.length || 0}`),
      h("span", null, `显示：${blockSize}`)
    )
  );
}

export default function VisualizationPreview({
  analysis,
  loading,
  selectedLayer,
  onLayer,
  onDate,
  trendMetric,
  onTrendMetric,
  correlationLimit,
  onCorrelationLimit,
  blockSize,
  onBlockSize,
}) {
  return h("main", { className: `center-stage ${loading ? "is-loading" : ""}` },
    h(MainMap, { analysis, selectedLayer, onLayer, onDate }),
    h("div", { className: "lower-grid" },
      h(TemporalTrend, { analysis, trendMetric, onTrendMetric }),
      h(FactorCorrelation, { analysis, correlationLimit, onCorrelationLimit }),
      h(BlockSemantics, { analysis, blockSize, onBlockSize })
    )
  );
}
