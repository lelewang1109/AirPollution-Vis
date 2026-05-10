import React, { useEffect, useMemo, useState } from "https://esm.sh/react@18.2.0";
import QueryInput from "../components/QueryInput.js";
import TaskGraphDisplay from "../components/TaskGraphDisplay.js";
import VisualizationPreview from "../components/VisualizationPreview.js";
import VisualizationStrategy from "../components/VisualizationStrategy.js";

const h = React.createElement;

const DEFAULT_QUERY = "请分析 2000-01-01 中国区域 PM2.5 空间分布，并展示高值区域、时间趋势与可能影响因子。";

function extractDate(query) {
  const match = String(query || "").match(/(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})/);
  if (!match) return "2000-01-01";
  return `${match[1]}-${String(Number(match[2])).padStart(2, "0")}-${String(Number(match[3])).padStart(2, "0")}`;
}

function inferVariable(query, catalog) {
  const text = String(query || "").toLowerCase();
  const names = (catalog?.variables || []).map((item) => item.name);
  const has = (name) => names.includes(name);
  if (text.includes("温度")) return names.includes("temp") ? "temp" : "PM2.5";
  if (text.includes("湿度")) return names.includes("rhum") ? "rhum" : "PM2.5";
  if (text.includes("风")) return has("wind") ? "wind" : "PM2.5";
  if (text.includes("降水") || text.includes("precipitation")) return has("prec") ? "prec" : "PM2.5";
  if (text.includes("降雨") || text.includes("rain")) return has("rain") ? "rain" : "PM2.5";
  if (text.includes("降雪") || text.includes("snow")) return has("snow") ? "snow" : "PM2.5";
  if (text.includes("气压") || text.includes("pressure")) return has("pres") ? "pres" : "PM2.5";
  if (text.includes("高程") || text.includes("地形") || text.includes("elevation")) return has("elevation") ? "elevation" : "PM2.5";
  if (text.includes("长波") || text.includes("longwave")) return has("lrad") ? "lrad" : "PM2.5";
  if (text.includes("短波") || text.includes("shortwave") || text.includes("辐射")) return has("srad") ? "srad" : "PM2.5";
  if (text.includes("比湿") || text.includes("specific humidity")) return has("shum") ? "shum" : "PM2.5";
  return has("PM2.5") ? "PM2.5" : names[0] || "PM2.5";
}

function inferRegion(query) {
  const text = String(query || "");
  if (text.includes("京津冀")) return "jingjinji";
  if (text.includes("华北")) return "north_china";
  if (text.includes("东北")) return "northeast";
  if (text.includes("华东") || text.includes("长三角")) return text.includes("长三角") ? "yangtze_delta" : "east_china";
  if (text.includes("华南") || text.includes("珠三角")) return text.includes("珠三角") ? "pearl_delta" : "south_china";
  if (text.includes("西南")) return "southwest";
  if (text.includes("西北")) return "northwest";
  return "china";
}

function buildMockAnalysis(query = DEFAULT_QUERY) {
  const lon = Array.from({ length: 92 }, (_, index) => 73.7 + index * 0.66);
  const lat = Array.from({ length: 54 }, (_, index) => 18.3 + index * 0.66);
  const z = lat.map((la) =>
    lon.map((lo) => {
      const northChina = Math.exp(-((lo - 116) ** 2 / 92 + (la - 38) ** 2 / 42)) * 165;
      const central = Math.exp(-((lo - 109) ** 2 / 180 + (la - 31) ** 2 / 52)) * 78;
      const basin = Math.exp(-((lo - 104) ** 2 / 60 + (la - 30) ** 2 / 28)) * 54;
      const westClean = (lo - 73) * 0.72 + Math.sin(la * 0.7) * 10;
      return Math.max(6, 22 + northChina + central + basin + westClean);
    })
  );
  const dates = Array.from({ length: 31 }, (_, index) => {
    const day = index + 22;
    if (day <= 31) return `1999-12-${String(day).padStart(2, "0")}`;
    return `2000-01-${String(day - 31).padStart(2, "0")}`;
  });
  const mean = dates.map((_, index) => Math.round((54 + index * 5.1 + Math.sin(index / 2.3) * 23 + (index > 7 && index < 17 ? 82 : 0)) * 10) / 10);
  const matrix = Array.from({ length: 9 }, (_, row) =>
    Array.from({ length: 14 }, (_, col) => {
      if ((row < 2 && col < 4) || (row > 6 && col > 11)) return null;
      return Math.max(0, Math.min(1, 0.24 + col / 18 + Math.exp(-((col - 9) ** 2 + (row - 4) ** 2) / 11) * 0.55 + Math.sin(row + col) * 0.08));
    })
  );
  const labelNames = {
    H: "高值热点型",
    L: "低值冷点型",
    G: "梯度过渡型",
    B: "边界突变型",
    D: "扩散分散型",
    U: "均匀稳定型",
    M: "混合复杂型",
    N: "噪声不确定型",
  };
  const labelMatrix = matrix.map((row, y) => row.map((item, x) => {
    if (item == null) return null;
    if (item > 0.78) return "H";
    if (x > 8 && y < 5) return "G";
    if ((x + y) % 7 === 0) return "D";
    if (item < 0.32) return "L";
    if (Math.abs(x - 7) <= 1 && y > 2) return "B";
    return item < 0.42 ? "U" : "M";
  }));
  const mockBlocks = labelMatrix.flatMap((row, y) => row.map((label, x) => {
    if (!label) return null;
    const saliency = matrix[y][x];
    const blockId = `B_${String(y).padStart(3, "0")}_${String(x).padStart(3, "0")}`;
    return {
      block_id: blockId,
      id: blockId,
      row: y,
      col: x,
      bbox: { lon_min: 73.7 + x * 4.36, lon_max: 78.0 + x * 4.36, lat_min: 18.3 + y * 3.9, lat_max: 22.1 + y * 3.9 },
      primary_label: label,
      primary_label_name: labelNames[label],
      secondary_labels: label === "H" ? ["G"] : label === "B" ? ["D"] : [],
      confidence: Math.max(0.48, Math.min(0.95, saliency)),
      scores: { H: 0.12, L: 0.08, G: 0.16, B: 0.1, D: 0.11, U: 0.14, M: 0.2, N: 0.02, [label]: Math.max(0.55, saliency) },
      features: {
        mean: 50 + saliency * 120,
        std: 8 + saliency * 28,
        min: 12 + saliency * 20,
        max: 88 + saliency * 170,
        gradient_mean: saliency * 0.42,
        edge_strength: label === "B" ? 0.72 : saliency * 0.32,
        hotspot_ratio: label === "H" ? 0.34 : 0.08,
        lowspot_ratio: label === "L" ? 0.31 : 0.05,
        connected_components_high: label === "D" ? 5 : label === "H" ? 1 : 2,
        connected_components_low: label === "D" ? 4 : label === "L" ? 1 : 2,
        missing_ratio: 0.02,
      },
      evidence: label === "H"
        ? ["高值像元占比为 34%", "高值连通域数量为 1", "最大值显著高于 block 均值", "缺失率较低，数据质量较好"]
        : ["规则分类器基于统计、梯度、边界和连通域特征生成标签", "缺失率较低，数据质量较好"],
      llm_explanation: `该 block 被识别为${labelNames[label]}，解释文本基于算法标签、置信度和证据生成。`,
      saliency,
      pattern_type: label,
      trend_label: "stable",
      uncertainty: "low",
    };
  })).filter(Boolean);

  return {
    catalog: {
      data_dir: "/Users/lele/Desktop/GridVis-LLM/data/2000",
      file_count: 152,
      date_start: "2000-01-01",
      date_end: "2000-05-31",
      shape: [353, 613],
      bounds: { lon_min: 73.65, lon_max: 134.85, lat_min: 18.25, lat_max: 53.55 },
    },
    selection: {
      variable: "PM2.5",
      label: "PM2.5",
      unit: "µg/m3",
      region: "china",
      region_label: "中国全域",
      latest_date: extractDate(query),
      selected_date: extractDate(query),
    },
    variables: [
      { name: "PM2.5", label: "PM2.5", unit: "µg/m3", mean: 73.6, selected: true },
      { name: "temp", label: "Temperature", unit: "K", mean: 282.4 },
      { name: "rhum", label: "Humidity", unit: "%", mean: 61.7 },
      { name: "pres", label: "Pressure", unit: "Pa", mean: 84500 },
      { name: "elevation", label: "Elevation", unit: "m", mean: 1186 },
      { name: "bcpr", label: "Bias-Corrected Precipitation", unit: "kg m-2 s-1", mean: 0.00002 },
      { name: "lrad", label: "Longwave Radiation", unit: "W m-2", mean: 283.1 },
      { name: "prec", label: "Precipitation", unit: "kg m-2 s-1", mean: 0.00003 },
      { name: "rain", label: "Rainfall", unit: "kg m-2 s-1", mean: 0.00002 },
      { name: "shum", label: "Specific Humidity", unit: "kg kg-1", mean: 0.004 },
      { name: "snow", label: "Snowfall", unit: "kg m-2 s-1", mean: 0.00001 },
      { name: "srad", label: "Shortwave Radiation", unit: "W m-2", mean: 172.6 },
      { name: "wind", label: "Wind", unit: "m s-1", mean: 3.8 },
    ],
    statistics: {
      overall: { mean: 73.6, max: 260.4, p95: 162.4 },
      latest: { mean: 73.6, max: 260.4, p95: 162.4 },
      quality: { finite_ratio: 1, valid_region_cells: 216389, total_cells: 216389 },
      temporal_trend: { direction: "increase", slope: 0.42, delta: 38.2 },
    },
    temporal: {
      dates,
      mean,
      p95: mean.map((value) => value + 64),
      max: mean.map((value) => value + 125),
      anomaly_days: [{ date: "2000-01-01", mean: 142.5, residual: 46.2 }],
    },
    maps: { latest: { lon, lat, z }, mean: { lon, lat, z } },
    hotspots: [
      { id: 1, lat: 38.2, lon: 116.5, mean: 162.4, peak: 260.4, cells: 92 },
      { id: 2, lat: 35.5, lon: 112.6, mean: 145.8, peak: 226.1, cells: 48 },
    ],
    block_semantics: {
      grid: { rows: 9, cols: 14 },
      matrix,
      label_matrix: labelMatrix,
      label_distribution: mockBlocks.reduce((acc, block) => ({ ...acc, [block.primary_label]: (acc[block.primary_label] || 0) + 1 }), {}),
      top_blocks: [
        ...mockBlocks.slice().sort((a, b) => b.saliency - a.saliency).slice(0, 3),
      ],
      blocks: mockBlocks,
      source_note: {
        label: "Pattern Label generated by Block Semantic Extractor.",
        explanation: "Explanation generated by LLM Assistant.",
        combined: "Algorithmic Label + LLM Explanation.",
      },
    },
    correlations: {
      daily: [
        { variable: "temp", label: "Temperature", correlation: -0.62 },
        { variable: "rhum", label: "Humidity", correlation: -0.48 },
        { variable: "pres", label: "Pressure", correlation: 0.31 },
        { variable: "elevation", label: "Elevation", correlation: -0.27 },
      ],
    },
    retrieval: {
      query_slice: extractDate(query),
      similar_time_slices: [
        { date: "2000-01-06", similarity: 0.93, rmse: 12.6 },
        { date: "1999-12-29", similarity: 0.89, rmse: 15.4 },
        { date: "2000-01-12", similarity: 0.84, rmse: 17.1 },
      ],
    },
    provenance: {
      source_file_count: 152,
      source_files: [
        "/Users/lele/Desktop/GridVis-LLM/data/2000/20000101.nc",
        "/Users/lele/Desktop/GridVis-LLM/data/2000/20000531.nc",
      ],
      pipeline: [
        "NetCDF catalog scan",
        "GridTensor construction",
        "Region mask",
        "Feature extraction",
        "Tool execution",
        "LLM-native figure binding",
      ],
      backend: "mock fallback / direct HTML mode",
    },
    llm: {
      query,
      task_graph: {
        task_type: "attribution",
        region: "中国全域",
        variables: ["PM2.5", "temp", "rhum", "pres", "elevation"],
        time_range: "2000-01-01",
        analysis_goals: ["空间分布", "高值区识别", "时间趋势", "影响因子相关性", "可解释溯源"],
      },
      visualization_strategy: {
        layout: "PM2.5 主地图 + 时间趋势 + 因子相关 + 语义块 + LLM 解释",
        views: ["Heatmap", "Time Series", "Correlation View", "Semantic Blocks"],
        responsibility: {
          label_generation: "Block Semantic Extractor algorithm",
          strategy_text: "LLM Assistant constrained by semantic-label rule library",
          chart_execution: "Tool execution layer",
        },
      },
      narrative:
        "基于当前网格数据，PM2.5 高值区主要集中于华北平原、汾渭平原及中东部城市群。整体空间分布呈现东部高、西部低的梯度格局；温度与湿度和 PM2.5 呈负相关，气压弱正相关。",
      uncertainty_risks: ["相关性不等同于因果归因，仍需结合排放源与边界层条件。"],
    },
  };
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

export default function LLMVisualizationPage() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [catalog, setCatalog] = useState(null);
  const [analysis, setAnalysis] = useState(() => buildMockAnalysis(DEFAULT_QUERY));
  const [variable, setVariable] = useState("PM2.5");
  const [region, setRegion] = useState("china");
  const [date, setDate] = useState("2000-01-01");
  const [loading, setLoading] = useState(false);
  const [apiState, setApiState] = useState("fallback");
  const [activeTab, setActiveTab] = useState("Data");
  const [sourceType, setSourceType] = useState("NC");
  const [datasetExpanded, setDatasetExpanded] = useState(false);
  const [adapterDetail, setAdapterDetail] = useState("Grid Tensor 构建");
  const [selectedGoals, setSelectedGoals] = useState(["Heatmap", "Time Series", "Correlation", "Explanation"]);
  const [savedConfig, setSavedConfig] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState("latest");
  const [trendMetric, setTrendMetric] = useState("mean");
  const [correlationLimit, setCorrelationLimit] = useState(4);
  const [blockSize, setBlockSize] = useState("1x1");
  const [assistantVote, setAssistantVote] = useState(null);
  const [strategyFocus, setStrategyFocus] = useState("Heatmap");
  const [traceFocus, setTraceFocus] = useState("Data Slice");
  const [statusMessage, setStatusMessage] = useState("");

  const canUseApi = typeof window !== "undefined" && window.location.protocol !== "file:";

  async function loadAnalysis(next = {}) {
    const nextQuery = next.query ?? query;
    const nextDate = next.date ?? extractDate(nextQuery) ?? date;
    const nextVariable = next.variable ?? variable;
    const nextRegion = next.region ?? region;
    setLoading(true);
    setDate(nextDate);
    setVariable(nextVariable);
    setRegion(nextRegion);
    try {
      if (!canUseApi) throw new Error("Direct file mode");
      const params = new URLSearchParams({
        query: nextQuery,
        variable: nextVariable,
        region: nextRegion,
        date: nextDate,
      });
      const payload = await fetchJson(`/api/analysis?${params.toString()}`);
      setAnalysis(payload);
      setApiState("connected");
      setStatusMessage(`已更新：${nextVariable} / ${nextRegion} / ${nextDate}`);
    } catch (error) {
      const fallback = buildMockAnalysis(nextQuery);
      fallback.runtime = { error: error.message };
      setAnalysis(fallback);
      setApiState(canUseApi ? "fallback" : "offline");
      setStatusMessage(`已切换为本地演示数据：${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function refreshCatalog() {
    try {
      if (!canUseApi) throw new Error("Direct file mode");
      const payload = await fetchJson("/api/catalog");
      setCatalog(payload);
      setApiState("connected");
      setStatusMessage(`Catalog 已刷新：${payload.file_count || 0} 个切片`);
    } catch (error) {
      setApiState(canUseApi ? "fallback" : "offline");
      setStatusMessage(`Catalog 刷新失败：${error.message}`);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      try {
        if (!canUseApi) throw new Error("Direct file mode");
        const payload = await fetchJson("/api/catalog");
        if (cancelled) return;
        setCatalog(payload);
        const initialVariable = inferVariable(DEFAULT_QUERY, payload);
        const initialRegion = inferRegion(DEFAULT_QUERY);
        setVariable(initialVariable);
        setRegion(initialRegion);
        await loadAnalysis({ query: DEFAULT_QUERY, variable: initialVariable, region: initialRegion, date: extractDate(DEFAULT_QUERY) });
      } catch {
        if (!cancelled) {
          setCatalog({
            variables: buildMockAnalysis().variables.map(({ name, label, unit }) => ({ name, label, unit })),
            regions: [
              { id: "china", label: "中国全域" },
              { id: "north_china", label: "华北地区" },
              { id: "jingjinji", label: "京津冀" },
            ],
          });
          setApiState(canUseApi ? "fallback" : "offline");
        }
      }
    }
    boot();
    return () => {
      cancelled = true;
    };
  }, []);

  const taskGraph = useMemo(() => analysis?.llm?.task_graph || {}, [analysis]);

  function runQuery() {
    const nextVariable = inferVariable(query, catalog);
    const nextRegion = inferRegion(query);
    loadAnalysis({ query, variable: nextVariable, region: nextRegion, date: extractDate(query) });
  }

  function toggleGoal(goal) {
    setSelectedGoals((current) =>
      current.includes(goal) ? current.filter((item) => item !== goal) : [...current, goal]
    );
  }

  function saveTaskConfig() {
    const config = {
      query,
      variable,
      region,
      date,
      selectedGoals,
      sourceType,
      savedAt: new Date().toISOString(),
    };
    setSavedConfig(config);
    setStatusMessage("任务配置已保存到当前会话");
  }

  function exportAnalysis() {
    const payload = {
      exported_at: new Date().toISOString(),
      ui_state: {
        query,
        variable,
        region,
        date,
        sourceType,
        selectedGoals,
        selectedLayer,
        trendMetric,
        correlationLimit,
        blockSize,
        strategyFocus,
        traceFocus,
      },
      analysis,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `gridvis_analysis_${String(date).replaceAll("-", "")}_${variable}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setStatusMessage("已导出当前分析 JSON");
  }

  return h("div", { className: "gridvis-app" },
    h(QueryInput, {
      query,
      onQuery: setQuery,
      onRun: runQuery,
        loading,
        apiState,
        activeTab,
        onTab: (tab) => {
          setActiveTab(tab);
          setStatusMessage(`已切换到 ${tab} 模块`);
        },
      }),
      statusMessage ? h("div", { className: "toast-line", role: "status" }, statusMessage) : null,
    h("div", { className: "workspace" },
      h(TaskGraphDisplay, {
        analysis,
        catalog,
        variable,
        region,
        date,
        onVariable: (value) => loadAnalysis({ variable: value, region, date, query }),
        onRegion: (value) => loadAnalysis({ variable, region: value, date, query }),
        onDate: (value) => loadAnalysis({ variable, region, date: value, query }),
        taskGraph,
        loading,
        activeTab,
        sourceType,
        onSourceType: setSourceType,
        onRefreshCatalog: refreshCatalog,
        datasetExpanded,
        onDatasetExpanded: setDatasetExpanded,
        adapterDetail,
        onAdapterDetail: setAdapterDetail,
        selectedGoals,
        onToggleGoal: toggleGoal,
        onSaveConfig: saveTaskConfig,
        savedConfig,
      }),
      h(VisualizationPreview, {
        analysis,
        loading,
        selectedLayer,
        onLayer: setSelectedLayer,
        onDate: (value) => loadAnalysis({ variable, region, date: value, query }),
        trendMetric,
        onTrendMetric: setTrendMetric,
        correlationLimit,
        onCorrelationLimit: setCorrelationLimit,
        blockSize,
        onBlockSize: setBlockSize,
        activeTab,
      }),
      h(VisualizationStrategy, {
        analysis,
        taskGraph,
        loading,
        selectedLayer,
        onLayer: setSelectedLayer,
        trendMetric,
        onTrendMetric: setTrendMetric,
        correlationLimit,
        onCorrelationLimit: setCorrelationLimit,
        blockSize,
        onBlockSize: setBlockSize,
        assistantVote,
        onAssistantVote: setAssistantVote,
        strategyFocus,
        onStrategyFocus: setStrategyFocus,
        traceFocus,
        onTraceFocus: setTraceFocus,
        onExport: exportAnalysis,
      })
    )
  );
}
