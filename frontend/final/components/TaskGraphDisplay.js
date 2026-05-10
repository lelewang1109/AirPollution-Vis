import React from "https://esm.sh/react@18.2.0";

const h = React.createElement;

const VARIABLE_LABELS = {
  "PM2.5": "PM2.5",
  temp: "Temperature",
  rhum: "Humidity",
  pres: "Pressure",
  elevation: "Elevation",
  bcpr: "Bias-Corrected Precipitation",
  lrad: "Longwave Radiation",
  wind: "Wind",
  prec: "Precipitation",
  rain: "Rain",
  shum: "Specific Humidity",
  snow: "Snowfall",
  srad: "Shortwave Radiation",
};

const VARIABLE_ORDER = ["PM2.5", "temp", "rhum", "pres", "wind", "prec", "rain", "bcpr", "shum", "snow", "lrad", "srad", "elevation"];

function displayVariable(item) {
  const name = typeof item === "string" ? item : item?.name;
  return VARIABLE_LABELS[name] || item?.label || name || "";
}

function orderedVariables(items) {
  const list = [...(items || [])];
  return list.sort((a, b) => {
    const ai = VARIABLE_ORDER.indexOf(a.name);
    const bi = VARIABLE_ORDER.indexOf(b.name);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });
}

function CheckIcon() {
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
    h("path", { d: "M20 6L9 17l-5-5" })
  );
}

function panel(title, children, footer = null, extraClass = "") {
  return h("section", { className: `side-panel ${extraClass}` },
    h("div", { className: "panel-heading" },
      h("h2", null, title)
    ),
    children,
    footer
  );
}

function DataAccess({ analysis, sourceType, onSourceType, onRefreshCatalog }) {
  const [showAdd, setShowAdd] = React.useState(false);
  const [fileName, setFileName] = React.useState("");
  return panel("Data Access",
    h(React.Fragment, null,
      h("div", { className: "muted-line" }, "已接入数据源"),
      h("div", { className: "format-row" },
        ["NC", "CSV", "TIF", "PNG"].map((label) =>
          h("button", {
            key: label,
            type: "button",
            className: `format-pill ${label.toLowerCase()} ${sourceType === label ? "active" : ""}`,
            onClick: () => onSourceType(label),
            title: `选择 ${label} 数据源类型`,
          }, label)
        )
      ),
      h("div", { className: "connection-row" },
        h("span", null, "连接状态"),
        h("strong", null, h("span", { className: "mini-dot" }), "Active")
      ),
      h("div", { className: "button-split" },
        h("button", { className: "ghost-button", type: "button", onClick: () => setShowAdd((value) => !value) }, showAdd ? "收起数据接入" : "+ Add Data Source"),
        h("button", { className: "ghost-button", type: "button", onClick: onRefreshCatalog }, "刷新 Catalog")
      ),
      showAdd ? h("div", { className: "inline-detail" },
        h("strong", null, `${sourceType} 待接入`),
        h("p", null, "当前后端读取 data/2000 下的 NetCDF 目录；选择文件会记录到前端待接入队列，不会伪装成已上传。"),
        h("input", {
          type: "file",
          accept: sourceType === "NC" ? ".nc,.cdf" : sourceType === "CSV" ? ".csv" : sourceType === "TIF" ? ".tif,.tiff" : ".png",
          onChange: (event) => setFileName(event.target.files?.[0]?.name || ""),
        }),
        fileName ? h("span", { className: "detail-chip" }, `已选择：${fileName}`) : null
      ) : null
    )
  );
}

function DatasetInfo({ analysis, expanded, onExpanded }) {
  const selection = analysis?.selection || {};
  const catalog = analysis?.catalog || {};
  const variables = orderedVariables(analysis?.variables || []).map(displayVariable).join(", ");
  return panel("Dataset",
    h(React.Fragment, null,
      h("dl", { className: "info-list" },
        h("div", null, h("dt", null, "Region:"), h("dd", null, selection.region_label || "China")),
        h("div", null, h("dt", null, "Resolution:"), h("dd", null, "0.1° × 0.1°")),
        h("div", null, h("dt", null, "Time:"), h("dd", null, selection.selected_date || selection.latest_date || catalog.date_start)),
        h("div", null, h("dt", null, "Variables:"), h("dd", null, variables || "PM2.5, Temperature, Humidity")),
        expanded ? h("div", null, h("dt", null, "Shape:"), h("dd", null, (catalog.shape || []).join(" × ") || "--")) : null,
        expanded ? h("div", null, h("dt", null, "Files:"), h("dd", null, catalog.file_count || analysis?.provenance?.source_file_count || "--")) : null,
        expanded ? h("div", null, h("dt", null, "Bounds:"), h("dd", null, catalog.bounds ? `${catalog.bounds.lon_min}~${catalog.bounds.lon_max}E, ${catalog.bounds.lat_min}~${catalog.bounds.lat_max}N` : "--")) : null
      ),
      h("button", { className: "ghost-button", type: "button", onClick: () => onExpanded(!expanded) }, expanded ? "Hide Dataset Info" : "View Dataset Info")
    ),
  );
}

function SemanticAdapter({ analysis, adapterDetail, onAdapterDetail }) {
  const blocks = analysis?.block_semantics?.blocks?.length || 0;
  const quality = analysis?.statistics?.quality || {};
  const rows = [
    ["Grid Tensor 构建", `${analysis?.catalog?.shape?.join("×") || "完成"}`],
    ["Spatial Features 提取", `${analysis?.hotspots?.length || 0} 热点`],
    ["Temporal Features 提取", `${analysis?.temporal?.dates?.length || 0} 日`],
    ["Block Semantics 解析", blocks ? `${blocks} 块` : "完成"],
  ];
  return panel("Grid Semantic Adapter",
    h(React.Fragment, null,
      h("div", { className: "status-list" },
        rows.map(([label, value]) =>
          h("button", {
            key: label,
            type: "button",
            className: adapterDetail === label ? "status-item active" : "status-item",
            onClick: () => onAdapterDetail(label),
          },
            h("span", null, label),
            h("strong", null, h(CheckIcon), value)
          )
        )
      ),
      h("div", { className: "inline-detail" },
        h("strong", null, adapterDetail),
        h("p", null, adapterDetail.includes("Grid") ? `有效网格占比 ${Math.round((quality.finite_ratio || 0) * 100)}%，变量已标准化为 GridTensor。`
          : adapterDetail.includes("Spatial") ? `当前区域识别到 ${analysis?.hotspots?.length || 0} 个热点连通域。`
          : adapterDetail.includes("Temporal") ? `时间序列覆盖 ${analysis?.catalog?.date_start || "--"} 至 ${analysis?.catalog?.date_end || "--"}。`
          : `语义块矩阵为 ${analysis?.block_semantics?.grid?.rows || 0} × ${analysis?.block_semantics?.grid?.cols || 0}。`)
      )
    )
  );
}

function TaskSetup({
  catalog,
  variable,
  region,
  date,
  onVariable,
  onRegion,
  onDate,
  loading,
  selectedGoals,
  onToggleGoal,
  onSaveConfig,
  savedConfig,
}) {
  const variables = orderedVariables(catalog?.variables?.length ? catalog.variables : [
    { name: "PM2.5", label: "PM2.5" },
    { name: "temp", label: "Temperature" },
    { name: "rhum", label: "Humidity" },
    { name: "pres", label: "Pressure" },
  ]);
  const regions = catalog?.regions?.length ? catalog.regions : [
    { id: "china", label: "中国 (CHN)" },
    { id: "north_china", label: "华北地区" },
    { id: "jingjinji", label: "京津冀" },
  ];
  const goals = ["Heatmap", "Time Series", "Correlation", "Explanation", "Semantic Blocks", "Trace"];

  return panel("Task Setup",
    h(React.Fragment, null,
      h("label", null, "变量选择"),
      h("div", { className: "chip-row" },
        variables.map((item) =>
          h("button", {
            key: item.name,
            type: "button",
            className: item.name === variable ? "chip active" : "chip",
            onClick: () => onVariable(item.name),
            disabled: loading,
          }, displayVariable(item))
        )
      ),
      h("label", null, "时间选择"),
      h("input", {
        type: "date",
        value: date,
        min: "2000-01-01",
        max: "2000-05-31",
        onChange: (event) => onDate(event.target.value),
        disabled: loading,
      }),
      h("label", null, "区域选择"),
      h("select", { value: region, onChange: (event) => onRegion(event.target.value), disabled: loading },
        regions.map((item) => h("option", { key: item.id, value: item.id }, item.label))
      ),
      h("label", null, "可视化目标（多选）"),
      h("div", { className: "goal-grid" },
        goals.map((goal) =>
          h("button", {
            key: goal,
            type: "button",
            className: selectedGoals.includes(goal) ? "goal active" : "goal",
            onClick: () => onToggleGoal(goal),
          }, goal)
        )
      ),
      h("button", { className: "save-button", type: "button", onClick: onSaveConfig }, "保存任务配置"),
      savedConfig ? h("span", { className: "detail-chip" }, `已保存 ${new Date(savedConfig.savedAt).toLocaleTimeString("zh-CN", { hour12: false })}`) : null
    )
  );
}

export default function TaskGraphDisplay(props) {
  return h("aside", { className: `left-rail active-${String(props.activeTab || "Data").toLowerCase()}` },
    h(DataAccess, props),
    h(DatasetInfo, { analysis: props.analysis, expanded: props.datasetExpanded, onExpanded: props.onDatasetExpanded }),
    h(SemanticAdapter, props),
    h(TaskSetup, props)
  );
}
