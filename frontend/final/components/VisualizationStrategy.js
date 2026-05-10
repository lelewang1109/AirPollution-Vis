import React from "https://esm.sh/react@18.2.0";

const h = React.createElement;

function Chevron() {
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, h("path", { d: "M9 18l6-6-6-6" }));
}

function MiniIcon({ path }) {
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, h("path", { d: path }));
}

function RightPanel({ title, icon, children }) {
  const [open, setOpen] = React.useState(true);
  return h("section", { className: "right-panel" },
    h("div", { className: "right-title" },
      h("span", { className: "right-icon" }, icon),
      h("h2", null, title),
      h("button", { type: "button", className: open ? "collapse-btn open" : "collapse-btn", onClick: () => setOpen((value) => !value) }, h(Chevron))
    ),
    open ? children : null
  );
}

function AssistantPanel({ analysis }) {
  const narrative = analysis?.llm?.narrative || analysis?.llm?.semantic_response || "正在等待 LLM 生成解释。";
  const points = narrative
    .replace(/。/g, "。\n")
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 6);
  const date = analysis?.selection?.selected_date || analysis?.selection?.latest_date;
  const variable = analysis?.selection?.label || analysis?.selection?.variable || "PM2.5";
  const metadata = analysis?.llm?.llm_metadata || {};
  const isDashScope = metadata.mode === "dashscope_agents" || metadata.provider === "dashscope";
  const modeLabel = isDashScope ? `DashScope · ${metadata.model || "qwen-plus"}` : `Rule fallback · ${metadata.mode || "local"}`;
  const modeText = isDashScope
    ? "当前解释由 DashScope agent 基于结构化 evidence 生成。"
    : "当前未检测到可用 API Key，解释使用本地规则 fallback。";
  return h(RightPanel, {
    title: "LLM Analysis Assistant",
    icon: h(MiniIcon, { path: "M12 2l2.2 5.1 5.5.5-4.2 3.6 1.2 5.4L12 13.8 7.3 16.6l1.2-5.4-4.2-3.6 5.5-.5L12 2z" }),
    children: h(React.Fragment, null,
      h("div", { className: "assistant-card" },
        h("div", { className: isDashScope ? "llm-mode real" : "llm-mode fallback" },
          h("strong", null, "Explanation mode"),
          h("span", null, modeLabel),
          h("p", null, modeText)
        ),
        h("p", null, `基于 ${date || "当前"} 中国区域 ${variable} 数据的分析结果如下：`),
        h("ul", null, points.map((point) => h("li", { key: point }, point))),
        h("p", null, "以上结论基于算法标签、结构化证据与模型解释生成，供可视分析参考。"),
        h("span", { className: "llm-badge" }, modeLabel),
        h("time", null, new Date().toLocaleTimeString("zh-CN", { hour12: false }))
      )
    )
  });
}

function normalizeView(view) {
  if (/time|时间|序列|timeline|retrieval/i.test(view)) return "Time Series";
  if (/corr|coupling|相关|因子/i.test(view)) return "Correlation View";
  if (/block|semantic|语义|矩阵/i.test(view)) return "Semantic Blocks";
  return "Heatmap";
}

function StrategyPanel({
  analysis,
  taskGraph,
  selectedLayer,
  onLayer,
  onTrendMetric,
  onCorrelationLimit,
  onBlockSize,
  strategyFocus,
  onStrategyFocus,
}) {
  const views = analysis?.llm?.visualization_strategy?.views || ["Heatmap", "Time Series", "Correlation View", "Semantic Blocks"];
  const semanticStrategy = analysis?.llm?.visualization_strategy?.semantic_strategy || {};
  const recommendations = semanticStrategy.recommendations || [];
  const responsibility = analysis?.llm?.visualization_strategy?.responsibility || semanticStrategy.responsibility || {};
  const goals = taskGraph?.analysis_goals || ["分析 PM2.5 空间分布、时间趋势与影响因子"];
  const nodes = [
    ["用户任务", goals[0] || "自然语言任务"],
    ["分析空间分布、时间趋势与影响因子", "LLM 策略生成"],
  ];
  const reasons = [
    "热力图可直观呈现空间分布与高值区域",
    "时间序列展示污染随时间变化趋势",
    "相关性分析识别影响因子的方向与强度",
    "语义块视图帮助理解通用空间形态差异",
  ];
  function activateView(view) {
    const normalized = normalizeView(view);
    onStrategyFocus(normalized);
    if (normalized === "Heatmap") onLayer(selectedLayer === "latest" ? "mean" : "latest");
    if (normalized === "Time Series") onTrendMetric("p95");
    if (normalized === "Correlation View") onCorrelationLimit(8);
    if (normalized === "Semantic Blocks") onBlockSize("2x2");
  }
  const recommendationPanel = recommendations.length
    ? h("div", { className: "reason-list semantic-recommendations" },
        h("strong", null, "由标签分布触发的策略"),
        h("ul", null, recommendations.slice(0, 4).map((item) =>
          h("li", { key: item.trigger_label },
            `${item.trigger_label} ${item.label_name} (${Math.round((item.ratio || 0) * 100)}%): ${(item.recommended_views || []).join(" / ")}`
          )
        ))
      )
    : null;

  return h(RightPanel, {
    title: "Visualization Strategy",
    icon: h(MiniIcon, { path: "M12 2l4 4-4 4-4-4 4-4zM4 14l4-4 4 4-4 4-4-4zM16 14l4-4 4 4-4 4-4-4z" }),
    children: h("div", { className: "strategy-box" },
      h("div", { className: "strategy-help" },
        h("strong", null, ""),
        h("p", null, ""),
        h("p", null, "交互方式：点击下面的视图卡片会反向驱动中间区域，例如 Heatmap 切换空间图层，Time Series 切到 P95 趋势，Correlation 展示更多相关因子，Semantic Blocks 放大块级语义视图。")
      ),
      h("div", { className: "intent-row" },
        nodes.map(([title, text]) => h("div", { key: title, className: "intent-chip" }, h("strong", null, title), h("span", null, text)))
      ),
      h("div", { className: "robot-node" },
        h("span", null, "规则约束下的 LLM 策略"),
        h("div", { className: "robot-face" },
          h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
            h("path", { d: "M8 8h8a4 4 0 0 1 4 4v4a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4v-4a4 4 0 0 1 4-4zM12 4v4M9 14h.1M15 14h.1M9 17h6" })
          )
        )
      ),
      h("div", { className: "view-row" },
        views.slice(0, 4).map((view) =>
          h("button", {
            type: "button",
            className: normalizeView(view) === strategyFocus ? "view-card active" : "view-card",
            key: view,
            onClick: () => activateView(view),
          },
            h("strong", null, view),
            h("span", null, {
              Heatmap: "空间分布",
              "Time Series": "时间趋势",
              "Correlation View": "影响因子相关性",
              "Semantic Blocks": "块级语义所属",
            }[view] || "证据视图")
          )
        )
      ),
      h("div", { className: "reason-list" },
        h("strong", null, "选择原因"),
        h("ul", null, reasons.map((item) => h("li", { key: item }, item)))
      ),
      h("div", { className: "strategy-contract" },
        h("strong", null, "职责边界"),
        h("p", null, `Label: ${responsibility.label_generation || "Block Semantic Extractor algorithm"}`),
        h("p", null, `Strategy: ${responsibility.strategy_text || "LLM Assistant constrained by semantic-label rule library"}`),
        h("p", null, `Charts: ${responsibility.chart_execution || "Tool execution layer"}`)
      ),
      recommendationPanel
    )
  });
}

function TracePanel({ analysis, traceFocus, onTraceFocus, onExport }) {
  const provenance = analysis?.provenance || {};
  const date = analysis?.selection?.selected_date || analysis?.selection?.latest_date;
  const variable = analysis?.selection?.variable || "PM2.5";
  const steps = [
    ["Data Slice", `${date} | China | ${variable}, Temp, Hum, Pres, Elev`],
    ["Region Mask", "China Boundary (CN_adm_0)"],
    ["Feature Extraction", "GridTensor + Spatial/Temporal Features"],
    ["Tool Execution", "Heatmap | TS Plot | Corr Plot | Block Summary"],
    ["Generated Figure", `LLM-native Figure (ID: fig_${String(date || "").replaceAll("-", "")}_001)`],
  ];
  return h(RightPanel, {
    title: "Trace & Provenance",
    icon: h(MiniIcon, { path: "M12 3v4M12 17v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M3 12h4M17 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z" }),
    children: h("div", { className: "trace-box" },
      h("div", { className: "trace-list" },
        steps.map(([label, text], index) =>
          h("div", { key: label, className: "trace-step" },
            h("button", { type: "button", onClick: () => onTraceFocus(label), className: traceFocus === label ? "active" : "" }, index + 1),
            h("strong", null, label),
            h("p", null, text),
            h("i", null, "✓")
          )
        )
      ),
      h("div", { className: "repro-row" },
        h("span", null, "可复现性："),
        h("strong", null, "100%")
      ),
      h("div", { className: "inline-detail" },
        h("strong", null, traceFocus),
        h("p", null, (provenance.pipeline || []).find((item) => item.toLowerCase().includes(traceFocus.split(" ")[0].toLowerCase())) || `${traceFocus} 已绑定当前分析证据。`)
      ),
      h("button", { className: "export-button", type: "button", onClick: onExport }, "导出分析记录（JSON）"),
      h("small", null, provenance.backend || "gridvis_server.py / Python HTTP API")
    )
  });
}

export default function VisualizationStrategy({
  analysis,
  taskGraph,
  loading,
  selectedLayer,
  onLayer,
  onTrendMetric,
  onCorrelationLimit,
  onBlockSize,
  assistantVote,
  onAssistantVote,
  strategyFocus,
  onStrategyFocus,
  traceFocus,
  onTraceFocus,
  onExport,
}) {
  return h("aside", { className: `right-rail ${loading ? "is-loading" : ""}` },
    h(AssistantPanel, { analysis }),
    h(StrategyPanel, { analysis, taskGraph, selectedLayer, onLayer, onTrendMetric, onCorrelationLimit, onBlockSize, strategyFocus, onStrategyFocus }),
    h(TracePanel, { analysis, traceFocus, onTraceFocus, onExport })
  );
}
