import React from "https://esm.sh/react@18.2.0";

const h = React.createElement;

const TABS = ["Data", "Analysis", "Visualization", "Trace", "Report"];
const METHOD_STEPS = [
  ["Data Access", "NetCDF catalog"],
  ["Grid Representation", "features"],
  ["Block Semantics", "labels"],
  ["LLM Planning", "strategy"],
  ["LLM-native Figure", "views"],
];

function SparkIcon() {
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
    h("path", { d: "M12 2l1.7 6.2L20 10l-6.3 1.8L12 18l-1.7-6.2L4 10l6.3-1.8L12 2z" }),
    h("path", { d: "M19 15l.8 2.7L22 18.5l-2.2.7L19 22l-.8-2.8-2.2-.7 2.2-.8L19 15z" })
  );
}

function PlayIcon() {
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
    h("path", { d: "M8 5v14l11-7L8 5z" })
  );
}

function UserIcon() {
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
    h("path", { d: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" }),
    h("path", { d: "M4 21a8 8 0 0 1 16 0" })
  );
}

export default function QueryInput({ query, onQuery, onRun, loading, apiState, activeTab, onTab }) {
  const statusLabel = apiState === "connected" ? "API Active" : apiState === "offline" ? "Local Demo" : "Fallback";
  const quickPrompts = [
    "分析中国区域 PM2.5 空间分布",
    "识别 block 语义标签和证据",
    "根据标签分布推荐可视化策略",
  ];
  return h("header", { className: "top-shell chat-shell" },
    h("div", { className: "top-strip" },
      h("div", { className: "brand" },
        h("div", { className: "brand-mark", "aria-hidden": "true" },
          h("span", null),
          h("span", null),
          h("span", null),
          h("span", null)
        ),
        h("div", null,
          h("h1", null, "AirPollution-Vis")
        )
      ),
      h("nav", { className: "nav-tabs", "aria-label": "分析阶段" },
        TABS.map((tab) =>
          h("button", {
            key: tab,
            type: "button",
            className: tab === activeTab ? "active" : "",
            onClick: () => onTab(tab),
          }, tab)
        )
      ),
      h("div", { className: "user-status", title: statusLabel },
        h("span", { className: `status-dot ${apiState}` }),
        h(UserIcon)
      )
    ),
    h("div", { className: "method-pipeline", "aria-label": "AirPollution-Vis 方法流程" },
      METHOD_STEPS.map(([title, subtitle], index) =>
        h("div", { key: title, className: "pipeline-step" },
          h("span", null, index + 1),
          h("strong", null, title),
          h("em", null, subtitle)
        )
      )
    ),
    h("section", { className: "chat-window", "aria-label": "LLM 对话式任务窗口" },
      h("div", { className: "chat-thread" },
        h("div", { className: "chat-bubble assistant" },
          h("strong", null, "LLM Assistant"),
          h("p", null, "把任务解析为变量、区域、时间、语义标签读取，并得到可视化策略，进行解释证据。")
        ),
        h("div", { className: "chat-bubble user" },
          h("strong", null, "Current task"),
          h("p", null, query || "请输入一个网格数据分析任务")
        )
      ),
      h("div", { className: "quick-prompts" },
        quickPrompts.map((prompt) =>
          h("button", { key: prompt, type: "button", onClick: () => onQuery(prompt) }, prompt)
        )
      ),
      h("div", { className: "chat-composer" },
        h("textarea", {
          value: query,
          onChange: (event) => onQuery(event.target.value),
          onKeyDown: (event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") onRun();
          },
          rows: 2,
          placeholder: "输入任务，例如：分析 2000-01-01 中国区域 PM2.5 空间分布，并解释 H/G/B/N 标签分布",
          "aria-label": "对话式自然语言分析任务",
        }),
        h("button", { className: "run-button", type: "button", onClick: onRun, disabled: loading },
          loading ? h(SparkIcon) : h(PlayIcon),
          loading ? "Analyzing" : "Run"
        )
      )
    )
  );
}
