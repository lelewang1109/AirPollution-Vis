import React from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";
import LLMVisualizationPage from "./pages/LLMVisualizationPage.js";

function App() {
  return React.createElement(LLMVisualizationPage);
}

createRoot(document.getElementById("root")).render(React.createElement(App));
