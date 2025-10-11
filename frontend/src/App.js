import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import SummaryView from "./components/SummaryView";
import ForecastChart from "./components/ForecastChart";
import ChatPanel from "./components/ChatPanel";
import "./App.css";

function App() {
  const [summaryData, setSummaryData] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="App" style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1 style={{ color: "#0a66c2" }}>AI Business Insight Dashboard</h1>
      <p>Upload your CSV file to generate AI-driven summaries and forecasts.</p>
      <UploadForm setSummaryData={setSummaryData} setForecastData={setForecastData} setLoading={setLoading} />
      {loading && <p>Processing... please wait.</p>}
      {summaryData && <SummaryView data={summaryData} />}
      {summaryData && <ChatPanel sessionId={summaryData.session_id} />}
      {forecastData && <ForecastChart data={forecastData} />}
    </div>
  );
}

export default App;
