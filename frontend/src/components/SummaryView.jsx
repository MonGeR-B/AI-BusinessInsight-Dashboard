import React from "react";

const SummaryView = ({ data }) => {
  return (
    <div style={{ marginTop: "30px", textAlign: "left" }}>
      <h2>AI Summary</h2>
      <p><strong>Session ID:</strong> {data.session_id || "not available"}</p>
      <p>{data.summary}</p>
      <h3>Detected Date Columns</h3>
      <pre>{JSON.stringify(data.detected_date_columns, null, 2)}</pre>
      <h3>Numeric Stats (sample)</h3>
      <pre>{JSON.stringify(data.numeric_stats, null, 2)}</pre>
    </div>
  );
};

export default SummaryView;
