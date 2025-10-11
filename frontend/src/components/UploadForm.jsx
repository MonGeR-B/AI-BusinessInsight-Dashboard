import React, { useState } from "react";
import axios from "axios";

const UploadForm = ({ setSummaryData, setForecastData, setLoading }) => {
  const [file, setFile] = useState(null);
  const backendUrl = "http://127.0.0.1:8000";

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select a CSV file first.");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const summaryRes = await axios.post(`${backendUrl}/ai/summary?use_mock=true`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSummaryData(summaryRes.data);


      const forecastRes = await axios.post(`${backendUrl}/forecast?horizon=3`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setForecastData(forecastRes.data);
    } catch (err) {
      console.error(err);
      alert("Error processing your file. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ margin: "20px 0" }}>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button type="submit" style={{ marginLeft: "10px" }}>Upload & Analyze</button>
    </form>
  );
};

export default UploadForm;
