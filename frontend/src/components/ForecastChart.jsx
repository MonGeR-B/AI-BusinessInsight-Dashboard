import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, Title, Tooltip, Legend);

const ForecastChart = ({ data }) => {
  const preds = data.predictions || [];
  const labels = preds.map(p => p.date || p.index);
  const values = preds.map(p => p.prediction);

  const chartData = {
    labels,
    datasets: [
      {
        label: `Forecast for ${data.target}`,
        data: values,
        borderColor: "rgba(75,192,192,1)",
        fill: false,
      },
    ],
  };

  return (
    <div style={{ marginTop: "40px" }}>
      <h2>Forecast Chart ({data.target})</h2>
      <Line data={chartData} />
    </div>
  );
};

export default ForecastChart;
