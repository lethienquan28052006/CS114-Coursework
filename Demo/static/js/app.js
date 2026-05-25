function renderContributionChart(canvasId, items) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !Array.isArray(items)) return;

  const labels = items.map((item) => item.reason || item.feature);
  const values = items.map((item) => item.contribution);
  const colors = values.map((value) => (value >= 0 ? "rgba(251, 113, 133, 0.82)" : "rgba(83, 182, 255, 0.82)"));

  new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Contribution",
          data: values,
          backgroundColor: colors,
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 8,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (context) => {
              const item = items[context.dataIndex];
              return `${item.feature}: ${Number(item.value).toFixed(3)}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.08)" },
          ticks: { color: "#dbe4ff" },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#dbe4ff" },
        },
      },
    },
  });
}
