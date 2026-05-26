function renderContributionChart(containerId, items) {
  const container = document.getElementById(containerId);
  if (!container || !Array.isArray(items) || items.length === 0) return;

  const maxAbs = Math.max(...items.map((item) => Math.abs(Number(item.contribution) || 0)), 0.001);
  container.innerHTML = "";

  items.forEach((item) => {
    const contribution = Number(item.contribution) || 0;
    const width = Math.max(3, Math.round((Math.abs(contribution) / maxAbs) * 100));
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = item.reason || item.feature || "Feature";

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = `bar-fill ${contribution >= 0 ? "bar-positive" : "bar-negative"}`;
    fill.style.width = `${width}%`;
    track.appendChild(fill);

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = contribution.toFixed(3);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  });
}
