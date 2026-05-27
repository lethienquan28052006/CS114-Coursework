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
    label.textContent = item.reason || item.feature_label || item.feature || "Feature";

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

const demoProfiles = {
  high: {
    gender: 0,
    Near_Location: 0,
    Partner: 0,
    Promo_friends: 0,
    Phone: 1,
    Contract_period: 1,
    Group_visits: 0,
    Age: 27,
    Avg_additional_charges_total: 62,
    Month_to_end_contract: 0.5,
    Lifetime: 1,
    Avg_class_frequency_total: 1.4,
    Avg_class_frequency_current_month: 0.2,
  },
  medium: {
    gender: 1,
    Near_Location: 1,
    Partner: 0,
    Promo_friends: 1,
    Phone: 1,
    Contract_period: 6,
    Group_visits: 0,
    Age: 42,
    Avg_additional_charges_total: 120,
    Month_to_end_contract: 2,
    Lifetime: 5,
    Avg_class_frequency_total: 2.1,
    Avg_class_frequency_current_month: 1.2,
  },
  low: {
    gender: 1,
    Near_Location: 1,
    Partner: 1,
    Promo_friends: 0,
    Phone: 1,
    Contract_period: 12,
    Group_visits: 1,
    Age: 31,
    Avg_additional_charges_total: 180.5,
    Month_to_end_contract: 8,
    Lifetime: 14,
    Avg_class_frequency_total: 2.8,
    Avg_class_frequency_current_month: 2.6,
  },
};

function fillDemoProfile(profileName) {
  const form = document.getElementById("prediction-form");
  const profile = demoProfiles[profileName];
  if (!form || !profile) return;

  Object.entries(profile).forEach(([name, value]) => {
    const field = form.elements[name];
    if (field) field.value = value;
  });
}

function initDemoProfiles() {
  document.querySelectorAll("[data-demo-profile]").forEach((button) => {
    button.addEventListener("click", () => fillDemoProfile(button.dataset.demoProfile));
  });
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const nextChar = line[index + 1];

    if (char === '"' && inQuotes && nextChar === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      values.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current.trim());
  return values;
}

function renderCsvPreview(headers, rows) {
  const table = document.getElementById("csv-preview-table");
  if (!table) return;

  table.innerHTML = "";
  if (!headers.length) return;

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.slice(0, 5).forEach((row) => {
    const tr = document.createElement("tr");
    headers.forEach((_, index) => {
      const td = document.createElement("td");
      td.textContent = row[index] || "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function initCsvPreview() {
  const input = document.getElementById("csv-file-input");
  const validation = document.getElementById("csv-validation");
  const submitButton = document.getElementById("batch-submit-button");
  const requiredColumns = window.requiredCsvColumns || [];

  if (!input || !validation || !submitButton) return;

  input.addEventListener("change", () => {
    submitButton.disabled = true;
    validation.className = "validation-panel muted-box";
    validation.textContent = "Đang đọc file CSV...";

    const file = input.files && input.files[0];
    if (!file) {
      validation.textContent = "Chưa chọn file CSV.";
      renderCsvPreview([], []);
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      validation.className = "validation-panel validation-error";
      validation.textContent = "File không hợp lệ. Vui lòng chọn file .csv.";
      renderCsvPreview([], []);
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      const lines = text.split(/\r?\n/).filter((line) => line.trim() !== "");
      if (!lines.length) {
        validation.className = "validation-panel validation-error";
        validation.textContent = "File CSV đang rỗng.";
        renderCsvPreview([], []);
        return;
      }

      const headers = parseCsvLine(lines[0]).map((header) => header.replace(/^\uFEFF/, ""));
      const rows = lines.slice(1, 6).map(parseCsvLine);
      const missing = requiredColumns.filter((column) => !headers.includes(column));

      renderCsvPreview(headers, rows);

      if (missing.length) {
        validation.className = "validation-panel validation-error";
        validation.textContent = `Thiếu ${missing.length} cột bắt buộc: ${missing.join(", ")}`;
        return;
      }

      validation.className = "validation-panel validation-success";
      validation.textContent = `File hợp lệ. Tìm thấy ${headers.length} cột và khoảng ${Math.max(lines.length - 1, 0)} dòng dữ liệu.`;
      submitButton.disabled = false;
    };

    reader.onerror = () => {
      validation.className = "validation-panel validation-error";
      validation.textContent = "Không thể đọc file CSV.";
      renderCsvPreview([], []);
    };

    reader.readAsText(file, "utf-8");
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initCsvPreview);
  document.addEventListener("DOMContentLoaded", initDemoProfiles);
} else {
  initCsvPreview();
  initDemoProfiles();
}
