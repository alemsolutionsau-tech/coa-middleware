require("dotenv").config();

const express = require("express");
const cors = require("cors");
const puppeteer = require("puppeteer");

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const PORT = process.env.PORT || 3000;

function esc(v = "") {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* =========================
   V7.1 INTELLIGENCE LAYER
========================= */

function computeDataConfidence(chemistry = {}, contaminants = {}) {
  let points = 0;
  let max = 10;

  if (chemistry.thc_total) points++;
  if (chemistry.total_terpenes) points++;
  if ((chemistry.top_terpenes || []).length >= 3) points++;
  if ((chemistry.top_cannabinoids || []).length >= 3) points++;

  if (contaminants.pesticides_status) points++;
  if (contaminants.heavy_metals_status) points++;
  if (contaminants.microbials_status) points++;
  if (contaminants.mycotoxins_status) points++;

  const pct = Math.round((points / max) * 100);

  let label = "Low";
  if (pct >= 75) label = "High";
  else if (pct >= 45) label = "Medium";

  return { pct, label };
}

function buildMissingDataInsights(chemistry = {}, contaminants = {}) {
  const missing = [];

  if (!chemistry.total_terpenes) {
    missing.push("Terpene data missing → effect interpretation limited");
  }

  if (!contaminants.pesticides_status) {
    missing.push("Pesticides not reported → safety unknown");
  }

  if (!contaminants.heavy_metals_status) {
    missing.push("Heavy metals missing → contamination risk unknown");
  }

  if (!contaminants.microbials_status) {
    missing.push("Microbial data missing → pathogen safety unknown");
  }

  if (!contaminants.mycotoxins_status) {
    missing.push("Mycotoxins missing → toxin risk unknown");
  }

  return missing;
}

function buildPremiumBadges(chemistry = {}, contaminants = {}) {
  const badges = [];

  const terp = Number(chemistry.total_terpenes || 0);

  if (terp > 2.5) badges.push("Terpene Richness");

  const fullSafety =
    contaminants.pesticides_status &&
    contaminants.heavy_metals_status &&
    contaminants.microbials_status &&
    contaminants.mycotoxins_status;

  if (fullSafety) badges.push("Complete Safety Panel");
  else badges.push("Incomplete COA");

  return badges;
}

/* =========================
   HTML RENDER
========================= */

function renderReportHTML(data = {}) {
  const chemistry = data.chemistry || {};
  const contaminants = data.contaminants || {};

  const confidence = computeDataConfidence(chemistry, contaminants);
  const missing = buildMissingDataInsights(chemistry, contaminants);
  const badges = buildPremiumBadges(chemistry, contaminants);

  return `
  <html>
  <head>
    <title>Alem COA Report</title>
    <style>
      body { font-family: Arial; background:#111; color:#fff; padding:40px;}
      .card { background:#1c1c1c; padding:20px; margin-bottom:20px; border-radius:12px;}
      .badge { background:#333; padding:6px 10px; margin-right:6px; border-radius:8px;}
      .green { color:#4caf50;}
      .yellow { color:#ffc107;}
      .red { color:#f44336;}
    </style>
  </head>

  <body>

    <h1>${esc(chemistry.product_name || "Unknown Product")}</h1>

    <div class="card">
      <strong>THC:</strong> ${chemistry.thc_total || "N/A"}% <br/>
      <strong>Terpenes:</strong> ${chemistry.total_terpenes || "N/A"}%
    </div>

    <div class="card">
      <h2>Data Confidence</h2>
      <div class="${confidence.label === "High" ? "green" : confidence.label === "Medium" ? "yellow" : "red"}">
        ${confidence.label} (${confidence.pct}%)
      </div>
    </div>

    <div class="card">
      <h2>Premium Signals</h2>
      ${badges.map(b => `<span class="badge">${b}</span>`).join("")}
    </div>

    <div class="card">
      <h2>Missing Data Insights</h2>
      ${
        missing.length > 0
          ? `<ul>${missing.map(m => `<li>${m}</li>`).join("")}</ul>`
          : `<p>No major gaps</p>`
      }
    </div>

  </body>
  </html>
  `;
}

/* =========================
   ROUTES
========================= */

app.get("/", (req, res) => {
  res.send("Alem COA Analyzer V7.1 Running");
});

app.post("/generate-report", (req, res) => {
  const data = req.body || {};
  const html = renderReportHTML(data);
  res.send(html);
});

app.get("/report/test", (req, res) => {
  const mock = {
    chemistry: {
      product_name: "Test Flower",
      thc_total: 24.8,
      total_terpenes: 2.9,
      top_terpenes: [{}, {}, {}],
      top_cannabinoids: [{}, {}, {}],
    },
    contaminants: {
      pesticides_status: "PASS",
      heavy_metals_status: "PASS",
      microbials_status: "PASS",
      mycotoxins_status: null,
    }
  };

  res.send(renderReportHTML(mock));
});

/* =========================
   START SERVER
========================= */

app.listen(PORT, () => {
  console.log("Server running on port " + PORT);
});