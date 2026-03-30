require("dotenv").config();

const express = require("express");
const cors = require("cors");
const puppeteer = require("puppeteer");

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const path = require("path");

// serve static files (your index.html, css, js)
app.use(express.static(path.join(__dirname, "public")));

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
  const max = 10;

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

  if (!chemistry.water_activity && !chemistry.moisture_content) {
    missing.push("Post-harvest data missing → stability interpretation limited");
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

  if (!chemistry.total_terpenes) badges.push("Interpretation Limited");

  return badges;
}

/* =========================
   HTML RENDER
========================= */

function renderLandingPage() {
  return `
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Alem COA Intelligence</title>
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: #f7f5ef;
        color: #171714;
      }
      .wrap {
        max-width: 900px;
        margin: 0 auto;
        padding: 48px 24px;
      }
      .card {
        background: #ffffff;
        border: 1px solid #e7e3d8;
        border-radius: 18px;
        padding: 32px;
      }
      h1 {
        margin: 0 0 10px;
        font-size: 36px;
      }
      p {
        color: #4a4a44;
        line-height: 1.65;
      }
      .row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 20px;
      }
      a, button {
        appearance: none;
        border: none;
        background: #0f6847;
        color: #fff;
        text-decoration: none;
        padding: 12px 18px;
        border-radius: 999px;
        font-size: 14px;
        cursor: pointer;
      }
      .secondary {
        background: transparent;
        color: #0f6847;
        border: 1px solid #cfe2d8;
      }
      code {
        background: #f3f0e8;
        padding: 2px 6px;
        border-radius: 6px;
      }
      .mini {
        margin-top: 22px;
        padding: 16px;
        background: #faf8f3;
        border: 1px solid #ece6da;
        border-radius: 14px;
      }
      .mini strong {
        display: block;
        margin-bottom: 8px;
      }
      textarea {
        width: 100%;
        min-height: 180px;
        margin-top: 12px;
        border-radius: 12px;
        border: 1px solid #ddd6c9;
        padding: 14px;
        font-family: monospace;
        font-size: 13px;
      }
      .result {
        margin-top: 16px;
      }
      @media (max-width: 640px) {
        h1 { font-size: 28px; }
        .card { padding: 22px; }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>Alem COA Intelligence</h1>
        <p>Your server is running correctly.</p>
        <p>This version supports:</p>
        <p>
          <code>GET /</code> landing page<br>
          <code>GET /health</code> health check<br>
          <code>POST /generate-report</code> render a report from JSON<br>
          <code>GET /report/test</code> open a sample report<br>
          <code>GET /pdf/test</code> export the sample report as PDF
        </p>

        <div class="row">
          <a href="/report/test">Open test report</a>
          <a href="/pdf/test" class="secondary">Open test PDF</a>
          <a href="/health" class="secondary">Health</a>
        </div>

        <div class="mini">
          <strong>Quick JSON test</strong>
          <p>Paste JSON below and click render. This posts to <code>/generate-report</code>.</p>
          <textarea id="jsonInput">{
  "chemistry": {
    "product_name": "Acai Berry",
    "thc_total": 24.8,
    "total_terpenes": 2.9,
    "top_terpenes": [{}, {}, {}],
    "top_cannabinoids": [{}, {}, {}],
    "water_activity": 0.59
  },
  "contaminants": {
    "pesticides_status": "PASS",
    "heavy_metals_status": "PASS",
    "microbials_status": "PASS",
    "mycotoxins_status": "PASS"
  }
}</textarea>
          <div class="row">
            <button onclick="submitJson()">Render report</button>
          </div>
          <div id="result" class="result"></div>
        </div>
      </div>
    </div>

    <script>
      async function submitJson() {
        const result = document.getElementById("result");
        result.innerHTML = "Rendering...";
        try {
          const raw = document.getElementById("jsonInput").value;
          const parsed = JSON.parse(raw);

          const res = await fetch("/generate-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(parsed)
          });

          const html = await res.text();
          const newWin = window.open();
          newWin.document.open();
          newWin.document.write(html);
          newWin.document.close();

          result.innerHTML = "Opened in a new tab.";
        } catch (err) {
          result.innerHTML = "Error: " + err.message;
        }
      }
    </script>
  </body>
  </html>
  `;
}

function renderReportHTML(data = {}) {
  const chemistry = data.chemistry || {};
  const contaminants = data.contaminants || {};

  const confidence = computeDataConfidence(chemistry, contaminants);
  const missing = buildMissingDataInsights(chemistry, contaminants);
  const badges = buildPremiumBadges(chemistry, contaminants);

  const confidenceClass =
    confidence.label === "High"
      ? "green"
      : confidence.label === "Medium"
      ? "yellow"
      : "red";

  return `
  <!DOCTYPE html>
  <html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Alem COA Report</title>
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: #111;
        color: #fff;
        padding: 40px 20px;
      }
      .container {
        max-width: 900px;
        margin: 0 auto;
      }
      .card {
        background: #1c1c1c;
        padding: 22px;
        margin-bottom: 20px;
        border-radius: 14px;
        border: 1px solid #2b2b2b;
      }
      .badge {
        display: inline-block;
        background: #333;
        padding: 6px 10px;
        margin-right: 6px;
        margin-bottom: 6px;
        border-radius: 999px;
        font-size: 13px;
      }
      .green { color: #4caf50; }
      .yellow { color: #ffc107; }
      .red { color: #f44336; }
      h1, h2 {
        margin-top: 0;
      }
      ul {
        padding-left: 20px;
        margin-bottom: 0;
      }
      .sub {
        color: #bbb;
        margin-top: -8px;
        margin-bottom: 24px;
      }
      .topbar {
        margin-bottom: 24px;
      }
      a {
        color: #9ad9b8;
        text-decoration: none;
      }
      .metric {
        display: inline-block;
        margin-right: 18px;
        margin-bottom: 8px;
      }
    </style>
  </head>

  <body>
    <div class="container">
      <div class="topbar">
        <a href="/">← Back to landing page</a>
      </div>

      <h1>${esc(chemistry.product_name || "Unknown Product")}</h1>
      <div class="sub">Alem COA Intelligence Report</div>

      <div class="card">
        <div class="metric"><strong>THC:</strong> ${chemistry.thc_total || "N/A"}%</div>
        <div class="metric"><strong>Terpenes:</strong> ${chemistry.total_terpenes || "N/A"}%</div>
        <div class="metric"><strong>Water activity:</strong> ${chemistry.water_activity || "N/A"}</div>
      </div>

      <div class="card">
        <h2>Data Confidence</h2>
        <div class="${confidenceClass}">
          <strong>${confidence.label}</strong> (${confidence.pct}%)
        </div>
      </div>

      <div class="card">
        <h2>Premium Signals</h2>
        ${
          badges.length
            ? badges.map((b) => `<span class="badge">${esc(b)}</span>`).join("")
            : `<p>No premium signals detected.</p>`
        }
      </div>

      <div class="card">
        <h2>Missing Data Insights</h2>
        ${
          missing.length > 0
            ? `<ul>${missing.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>`
            : `<p>No major gaps.</p>`
        }
      </div>

      <div class="card">
        <h2>Raw Safety Panel</h2>
        <p><strong>Pesticides:</strong> ${esc(contaminants.pesticides_status || "N/A")}</p>
        <p><strong>Heavy metals:</strong> ${esc(contaminants.heavy_metals_status || "N/A")}</p>
        <p><strong>Microbials:</strong> ${esc(contaminants.microbials_status || "N/A")}</p>
        <p><strong>Mycotoxins:</strong> ${esc(contaminants.mycotoxins_status || "N/A")}</p>
      </div>
    </div>
  </body>
  </html>
  `;
}

/* =========================
   ROUTES
========================= */

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.get("/health", (req, res) => {
  res.json({
    ok: true,
    service: "alem-coa-analyzer-v7.1",
    port: PORT
  });
});

app.post("/generate-report", (req, res) => {
  try {
    const data = req.body || {};
    const html = renderReportHTML(data);
    res.send(html);
  } catch (error) {
    console.error("generate-report error:", error);
    res.status(500).send("Failed to generate report");
  }
});

app.get("/report/test", (req, res) => {
  const mock = {
    chemistry: {
      product_name: "Test Flower",
      thc_total: 24.8,
      total_terpenes: 2.9,
      top_terpenes: [{}, {}, {}],
      top_cannabinoids: [{}, {}, {}],
      water_activity: 0.59
    },
    contaminants: {
      pesticides_status: "PASS",
      heavy_metals_status: "PASS",
      microbials_status: "PASS",
      mycotoxins_status: null
    }
  };

  res.send(renderReportHTML(mock));
});

app.get("/pdf/test", async (req, res) => {
  let browser;
  try {
    const mock = {
      chemistry: {
        product_name: "Test Flower",
        thc_total: 24.8,
        total_terpenes: 2.9,
        top_terpenes: [{}, {}, {}],
        top_cannabinoids: [{}, {}, {}],
        water_activity: 0.59
      },
      contaminants: {
        pesticides_status: "PASS",
        heavy_metals_status: "PASS",
        microbials_status: "PASS",
        mycotoxins_status: null
      }
    };

    const html = renderReportHTML(mock);

    browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"]
    });

    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "networkidle0" });

    const pdf = await page.pdf({
      format: "A4",
      printBackground: true,
      margin: {
        top: "12mm",
        right: "12mm",
        bottom: "12mm",
        left: "12mm"
      }
    });

    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", 'inline; filename="alem-test-report.pdf"');
    res.send(pdf);
  } catch (error) {
    console.error("pdf/test error:", error);
    res.status(500).send("Failed to generate PDF");
  } finally {
    if (browser) {
      try {
        await browser.close();
      } catch (_) {}
    }
  }
});

/* =========================
   START SERVER
========================= */

app.listen(PORT, () => {
  console.log("Server running on port " + PORT);
});