require("dotenv").config();

const express = require("express");
const cors = require("cors");
const puppeteer = require("puppeteer");
const axios = require("axios");
const OpenAI = require("openai");
const {
  DocumentAnalysisClient,
  AzureKeyCredential,
} = require("@azure/ai-form-recognizer");

const supabase = require("./supabase");

const app = express();

app.use(cors());
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true }));

const PORT = process.env.PORT || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-5";
const MAX_OCR_CHARS_FOR_OPENAI = Number(
  process.env.MAX_OCR_CHARS_FOR_OPENAI || 12000
);
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 20000);

if (
  !process.env.AZURE_DOC_INTELLIGENCE_ENDPOINT ||
  !process.env.AZURE_DOC_INTELLIGENCE_KEY
) {
  throw new Error("Missing Azure Document Intelligence environment variables");
}

if (!process.env.OPENAI_API_KEY) {
  throw new Error("Missing OPENAI_API_KEY in .env");
}

const azureClient = new DocumentAnalysisClient(
  process.env.AZURE_DOC_INTELLIGENCE_ENDPOINT,
  new AzureKeyCredential(process.env.AZURE_DOC_INTELLIGENCE_KEY)
);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const extractionPrompt = `
You MUST return a JSON object. Do not think step-by-step. Do not hide output. Respond immediately.

You are the Alem Solutions COA extraction engine.

Task:
Read OCR text from a cannabis Certificate of Analysis and return EXACTLY ONE valid JSON object.

CRITICAL RULES:
- Return valid JSON only.
- No markdown.
- No comments.
- No trailing commas.
- No prose before or after the JSON.
- If a field is missing, return "" for strings, [] for arrays.
- If uncertain, leave the field empty instead of guessing.
- Keep arrays short and useful.
- top_cannabinoids: max 8 items
- top_terpenes: max 8 items
- positive_flags: max 6 items
- warning_flags: max 6 items
- scientific_references must be a short plain-text summary, not citations list
- Output must be parseable by JSON.parse()

Return this exact shape:

{
  "product_name": "",
  "batch_number": "",
  "coa_report_date": "",
  "product_type": "",
  "laboratory_name": "",
  "opening_statement": "",
  "overall_score": "",
  "thc_total": "",
  "cbd_total": "",
  "total_terpenes": "",
  "minor_cannabinoids": "",
  "contaminant_overview": "",
  "lab_quality_summary": "",
  "scientific_references": "",
  "top_cannabinoids": [
    {
      "name": "",
      "value": "",
      "unit": "",
      "notes": ""
    }
  ],
  "top_terpenes": [
    {
      "name": "",
      "value": "",
      "unit": ""
    }
  ],
  "positive_flags": [""],
  "warning_flags": [""]
}
`;

function esc(v = "") {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sanitizeFileName(name = "") {
  return String(name || "")
    .replace(/[^a-zA-Z0-9-_.]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function numberFromPercentString(v) {
  if (v === null || v === undefined) return 0;
  const match = String(v).match(/-?\\d+(\\.\\d+)?/);
  return match ? Number(match[0]) : 0;
}

function getSafeArray(arr) {
  return Array.isArray(arr) ? arr : [];
}

function detectMimeType(url = "", headerContentType = "") {
  const lowerUrl = String(url || "").toLowerCase();
  const lowerHeader = String(headerContentType || "").toLowerCase();

  if (lowerHeader.includes("application/pdf") || lowerUrl.endsWith(".pdf")) {
    return "application/pdf";
  }
  if (lowerHeader.includes("image/png") || lowerUrl.endsWith(".png")) {
    return "image/png";
  }
  if (
    lowerHeader.includes("image/jpeg") ||
    lowerUrl.endsWith(".jpg") ||
    lowerUrl.endsWith(".jpeg")
  ) {
    return "image/jpeg";
  }
  if (
    lowerHeader.includes("image/tiff") ||
    lowerUrl.endsWith(".tif") ||
    lowerUrl.endsWith(".tiff")
  ) {
    return "image/tiff";
  }

  return "application/octet-stream";
}

function getBaseUrl(req) {
  const proto =
    req.headers["x-forwarded-proto"] ||
    req.protocol ||
    "https";
  return `${proto}://${req.get("host")}`;
}

function buildReportUrl(req, documentId) {
  return `${getBaseUrl(req)}/report/${documentId}`;
}

function parseIncomingBody(rawBody) {
  if (rawBody === undefined || rawBody === null) {
    throw new Error("Request body is empty");
  }

  let payload = rawBody;

  if (typeof rawBody === "string") {
    const trimmed = rawBody.trim();

    if (!trimmed || trimmed === "null" || trimmed === "undefined") {
      throw new Error("Request body is empty or null");
    }

    try {
      payload = JSON.parse(trimmed);
    } catch (err) {
      throw new Error(`Request body is not valid JSON: ${err.message}`);
    }
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Parsed JSON is null or not a valid object");
  }

  if (payload.report_json && typeof payload.report_json === "object") {
    return {
      fileName:
        sanitizeFileName(
          payload.file_name ||
            payload.report_json?.product_name ||
            `coa-${Date.now()}`
        ) + ".pdf",
      data: payload.report_json,
    };
  }

  if (
    payload.report_json_string &&
    typeof payload.report_json_string === "string"
  ) {
    let parsedInner;
    try {
      parsedInner = JSON.parse(payload.report_json_string);
    } catch (err) {
      throw new Error(`report_json_string is not valid JSON: ${err.message}`);
    }

    if (
      !parsedInner ||
      typeof parsedInner !== "object" ||
      Array.isArray(parsedInner)
    ) {
      throw new Error("report_json_string parsed to null or invalid object");
    }

    return {
      fileName:
        sanitizeFileName(
          payload.file_name || parsedInner?.product_name || `coa-${Date.now()}`
        ) + ".pdf",
      data: parsedInner,
    };
  }

  if (payload.product_name || payload.top_cannabinoids || payload.top_terpenes) {
    return {
      fileName:
        sanitizeFileName(
          payload.file_name || payload.product_name || `coa-${Date.now()}`
        ) + ".pdf",
      data: payload,
    };
  }

  throw new Error("Missing valid report_json or report_json_string");
}

function renderList(items = [], emptyText = "Not reported") {
  if (!Array.isArray(items) || !items.length) {
    return `<div class="muted">${esc(emptyText)}</div>`;
  }

  return `<ul>${items
    .map((x) => `<li>${esc(typeof x === "string" ? x : JSON.stringify(x))}</li>`)
    .join("")}</ul>`;
}

function renderMetric(label, value) {
  return `
    <div class="metric">
      <div class="metric-label">${esc(label)}</div>
      <div class="metric-value">${esc(value || "Not reported")}</div>
    </div>
  `;
}

function renderCannabinoidTable(items = []) {
  const rows = getSafeArray(items);

  if (!rows.length) {
    return `<div class="muted">No cannabinoid rows reported.</div>`;
  }

  return `
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Value</th>
          <th>Unit</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (item) => `
            <tr>
              <td>${esc(item?.name)}</td>
              <td>${esc(item?.value)}</td>
              <td>${esc(item?.unit)}</td>
              <td>${esc(item?.notes)}</td>
            </tr>
          `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderTerpeneTable(items = []) {
  const rows = getSafeArray(items);

  if (!rows.length) {
    return `<div class="muted">No terpene rows reported.</div>`;
  }

  return `
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Value</th>
          <th>Unit</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (item) => `
            <tr>
              <td>${esc(item?.name)}</td>
              <td>${esc(item?.value)}</td>
              <td>${esc(item?.unit)}</td>
            </tr>
          `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderActionBar({ documentId, pdfUrl }) {
  if (!documentId) return "";

  return `
    <div class="action-bar">
      <button id="generatePdfBtn" class="btn btn-primary">Generate PDF</button>
      ${
        pdfUrl
          ? `<a class="btn btn-secondary" href="${esc(pdfUrl)}" target="_blank" rel="noopener noreferrer">Open current PDF</a>`
          : ""
      }
      <span id="pdfStatus" class="action-status"></span>
    </div>

    <script>
      (function () {
        const btn = document.getElementById("generatePdfBtn");
        const status = document.getElementById("pdfStatus");

        if (!btn) return;

        btn.addEventListener("click", async function () {
          try {
            btn.disabled = true;
            btn.textContent = "Generating PDF...";
            status.textContent = "";

            const response = await fetch("/generate-pdf/${documentId}", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              }
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
              throw new Error(result.error || "Failed to generate PDF");
            }

            status.textContent = "PDF ready";
            btn.textContent = "Generate PDF again";

            if (result.pdf_url) {
              window.open(result.pdf_url, "_blank");
            }
          } catch (err) {
            console.error(err);
            status.textContent = err.message || "PDF generation failed";
            btn.textContent = "Generate PDF";
          } finally {
            btn.disabled = false;
          }
        });
      })();
    </script>
  `;
}

function renderReportHTML(data = {}, options = {}) {
  const topCannabinoids = getSafeArray(data?.top_cannabinoids);
  const topTerpenes = getSafeArray(data?.top_terpenes);
  const positiveFlags = getSafeArray(data?.positive_flags);
  const warningFlags = getSafeArray(data?.warning_flags);

  const thcNumber = numberFromPercentString(data?.thc_total);
  const cbdNumber = numberFromPercentString(data?.cbd_total);
  const terpNumber = numberFromPercentString(data?.total_terpenes);

  const productName = data?.product_name || "Cannabis Product";
  const productType = data?.product_type || "Product type not reported";
  const batchNumber = data?.batch_number || "Not reported";
  const laboratoryName = data?.laboratory_name || "Lab not reported";
  const reportDate = data?.coa_report_date || "Date not reported";
  const subtitle =
    data?.opening_statement ||
    "Chemistry translated into practical intelligence from the uploaded COA.";

  const overallScoreText = data?.overall_score || "Interpretive summary pending";

  const regulatoryStatus = warningFlags.length
    ? "Inconclusive"
    : positiveFlags.length
    ? "Approved"
    : "Under Review";

  const statusClass =
    regulatoryStatus === "Approved"
      ? "approved"
      : regulatoryStatus === "Inconclusive"
      ? "inconclusive"
      : "review";

  const dominantTerpene = topTerpenes[0]?.name || "Unknown";
  const secondTerpene = topTerpenes[1]?.name || "";
  const thirdTerpene = topTerpenes[2]?.name || "";

  const aromaticProfile = topTerpenes.length
    ? topTerpenes
        .slice(0, 4)
        .map((x) => x?.name)
        .filter(Boolean)
        .join(" • ")
    : "Aromatic profile not available";

  const derivedProfileTag =
    thcNumber >= 24
      ? "High potency"
      : thcNumber >= 18
      ? "Moderate-high potency"
      : thcNumber > 0
      ? "Moderate potency"
      : "Profile pending";

  const experienceDirection =
    dominantTerpene.toLowerCase().includes("terpinolene")
      ? "More uplifting / mentally active leaning"
      : dominantTerpene.toLowerCase().includes("myrcene")
      ? "More body-heavy / calming leaning"
      : dominantTerpene.toLowerCase().includes("limonene")
      ? "Brighter / mood-forward leaning"
      : dominantTerpene.toLowerCase().includes("caryophyllene")
      ? "Balanced / spicy profile"
      : "General chemistry-led profile";

  const thcBand =
    thcNumber >= 24 ? "THC-H" : thcNumber >= 18 ? "THC-M" : thcNumber > 0 ? "THC-L" : "THC-?";
  const terpBand =
    terpNumber >= 3 ? "TERP-H" : terpNumber >= 1.5 ? "TERP-M" : terpNumber > 0 ? "TERP-L" : "TERP-?";
  const cbdBand =
    cbdNumber >= 5 ? "CBD-H" : cbdNumber >= 1 ? "CBD-M" : cbdNumber > 0 ? "CBD-L" : "CBD-LOW";

  const fingerprintTags = [
    thcBand,
    terpBand,
    cbdBand,
    `${String(dominantTerpene || "UNKNOWN").toUpperCase().replace(/[^A-Z0-9]+/g, "-")}-DOM`,
    data?.minor_cannabinoids ? "MC+" : "MC-LIMITED",
    warningFlags.length ? "REVIEW" : "CLEAN",
  ];

  const dataConfidenceScore = Math.max(
    52,
    Math.min(
      98,
      50 +
        (topCannabinoids.length ? 12 : 0) +
        (topTerpenes.length ? 12 : 0) +
        (data?.thc_total ? 8 : 0) +
        (data?.total_terpenes ? 8 : 0) +
        (data?.contaminant_overview ? 4 : 0) +
        (data?.lab_quality_summary ? 4 : 0)
    )
  );

  const cannabinoidVisibility = topCannabinoids.length ? "High" : data?.thc_total || data?.cbd_total ? "Moderate" : "Limited";
  const terpeneVisibility = topTerpenes.length ? "High" : data?.total_terpenes ? "Moderate" : "Limited";

  const completenessItems = [
    {
      label: "Cannabinoids",
      value:
        topCannabinoids.length || data?.thc_total || data?.cbd_total
          ? "Complete"
          : "Limited",
      className:
        topCannabinoids.length || data?.thc_total || data?.cbd_total
          ? "good"
          : "warn",
    },
    {
      label: "Terpenes",
      value: topTerpenes.length || data?.total_terpenes ? "Complete" : "Limited",
      className: topTerpenes.length || data?.total_terpenes ? "good" : "warn",
    },
    {
      label: "Contaminants",
      value: data?.contaminant_overview ? "Reported" : "Missing",
      className: data?.contaminant_overview ? "good" : "warn",
    },
    {
      label: "Lab Metadata",
      value:
        data?.laboratory_name && data?.coa_report_date ? "Complete" : "Partial",
      className:
        data?.laboratory_name && data?.coa_report_date ? "good" : "warn",
    },
  ];

  const coaQualityScore = Math.max(
    4,
    Math.min(
      10,
      (topCannabinoids.length ? 2 : 0) +
        (topTerpenes.length ? 2 : 0) +
        (data?.contaminant_overview ? 2 : 0) +
        (data?.lab_quality_summary ? 2 : 0) +
        (data?.laboratory_name ? 1 : 0) +
        (data?.coa_report_date ? 1 : 0)
    )
  );

  // Patient-safe AI summaries
  const patientSafeSummary = [
    thcNumber >= 24
      ? "This appears to be a high-THC product and may feel strong for people with lower tolerance."
      : thcNumber >= 18
      ? "This appears to be a moderate-to-high THC product."
      : thcNumber > 0
      ? "This appears to be a more moderate potency product."
      : "Potency could not be clearly determined from the extracted data.",
    dominantTerpene && dominantTerpene !== "Unknown"
      ? `${dominantTerpene} appears to be the dominant terpene in this sample.`
      : "A dominant terpene could not be clearly identified.",
    terpNumber >= 2
      ? "Visible terpene content suggests a more expressive aromatic profile."
      : terpNumber > 0
      ? "Terpene content appears present but not especially high."
      : "Terpene intensity could not be clearly assessed.",
    warningFlags.length
      ? "Some parts of the report should be read cautiously because certain fields were limited or flagged."
      : "No major caution flags were detected in the structured output.",
  ];

  const clinicianInsights = [
    cbdNumber <= 1
      ? "Low CBD suggests limited direct THC modulation."
      : "Visible CBD may contribute some modulation of THC-dominant effects.",
    thcNumber >= 24
      ? "High THC concentration warrants tolerance-aware prescribing considerations."
      : "THC potency appears less extreme than ultra-high THC flower products.",
    data?.minor_cannabinoids
      ? "Minor cannabinoids are present and may add profile complexity."
      : "Minor cannabinoid depth was not clearly reported.",
    data?.contaminant_overview
      ? "Contaminant summary was reported and should be reviewed alongside the source COA."
      : "Contaminant interpretation is limited because no structured overview was extracted.",
  ];

  const industryInsights = [
    dominantTerpene && dominantTerpene !== "Unknown"
      ? `${dominantTerpene}-led chemistry can support differentiated market positioning.`
      : "Terpene dominance is not strong enough in the current extraction to support a clear positioning angle.",
    terpNumber >= 2.5
      ? "Stronger terpene density supports premium aromatic positioning."
      : "Moderate terpene density may require positioning around effect profile rather than aroma intensity alone.",
    data?.minor_cannabinoids
      ? "Minor cannabinoids add product-story depth for brand and category differentiation."
      : "Limited visible minor cannabinoid depth may reduce differentiation narrative.",
    warningFlags.length
      ? "Commercial claims should be conservative until flagged issues are clarified."
      : "Structured output suggests cleaner messaging conditions for educational positioning.",
  ];

  const enthusiastInsights = [
    aromaticProfile !== "Aromatic profile not available"
      ? `Top aromatic signals: ${aromaticProfile}.`
      : "Aromatic structure could not be confidently extracted.",
    experienceDirection,
    thirdTerpene
      ? `Layered terpene architecture includes ${dominantTerpene}, ${secondTerpene}, and ${thirdTerpene}.`
      : secondTerpene
      ? `Leading terpene pairing includes ${dominantTerpene} and ${secondTerpene}.`
      : `Primary terpene signal centers on ${dominantTerpene}.`,
    data?.minor_cannabinoids
      ? "Minor cannabinoids suggest added complexity beyond THC alone."
      : "Profile appears more driven by the major compounds captured in the report.",
  ];

  // Benchmarking engine
  const benchmarkCards = [
    {
      label: "Potency Position",
      value:
        thcNumber >= 24
          ? "High-tier"
          : thcNumber >= 18
          ? "Mid-high"
          : thcNumber > 0
          ? "Moderate"
          : "Unknown",
    },
    {
      label: "Aroma Density",
      value:
        terpNumber >= 3
          ? "High"
          : terpNumber >= 1.5
          ? "Moderate"
          : terpNumber > 0
          ? "Light"
          : "Unknown",
    },
    {
      label: "Differentiation",
      value:
        data?.minor_cannabinoids && topTerpenes.length >= 3
          ? "Moderate-high"
          : topTerpenes.length >= 2
          ? "Moderate"
          : "Limited",
    },
    {
      label: "Database Readiness",
      value:
        topCannabinoids.length && topTerpenes.length ? "Benchmark-ready" : "Partial",
    },
  ];

  // Compare mode (UI only for now)
  const compareCards = [
    {
      label: "vs High-THC flower",
      value:
        thcNumber >= 24
          ? "Aligned"
          : thcNumber >= 18
          ? "Slightly lighter"
          : "Below category",
    },
    {
      label: "vs aromatic premium flower",
      value:
        terpNumber >= 3
          ? "Aligned"
          : terpNumber >= 1.5
          ? "Moderate"
          : "Below category",
    },
    {
      label: "vs differentiated chemotype",
      value:
        data?.minor_cannabinoids || topTerpenes.length >= 4
          ? "More distinctive"
          : "More standard",
    },
    {
      label: "vs data-rich COA",
      value:
        topCannabinoids.length && topTerpenes.length && data?.contaminant_overview
          ? "Strong"
          : "Partial",
    },
  ];

  const similaritySignals = [
    dominantTerpene && dominantTerpene !== "Unknown"
      ? `Chemotype anchor: ${dominantTerpene}-dominant`
      : "Chemotype anchor unavailable",
    thcNumber >= 24
      ? "Similarity cluster: higher-THC flower"
      : thcNumber >= 18
      ? "Similarity cluster: standard THC flower"
      : "Similarity cluster: lighter potency flower",
    terpNumber >= 2
      ? "Comparable set: aromatic-forward profiles"
      : "Comparable set: lighter terpene expression",
  ];

  const renderPillList = (items = [], emptyText = "Not reported") => {
    if (!Array.isArray(items) || !items.length) {
      return `<div class="muted">${esc(emptyText)}</div>`;
    }
    return `
      <div class="pill-wrap">
        ${items.map((item) => `<span class="pill">${esc(item)}</span>`).join("")}
      </div>
    `;
  };

  const renderCompoundPills = (items = [], type = "cannabinoid") => {
    if (!Array.isArray(items) || !items.length) {
      return `<div class="muted">No ${esc(type)} data reported.</div>`;
    }

    return `
      <div class="compound-grid">
        ${items
          .map((item) => {
            const name = item?.name || "";
            const value = item?.value || "";
            const unit = item?.unit || "";
            const notes = item?.notes || "";

            return `
              <div class="compound-card">
                <div class="compound-name">${esc(name || "Unnamed")}</div>
                <div class="compound-value">${esc([value, unit].filter(Boolean).join(" ")) || "—"}</div>
                ${
                  notes
                    ? `<div class="compound-note">${esc(notes)}</div>`
                    : ""
                }
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  };

  const renderFullTerpeneList = (items = []) => {
    if (!Array.isArray(items) || !items.length) {
      return `<div class="muted">No terpene list reported.</div>`;
    }

    return `
      <div class="terpene-list-grid">
        ${items
          .map(
            (item) => `
            <div class="terpene-list-item">
              <span>${esc(item?.name || "Unnamed")}</span>
              <strong>${esc([item?.value, item?.unit].filter(Boolean).join(" ")) || "—"}</strong>
            </div>
          `
          )
          .join("")}
      </div>
    `;
  };

  const renderBar = (label, valueText, widthPercent, tone = "") => `
    <div class="bar-row">
      <div class="bar-row-head">
        <span>${esc(label)}</span>
        <strong>${esc(valueText || "0")}</strong>
      </div>
      <div class="bar-shell ${tone}">
        <span style="width:${Math.max(0, Math.min(100, widthPercent))}%"></span>
      </div>
    </div>
  `;

  const rawJson = esc(JSON.stringify(data, null, 2));

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${esc(productName)}</title>
<style>
  :root {
    --bg: #f5f3ef;
    --panel: #ffffff;
    --panel-2: #fbfaf7;
    --line: #e5dfd5;
    --text: #191816;
    --muted: #706c65;
    --green: #1f3d2b;
    --green-2: #2e5c42;
    --soft-green: #e7eee8;
    --teal: #6b9f97;
    --gold: #b5934c;
    --danger: #a55757;
    --warn: #a67e2d;
    --shadow: 0 14px 38px rgba(32, 31, 28, 0.06);
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0;
    padding: 0;
    background:
      radial-gradient(circle at top right, rgba(31,61,43,0.05), transparent 22%),
      radial-gradient(circle at top left, rgba(107,159,151,0.05), transparent 22%),
      linear-gradient(180deg, #f7f5f1 0%, #f4f1eb 100%);
    color: var(--text);
    font-family: Inter, Arial, Helvetica, sans-serif;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  .shell {
    max-width: 1380px;
    margin: 0 auto;
    padding: 24px 20px 42px;
  }

  .action-bar {
    position: sticky;
    top: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 18px;
    padding: 14px 16px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow);
  }

  .btn {
    appearance: none;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #fff;
    color: var(--text);
  }

  .btn-primary {
    background: linear-gradient(90deg, var(--green), var(--green-2));
    color: #fff;
    border-color: var(--green);
  }

  .btn-secondary {
    background: rgba(255,255,255,0.88);
    color: var(--text);
    border: 1px solid var(--line);
  }

  .btn:disabled {
    opacity: 0.65;
    cursor: wait;
  }

  .action-status {
    color: var(--muted);
    font-size: 14px;
  }

  .hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 30px;
    padding: 32px 28px 26px;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.98)),
      radial-gradient(circle at right top, rgba(31,61,43,0.08), transparent 30%);
    box-shadow: var(--shadow);
  }

  .hero::after {
    content: "";
    position: absolute;
    right: -90px;
    top: -90px;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(31,61,43,0.08), transparent 70%);
  }

  .hero-top {
    display: flex;
    justify-content: space-between;
    gap: 24px;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .brand-mark {
    font-size: 12px;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--green);
    margin-bottom: 10px;
    font-weight: 700;
  }

  .hero h1 {
    margin: 0;
    font-size: 48px;
    line-height: 1.02;
    letter-spacing: -0.03em;
    color: var(--text);
  }

  .hero-subtitle {
    margin-top: 14px;
    max-width: 820px;
    color: #3f3b35;
    font-size: 16px;
    line-height: 1.75;
  }

  .hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: #fff;
    color: var(--text);
    font-size: 13px;
    font-weight: 700;
  }

  .status-badge.approved {
    background: rgba(31,61,43,0.08);
    border-color: rgba(31,61,43,0.20);
    color: var(--green);
  }

  .status-badge.inconclusive {
    background: rgba(181,147,76,0.10);
    border-color: rgba(181,147,76,0.24);
    color: #876822;
  }

  .status-badge.review {
    background: rgba(107,159,151,0.10);
    border-color: rgba(107,159,151,0.24);
    color: #456d67;
  }

  .hero-score {
    min-width: 260px;
    border: 1px solid var(--line);
    border-radius: 26px;
    padding: 18px 18px 16px;
    background: #fff;
    text-align: center;
    box-shadow: 0 8px 24px rgba(32, 31, 28, 0.04);
  }

  .hero-score-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 8px;
  }

  .hero-score-value {
    font-size: 24px;
    font-weight: 800;
    color: var(--green);
    line-height: 1.2;
  }

  .hero-score-note {
    margin-top: 8px;
    font-size: 13px;
    line-height: 1.5;
    color: #5b5750;
  }

  .section {
    margin-top: 22px;
  }

  .section-title {
    margin: 0 0 14px;
    font-size: 30px;
    color: var(--text);
    letter-spacing: -0.03em;
  }

  .section-title.small {
    font-size: 24px;
  }

  .section-sub {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 16px;
  }

  .card {
    border: 1px solid var(--line);
    border-radius: 24px;
    background: var(--panel);
    padding: 20px;
    box-shadow: var(--shadow);
  }

  .card-soft {
    background: var(--panel-2);
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1.08fr 0.92fr;
    gap: 18px;
  }

  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }

  .meta-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
  }

  .meta-item {
    border-bottom: 1px solid var(--line);
    padding-bottom: 10px;
  }

  .meta-label {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }

  .meta-value {
    font-size: 15px;
    font-weight: 700;
    line-height: 1.5;
  }

  .top-metric {
    border-radius: 22px;
    padding: 16px 18px;
    background: #fff;
    border: 1px solid var(--line);
  }

  .top-metric.thc {
    background: linear-gradient(180deg, #f2f6f3, #ffffff);
  }

  .top-metric.cbd {
    background: linear-gradient(180deg, #faf6ec, #ffffff);
  }

  .top-metric.terps {
    background: linear-gradient(180deg, #eef5f3, #ffffff);
  }

  .top-metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }

  .top-metric-value {
    font-size: 34px;
    font-weight: 800;
    line-height: 1.05;
    color: var(--text);
  }

  .minor-box {
    margin-top: 14px;
    padding: 16px;
    border-radius: 18px;
    background: #f8f6f1;
    border: 1px solid var(--line);
  }

  .minor-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .minor-copy {
    color: var(--text);
    line-height: 1.7;
    font-size: 14px;
  }

  .bar-row + .bar-row {
    margin-top: 14px;
  }

  .bar-row-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 14px;
    margin-bottom: 7px;
  }

  .bar-row-head strong {
    font-size: 14px;
    color: var(--text);
  }

  .bar-shell {
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: #f1eee8;
    border: 1px solid #e9e3d7;
  }

  .bar-shell span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--green-2), #6f907c);
  }

  .bar-shell.gold span {
    background: linear-gradient(90deg, #b5934c, #dcc189);
  }

  .bar-shell.teal span {
    background: linear-gradient(90deg, #6b9f97, #9dc1bb);
  }

  .compound-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .compound-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px;
    background: var(--panel-2);
  }

  .compound-name {
    font-size: 13px;
    font-weight: 700;
    color: var(--green);
    margin-bottom: 6px;
  }

  .compound-value {
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
  }

  .compound-note {
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.5;
  }

  .terpene-layout {
    display: grid;
    grid-template-columns: 0.95fr 1.05fr;
    gap: 18px;
  }

  .terpene-hero-box {
    background: linear-gradient(180deg, #f6f8f7, #ffffff);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 22px;
  }

  .terpene-hero-title {
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .terpene-big-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .terpene-big {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--line);
    font-size: 15px;
  }

  .terpene-big strong {
    color: var(--text);
  }

  .aroma-box {
    margin-top: 16px;
    padding: 14px 16px;
    border-radius: 18px;
    background: #fbfaf7;
    border: 1px solid var(--line);
  }

  .aroma-title {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .aroma-copy {
    color: var(--text);
    line-height: 1.7;
    font-size: 15px;
  }

  .terpene-list-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px 16px;
  }

  .terpene-list-item {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--line);
    font-size: 14px;
  }

  .terpene-list-item strong {
    color: var(--text);
    white-space: nowrap;
  }

  .reg-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }

  .reg-status-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .status-card {
    border-radius: 18px;
    padding: 16px 12px;
    text-align: center;
    border: 1px solid var(--line);
    background: #fff;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .status-card.active.approved {
    background: rgba(31,61,43,0.08);
    border-color: rgba(31,61,43,0.20);
    color: var(--green);
  }

  .status-card.active.inconclusive {
    background: rgba(181,147,76,0.10);
    border-color: rgba(181,147,76,0.24);
    color: #876822;
  }

  .status-card.active.review {
    background: rgba(107,159,151,0.10);
    border-color: rgba(107,159,151,0.24);
    color: #456d67;
  }

  .check-list {
    display: grid;
    gap: 10px;
  }

  .check-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 16px;
    background: #fff;
    border: 1px solid var(--line);
    font-size: 14px;
  }

  .check-pill {
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .check-pill.good {
    background: rgba(31,61,43,0.08);
    color: var(--green);
    border: 1px solid rgba(31,61,43,0.18);
  }

  .check-pill.warn {
    background: rgba(181,147,76,0.10);
    color: #876822;
    border: 1px solid rgba(181,147,76,0.22);
  }

  .score-ring {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 220px;
  }

  .score-ring-inner {
    width: 190px;
    height: 190px;
    border-radius: 50%;
    border: 10px solid rgba(31,61,43,0.14);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: radial-gradient(circle at center, rgba(31,61,43,0.05), rgba(255,255,255,1));
    box-shadow: inset 0 0 30px rgba(0,0,0,0.03);
  }

  .score-ring-value {
    font-size: 44px;
    font-weight: 800;
    color: var(--green);
    line-height: 1;
  }

  .score-ring-sub {
    margin-top: 8px;
    font-size: 12px;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .pill-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }

  .pill {
    display: inline-flex;
    align-items: center;
    padding: 10px 12px;
    border-radius: 999px;
    background: #fff;
    border: 1px solid var(--line);
    color: var(--text);
    font-size: 13px;
    line-height: 1.3;
  }

  .insight-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
  }

  .insight-card {
    min-height: 100%;
  }

  .insight-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--green);
    margin-bottom: 12px;
    font-weight: 700;
  }

  .copy {
    font-size: 14px;
    line-height: 1.8;
    color: #2f2b26;
  }

  .muted {
    font-size: 14px;
    line-height: 1.7;
    color: var(--muted);
  }

  .tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 16px;
  }

  .tab-btn {
    border: 1px solid var(--line);
    background: #fff;
    color: var(--text);
    padding: 10px 14px;
    border-radius: 999px;
    cursor: pointer;
    font-weight: 700;
  }

  .tab-btn.active {
    background: var(--green);
    color: #fff;
    border-color: var(--green);
  }

  .tab-panel {
    display: none;
  }

  .tab-panel.active {
    display: block;
  }

  .insight-list {
    margin: 0;
    padding-left: 20px;
  }

  .insight-list li {
    margin: 10px 0;
    color: #2e2a25;
    line-height: 1.7;
  }

  .fingerprint-hero {
    display: grid;
    grid-template-columns: 0.95fr 1.05fr;
    gap: 18px;
    align-items: stretch;
  }

  .fingerprint-visual {
    position: relative;
    min-height: 320px;
    border-radius: 24px;
    border: 1px solid var(--line);
    overflow: hidden;
    background:
      radial-gradient(circle at center, rgba(31,61,43,0.06) 0, rgba(31,61,43,0.06) 1px, transparent 1px),
      linear-gradient(180deg, #fbfaf7, #ffffff);
    background-size: 28px 28px, auto;
  }

  .fingerprint-center {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
  }

  .fingerprint-ring {
    width: 230px;
    height: 230px;
    border: 1px dashed rgba(31,61,43,0.22);
    border-radius: 50%;
    position: relative;
    display: grid;
    place-items: center;
  }

  .fingerprint-ring::before,
  .fingerprint-ring::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    border: 1px dashed rgba(31,61,43,0.16);
  }

  .fingerprint-ring::before {
    inset: 22px;
  }

  .fingerprint-ring::after {
    inset: 48px;
  }

  .fingerprint-core {
    width: 114px;
    height: 114px;
    border-radius: 50%;
    background: var(--green);
    color: #fff;
    display: grid;
    place-items: center;
    text-align: center;
    padding: 12px;
    font-weight: 800;
    font-size: 13px;
    box-shadow: 0 12px 28px rgba(31,61,43,0.18);
  }

  .fp-signal {
    position: absolute;
    padding: 7px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.96);
    border: 1px solid var(--line);
    font-size: 12px;
    color: var(--muted);
    box-shadow: 0 8px 20px rgba(0,0,0,0.03);
  }

  .fp-a { top: 20px; left: 22px; }
  .fp-b { top: 34px; right: 22px; }
  .fp-c { bottom: 26px; left: 26px; }
  .fp-d { bottom: 34px; right: 20px; }

  .fingerprint-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }

  .fingerprint-tag {
    padding: 10px 12px;
    border-radius: 999px;
    background: var(--soft-green);
    color: var(--green);
    border: 1px solid rgba(31,61,43,0.12);
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.03em;
  }

  .benchmark-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }

  .benchmark-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px;
    background: #fff;
  }

  .benchmark-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .benchmark-value {
    font-size: 20px;
    font-weight: 800;
    color: var(--text);
  }

  .compare-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }

  .compare-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px;
    background: #fff;
  }

  .compare-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .compare-value {
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
  }

  .future-box {
    border-radius: 24px;
    border: 1px dashed rgba(31,61,43,0.24);
    padding: 20px;
    background: rgba(255,255,255,0.5);
  }

  .future-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--green);
    margin-bottom: 10px;
    font-weight: 700;
  }

  .raw-box {
    display: none;
    margin-top: 22px;
  }

  .raw-box pre {
    margin: 0;
    padding: 18px;
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 18px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.6;
    color: #2f2b26;
  }

  .footer {
    margin-top: 28px;
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.8;
    padding: 0 10px;
  }

  @media (max-width: 1100px) {
    .grid-2,
    .reg-grid,
    .terpene-layout,
    .insight-grid,
    .fingerprint-hero {
      grid-template-columns: 1fr;
    }

    .benchmark-grid,
    .compare-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .grid-3 {
      grid-template-columns: 1fr;
    }

    .meta-grid,
    .compound-grid,
    .terpene-list-grid {
      grid-template-columns: 1fr;
    }

    .hero h1 {
      font-size: 38px;
    }
  }

  @media (max-width: 700px) {
    .shell {
      padding: 16px 14px 34px;
    }

    .hero {
      padding: 24px 20px 22px;
      border-radius: 24px;
    }

    .hero h1 {
      font-size: 31px;
    }

    .benchmark-grid,
    .compare-grid,
    .reg-status-cards {
      grid-template-columns: 1fr;
    }

    .top-metric-value {
      font-size: 28px;
    }
  }

  @media print {
    .action-bar {
      display: none;
    }
    .shell {
      max-width: none;
      padding: 0;
    }
  }
</style>
</head>
<body>
  <div class="shell">
    ${
      options.documentId
        ? `
        <div class="action-bar">
          <button id="generatePdfBtn" class="btn btn-primary">Generate PDF</button>
          ${
            options.pdfUrl
              ? `<a class="btn btn-secondary" href="${esc(options.pdfUrl)}" target="_blank" rel="noopener noreferrer">Open current PDF</a>`
              : ""
          }
          <button id="toggleRawBtn" class="btn btn-secondary" type="button">Raw Data</button>
          <span id="pdfStatus" class="action-status"></span>
        </div>
      `
        : ""
    }

    <section class="hero">
      <div class="hero-top">
        <div>
          <div class="brand-mark">ALEM COA Intelligence Interface</div>
          <h1>${esc(productName)}</h1>
          <div class="hero-subtitle">${esc(subtitle)}</div>

          <div class="hero-badges">
            <span class="badge">Batch ${esc(batchNumber)}</span>
            <span class="badge">${esc(productType)}</span>
            <span class="badge">${esc(laboratoryName)}</span>
            <span class="badge">${esc(reportDate)}</span>
            <span class="badge">${esc(derivedProfileTag)}</span>
            <span class="badge">Confidence ${esc(`${dataConfidenceScore}%`)}</span>
            <span class="badge status-badge ${statusClass}">${esc(regulatoryStatus)}</span>
          </div>
        </div>

        <div class="hero-score">
          <div class="hero-score-label">AI Chemical Brief</div>
          <div class="hero-score-value">${esc(overallScoreText)}</div>
          <div class="hero-score-note">
            Educational, clinical, enthusiast, and industry-facing intelligence generated from the structured COA data.
          </div>
        </div>
      </div>
    </section>

    <section class="section grid-2">
      <div class="card">
        <h2 class="section-title">Batch & document details</h2>
        <div class="meta-grid">
          <div class="meta-item">
            <div class="meta-label">Batch number</div>
            <div class="meta-value">${esc(batchNumber)}</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">COA report date</div>
            <div class="meta-value">${esc(reportDate)}</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Product type</div>
            <div class="meta-value">${esc(productType)}</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Laboratory name</div>
            <div class="meta-value">${esc(laboratoryName)}</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2 class="section-title">Patient-safe summary</h2>
        <ul class="insight-list">
          ${patientSafeSummary.map((x) => `<li>${esc(x)}</li>`).join("")}
        </ul>
      </div>
    </section>

    <section class="section">
      <div class="card card-soft">
        <h2 class="section-title">Audience lens</h2>
        <div class="section-sub">The same chemistry interpreted through different user contexts.</div>

        <div class="tabs">
          <button class="tab-btn active" data-tab="patient">Patient View</button>
          <button class="tab-btn" data-tab="clinician">Clinician View</button>
          <button class="tab-btn" data-tab="industry">Industry View</button>
          <button class="tab-btn" data-tab="enthusiast">Enthusiast View</button>
        </div>

        <div class="tab-panel active" id="tab-patient">
          <ul class="insight-list">
            ${patientSafeSummary.map((x) => `<li>${esc(x)}</li>`).join("")}
          </ul>
        </div>

        <div class="tab-panel" id="tab-clinician">
          <ul class="insight-list">
            ${clinicianInsights.map((x) => `<li>${esc(x)}</li>`).join("")}
          </ul>
        </div>

        <div class="tab-panel" id="tab-industry">
          <ul class="insight-list">
            ${industryInsights.map((x) => `<li>${esc(x)}</li>`).join("")}
          </ul>
        </div>

        <div class="tab-panel" id="tab-enthusiast">
          <ul class="insight-list">
            ${enthusiastInsights.map((x) => `<li>${esc(x)}</li>`).join("")}
          </ul>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Chemical fingerprint</h2>
        <div class="fingerprint-hero">
          <div class="fingerprint-visual">
            <div class="fingerprint-center">
              <div class="fingerprint-ring">
                <div class="fingerprint-core">
                  Fingerprint<br/>Engine
                </div>
              </div>
            </div>
            <div class="fp-signal fp-a">THC ${esc(data?.thc_total || "N/A")}</div>
            <div class="fp-signal fp-b">Terpenes ${esc(data?.total_terpenes || "N/A")}</div>
            <div class="fp-signal fp-c">Lead ${esc(dominantTerpene)}</div>
            <div class="fp-signal fp-d">${esc(data?.minor_cannabinoids ? "Minor cannabinoids present" : "Minor cannabinoids limited")}</div>
          </div>

          <div>
            <div class="section-sub">A compact fingerprint layer that prepares the product for future similarity matching, clustering, and compare mode.</div>
            <div class="fingerprint-tags">
              ${fingerprintTags.map((tag) => `<span class="fingerprint-tag">${esc(tag)}</span>`).join("")}
            </div>

            <div class="card card-soft" style="margin-top:16px;">
              <div class="insight-label">Similarity signals</div>
              ${renderPillList(similaritySignals, "Similarity analysis pending")}
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Benchmarking engine</h2>
        <div class="section-sub">This section prepares the report for future compare mode, database ranking, and profile clustering.</div>
        <div class="benchmark-grid">
          ${benchmarkCards
            .map(
              (item) => `
            <div class="benchmark-card">
              <div class="benchmark-label">${esc(item.label)}</div>
              <div class="benchmark-value">${esc(item.value)}</div>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Compare mode</h2>
        <div class="section-sub">UI-ready comparison layer against broad internal benchmark categories.</div>
        <div class="compare-grid">
          ${compareCards
            .map(
              (item) => `
            <div class="compare-card">
              <div class="compare-label">${esc(item.label)}</div>
              <div class="compare-value">${esc(item.value)}</div>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Cannabinoid profile</h2>
        <div class="grid-3">
          <div class="top-metric thc">
            <div class="top-metric-label">THC</div>
            <div class="top-metric-value">${esc(data?.thc_total || "Not reported")}</div>
          </div>
          <div class="top-metric cbd">
            <div class="top-metric-label">CBD</div>
            <div class="top-metric-value">${esc(data?.cbd_total || "Not reported")}</div>
          </div>
          <div class="top-metric terps">
            <div class="top-metric-label">Terpenes</div>
            <div class="top-metric-value">${esc(data?.total_terpenes || "Not reported")}</div>
          </div>
        </div>

        <div class="grid-2" style="margin-top:18px;">
          <div>
            ${renderBar("THC", data?.thc_total || "0", thcNumber * 3, "")}
            ${renderBar("CBD", data?.cbd_total || "0", cbdNumber * 10, "gold")}
            ${renderBar("Total terpenes", data?.total_terpenes || "0", terpNumber * 20, "teal")}

            <div class="minor-box">
              <div class="minor-title">Minor cannabinoids</div>
              <div class="minor-copy">${esc(data?.minor_cannabinoids || "No minor cannabinoid summary reported.")}</div>
            </div>
          </div>

          <div>
            <div class="section-sub">Primary cannabinoids extracted from the COA.</div>
            ${renderCompoundPills(topCannabinoids, "cannabinoid")}
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Terpenes</h2>
        <div class="terpene-layout">
          <div class="terpene-hero-box">
            <div class="terpene-hero-title">Top terpenes</div>
            <div class="terpene-big-list">
              ${
                topTerpenes.length
                  ? topTerpenes
                      .slice(0, 5)
                      .map(
                        (item) => `
                        <div class="terpene-big">
                          <span>${esc(item?.name || "Unnamed")}</span>
                          <strong>${esc([item?.value, item?.unit].filter(Boolean).join(" ")) || "—"}</strong>
                        </div>
                      `
                      )
                      .join("")
                  : `<div class="muted">No terpene rows reported.</div>`
              }
            </div>

            <div class="aroma-box">
              <div class="aroma-title">Aromatic profile</div>
              <div class="aroma-copy">${esc(aromaticProfile)}</div>
            </div>
          </div>

          <div>
            <div class="section-sub">Full terpene list extracted from the COA.</div>
            ${renderFullTerpeneList(topTerpenes)}
          </div>
        </div>
      </div>
    </section>

    <section class="section reg-grid">
      <div class="card">
        <h2 class="section-title small">Regulatory complexity score</h2>

        <div class="reg-status-cards">
          <div class="status-card ${regulatoryStatus === "Approved" ? `active approved` : ""}">Approved</div>
          <div class="status-card ${regulatoryStatus === "Inconclusive" ? `active inconclusive` : ""}">Inconclusive</div>
          <div class="status-card ${regulatoryStatus === "Under Review" ? `active review` : ""}">Under review</div>
        </div>

        <div class="copy">
          ${esc(
            data?.contaminant_overview ||
              "Contaminant or TGO-related interpretation was not fully reported in the structured extraction."
          )}
        </div>

        <div class="section-sub" style="margin-top:16px;">Data completeness</div>
        <div class="check-list">
          ${completenessItems
            .map(
              (item) => `
              <div class="check-item">
                <span>${esc(item.label)}</span>
                <span class="check-pill ${item.className}">${esc(item.value)}</span>
              </div>
            `
            )
            .join("")}
        </div>
      </div>

      <div class="card">
        <h2 class="section-title small">COA quality score</h2>
        <div class="score-ring">
          <div class="score-ring-inner">
            <div class="score-ring-value">${esc(`${coaQualityScore}/10`)}</div>
            <div class="score-ring-sub">Document quality</div>
          </div>
        </div>
        <div class="copy" style="text-align:center;">
          A heuristic quality score based on how complete and interpretable the extracted COA data appears.
        </div>
      </div>
    </section>

    <section class="section grid-2">
      <div class="card">
        <div class="insight-label">Confidence layer</div>
        <div class="copy"><strong>Extraction confidence:</strong> ${esc(`${dataConfidenceScore}%`)}</div>
        <div class="copy" style="margin-top:8px;"><strong>Cannabinoid visibility:</strong> ${esc(cannabinoidVisibility)}</div>
        <div class="copy" style="margin-top:8px;"><strong>Terpene visibility:</strong> ${esc(terpeneVisibility)}</div>
        <div class="copy" style="margin-top:8px;"><strong>Interpretation note:</strong> ${esc(warningFlags.length ? "Some outputs should be read cautiously due to flagged or incomplete areas." : "Interpretation is reasonably supported by the available structured fields.")}</div>
      </div>

      <div class="card">
        <div class="insight-label">Scientific context</div>
        <div class="copy">${esc(data?.scientific_references || "No scientific notes included.")}</div>
      </div>
    </section>

    <section class="section">
      <div class="insight-grid">
        <div class="card insight-card">
          <div class="insight-label">Experience profile</div>
          ${renderPillList(positiveFlags.length ? positiveFlags : patientSafeSummary.slice(0, 3), "No experiential interpretation available")}
        </div>

        <div class="card insight-card">
          <div class="insight-label">Potential use cases</div>
          ${renderPillList(
            warningFlags.length
              ? [
                  "Requires clinician review",
                  "Check source COA before relying on product-positioning claims",
                ]
              : [
                  thcNumber >= 24 ? "Higher-intensity use profile" : "Moderate potency profile",
                  terpNumber >= 2 ? "Aroma-forward positioning" : "Subtler aromatic profile",
                  data?.minor_cannabinoids ? "Complexity-driven education angle" : "Major-compound-driven profile",
                ],
            "No use-case guidance available"
          )}
        </div>

        <div class="card insight-card">
          <div class="insight-label">Market interpretation</div>
          <div class="copy">
            ${esc(
              data?.lab_quality_summary ||
                (data?.minor_cannabinoids
                  ? "This profile has enough visible chemical depth to support stronger education and product-storytelling angles."
                  : "Market differentiation appears more dependent on dominant potency and terpene architecture than minor cannabinoid depth.")
            )}
          </div>
        </div>
      </div>
    </section>

    <section class="section grid-2">
      <div class="card">
        <div class="insight-label">Positive signals</div>
        ${renderPillList(positiveFlags, "No positive flags reported")}
      </div>

      <div class="card">
        <div class="insight-label">Watchouts</div>
        ${renderPillList(warningFlags, "No warning flags reported")}
      </div>
    </section>

    <section class="section">
      <div class="future-box">
        <div class="future-title">Next intelligence layer</div>
        <div class="copy">
          This page is now ready for future additions such as true batch-to-batch compare mode, database-wide similarity ranking, fingerprint radar plotting, strain/lineage matching, and chemotype clustering.
        </div>
      </div>
    </section>

    <section id="rawDataBox" class="raw-box">
      <pre>${rawJson}</pre>
    </section>

    <div class="footer">
      Medicinal cannabis is a regulated treatment option in Australia and this report is an educational interpretive layer only. It does not replace physician advice, pharmacist counselling, regulatory review, or direct laboratory confirmation.
    </div>
  </div>

  <script>
    (function () {
      const tabButtons = document.querySelectorAll(".tab-btn");
      const tabPanels = document.querySelectorAll(".tab-panel");

      tabButtons.forEach((btn) => {
        btn.addEventListener("click", function () {
          const tab = btn.getAttribute("data-tab");

          tabButtons.forEach((b) => b.classList.remove("active"));
          tabPanels.forEach((p) => p.classList.remove("active"));

          btn.classList.add("active");
          const target = document.getElementById("tab-" + tab);
          if (target) target.classList.add("active");
        });
      });

      const pdfBtn = document.getElementById("generatePdfBtn");
      const pdfStatus = document.getElementById("pdfStatus");
      const rawBtn = document.getElementById("toggleRawBtn");
      const rawBox = document.getElementById("rawDataBox");

      if (rawBtn && rawBox) {
        rawBtn.addEventListener("click", function () {
          const hidden = rawBox.style.display === "none" || !rawBox.style.display;
          rawBox.style.display = hidden ? "block" : "none";
          if (hidden) rawBox.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      }

      if (pdfBtn) {
        pdfBtn.addEventListener("click", async function () {
          try {
            pdfBtn.disabled = true;
            pdfBtn.textContent = "Generating PDF...";
            if (pdfStatus) pdfStatus.textContent = "";

            const response = await fetch("/generate-pdf/${options.documentId || ""}", {
              method: "POST",
              headers: { "Content-Type": "application/json" }
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
              throw new Error(result.error || "Failed to generate PDF");
            }

            if (pdfStatus) pdfStatus.textContent = "PDF ready";
            pdfBtn.textContent = "Generate PDF again";

            if (result.pdf_url) {
              window.open(result.pdf_url, "_blank");
            }
          } catch (err) {
            console.error(err);
            if (pdfStatus) pdfStatus.textContent = err.message || "PDF generation failed";
            pdfBtn.textContent = "Generate PDF";
          } finally {
            pdfBtn.disabled = false;
          }
        });
      }
    })();
  </script>
</body>
</html>
`;
}

function extractJSONObject(text) {
  if (!text || typeof text !== "string") {
    throw new Error("No text received from OpenAI");
  }

  const trimmed = text.trim();

  if (!trimmed) {
    throw new Error("OpenAI returned empty text");
  }

  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");

  if (firstBrace === -1) {
    throw new Error("No opening JSON brace found in OpenAI output");
  }

  if (lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error("No closing JSON brace found in OpenAI output");
  }

  return trimmed.slice(firstBrace, lastBrace + 1);
}

function normalizeParsedCOA(data = {}) {
  const toString = (v) => (v === null || v === undefined ? "" : String(v));
  const toArray = (v) => (Array.isArray(v) ? v : []);

  return {
    product_name: toString(data.product_name),
    batch_number: toString(data.batch_number),
    coa_report_date: toString(data.coa_report_date),
    product_type: toString(data.product_type),
    laboratory_name: toString(data.laboratory_name),
    opening_statement: toString(data.opening_statement),
    overall_score: toString(data.overall_score),
    thc_total: toString(data.thc_total),
    cbd_total: toString(data.cbd_total),
    total_terpenes: toString(data.total_terpenes),
    minor_cannabinoids: toString(data.minor_cannabinoids),
    contaminant_overview: toString(data.contaminant_overview),
    lab_quality_summary: toString(data.lab_quality_summary),
    scientific_references: toString(data.scientific_references),

    top_cannabinoids: toArray(data.top_cannabinoids)
      .slice(0, 8)
      .map((item) => ({
        name: toString(item?.name),
        value: toString(item?.value),
        unit: toString(item?.unit),
        notes: toString(item?.notes),
      })),

    top_terpenes: toArray(data.top_terpenes)
      .slice(0, 8)
      .map((item) => ({
        name: toString(item?.name),
        value: toString(item?.value),
        unit: toString(item?.unit),
      })),

    positive_flags: toArray(data.positive_flags)
      .slice(0, 6)
      .map((x) => toString(x)),

    warning_flags: toArray(data.warning_flags)
      .slice(0, 6)
      .map((x) => toString(x)),
  };
}

function safeSnippet(value, max = 1000) {
  return String(value || "").slice(0, max);
}

function prepareOCRTextForModel(cleanText = "") {
  const text = String(cleanText || "").trim();
  if (!text) return "";

  if (text.length <= MAX_OCR_CHARS_FOR_OPENAI) {
    return text;
  }

  const headLength = Math.floor(MAX_OCR_CHARS_FOR_OPENAI * 0.7);
  const tailLength = MAX_OCR_CHARS_FOR_OPENAI - headLength;

  return [
    text.slice(0, headLength),
    "\\n\\n[TRUNCATED FOR MODEL INPUT]\\n\\n",
    text.slice(-tailLength),
  ].join("");
}

function extractTextFromOpenAIResponse(response) {
  try {
    if (!response || typeof response !== "object") {
      return "";
    }

    if (
      typeof response.output_text === "string" &&
      response.output_text.trim()
    ) {
      return response.output_text.trim();
    }

    const chunks = [];

    if (Array.isArray(response.output)) {
      for (const item of response.output) {
        if (!item || !Array.isArray(item.content)) continue;

        for (const part of item.content) {
          if (!part) continue;

          if (typeof part.text === "string" && part.text.trim()) {
            chunks.push(part.text.trim());
            continue;
          }

          if (
            part.type === "output_text" &&
            typeof part.text === "string" &&
            part.text.trim()
          ) {
            chunks.push(part.text.trim());
          }
        }
      }
    }

    return chunks.join("\\n").trim();
  } catch (err) {
    console.error("extractTextFromOpenAIResponse error:", err.message);
    return "";
  }
}

function logOpenAIResponseMeta(label, response) {
  try {
    console.log(`${label} RESPONSE ID:`, response?.id || "");
    console.log(`${label} RESPONSE STATUS:`, response?.status || "");
    console.log(`${label} RESPONSE MODEL:`, response?.model || "");
    console.log(
      `${label} RESPONSE OUTPUT_TEXT EXISTS:`,
      typeof response?.output_text === "string"
    );
    console.log(
      `${label} RESPONSE OUTPUT COUNT:`,
      Array.isArray(response?.output) ? response.output.length : 0
    );

    if (response?.incomplete_details) {
      console.log(
        `${label} RESPONSE INCOMPLETE DETAILS:`,
        JSON.stringify(response.incomplete_details)
      );
    }

    if (response?.error) {
      console.log(`${label} RESPONSE ERROR:`, JSON.stringify(response.error));
    }
  } catch (err) {
    console.error("logOpenAIResponseMeta error:", err.message);
  }
}

async function callOpenAIForJSON(systemPrompt, userText, maxOutputTokens = 1200) {
  console.log("🚀 Sending request to OpenAI...");

  const response = await Promise.race([
    openai.responses.create({
      model: OPENAI_MODEL,
      store: false,
      text: {
        format: { type: "text" },
      },
      reasoning: { effort: "minimal" },
      max_output_tokens: maxOutputTokens,
      input: [
        {
          role: "system",
          content: systemPrompt,
        },
        {
          role: "user",
          content: userText,
        },
      ],
    }),
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`OpenAI timeout after ${OPENAI_TIMEOUT_MS}ms`)),
        OPENAI_TIMEOUT_MS
      )
    ),
  ]);

  console.log("✅ OpenAI responded");
  return response;
}

async function repairJSONToSchema(cleanText = "") {
  const modelInput = prepareOCRTextForModel(cleanText);

  const repairPrompt = `
You are repairing a failed COA extraction.

Return EXACTLY ONE valid JSON object in this exact shape:

{
  "product_name": "",
  "batch_number": "",
  "coa_report_date": "",
  "product_type": "",
  "laboratory_name": "",
  "opening_statement": "",
  "overall_score": "",
  "thc_total": "",
  "cbd_total": "",
  "total_terpenes": "",
  "minor_cannabinoids": "",
  "contaminant_overview": "",
  "lab_quality_summary": "",
  "scientific_references": "",
  "top_cannabinoids": [
    {
      "name": "",
      "value": "",
      "unit": "",
      "notes": ""
    }
  ],
  "top_terpenes": [
    {
      "name": "",
      "value": "",
      "unit": ""
    }
  ],
  "positive_flags": [""],
  "warning_flags": [""]
}

STRICT RULES:
- Valid JSON only
- No markdown
- No comments
- No trailing commas
- If uncertain, leave fields empty
- Keep arrays under 8 items
- Output must parse with JSON.parse()
`;

  const response = await callOpenAIForJSON(repairPrompt, modelInput, 1200);

  logOpenAIResponseMeta("REPAIR", response);

  const repairedText = extractTextFromOpenAIResponse(response);

  console.log("REPAIR OUTPUT LENGTH:", repairedText ? repairedText.length : 0);
  console.log("REPAIR OUTPUT START:", safeSnippet(repairedText, 1000));
  console.log("REPAIR OUTPUT END:", String(repairedText || "").slice(-1000));

  if (!repairedText.trim()) {
    console.error(
      "REPAIR FULL OPENAI RESPONSE:",
      JSON.stringify(response, null, 2)
    );
    throw new Error("No text received from OpenAI during repair");
  }

  const cleaned = extractJSONObject(repairedText);
  return JSON.parse(cleaned);
}

async function parseCOAWithOpenAI(cleanText = "") {
  if (!cleanText || !String(cleanText).trim()) {
    throw new Error("cleanText is empty");
  }

  const modelInput = prepareOCRTextForModel(cleanText);

  try {
    console.log("OCR TEXT ORIGINAL LENGTH:", String(cleanText).length);
    console.log("OCR TEXT MODEL INPUT LENGTH:", String(modelInput).length);
    console.log("OCR TEXT PREVIEW:", safeSnippet(modelInput, 1500));

    const response = await callOpenAIForJSON(extractionPrompt, modelInput, 1200);

    logOpenAIResponseMeta("RAW", response);

    const rawText = extractTextFromOpenAIResponse(response);

    console.log("RAW OPENAI OUTPUT LENGTH:", rawText ? rawText.length : 0);
    console.log("RAW OPENAI OUTPUT START:", safeSnippet(rawText, 1000));
    console.log("RAW OPENAI OUTPUT END:", String(rawText || "").slice(-1000));

    if (!rawText.trim()) {
      console.warn("⚠️ Empty OpenAI output, retrying with fallback prompt...");

      const retry = await callOpenAIForJSON(
        "Return only one valid JSON object. No thinking. No explanation. No markdown.",
        modelInput.slice(0, 9000),
        900
      );

      logOpenAIResponseMeta("RETRY", retry);

      const retryText = extractTextFromOpenAIResponse(retry);

      console.log("RETRY OUTPUT LENGTH:", retryText ? retryText.length : 0);
      console.log("RETRY OUTPUT START:", safeSnippet(retryText, 1000));
      console.log("RETRY OUTPUT END:", String(retryText || "").slice(-1000));

      if (retryText.trim()) {
        let retryJson = extractJSONObject(retryText).trim();

        if (retryJson.endsWith(",")) {
          retryJson = retryJson.slice(0, -1);
        }

        return normalizeParsedCOA(JSON.parse(retryJson));
      }

      throw new Error("No text received from OpenAI after retry");
    }

    let jsonText = extractJSONObject(rawText).trim();

    if (jsonText.endsWith(",")) {
      jsonText = jsonText.slice(0, -1);
    }

    return normalizeParsedCOA(JSON.parse(jsonText));
  } catch (parseError) {
    console.warn("JSON parse failed, attempting repair...");
    console.warn("PRIMARY PARSE ERROR:", parseError.message);

    try {
      const repaired = await repairJSONToSchema(cleanText);
      console.log("JSON repaired successfully");
      return normalizeParsedCOA(repaired);
    } catch (repairError) {
      console.error("JSON repair failed:", repairError.message);

      return normalizeParsedCOA({
        product_name: "",
        batch_number: "",
        coa_report_date: "",
        product_type: "",
        laboratory_name: "",
        opening_statement: "Partial parse only.",
        overall_score: "",
        thc_total: "",
        cbd_total: "",
        total_terpenes: "",
        minor_cannabinoids: "",
        contaminant_overview: "",
        lab_quality_summary: "",
        scientific_references: "",
        top_cannabinoids: [],
        top_terpenes: [],
        positive_flags: [],
        warning_flags: [
          "COA parsed partially due to malformed, timeout, or empty model output",
        ],
      });
    }
  }
}

async function safeParseCOA(cleanText = "") {
  try {
    return await parseCOAWithOpenAI(cleanText);
  } catch (error) {
    console.error("safeParseCOA fallback triggered:", error.message);

    return normalizeParsedCOA({
      product_name: "",
      batch_number: "",
      coa_report_date: "",
      product_type: "",
      laboratory_name: "",
      opening_statement: "COA could not be fully parsed automatically.",
      overall_score: "",
      thc_total: "",
      cbd_total: "",
      total_terpenes: "",
      minor_cannabinoids: "",
      contaminant_overview: "",
      lab_quality_summary: "",
      scientific_references: "",
      top_cannabinoids: [],
      top_terpenes: [],
      positive_flags: [],
      warning_flags: ["Automatic extraction failed"],
    });
  }
}

async function extractDocumentFromUrl(fileUrl) {
  if (!fileUrl || typeof fileUrl !== "string") {
    throw new Error("fileUrl must be a non-empty string");
  }

  const fileResponse = await axios.get(fileUrl, {
    responseType: "arraybuffer",
    timeout: 60000,
    maxRedirects: 5,
    validateStatus: (status) => status >= 200 && status < 300,
  });

  const mimeType = detectMimeType(
    fileUrl,
    fileResponse.headers["content-type"]
  );

  const poller = await azureClient.beginAnalyzeDocument(
    "prebuilt-layout",
    fileResponse.data,
    {
      contentType: mimeType,
    }
  );

  const result = await poller.pollUntilDone();

  const pages = (result.pages || []).map((page) => {
    const text = (page.lines || []).map((line) => line.content).join("\\n");

    return {
      page_number: page.pageNumber,
      width: page.width,
      height: page.height,
      unit: page.unit,
      text,
      lines: (page.lines || []).map((line) => ({
        content: line.content,
      })),
    };
  });

  const tables = (result.tables || []).map((table, tableIndex) => ({
    table_index: tableIndex + 1,
    row_count: table.rowCount,
    column_count: table.columnCount,
    cells: (table.cells || []).map((cell) => ({
      row_index: cell.rowIndex,
      column_index: cell.columnIndex,
      content: cell.content,
    })),
  }));

  const plainText = pages.map((p) => p.text).join("\\n\\n");

  return {
    mimeType,
    pages,
    tables,
    plain_text: plainText,
  };
}

async function buildPdfBufferFromHtml(html) {
  let browser;

  try {
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();

    await page.setViewport({
      width: 1400,
      height: 2000,
      deviceScaleFactor: 1,
    });

    await page.setContent(html, { waitUntil: "networkidle0" });
    await page.emulateMediaType("screen");

    const pageHeight = await page.evaluate(() => {
      const body = document.body;
      const htmlEl = document.documentElement;
      return Math.max(
        body.scrollHeight,
        body.offsetHeight,
        htmlEl.clientHeight,
        htmlEl.scrollHeight,
        htmlEl.offsetHeight
      );
    });

    const pdfHeight = Math.max(pageHeight + 30, 2200);

    const pdfBuffer = await page.pdf({
      printBackground: true,
      width: "1280px",
      height: `${pdfHeight}px`,
      margin: {
        top: "0px",
        right: "0px",
        bottom: "0px",
        left: "0px",
      },
      pageRanges: "1",
    });

    await browser.close();
    return pdfBuffer;
  } catch (error) {
    if (browser) {
      try {
        await browser.close();
      } catch (closeErr) {
        console.error("Error closing browser:", closeErr.message);
      }
    }
    throw error;
  }
}

async function uploadPdfToSupabase(fileName, pdfBuffer) {
  const bucket = process.env.SUPABASE_BUCKET;
  if (!bucket) {
    throw new Error("Missing SUPABASE_BUCKET in .env");
  }

  const { error: uploadError } = await supabase.storage
    .from(bucket)
    .upload(fileName, pdfBuffer, {
      contentType: "application/pdf",
      upsert: true,
    });

  if (uploadError) {
    throw uploadError;
  }

  const { data: publicUrlData } = supabase.storage
    .from(bucket)
    .getPublicUrl(fileName);

  const pdfUrl = publicUrlData?.publicUrl;

  if (!pdfUrl) {
    throw new Error("Could not generate public PDF URL");
  }

  return pdfUrl;
}

async function updateDocumentStatus(documentId, values = {}) {
  if (!documentId) return;
  const { error } = await supabase.from("documents").update(values).eq("id", documentId);
  if (error) {
    throw new Error(`Could not update document ${documentId}: ${error.message}`);
  }
}

async function getDocumentById(documentId) {
  const { data, error } = await supabase
    .from("documents")
    .select("*")
    .eq("id", documentId)
    .single();

  if (error) {
    throw new Error(`Could not fetch document ${documentId}: ${error.message}`);
  }

  return data;
}

async function processCOAJob({
  documentId,
  fileUrl,
  originalFilename,
  documentType = "coa",
}) {
  try {
    console.log("🚀 Processing async job:", documentId);

    await updateDocumentStatus(documentId, {
      status: "processing",
      original_filename: originalFilename,
      document_type: documentType,
    });

    const extracted = await extractDocumentFromUrl(fileUrl);
    const parsedJson = await safeParseCOA(extracted.plain_text);

    await updateDocumentStatus(documentId, {
      status: "completed",
      mime_type: extracted.mimeType,
      extracted_text: extracted.plain_text,
      parsed_json: {
        ...parsedJson,
        extraction_meta: {
          source_url: fileUrl,
          engine_used: "azure-document-intelligence",
          model_used: "prebuilt-layout",
          page_count: extracted.pages.length,
          tables_found: extracted.tables.length,
        },
      },
    });

    console.log("✅ Async job completed:", documentId);

    return {
      document_id: documentId,
      parsed_json: parsedJson,
    };
  } catch (error) {
    console.error("❌ Async job failed:", documentId, error.message);

    try {
      await updateDocumentStatus(documentId, {
        status: "failed",
      });
    } catch (statusErr) {
      console.error("Failed to mark async job as failed:", statusErr.message);
    }

    throw error;
  }
}

async function generatePdfForDocument(documentId) {
  const row = await getDocumentById(documentId);

  const parsedJson = row?.parsed_json || {};
  if (!parsedJson || typeof parsedJson !== "object") {
    throw new Error("Document has no parsed_json to render");
  }

  const pdfBaseName = sanitizeFileName(
    parsedJson?.product_name || row?.original_filename || `coa-${Date.now()}`
  );
  const fileName = `${pdfBaseName}.pdf`;

  const html = renderReportHTML(parsedJson, {
    documentId,
    pdfUrl: null,
  });

  const pdfBuffer = await buildPdfBufferFromHtml(html);
  const pdfUrl = await uploadPdfToSupabase(fileName, pdfBuffer);

  await updateDocumentStatus(documentId, {
    storage_path: fileName,
    parsed_json: {
      ...parsedJson,
      pdf_url: pdfUrl,
    },
  });

  return {
    file_name: fileName,
    pdf_url: pdfUrl,
  };
}

app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Middleware is running",
  });
});

app.get("/health", async (req, res) => {
  try {
    const { error } = await supabase.from("documents").select("id").limit(1);

    res.json({
      success: true,
      status: "ok",
      hasSupabaseUrl: !!process.env.SUPABASE_URL,
      hasSupabaseKey:
        !!process.env.SUPABASE_SERVICE_ROLE_KEY ||
        !!process.env.SUPABASE_ANON_KEY,
      hasAzureEndpoint: !!process.env.AZURE_DOC_INTELLIGENCE_ENDPOINT,
      hasAzureKey: !!process.env.AZURE_DOC_INTELLIGENCE_KEY,
      hasOpenAIKey: !!process.env.OPENAI_API_KEY,
      bucket: process.env.SUPABASE_BUCKET || null,
      db_connected: !error,
      db_error: error ? error.message : null,
    });
  } catch (err) {
    res.json({
      success: false,
      status: "error",
      message: err.message,
    });
  }
});

app.get("/routes-check", (req, res) => {
  res.json({
    ok: true,
    routes: [
      "/",
      "/health",
      "/routes-check",
      "/extract-from-url",
      "/extract-and-save",
      "/extract-parse-and-save",
      "/full-coa-pipeline",
      "/start-coa-job",
      "/job-status",
      "/report/:id",
      "/generate-pdf/:id",
      "/generate-report",
      "/test",
    ],
  });
});

app.get("/report/:id", async (req, res) => {
  try {
    const id = req.params.id;
    if (!id) {
      return res.status(400).send("Missing document id");
    }

    const data = await getDocumentById(id);
    const parsedJson = data?.parsed_json || null;

    if (!parsedJson) {
      return res.status(404).send("Report data not found");
    }

    let pdfUrl = parsedJson?.pdf_url || null;

    if (!pdfUrl && data?.storage_path) {
      const { data: publicUrlData } = supabase.storage
        .from(process.env.SUPABASE_BUCKET)
        .getPublicUrl(data.storage_path);

      pdfUrl = publicUrlData?.publicUrl || null;
    }

    const html = renderReportHTML(parsedJson, {
      documentId: id,
      pdfUrl,
    });

    res.setHeader("Content-Type", "text/html; charset=utf-8");
    return res.status(200).send(html);
  } catch (error) {
    console.error("ERROR IN /report/:id:", error);
    return res.status(500).send(`Report error: ${esc(error.message || "Unknown error")}`);
  }
});

app.post("/generate-pdf/:id", async (req, res) => {
  try {
    const id = req.params.id;

    if (!id) {
      return res.status(400).json({
        success: false,
        error: "Document id is required",
      });
    }

    const result = await generatePdfForDocument(id);

    return res.json({
      success: true,
      document_id: id,
      file_name: result.file_name,
      pdf_url: result.pdf_url,
    });
  } catch (error) {
    console.error("ERROR IN /generate-pdf/:id:", error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown PDF generation error",
    });
  }
});

app.post("/extract-from-url", async (req, res) => {
  try {
    const fileUrl = req.body?.file_url;

    if (!fileUrl) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const extracted = await extractDocumentFromUrl(fileUrl);

    return res.json({
      success: true,
      engine_used: "azure-document-intelligence",
      model_used: "prebuilt-layout",
      source_url: fileUrl,
      mime_type: extracted.mimeType,
      plain_text: extracted.plain_text,
      pages: extracted.pages,
      tables: extracted.tables,
      metadata: {
        page_count: extracted.pages.length,
      },
    });
  } catch (error) {
    console.error("ERROR IN /extract-from-url:", error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown extraction error",
      details: error?.response?.data || null,
    });
  }
});

app.post("/extract-and-save", async (req, res) => {
  try {
    const fileUrl = req.body?.file_url;
    const originalFilename =
      req.body?.original_filename ||
      fileUrl?.split("/").pop() ||
      `upload-${Date.now()}`;
    const documentType = req.body?.document_type || "coa";

    if (!fileUrl) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const extracted = await extractDocumentFromUrl(fileUrl);

    const { data: insertedRow, error: insertError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename: originalFilename,
          mime_type: extracted.mimeType,
          document_type: documentType,
          status: "extracted",
          extracted_text: extracted.plain_text,
          parsed_json: {
            source_url: fileUrl,
            engine_used: "azure-document-intelligence",
            model_used: "prebuilt-layout",
            pages: extracted.pages,
            tables: extracted.tables,
            metadata: {
              page_count: extracted.pages.length,
            },
          },
        },
      ])
      .select()
      .single();

    if (insertError) {
      throw new Error(
        `Could not save extracted document: ${insertError.message}`
      );
    }

    return res.json({
      success: true,
      document_id: insertedRow.id,
      source_url: fileUrl,
      mime_type: extracted.mimeType,
      plain_text: extracted.plain_text,
      page_count: extracted.pages.length,
      html_url: buildReportUrl(req, insertedRow.id),
    });
  } catch (error) {
    console.error("ERROR IN /extract-and-save:", error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown extraction/save error",
    });
  }
});

app.post("/extract-parse-and-save", async (req, res) => {
  try {
    const fileUrl = req.body?.file_url;
    const originalFilename =
      req.body?.original_filename ||
      fileUrl?.split("/").pop() ||
      `upload-${Date.now()}`;
    const documentType = req.body?.document_type || "coa";

    if (!fileUrl) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const extracted = await extractDocumentFromUrl(fileUrl);
    const parsedJson = await safeParseCOA(extracted.plain_text);

    const { data: insertedRow, error: insertError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename: originalFilename,
          mime_type: extracted.mimeType,
          document_type: documentType,
          status: "completed",
          extracted_text: extracted.plain_text,
          parsed_json: {
            ...parsedJson,
            extraction_meta: {
              source_url: fileUrl,
              engine_used: "azure-document-intelligence",
              model_used: "prebuilt-layout",
              page_count: extracted.pages.length,
              tables_found: extracted.tables.length,
            },
          },
        },
      ])
      .select()
      .single();

    if (insertError) {
      throw new Error(`Could not save parsed document: ${insertError.message}`);
    }

    return res.json({
      success: true,
      document_id: insertedRow.id,
      source_url: fileUrl,
      mime_type: extracted.mimeType,
      plain_text: extracted.plain_text,
      parsed_json: parsedJson,
      html_url: buildReportUrl(req, insertedRow.id),
    });
  } catch (error) {
    console.error("ERROR IN /extract-parse-and-save:", error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown parse error",
    });
  }
});

app.post("/test", async (req, res) => {
  res.json({
    success: true,
    message: "Server is working",
    received: req.body,
  });
});

app.post("/start-coa-job", async (req, res) => {
  try {
    const fileUrl = req.body?.file_url;
    const originalFilename =
      req.body?.original_filename ||
      fileUrl?.split("/").pop() ||
      `upload-${Date.now()}`;
    const documentType = req.body?.document_type || "coa";

    if (!fileUrl) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const { data: row, error } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename: originalFilename,
          document_type: documentType,
          status: "queued",
          parsed_json: {
            source_url: fileUrl,
          },
        },
      ])
      .select()
      .single();

    if (error) {
      throw new Error(`Could not create async job: ${error.message}`);
    }

    processCOAJob({
      documentId: row.id,
      fileUrl,
      originalFilename,
      documentType,
    }).catch((err) => {
      console.error("Detached background job failed:", err.message);
    });

    return res.json({
      success: true,
      job_id: row.id,
      document_id: row.id,
      status: "queued",
      html_url: buildReportUrl(req, row.id),
    });
  } catch (error) {
    console.error("ERROR IN /start-coa-job:", error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown async job error",
    });
  }
});

app.get("/job-status", async (req, res) => {
  try {
    const id = req.query?.id || req.query?.document_id || req.query?.job_id;

    if (!id) {
      return res.status(400).json({
        success: false,
        error: "id is required",
      });
    }

    const data = await getDocumentById(id);

    let pdfUrl = data?.parsed_json?.pdf_url || null;

    if (!pdfUrl && data?.storage_path) {
      const { data: publicUrlData } = supabase.storage
        .from(process.env.SUPABASE_BUCKET)
        .getPublicUrl(data.storage_path);

      pdfUrl = publicUrlData?.publicUrl || null;
    }

    return res.json({
      success: true,
      document_id: data.id,
      status: data.status,
      file_name: data.storage_path || null,
      pdf_url: pdfUrl,
      html_url: buildReportUrl(req, data.id),
      parsed_json: data.parsed_json || null,
    });
  } catch (error) {
    console.error("ERROR IN /job-status:", error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown job status error",
    });
  }
});

app.post("/full-coa-pipeline", async (req, res) => {
  let documentId = null;

  try {
    const fileUrl = req.body?.file_url;
    const originalFilename =
      req.body?.original_filename ||
      fileUrl?.split("/").pop() ||
      `upload-${Date.now()}`;
    const documentType = req.body?.document_type || "coa";

    if (!fileUrl) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const { data: row, error } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename: originalFilename,
          document_type: documentType,
          status: "queued",
          parsed_json: {
            source_url: fileUrl,
          },
        },
      ])
      .select()
      .single();

    if (error) {
      throw new Error(`Could not create pipeline document row: ${error.message}`);
    }

    documentId = row.id;

    processCOAJob({
      documentId,
      fileUrl,
      originalFilename,
      documentType,
    }).catch((err) => {
      console.error("Detached pipeline job failed:", err.message);
    });

    return res.json({
      success: true,
      async: true,
      document_id: documentId,
      job_id: documentId,
      status: "queued",
      html_url: buildReportUrl(req, documentId),
      message: "COA job started. Check /job-status with this id.",
    });
  } catch (error) {
    console.error("ERROR IN /full-coa-pipeline:", error);

    if (documentId) {
      try {
        await updateDocumentStatus(documentId, {
          status: "failed",
        });
      } catch (statusErr) {
        console.error("Failed to mark document as failed:", statusErr.message);
      }
    }

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown full pipeline error",
    });
  }
});

app.post("/generate-report", async (req, res) => {
  let documentId = null;

  try {
    const { data } = parseIncomingBody(req.body);

    const { data: documentRow, error: documentError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename:
            sanitizeFileName(data?.product_name || `coa-${Date.now()}`) + ".pdf",
          mime_type: "text/html",
          document_type: "coa",
          status: "completed",
          parsed_json: data,
        },
      ])
      .select()
      .single();

    if (documentError) {
      throw new Error(`Could not create document row: ${documentError.message}`);
    }

    documentId = documentRow.id;

    return res.json({
      success: true,
      document_id: documentId,
      html_url: buildReportUrl(req, documentId),
      message: "HTML report created successfully",
    });
  } catch (error) {
    console.error("ERROR IN /generate-report:", error);

    if (documentId) {
      try {
        await updateDocumentStatus(documentId, {
          status: "failed",
        });
      } catch (statusErr) {
        console.error("Failed to mark generated report as failed:", statusErr.message);
      }
    }

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown server error",
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 