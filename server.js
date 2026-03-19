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

  const aromaticProfile = topTerpenes.length
    ? topTerpenes
        .slice(0, 4)
        .map((x) => x?.name)
        .filter(Boolean)
        .join(" • ")
    : "Aromatic profile not available";

  const experienceProfile = positiveFlags.length
    ? positiveFlags.slice(0, 4)
    : [
        "Chemotype review in progress",
        "Awaiting more interpretive signals",
      ];

  const useCases = warningFlags.length
    ? ["Requires clinician review", "Check lab details before relying on use-case claims"]
    : positiveFlags.slice(0, 3).length
    ? positiveFlags.slice(0, 3)
    : ["General interpretive guidance only"];

  const coaQualityScore = Math.max(
    4,
    Math.min(
      10,
      (
        (topCannabinoids.length ? 2 : 0) +
        (topTerpenes.length ? 2 : 0) +
        (data?.contaminant_overview ? 2 : 0) +
        (data?.lab_quality_summary ? 2 : 0) +
        (data?.laboratory_name ? 1 : 0) +
        (data?.coa_report_date ? 1 : 0)
      )
    )
  );

  const completenessItems = [
    {
      label: "Cannabinoids",
      value: topCannabinoids.length || data?.thc_total || data?.cbd_total ? "Complete" : "Limited",
      className: topCannabinoids.length || data?.thc_total || data?.cbd_total ? "good" : "warn",
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
      value: data?.laboratory_name && data?.coa_report_date ? "Complete" : "Partial",
      className: data?.laboratory_name && data?.coa_report_date ? "good" : "warn",
    },
  ];

  const renderTopMetric = (label, value, accentClass = "") => `
    <div class="top-metric ${accentClass}">
      <div class="top-metric-label">${esc(label)}</div>
      <div class="top-metric-value">${esc(value || "Not reported")}</div>
    </div>
  `;

  const renderPillList = (items = [], emptyText = "Not reported") => {
    if (!Array.isArray(items) || !items.length) {
      return `<div class="muted">${esc(emptyText)}</div>`;
    }
    return `
      <div class="pill-wrap">
        ${items
          .map((item) => `<span class="pill">${esc(item)}</span>`)
          .join("")}
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

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${esc(data?.product_name || "ALEM COA Intelligence Report")}</title>
<style>
  :root {
    --bg: #071d24;
    --bg-2: #0b2830;
    --panel: rgba(255,255,255,0.05);
    --panel-2: rgba(255,255,255,0.08);
    --line: rgba(255,255,255,0.10);
    --text: #edf6f2;
    --muted: #a8c3bc;
    --green: #8fd7a7;
    --green-2: #6ec594;
    --teal: #7fd7d4;
    --gold: #d8bf6a;
    --navy-card: #173c63;
    --white: #ffffff;
    --danger: #ef8d8d;
    --warn: #f0d18b;
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0;
    padding: 0;
    background:
      radial-gradient(circle at top right, rgba(111, 213, 164, 0.10), transparent 24%),
      radial-gradient(circle at top left, rgba(87, 164, 210, 0.09), transparent 28%),
      linear-gradient(180deg, #051720 0%, #08202a 50%, #061922 100%);
    color: var(--text);
    font-family: Arial, Helvetica, sans-serif;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
      linear-gradient(rgba(7,29,36,0.72), rgba(7,29,36,0.90)),
      url("https://images.unsplash.com/photo-1603909223429-69bb7101f420?auto=format&fit=crop&w=1200&q=60") center/cover no-repeat;
    opacity: 0.12;
  }

  .shell {
    position: relative;
    z-index: 1;
    max-width: 1320px;
    margin: 0 auto;
    padding: 24px 22px 42px;
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
    background: rgba(5, 18, 24, 0.84);
    backdrop-filter: blur(10px);
  }

  .btn {
    appearance: none;
    border: 0;
    border-radius: 999px;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .btn-primary {
    background: linear-gradient(90deg, var(--green), #b0f0c4);
    color: #052129;
  }

  .btn-secondary {
    background: rgba(255,255,255,0.07);
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
    border: 1px solid rgba(143,215,167,0.18);
    border-radius: 30px;
    padding: 34px 30px 28px;
    background:
      linear-gradient(135deg, rgba(7,29,36,0.95), rgba(10,36,44,0.90)),
      radial-gradient(circle at right top, rgba(143,215,167,0.16), transparent 32%);
    box-shadow: 0 18px 50px rgba(0,0,0,0.25);
  }

  .hero::after {
    content: "";
    position: absolute;
    right: -80px;
    top: -80px;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(143,215,167,0.13), transparent 70%);
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
    letter-spacing: -0.02em;
  }

  .hero-subtitle {
    margin-top: 14px;
    max-width: 780px;
    color: #d6e6df;
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
    background: rgba(255,255,255,0.05);
    color: var(--text);
    font-size: 13px;
    font-weight: 700;
  }

  .status-badge.approved {
    background: rgba(110, 197, 148, 0.14);
    border-color: rgba(110, 197, 148, 0.35);
    color: #c9f2d8;
  }

  .status-badge.inconclusive {
    background: rgba(216, 191, 106, 0.14);
    border-color: rgba(216, 191, 106, 0.35);
    color: #f3e4b7;
  }

  .status-badge.review {
    background: rgba(127, 215, 212, 0.14);
    border-color: rgba(127, 215, 212, 0.35);
    color: #d2f4f3;
  }

  .hero-score {
    min-width: 220px;
    border: 1px solid var(--line);
    border-radius: 26px;
    padding: 18px 18px 16px;
    background: rgba(255,255,255,0.05);
    text-align: center;
  }

  .hero-score-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 8px;
  }

  .hero-score-value {
    font-size: 30px;
    font-weight: 800;
    color: var(--green);
    line-height: 1.1;
  }

  .hero-score-note {
    margin-top: 8px;
    font-size: 13px;
    line-height: 1.5;
    color: #d5e2dc;
  }

  .section {
    margin-top: 22px;
  }

  .section-title {
    margin: 0 0 14px;
    font-size: 30px;
    color: var(--white);
    letter-spacing: -0.02em;
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
    box-shadow: 0 10px 24px rgba(0,0,0,0.14);
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 18px;
  }

  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }

  .grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
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
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--line);
  }

  .top-metric.thc {
    background: linear-gradient(180deg, rgba(23,60,99,0.85), rgba(23,60,99,0.65));
  }

  .top-metric.cbd {
    background: linear-gradient(180deg, rgba(216,191,106,0.18), rgba(216,191,106,0.10));
  }

  .top-metric.terps {
    background: linear-gradient(180deg, rgba(110,197,148,0.22), rgba(110,197,148,0.10));
  }

  .top-metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #dce9e5;
    margin-bottom: 10px;
  }

  .top-metric-value {
    font-size: 34px;
    font-weight: 800;
    line-height: 1.05;
  }

  .minor-box {
    margin-top: 14px;
    padding: 16px;
    border-radius: 18px;
    background: rgba(23,60,99,0.72);
    border: 1px solid rgba(255,255,255,0.10);
  }

  .minor-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #dce9e5;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .minor-copy {
    color: #f1f6f3;
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
    color: var(--white);
  }

  .bar-shell {
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.06);
  }

  .bar-shell span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--green-2), #c0f0cf);
  }

  .bar-shell.gold span {
    background: linear-gradient(90deg, #d2b75d, #f2df9a);
  }

  .bar-shell.teal span {
    background: linear-gradient(90deg, #60cfd0, #9febea);
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
    background: rgba(255,255,255,0.04);
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
    color: var(--white);
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
    background: linear-gradient(180deg, rgba(23,60,99,0.92), rgba(23,60,99,0.72));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 22px;
  }

  .terpene-hero-title {
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #d7ebe3;
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
    border-bottom: 1px solid rgba(255,255,255,0.10);
    font-size: 15px;
  }

  .terpene-big strong {
    color: var(--white);
  }

  .aroma-box {
    margin-top: 16px;
    padding: 14px 16px;
    border-radius: 18px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
  }

  .aroma-title {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .aroma-copy {
    color: var(--white);
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
    color: var(--white);
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
    background: rgba(255,255,255,0.04);
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .status-card.active.approved {
    background: rgba(110,197,148,0.16);
    border-color: rgba(110,197,148,0.34);
    color: #d2f5de;
  }

  .status-card.active.inconclusive {
    background: rgba(216,191,106,0.16);
    border-color: rgba(216,191,106,0.34);
    color: #f3e4b7;
  }

  .status-card.active.review {
    background: rgba(127,215,212,0.16);
    border-color: rgba(127,215,212,0.34);
    color: #d2f4f3;
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
    background: rgba(255,255,255,0.04);
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
    background: rgba(110,197,148,0.16);
    color: #cef4db;
    border: 1px solid rgba(110,197,148,0.30);
  }

  .check-pill.warn {
    background: rgba(216,191,106,0.16);
    color: #f2e4b8;
    border: 1px solid rgba(216,191,106,0.30);
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
    border: 10px solid rgba(143,215,167,0.20);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background:
      radial-gradient(circle at center, rgba(143,215,167,0.14), rgba(255,255,255,0.02));
    box-shadow: inset 0 0 30px rgba(0,0,0,0.12);
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
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--line);
    color: var(--white);
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
    color: #e4eeea;
  }

  .muted {
    font-size: 14px;
    line-height: 1.7;
    color: var(--muted);
  }

  .future-box {
    border-radius: 24px;
    border: 1px dashed rgba(143,215,167,0.30);
    padding: 20px;
    background: rgba(255,255,255,0.03);
  }

  .future-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--green);
    margin-bottom: 10px;
    font-weight: 700;
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
    .insight-grid {
      grid-template-columns: 1fr;
    }

    .grid-4 {
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

    .grid-4,
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
    body::before {
      opacity: 0.08;
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
          <span id="pdfStatus" class="action-status"></span>
        </div>
      `
        : ""
    }

    <section class="hero">
      <div class="hero-top">
        <div>
          <div class="brand-mark">ALEM COA Intelligence Report</div>
          <h1>${esc(data?.product_name || "Cannabis Product")}</h1>
          <div class="hero-subtitle">
            ${esc(
              data?.opening_statement ||
                "Medicinal cannabis interpretive report generated from certificate of analysis data."
            )}
          </div>

          <div class="hero-badges">
            <span class="badge">Batch ${esc(data?.batch_number || "Not reported")}</span>
            <span class="badge">${esc(data?.product_type || "Product type not reported")}</span>
            <span class="badge">${esc(data?.laboratory_name || "Lab not reported")}</span>
            <span class="badge">${esc(data?.coa_report_date || "Date not reported")}</span>
            <span class="badge status-badge ${statusClass}">${esc(regulatoryStatus)}</span>
          </div>
        </div>

        <div class="hero-score">
          <div class="hero-score-label">Overall Intelligence Score</div>
          <div class="hero-score-value">${esc(overallScoreText)}</div>
          <div class="hero-score-note">
            Educational interpretive summary aligned to the Alem COA Analyzer style.
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
            <div class="meta-value">${esc(data?.batch_number || "Not reported")}</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">COA report date</div>
            <div class="meta-value">${esc(data?.coa_report_date || "Not reported")}</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Product type</div>
            <div class="meta-value">${esc(data?.product_type || "Not reported")}</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Laboratory name</div>
            <div class="meta-value">${esc(data?.laboratory_name || "Not reported")}</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2 class="section-title">Interpretive summary</h2>
        <div class="copy"><strong>Overall score:</strong> ${esc(data?.overall_score || "Not reported")}</div>
        <div class="copy" style="margin-top:10px;"><strong>Contaminant overview:</strong> ${esc(data?.contaminant_overview || "Not reported")}</div>
        <div class="copy" style="margin-top:10px;"><strong>Lab quality summary:</strong> ${esc(data?.lab_quality_summary || "Not reported")}</div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Cannabinoid profile</h2>
        <div class="grid-3">
          ${renderTopMetric("THC", data?.thc_total, "thc")}
          ${renderTopMetric("CBD", data?.cbd_total, "cbd")}
          ${renderTopMetric("Terpenes", data?.total_terpenes, "terps")}
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

    <section class="section">
      <div class="insight-grid">
        <div class="card insight-card">
          <div class="insight-label">Experience profile</div>
          ${renderPillList(experienceProfile, "No experiential interpretation available")}
        </div>

        <div class="card insight-card">
          <div class="insight-label">Potential use cases</div>
          ${renderPillList(useCases, "No use-case guidance available")}
        </div>

        <div class="card insight-card">
          <div class="insight-label">Scientific notes</div>
          <div class="copy">${esc(data?.scientific_references || "No scientific notes included.")}</div>
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
        <div class="future-title">Chemical fingerprint</div>
        <div class="copy">
          This section is reserved for your future fingerprint radar, chemotype clustering, strain similarity, and breeding signal detection layer.
        </div>
      </div>
    </section>

    <div class="footer">
      Medicinal cannabis is a regulated treatment option in Australia and this report is an educational interpretive layer only. It does not replace physician advice, pharmacist counselling, regulatory review, or direct laboratory confirmation.
    </div>
  </div>

  ${
    options.documentId
      ? `
      <script>
        (function () {
          const btn = document.getElementById("generatePdfBtn");
          const status = document.getElementById("pdfStatus");

          if (!btn) return;

          btn.addEventListener("click", async function () {
            try {
              btn.disabled = true;
              btn.textContent = "Generating PDF...";
              if (status) status.textContent = "";

              const response = await fetch("/generate-pdf/${options.documentId}", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json"
                }
              });

              const result = await response.json();

              if (!response.ok || !result.success) {
                throw new Error(result.error || "Failed to generate PDF");
              }

              if (status) status.textContent = "PDF ready";
              btn.textContent = "Generate PDF again";

              if (result.pdf_url) {
                window.open(result.pdf_url, "_blank");
              }
            } catch (err) {
              console.error(err);
              if (status) status.textContent = err.message || "PDF generation failed";
              btn.textContent = "Generate PDF";
            } finally {
              btn.disabled = false;
            }
          });
        })();
      </script>
    `
      : ""
  }
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