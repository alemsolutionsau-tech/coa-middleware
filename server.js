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
const MAX_OCR_CHARS_FOR_OPENAI = Number(process.env.MAX_OCR_CHARS_FOR_OPENAI || 25000);

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
  const match = String(v).match(/-?\d+(\.\d+)?/);
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

function renderReportHTML(data = {}) {
  const topCannabinoids = getSafeArray(data?.top_cannabinoids);
  const topTerpenes = getSafeArray(data?.top_terpenes);

  const thcNumber = numberFromPercentString(data?.thc_total);
  const cbdNumber = numberFromPercentString(data?.cbd_total);
  const terpNumber = numberFromPercentString(data?.total_terpenes);

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>${esc(data?.product_name || "COA Intelligence Report")}</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 0;
    background: #0b1110;
    color: #edf3ef;
    font-family: Arial, Helvetica, sans-serif;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .page {
    width: 100%;
    padding: 36px 42px 42px;
  }
  .hero {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 28px;
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  }
  .brand {
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #97aca2;
    margin-bottom: 10px;
  }
  h1 {
    margin: 0 0 10px;
    font-size: 42px;
    line-height: 1.05;
  }
  h2 {
    margin: 0 0 12px;
    font-size: 22px;
  }
  .sub {
    color: #cad7d1;
    line-height: 1.7;
    font-size: 15px;
  }
  .section {
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.08);
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 18px;
  }
  .metric {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px;
    background: rgba(255,255,255,0.03);
  }
  .metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #97aca2;
    margin-bottom: 8px;
  }
  .metric-value {
    font-size: 24px;
    font-weight: 700;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }
  .card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    background: rgba(255,255,255,0.02);
  }
  .muted {
    color: #97aca2;
    font-size: 14px;
    line-height: 1.6;
  }
  .copy {
    color: #dbe6e0;
    line-height: 1.75;
    font-size: 14px;
  }
  .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 16px;
  }
  .meta-item {
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 8px;
  }
  .meta-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #97aca2;
    margin-bottom: 6px;
  }
  .meta-value {
    font-size: 14px;
    font-weight: 700;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th, td {
    text-align: left;
    padding: 10px 8px;
    vertical-align: top;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  th {
    color: #97aca2;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .bar-wrap {
    margin-top: 12px;
  }
  .bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 6px;
    color: #dbe6e0;
  }
  .bar {
    height: 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
  }
  .bar > span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, #6bcf72, #9bf19a);
  }
  ul {
    margin: 8px 0 0 18px;
    padding: 0;
    line-height: 1.7;
  }
  .footer {
    margin-top: 24px;
    color: #97aca2;
    font-size: 11px;
    line-height: 1.7;
    text-align: center;
  }
  @page { margin: 0; }
</style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div class="brand">Alem Solutions · COA Intelligence Report</div>
      <h1>${esc(data?.product_name || "Cannabis Product")}</h1>
      <div class="sub">${esc(
        data?.opening_statement ||
          data?.overall_score ||
          "Interpretive certificate of analysis summary."
      )}</div>

      <div class="metrics">
        ${renderMetric("Total THC", data?.thc_total)}
        ${renderMetric("Total CBD", data?.cbd_total)}
        ${renderMetric("Total Terpenes", data?.total_terpenes)}
        ${renderMetric("Laboratory", data?.laboratory_name)}
      </div>
    </div>

    <div class="section grid-2">
      <div class="card">
        <h2>Batch & document details</h2>
        <div class="meta-grid">
          <div class="meta-item"><div class="meta-label">Batch</div><div class="meta-value">${esc(data?.batch_number || "Not reported")}</div></div>
          <div class="meta-item"><div class="meta-label">COA date</div><div class="meta-value">${esc(data?.coa_report_date || "Not reported")}</div></div>
          <div class="meta-item"><div class="meta-label">Product type</div><div class="meta-value">${esc(data?.product_type || "Not reported")}</div></div>
          <div class="meta-item"><div class="meta-label">Lab</div><div class="meta-value">${esc(data?.laboratory_name || "Not reported")}</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Interpretive summary</h2>
        <div class="copy"><strong>Overall score:</strong> ${esc(data?.overall_score || "Not reported")}</div>
        <div class="copy" style="margin-top:10px;"><strong>Minor cannabinoids:</strong> ${esc(data?.minor_cannabinoids || "Not reported")}</div>
        <div class="copy" style="margin-top:10px;"><strong>Contaminant overview:</strong> ${esc(data?.contaminant_overview || "Not reported")}</div>
        <div class="copy" style="margin-top:10px;"><strong>Lab quality summary:</strong> ${esc(data?.lab_quality_summary || "Not reported")}</div>
      </div>
    </div>

    <div class="section grid-2">
      <div class="card">
        <h2>Potency bars</h2>

        <div class="bar-wrap">
          <div class="bar-label"><span>THC</span><span>${esc(data?.thc_total || "0")}</span></div>
          <div class="bar"><span style="width:${Math.max(0, Math.min(100, thcNumber * 3))}%"></span></div>
        </div>

        <div class="bar-wrap">
          <div class="bar-label"><span>CBD</span><span>${esc(data?.cbd_total || "0")}</span></div>
          <div class="bar"><span style="width:${Math.max(0, Math.min(100, cbdNumber * 10))}%"></span></div>
        </div>

        <div class="bar-wrap">
          <div class="bar-label"><span>Total terpenes</span><span>${esc(data?.total_terpenes || "0")}</span></div>
          <div class="bar"><span style="width:${Math.max(0, Math.min(100, terpNumber * 20))}%"></span></div>
        </div>
      </div>

      <div class="card">
        <h2>Scientific notes</h2>
        <div class="copy">${esc(
          data?.scientific_references || "No scientific references included."
        )}</div>
      </div>
    </div>

    <div class="section grid-2">
      <div class="card">
        <h2>Top cannabinoids</h2>
        ${renderCannabinoidTable(topCannabinoids)}
      </div>

      <div class="card">
        <h2>Top terpenes</h2>
        ${renderTerpeneTable(topTerpenes)}
      </div>
    </div>

    <div class="section grid-2">
      <div class="card">
        <h2>Positive flags</h2>
        ${renderList(data?.positive_flags, "No positive flags reported")}
      </div>

      <div class="card">
        <h2>Watchouts</h2>
        ${renderList(data?.warning_flags, "No warning flags reported")}
      </div>
    </div>

    <div class="footer">
      This report is an educational interpretive layer based on certificate of analysis data and does not replace physician advice, pharmacist counselling, regulatory review, or direct laboratory confirmation.
    </div>
  </div>
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
    "\n\n[TRUNCATED FOR MODEL INPUT]\n\n",
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

    return chunks.join("\n").trim();
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

async function callOpenAIForJSON(systemPrompt, userText, maxOutputTokens = 2200) {
  const response = await openai.responses.create({
  model: OPENAI_MODEL,
  store: false,

  // 🔥 THIS FIXES YOUR ISSUE
  text: {
    format: {
      type: "text"
    }
  },

  reasoning: {
    effort: "low" // also reduce this
  },

  max_output_tokens: 2200,

  input: [
    {
      role: "system",
      content: extractionPrompt
    },
    {
      role: "user",
      content: modelInput
    }
  ]
});

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

  const response = await callOpenAIForJSON(repairPrompt, modelInput, 2200);

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

async function parseCOAWithOpenAI(cleanText = "") {
  if (!cleanText || !String(cleanText).trim()) {
    throw new Error("cleanText is empty");
  }

  const modelInput = prepareOCRTextForModel(cleanText);

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

  try {
    console.log("OCR TEXT ORIGINAL LENGTH:", String(cleanText).length);
    console.log("OCR TEXT MODEL INPUT LENGTH:", String(modelInput).length);
    console.log("OCR TEXT PREVIEW:", safeSnippet(modelInput, 1500));

    const response = await callOpenAIForJSON(extractionPrompt, modelInput, 2200);

    logOpenAIResponseMeta("RAW", response);

    const rawText = extractTextFromOpenAIResponse(response);

    console.log("RAW OPENAI OUTPUT LENGTH:", rawText ? rawText.length : 0);
    console.log("RAW OPENAI OUTPUT START:", safeSnippet(rawText, 1000));
    console.log("RAW OPENAI OUTPUT END:", String(rawText || "").slice(-1000));

if (!rawText.trim()) {
  console.warn("⚠️ Empty OpenAI output, retrying with fallback prompt...");

  const retry = await openai.responses.create({
    model: OPENAI_MODEL,

    text: {
      format: { type: "text" }
    },

    reasoning: { effort: "minimal" },
    max_output_tokens: 1500,

    input: [
      {
        role: "system",
        content: "Return only JSON. No thinking. No explanation."
      },
      {
        role: "user",
        content: modelInput.slice(0, 15000)
      }
    ]
  });

  const retryText = extractTextFromOpenAIResponse(retry);

  console.log("RETRY OUTPUT LENGTH:", retryText.length);

  if (retryText.trim()) {
    let retryJson = extractJSONObject(retryText);

    if (retryJson.endsWith(",")) {
      retryJson = retryJson.slice(0, -1);
    }

    return normalizeParsedCOA(JSON.parse(retryJson));
  }

  console.error("❌ Retry also failed");

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
          "COA parsed partially due to malformed or empty model output",
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
    const text = (page.lines || []).map((line) => line.content).join("\n");

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

  const plainText = pages.map((p) => p.text).join("\n\n");

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
      "/generate-report",
      "/test",
    ],
  });
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
          status: "parsed",
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

    const extracted = await extractDocumentFromUrl(fileUrl);
    const parsedJson = await safeParseCOA(extracted.plain_text);

    const pdfBaseName = sanitizeFileName(
      parsedJson?.product_name || originalFilename || `coa-${Date.now()}`
    );
    const fileName = `${pdfBaseName}.pdf`;

    const { data: documentRow, error: documentError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename: originalFilename,
          mime_type: extracted.mimeType,
          document_type: documentType,
          status: "processing",
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

    if (documentError) {
      throw new Error(
        `Could not create pipeline document row: ${documentError.message}`
      );
    }

    documentId = documentRow.id;

    const html = renderReportHTML(parsedJson);
    const pdfBuffer = await buildPdfBufferFromHtml(html);
    const pdfUrl = await uploadPdfToSupabase(fileName, pdfBuffer);

    const { error: updateError } = await supabase
      .from("documents")
      .update({
        status: "completed",
        storage_path: fileName,
        mime_type: "application/pdf",
      })
      .eq("id", documentId);

    if (updateError) {
      throw new Error(`Could not update completed document: ${updateError.message}`);
    }

    return res.json({
      success: true,
      document_id: documentId,
      file_name: fileName,
      pdf_url: pdfUrl,
      parsed_json: parsedJson,
    });
  } catch (error) {
    console.error("ERROR IN /full-coa-pipeline:", error);

    if (documentId) {
      try {
        await supabase
          .from("documents")
          .update({
            status: "failed",
          })
          .eq("id", documentId);
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
    const { data, fileName } = parseIncomingBody(req.body);

    const { data: documentRow, error: documentError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename: fileName,
          mime_type: "application/pdf",
          document_type: "coa",
          status: "processing",
        },
      ])
      .select()
      .single();

    if (documentError) {
      throw new Error(`Could not create document row: ${documentError.message}`);
    }

    documentId = documentRow.id;

    const html = renderReportHTML(data);
    const pdfBuffer = await buildPdfBufferFromHtml(html);
    const pdfUrl = await uploadPdfToSupabase(fileName, pdfBuffer);

    const { error: updateError } = await supabase
      .from("documents")
      .update({
        status: "completed",
        storage_path: fileName,
        extracted_text: null,
        parsed_json: data,
      })
      .eq("id", documentId);

    if (updateError) {
      throw new Error(`Could not update generated report row: ${updateError.message}`);
    }

    return res.json({
      success: true,
      file_name: fileName,
      pdf_url: pdfUrl,
    });
  } catch (error) {
    console.error("ERROR IN /generate-report:", error);

    if (documentId) {
      try {
        await supabase
          .from("documents")
          .update({
            status: "failed",
          })
          .eq("id", documentId);
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