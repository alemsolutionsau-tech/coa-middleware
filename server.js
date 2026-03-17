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
          payload.file_name || payload.report_json?.product_name || `coa-${Date.now()}`
        ) + ".pdf",
      data: payload.report_json,
    };
  }

  if (payload.report_json_string && typeof payload.report_json_string === "string") {
    let parsedInner;
    try {
      parsedInner = JSON.parse(payload.report_json_string);
    } catch (err) {
      throw new Error(`report_json_string is not valid JSON: ${err.message}`);
    }

    if (!parsedInner || typeof parsedInner !== "object" || Array.isArray(parsedInner)) {
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
        sanitizeFileName(payload.file_name || payload.product_name || `coa-${Date.now()}`) +
        ".pdf",
      data: payload,
    };
  }

  throw new Error("Missing valid report_json or report_json_string");
}

function renderList(items = [], emptyText = "Not reported") {
  if (!Array.isArray(items) || !items.length) return `<div class="muted">${esc(emptyText)}</div>`;
  return `<ul>${items.map((x) => `<li>${esc(typeof x === "string" ? x : JSON.stringify(x))}</li>`).join("")}</ul>`;
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
      <div class="sub">${esc(data?.opening_statement || data?.overall_score || "Interpretive certificate of analysis summary.")}</div>

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
        <div class="copy">${esc(data?.scientific_references || "No scientific references included.")}</div>
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

async function parseCOAWithOpenAI(cleanText = "") {
  if (!cleanText || !String(cleanText).trim()) {
    throw new Error("cleanText is empty");
  }

  const response = await openai.responses.create({
    model: OPENAI_MODEL,
    store: false,
    reasoning: {
      effort: "medium",
    },
    max_output_tokens: 4000,
    text: {
      format: {
        type: "json_schema",
        name: "coa_report",
        schema: {
          type: "object",
          properties: {
            product_name: { type: "string" },
            batch_number: { type: "string" },
            coa_report_date: { type: "string" },
            product_type: { type: "string" },
            laboratory_name: { type: "string" },
            overall_score: { type: "string" },
            opening_statement: { type: "string" },
            thc_total: { type: "string" },
            cbd_total: { type: "string" },
            total_terpenes: { type: "string" },
            minor_cannabinoids: { type: "string" },
            contaminant_overview: { type: "string" },
            lab_quality_summary: { type: "string" },
            scientific_references: { type: "string" },
            positive_flags: {
              type: "array",
              items: { type: "string" },
            },
            warning_flags: {
              type: "array",
              items: { type: "string" },
            },
            top_cannabinoids: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  name: { type: "string" },
                  value: { type: "string" },
                  unit: { type: "string" },
                  notes: { type: "string" },
                },
                required: ["name", "value", "unit", "notes"],
                additionalProperties: false,
              },
            },
            top_terpenes: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  name: { type: "string" },
                  value: { type: "string" },
                  unit: { type: "string" },
                },
                required: ["name", "value", "unit"],
                additionalProperties: false,
              },
            },
          },
          required: [
            "product_name",
            "batch_number",
            "coa_report_date",
            "product_type",
            "laboratory_name",
            "overall_score",
            "opening_statement",
            "thc_total",
            "cbd_total",
            "total_terpenes",
            "minor_cannabinoids",
            "contaminant_overview",
            "lab_quality_summary",
            "scientific_references",
            "positive_flags",
            "warning_flags",
            "top_cannabinoids",
            "top_terpenes",
          ],
          additionalProperties: false,
        },
      },
    },
    input: [
      {
        role: "system",
        content: [
          {
            type: "input_text",
            text:
              "You are the Alem Solutions COA parsing engine. Read the extracted OCR text and return only structured JSON matching the schema. Do not invent facts. If a field is not explicit, return an empty string or empty array.",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: cleanText,
          },
        ],
      },
    ],
  });

  const outputText = response.output_text;

  if (!outputText) {
    throw new Error("OpenAI returned empty output_text");
  }

  try {
    return JSON.parse(outputText);
  } catch (err) {
    throw new Error(`OpenAI output was not valid JSON: ${err.message}`);
  }
}

async function extractDocumentFromUrl(file_url) {
  const fileResponse = await axios.get(file_url, {
    responseType: "arraybuffer",
    timeout: 60000,
  });

  const mimeType = detectMimeType(
    file_url,
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

  const plain_text = pages.map((p) => p.text).join("\n\n");

  return {
    mimeType,
    pages,
    tables,
    plain_text,
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
        !!process.env.SUPABASE_SERVICE_ROLE_KEY || !!process.env.SUPABASE_ANON_KEY,
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
      "/generate-report",
    ],
  });
});

app.post("/extract-from-url", async (req, res) => {
  try {
    const file_url = req.body?.file_url;

    if (!file_url) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const extracted = await extractDocumentFromUrl(file_url);

    return res.json({
      success: true,
      engine_used: "azure-document-intelligence",
      model_used: "prebuilt-layout",
      source_url: file_url,
      mime_type: extracted.mimeType,
      plain_text: extracted.plain_text,
      pages: extracted.pages,
      tables: extracted.tables,
      metadata: {
        page_count: extracted.pages.length,
      },
    });
  } catch (error) {
    console.error("ERROR IN /extract-from-url:");
    console.error(error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown extraction error",
      details: error?.response?.data || null,
    });
  }
});

app.post("/extract-and-save", async (req, res) => {
  try {
    const file_url = req.body?.file_url;
    const original_filename =
      req.body?.original_filename || file_url?.split("/").pop() || `upload-${Date.now()}`;
    const document_type = req.body?.document_type || "coa";

    if (!file_url) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const extracted = await extractDocumentFromUrl(file_url);

    const { data: insertedRow, error: insertError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename,
          mime_type: extracted.mimeType,
          document_type,
          status: "extracted",
          extracted_text: extracted.plain_text,
          parsed_json: {
            source_url: file_url,
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
      throw new Error(`Could not save extracted document: ${insertError.message}`);
    }

    return res.json({
      success: true,
      document_id: insertedRow.id,
      source_url: file_url,
      mime_type: extracted.mimeType,
      plain_text: extracted.plain_text,
      page_count: extracted.pages.length,
    });
  } catch (error) {
    console.error("ERROR IN /extract-and-save:");
    console.error(error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown extraction/save error",
    });
  }
});

app.post("/extract-parse-and-save", async (req, res) => {
  try {
    const file_url = req.body?.file_url;
    const original_filename =
      req.body?.original_filename || file_url?.split("/").pop() || `upload-${Date.now()}`;
    const document_type = req.body?.document_type || "coa";

    if (!file_url) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const extracted = await extractDocumentFromUrl(file_url);
    const parsed_json = await parseCOAWithOpenAI(extracted.plain_text);

    const { data: insertedRow, error: insertError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename,
          mime_type: extracted.mimeType,
          document_type,
          status: "parsed",
          extracted_text: extracted.plain_text,
          parsed_json: {
            ...parsed_json,
            extraction_meta: {
              source_url: file_url,
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
      source_url: file_url,
      mime_type: extracted.mimeType,
      plain_text: extracted.plain_text,
      parsed_json,
    });
  } catch (error) {
    console.error("ERROR IN /extract-parse-and-save:");
    console.error(error);

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown parse error",
    });
  }
});
app.post("/full-coa-pipeline", async (req, res) => {
  let browser;

  try {
    const file_url = req.body?.file_url;
    const original_filename =
      req.body?.original_filename || file_url?.split("/").pop() || `upload-${Date.now()}`;
    const document_type = req.body?.document_type || "coa";

    if (!file_url) {
      return res.status(400).json({
        success: false,
        error: "file_url is required",
      });
    }

    const extracted = await extractDocumentFromUrl(file_url);
    const parsed_json = await parseCOAWithOpenAI(extracted.plain_text);

    const pdfBaseName = sanitizeFileName(
      parsed_json?.product_name || original_filename || `coa-${Date.now()}`
    );
    const fileName = `${pdfBaseName}.pdf`;

    const { data: documentRow, error: documentError } = await supabase
      .from("documents")
      .insert([
        {
          user_id: "bubble-user",
          source: "bubble",
          original_filename,
          mime_type: extracted.mimeType,
          document_type,
          status: "processing",
          extracted_text: extracted.plain_text,
          parsed_json: {
            ...parsed_json,
            extraction_meta: {
              source_url: file_url,
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
      throw new Error(`Could not create pipeline document row: ${documentError.message}`);
    }

    const documentId = documentRow.id;
    const html = renderReportHTML(parsed_json);

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
      const html = document.documentElement;
      return Math.max(
        body.scrollHeight,
        body.offsetHeight,
        html.clientHeight,
        html.scrollHeight,
        html.offsetHeight
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

    await supabase
      .from("documents")
      .update({
        status: "completed",
        storage_path: fileName,
        mime_type: "application/pdf",
      })
      .eq("id", documentId);

    await browser.close();
    browser = null;

    return res.json({
      success: true,
      document_id: documentId,
      file_name: fileName,
      pdf_url: pdfUrl,
      parsed_json,
    });
  } catch (error) {
    console.error("ERROR IN /full-coa-pipeline:");
    console.error(error);

    if (browser) {
      try {
        await browser.close();
      } catch (closeErr) {
        console.error("ERROR CLOSING BROWSER:", closeErr);
      }
    }

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown full pipeline error",
    });
  }
});

app.post("/generate-report", async (req, res) => {
  let browser;

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

    const documentId = documentRow.id;
    const html = renderReportHTML(data);

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
      const html = document.documentElement;
      return Math.max(
        body.scrollHeight,
        body.offsetHeight,
        html.clientHeight,
        html.scrollHeight,
        html.offsetHeight
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

    await supabase
      .from("documents")
      .update({
        status: "completed",
        storage_path: fileName,
        extracted_text: null,
        parsed_json: data,
      })
      .eq("id", documentId);

    await browser.close();
    browser = null;

    return res.json({
      success: true,
      file_name: fileName,
      pdf_url: pdfUrl,
    });
  } catch (error) {
    console.error("ERROR IN /generate-report:");
    console.error(error);

    if (browser) {
      try {
        await browser.close();
      } catch (closeErr) {
        console.error("ERROR CLOSING BROWSER:", closeErr);
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