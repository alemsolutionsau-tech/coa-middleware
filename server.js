require("dotenv").config();

const path = require("path");
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const puppeteer = require("puppeteer");
const axios = require("axios");
const OpenAI = require("openai");
const {
  DocumentAnalysisClient,
  AzureKeyCredential,
} = require("@azure/ai-form-recognizer");

const supabase = require("./supabase");

const app = express();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = [
      "application/pdf",
      "image/png",
      "image/jpeg",
      "image/jpg",
      "image/tiff",
    ];
    if (allowed.includes(file.mimetype)) return cb(null, true);
    cb(new Error("Only PDF, PNG, JPG, JPEG, and TIFF files are allowed"));
  },
});

app.use(cors());
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-5";
const MAX_OCR_CHARS_FOR_OPENAI = Number(
  process.env.MAX_OCR_CHARS_FOR_OPENAI || 12000
);
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 20000);
const SUPABASE_BUCKET = process.env.SUPABASE_BUCKET || "raw_documents";

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

function getBaseUrl(req) {
  const proto = req.headers["x-forwarded-proto"] || req.protocol || "https";
  return `${proto}://${req.get("host")}`;
}

function buildReportUrl(req, id) {
  return `${getBaseUrl(req)}/report/${id}`;
}

function buildPdfUrl(req, id) {
  return `${getBaseUrl(req)}/pdf/${id}`;
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
    if (!response || typeof response !== "object") return "";

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

function extractJSONObject(text = "") {
  const trimmed = String(text || "").trim();

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
  console.log("🚀 Sending request to OpenAI.");

  const response = await Promise.race([
    openai.responses.create({
      model: OPENAI_MODEL,
      store: false,
      text: { format: { type: "text" } },
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

Rules:
- Return JSON only
- No markdown
- No explanation
- Fill missing values conservatively
`;

  const response = await callOpenAIForJSON(repairPrompt, modelInput, 1200);
  logOpenAIResponseMeta("REPAIR", response);

  const rawText = extractTextFromOpenAIResponse(response);
  const jsonText = extractJSONObject(rawText);
  return normalizeParsedCOA(JSON.parse(jsonText));
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
        if (retryJson.endsWith(",")) retryJson = retryJson.slice(0, -1);
        return normalizeParsedCOA(JSON.parse(retryJson));
      }

      throw new Error("No text received from OpenAI after retry");
    }

    let jsonText = extractJSONObject(rawText).trim();
    if (jsonText.endsWith(",")) jsonText = jsonText.slice(0, -1);

    return normalizeParsedCOA(JSON.parse(jsonText));
  } catch (parseError) {
    console.warn("JSON parse failed, attempting repair.");
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
    const text = (page.lines || []).map((line) => line.content).join("\n");

    return {
      page_number: page.pageNumber,
      width: page.width,
      height: page.height,
      unit: page.unit,
      text,
    };
  });

  const plainText = pages.map((p) => p.text).join("\n\n").trim();

  const tables = (result.tables || []).map((table, index) => ({
    table_index: index + 1,
    row_count: table.rowCount,
    column_count: table.columnCount,
  }));

  return {
    mimeType,
    plain_text: plainText,
    pages,
    tables,
    raw: result,
  };
}

async function uploadBufferToSupabase({
  buffer,
  originalName,
  mimeType,
  folder = "raw_documents",
}) {
  const safeName = sanitizeFileName(originalName || `upload-${Date.now()}`);
  const storagePath = `${folder}/${Date.now()}-${safeName}`;

  const { error: uploadError } = await supabase.storage
    .from(SUPABASE_BUCKET)
    .upload(storagePath, buffer, {
      contentType: mimeType || "application/octet-stream",
      upsert: false,
    });

  if (uploadError) {
    throw new Error(`Supabase storage upload failed: ${uploadError.message}`);
  }

  const { data: publicData } = supabase.storage
    .from(SUPABASE_BUCKET)
    .getPublicUrl(storagePath);

  const publicUrl = publicData?.publicUrl;
  if (!publicUrl) {
    throw new Error("Failed to generate public URL");
  }

  return {
    storagePath,
    publicUrl,
  };
}

async function insertCOAReport({
  parsedJson,
  sourceUrl,
  storagePath,
  originalFilename,
  mimeType,
}) {
  const payload = {
    report_json: {
      ...parsedJson,
      _meta: {
        source_url: sourceUrl || "",
        storage_path: storagePath || "",
        original_filename: originalFilename || "",
        mime_type: mimeType || "",
        saved_at: new Date().toISOString(),
      },
    },
    overall_score: parsedJson.overall_score || null,
    report_confidence_score: null,
    chemotype_identity: null,
    chemotype_descriptor: null,
    fingerprint_id: null,
  };

  const { data, error } = await supabase
    .from("coa_ai_reports")
    .insert([payload])
    .select()
    .single();

  if (error) {
    throw new Error(`Supabase insert failed: ${error.message}`);
  }

  return data;
}

async function getReportById(id) {
  const { data, error } = await supabase
    .from("coa_ai_reports")
    .select("*")
    .eq("id", id)
    .single();

  if (error) {
    throw new Error(`Could not load report: ${error.message}`);
  }

  return data;
}

function renderMetric(label, value) {
  return `
    <div class="metric">
      <div class="metric-label">${esc(label)}</div>
      <div class="metric-value">${esc(value || "Not reported")}</div>
    </div>
  `;
}

function renderList(items = [], emptyText = "Not reported") {
  if (!Array.isArray(items) || !items.length) {
    return `<div class="muted">${esc(emptyText)}</div>`;
  }

  return `<ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
}

function renderCannabinoids(items = []) {
  if (!Array.isArray(items) || !items.length) {
    return `<div class="muted">No cannabinoid rows reported.</div>`;
  }

  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Name</th><th>Value</th><th>Unit</th><th>Notes</th></tr>
        </thead>
        <tbody>
          ${items
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
    </div>
  `;
}

function renderTerpenes(items = []) {
  if (!Array.isArray(items) || !items.length) {
    return `<div class="muted">No terpene rows reported.</div>`;
  }

  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Name</th><th>Value</th><th>Unit</th></tr>
        </thead>
        <tbody>
          ${items
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
    </div>
  `;
}

function renderReportHTML(data = {}, options = {}) {
  const topCannabinoids = Array.isArray(data.top_cannabinoids)
    ? data.top_cannabinoids
    : [];
  const topTerpenes = Array.isArray(data.top_terpenes) ? data.top_terpenes : [];
  const positiveFlags = Array.isArray(data.positive_flags)
    ? data.positive_flags
    : [];
  const warningFlags = Array.isArray(data.warning_flags)
    ? data.warning_flags
    : [];

  const title = data.product_name || "COA Report";

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${esc(title)}</title>
<style>
  :root {
    --bg: #f5f3ef;
    --panel: #ffffff;
    --line: #e5dfd5;
    --text: #191816;
    --muted: #706c65;
    --green: #1f3d2b;
    --green-2: #2e5c42;
    --shadow: 0 14px 38px rgba(32,31,28,.06);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Inter, Arial, sans-serif;
    background: linear-gradient(180deg, #f7f5f1 0%, #f4f1eb 100%);
    color: var(--text);
  }
  .shell { max-width: 1180px; margin: 0 auto; padding: 24px 18px 50px; }
  .topbar {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    border: 1px solid var(--line);
    padding: 12px 18px;
    text-decoration: none;
    background: #fff;
    color: var(--text);
    font-weight: 700;
    cursor: pointer;
  }
  .btn-primary {
    background: linear-gradient(90deg, var(--green), var(--green-2));
    color: #fff;
    border-color: var(--green);
  }
  .hero, .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 24px;
    box-shadow: var(--shadow);
  }
  .hero { padding: 28px; margin-bottom: 18px; }
  .eyebrow {
    color: var(--green);
    text-transform: uppercase;
    letter-spacing: .18em;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 10px;
  }
  h1 {
    margin: 0 0 12px;
    font-size: 42px;
    letter-spacing: -.03em;
  }
  .subtitle { color: #45403a; line-height: 1.75; }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 18px;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 18px;
  }
  .card { padding: 20px; }
  .section-title {
    margin: 0 0 14px;
    font-size: 26px;
    letter-spacing: -.03em;
  }
  .metric {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px;
    background: #fff;
  }
  .metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 800;
  }
  .muted { color: var(--muted); line-height: 1.7; }
  ul { margin: 0; padding-left: 18px; }
  li { margin: 8px 0; line-height: 1.65; }
  .table-wrap { overflow: auto; }
  table { width: 100%; border-collapse: collapse; }
  th, td {
    text-align: left;
    padding: 12px;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
  }
  th { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    font-size: 12px;
    line-height: 1.6;
  }
  @media (max-width: 900px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
    h1 { font-size: 32px; }
  }
</style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <a class="btn" href="/">New Upload</a>
      ${
        options.documentId
          ? `<a class="btn btn-primary" href="/pdf/${options.documentId}" target="_blank" rel="noopener noreferrer">Open PDF</a>`
          : ""
      }
    </div>

    <section class="hero">
      <div class="eyebrow">ALEM COA Intelligence</div>
      <h1>${esc(title)}</h1>
      <div class="subtitle">${esc(
        data.opening_statement ||
          "Chemistry translated into practical intelligence from the uploaded COA."
      )}</div>

      <div class="grid-3">
        ${renderMetric("THC Total", data.thc_total)}
        ${renderMetric("CBD Total", data.cbd_total)}
        ${renderMetric("Total Terpenes", data.total_terpenes)}
      </div>
    </section>

    <div class="grid-2">
      <section class="card">
        <h2 class="section-title">Batch details</h2>
        <div class="muted">
          <strong>Batch:</strong> ${esc(data.batch_number || "Not reported")}<br />
          <strong>Report date:</strong> ${esc(data.coa_report_date || "Not reported")}<br />
          <strong>Product type:</strong> ${esc(data.product_type || "Not reported")}<br />
          <strong>Lab:</strong> ${esc(data.laboratory_name || "Not reported")}
        </div>
      </section>

      <section class="card">
        <h2 class="section-title">Summary</h2>
        <div class="muted">
          <strong>Overall score:</strong> ${esc(data.overall_score || "Not reported")}<br /><br />
          <strong>Minor cannabinoids:</strong> ${esc(
            data.minor_cannabinoids || "Not reported"
          )}<br /><br />
          <strong>Contaminants:</strong> ${esc(
            data.contaminant_overview || "Not reported"
          )}
        </div>
      </section>
    </div>

    <div class="grid-2">
      <section class="card">
        <h2 class="section-title">Top cannabinoids</h2>
        ${renderCannabinoids(topCannabinoids)}
      </section>

      <section class="card">
        <h2 class="section-title">Top terpenes</h2>
        ${renderTerpenes(topTerpenes)}
      </section>
    </div>

    <div class="grid-2">
      <section class="card">
        <h2 class="section-title">Positive flags</h2>
        ${renderList(positiveFlags, "No positive flags reported.")}
      </section>

      <section class="card">
        <h2 class="section-title">Warning flags</h2>
        ${renderList(warningFlags, "No warning flags reported.")}
      </section>
    </div>

    <section class="card" style="margin-top:18px;">
      <h2 class="section-title">Lab quality summary</h2>
      <div class="muted">${esc(data.lab_quality_summary || "Not reported")}</div>
    </section>

    <section class="card" style="margin-top:18px;">
      <h2 class="section-title">Scientific references</h2>
      <div class="muted">${esc(data.scientific_references || "Not reported")}</div>
    </section>

    <section class="card" style="margin-top:18px;">
      <h2 class="section-title">Raw JSON</h2>
      <pre>${esc(JSON.stringify(data, null, 2))}</pre>
    </section>
  </div>
</body>
</html>
  `;
}

app.get("/", (req, res) => {
  const landing = path.join(__dirname, "public", "index.html");
  res.sendFile(landing);
});

app.get("/health", async (req, res) => {
  try {
    return res.json({
      success: true,
      message: "Middleware is running",
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: error.message || "Health check failed",
    });
  }
});

app.post("/upload-coa", upload.single("file"), async (req, res) => {
  try {
    console.log("📥 Upload received");

    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: "No file uploaded",
      });
    }

    const originalFilename =
      req.file.originalname || `upload-${Date.now()}.pdf`;
    const mimeType = req.file.mimetype || "application/octet-stream";

    const { publicUrl, storagePath } = await uploadBufferToSupabase({
      buffer: req.file.buffer,
      originalName: originalFilename,
      mimeType,
      folder: "raw_documents",
    });

    console.log("🔍 Sending to Azure OCR...");
    const extracted = await extractDocumentFromUrl(publicUrl);
    console.log("✅ OCR complete");

    console.log("🧠 Sending to OpenAI...");
    const parsedJson = await safeParseCOA(extracted.plain_text);
    console.log("✅ AI parsing complete");

    const insertedRow = await insertCOAReport({
      parsedJson,
      sourceUrl: publicUrl,
      storagePath,
      originalFilename,
      mimeType: extracted.mimeType || mimeType,
    });

    const reportUrl = buildReportUrl(req, insertedRow.id);

    return res.json({
      success: true,
      id: insertedRow.id,
      report_url: reportUrl,
      html_url: reportUrl,
      pdf_url: buildPdfUrl(req, insertedRow.id),
    });
  } catch (error) {
    console.error("❌ Upload pipeline error:", error.message);
    return res.status(500).json({
      success: false,
      error: error.message || "Upload pipeline failed",
    });
  }
});

app.get("/report/:id", async (req, res) => {
  try {
    const row = await getReportById(req.params.id);
    const data = row?.report_json || {};

    return res.send(
      renderReportHTML(data, {
        documentId: row.id,
      })
    );
  } catch (error) {
    console.error("ERROR IN /report/:id:", error.message);
    return res.status(404).send("Report not found");
  }
});

app.get("/pdf/:id", async (req, res) => {
  let browser;

  try {
    const row = await getReportById(req.params.id);
    const html = renderReportHTML(row?.report_json || {}, {
      documentId: row.id,
    });

    browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "networkidle0" });

    const pdfBuffer = await page.pdf({
      format: "A4",
      printBackground: true,
      margin: {
        top: "16mm",
        right: "12mm",
        bottom: "16mm",
        left: "12mm",
      },
    });

    res.setHeader("Content-Type", "application/pdf");
    res.setHeader(
      "Content-Disposition",
      `inline; filename="${sanitizeFileName(
        row?.report_json?.product_name || req.params.id
      )}.pdf"`
    );

    return res.send(pdfBuffer);
  } catch (error) {
    console.error("ERROR IN /pdf/:id:", error.message);
    return res.status(500).json({
      success: false,
      error: error.message || "PDF generation failed",
    });
  } finally {
    if (browser) {
      try {
        await browser.close();
      } catch (_) {}
    }
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});