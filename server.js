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

  const toNum = (value) => {
    if (value === null || value === undefined) return 0;
    const raw = String(value).trim().toLowerCase();
    if (!raw || raw === "nd" || raw.includes("not detected") || raw.includes("<loq")) return 0;
    const match = raw.match(/-?\d+(\.\d+)?/);
    return match ? Number(match[0]) : 0;
  };

  const thc = toNum(data.thc_total);
  const cbd = toNum(data.cbd_total);
  const terps = toNum(data.total_terpenes);

  const dominantTerpene = topTerpenes[0]?.name || "";
  const secondTerpene = topTerpenes[1]?.name || "";
  const aromaticProfile = topTerpenes.length
    ? topTerpenes.slice(0, 5).map((t) => t?.name).filter(Boolean).join(" • ")
    : "Not clearly reported";

  const hasMinorCannabinoids = !!String(data.minor_cannabinoids || "").trim();
  const hasContaminants = !!String(data.contaminant_overview || "").trim();

  function classifyPotency(v) {
    if (v >= 28) return { label: "Very high", note: "Upper-end flower potency." };
    if (v >= 24) return { label: "High", note: "High-THC flower profile." };
    if (v >= 18) return { label: "Moderate-high", note: "Meaningful THC strength." };
    if (v > 0) return { label: "Moderate", note: "More moderate THC intensity." };
    return { label: "Unknown", note: "Potency could not be classified." };
  }

  function classifyTerpenes(v) {
    if (v >= 3) return { label: "High", note: "Strong aromatic density." };
    if (v >= 1.5) return { label: "Moderate-high", note: "Meaningful terpene presence." };
    if (v > 0) return { label: "Light", note: "Lighter aromatic expression." };
    return { label: "Unknown", note: "Terpene density unclear." };
  }

  function inferDirection(lead = "") {
    const t = String(lead || "").toLowerCase();

    if (t.includes("terpinolene")) {
      return {
        label: "Uplifting / mentally active",
        note: "Terpinolene-led profiles often feel brighter and less body-heavy."
      };
    }
    if (t.includes("myrcene")) {
      return {
        label: "Calming / body-heavy",
        note: "Myrcene-forward profiles often lean more calming and physical."
      };
    }
    if (t.includes("limonene")) {
      return {
        label: "Bright / mood-forward",
        note: "Limonene-heavy profiles often read brighter and more energetic."
      };
    }
    if (t.includes("pinene")) {
      return {
        label: "Clear / alert",
        note: "Pinene-heavy profiles are often associated with clarity and alertness."
      };
    }
    if (t.includes("caryophyllene")) {
      return {
        label: "Balanced / structured",
        note: "Caryophyllene-led profiles often feel balanced and grounded."
      };
    }

    return {
      label: "Chemistry-led",
      note: "A stronger directional read would require clearer terpene dominance."
    };
  }

  function classifyCompleteness() {
    const score =
      (topCannabinoids.length ? 1 : 0) +
      (topTerpenes.length ? 1 : 0) +
      (data.thc_total ? 1 : 0) +
      (data.total_terpenes ? 1 : 0) +
      (hasContaminants ? 1 : 0) +
      (data.laboratory_name ? 1 : 0);

    if (score >= 5) return { label: "Strong", note: "Good structured chemistry coverage." };
    if (score >= 3) return { label: "Moderate", note: "Useful but not fully rich COA coverage." };
    return { label: "Limited", note: "Interpretation constrained by missing data." };
  }

  function buildHeroSummary() {
    const potency = classifyPotency(thc).label.toLowerCase();
    const aroma = classifyTerpenes(terps).label.toLowerCase();
    const lead = dominantTerpene ? `${dominantTerpene}-led` : "chemistry-led";

    return `${
      potency === "unknown" ? "Cannabis flower" : `${classifyPotency(thc).label} potency`
    }, ${lead} profile with ${aroma} terpene intensity and ${
      hasMinorCannabinoids ? "visible minor cannabinoid depth." : "a primary focus on major compounds."
    }`;
  }

  function buildWhatStandsOut() {
    const bullets = [];

    if (thc >= 24) {
      bullets.push("THC sits firmly in the high range, suggesting a stronger-than-average flower profile.");
    } else if (thc >= 18) {
      bullets.push("THC sits in the moderate-high range, indicating meaningful potency without being ultra-high.");
    }

    if (dominantTerpene) {
      bullets.push(`${dominantTerpene} appears to be the leading terpene, shaping the profile direction and aromatic identity.`);
    }

    if (terps >= 1.5) {
      bullets.push("Total terpene content suggests a meaningful aromatic presence rather than a flat chemistry profile.");
    }

    if (hasMinorCannabinoids) {
      bullets.push("Minor cannabinoids add depth beyond THC alone, which may support a more differentiated chemotype.");
    }

    if (!bullets.length) {
      bullets.push("This report is readable, but the chemistry data are not rich enough to support stronger interpretation.");
    }

    return bullets.slice(0, 4);
  }

  function renderFlagPills(items, tone = "good", emptyText = "None reported") {
    if (!items.length) {
      return `<div class="muted">${esc(emptyText)}</div>`;
    }

    return `
      <div class="pill-wrap">
        ${items
          .map((item) => `<span class="pill ${tone}">${esc(item)}</span>`)
          .join("")}
      </div>
    `;
  }

  function renderTerpeneBars(items = []) {
    if (!items.length) {
      return `<div class="muted">No terpene rows reported.</div>`;
    }

    const maxVal = Math.max(
      ...items.map((item) => {
        const num = toNum(item?.value);
        return num > 0 ? num : 0;
      }),
      0.01
    );

    return `
      <div class="bars">
        ${items
          .slice(0, 8)
          .map((item) => {
            const num = toNum(item?.value);
            const width = Math.max(6, (num / maxVal) * 100);
            return `
              <div class="bar-row">
                <div class="bar-head">
                  <span>${esc(item?.name || "Unnamed")}</span>
                  <strong>${esc(
                    [item?.value, item?.unit].filter(Boolean).join(" ") || "—"
                  )}</strong>
                </div>
                <div class="bar-shell">
                  <span style="width:${width}%"></span>
                </div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function renderCannabinoidCards(items = []) {
    if (!items.length) {
      return `<div class="muted">No cannabinoid rows reported.</div>`;
    }

    return `
      <div class="compound-grid">
        ${items
          .slice(0, 8)
          .map(
            (item) => `
            <div class="compound-card">
              <div class="compound-name">${esc(item?.name || "Unnamed")}</div>
              <div class="compound-value">${esc(
                [item?.value, item?.unit].filter(Boolean).join(" ") || "—"
              )}</div>
              ${
                item?.notes
                  ? `<div class="compound-note">${esc(item.notes)}</div>`
                  : ""
              }
            </div>
          `
          )
          .join("")}
      </div>
    `;
  }

  const potency = classifyPotency(thc);
  const aroma = classifyTerpenes(terps);
  const direction = inferDirection(dominantTerpene);
  const completeness = classifyCompleteness();
  const standoutBullets = buildWhatStandsOut();

  const rawJson = esc(JSON.stringify(data, null, 2));

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${esc(data.product_name || "COA Report")}</title>
<style>
  :root {
    --bg: #f5f1e8;
    --bg2: #faf8f3;
    --panel: rgba(255,255,255,0.88);
    --line: rgba(25,24,22,0.09);
    --text: #191816;
    --muted: #6e6961;
    --green: #1f3d2b;
    --green2: #2f5a43;
    --gold: #b99657;
    --goodBg: rgba(31,61,43,0.08);
    --warnBg: rgba(185,150,87,0.12);
    --shadow: 0 16px 45px rgba(25,24,22,0.07);
    --radius: 24px;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: Inter, Arial, sans-serif;
    color: var(--text);
    background:
      radial-gradient(circle at top left, rgba(185,150,87,0.10), transparent 25%),
      radial-gradient(circle at 85% 10%, rgba(31,61,43,0.07), transparent 20%),
      linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%);
  }

  .shell {
    max-width: 1240px;
    margin: 0 auto;
    padding: 24px 18px 60px;
  }

  .topbar {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 18px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    border-radius: 999px;
    border: 1px solid var(--line);
    padding: 12px 18px;
    background: rgba(255,255,255,0.9);
    color: var(--text);
    font-weight: 700;
    cursor: pointer;
  }

  .btn-primary {
    background: linear-gradient(90deg, var(--green), var(--green2));
    color: #fff;
    border-color: var(--green);
  }

  .hero,
  .card {
    border: 1px solid var(--line);
    border-radius: 28px;
    background: var(--panel);
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
  }

  .hero {
    padding: 30px;
    overflow: hidden;
    position: relative;
  }

  .hero::after {
    content: "";
    position: absolute;
    right: -40px;
    bottom: -50px;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(185,150,87,0.16), transparent 65%);
    pointer-events: none;
  }

  .eyebrow {
    color: var(--green);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 12px;
  }

  .hero-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 20px;
    align-items: start;
  }

  h1 {
    margin: 0 0 12px;
    font-size: clamp(2.1rem, 4vw, 3.6rem);
    line-height: 0.95;
    letter-spacing: -0.05em;
    max-width: 10ch;
  }

  .hero-summary {
    font-size: 1.02rem;
    color: #423d36;
    line-height: 1.8;
    max-width: 62ch;
  }

  .hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    padding: 10px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.88);
    border: 1px solid var(--line);
    font-size: 12px;
    font-weight: 700;
  }

  .hero-score {
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 18px;
    background: rgba(255,255,255,0.92);
  }

  .hero-score-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }

  .hero-score-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--green);
    line-height: 1.1;
  }

  .hero-score-note {
    margin-top: 8px;
    color: var(--muted);
    line-height: 1.65;
    font-size: 14px;
  }

  .section {
    margin-top: 18px;
  }

  .grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }

  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }

  .card {
    padding: 22px;
  }

  .section-title {
    margin: 0 0 14px;
    font-size: 1.55rem;
    letter-spacing: -0.04em;
  }

  .section-sub {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 14px;
  }

  .insight-card {
    padding: 18px;
    border-radius: 22px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.82);
  }

  .insight-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
    font-weight: 800;
  }

  .insight-value {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
  }

  .insight-note {
    color: var(--muted);
    line-height: 1.65;
    font-size: 14px;
  }

  .metric {
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 18px;
    background: rgba(255,255,255,0.82);
  }

  .metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }

  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.05;
  }

  .muted {
    color: var(--muted);
    line-height: 1.75;
    font-size: 14px;
  }

  .detail-list {
    display: grid;
    gap: 12px;
  }

  .detail-item {
    padding-bottom: 10px;
    border-bottom: 1px solid var(--line);
  }

  .detail-key {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .detail-value {
    font-size: 15px;
    font-weight: 700;
    line-height: 1.6;
  }

  .standout-list {
    margin: 0;
    padding-left: 18px;
  }

  .standout-list li {
    margin: 10px 0;
    line-height: 1.75;
  }

  .bars {
    display: grid;
    gap: 14px;
  }

  .bar-row { display: grid; gap: 7px; }

  .bar-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 14px;
  }

  .bar-head strong {
    color: var(--text);
    white-space: nowrap;
  }

  .bar-shell {
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: #eee8dd;
    border: 1px solid #e3dccc;
  }

  .bar-shell span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--green), var(--gold));
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
    background: rgba(255,255,255,0.82);
  }

  .compound-name {
    font-size: 13px;
    font-weight: 800;
    color: var(--green);
    margin-bottom: 6px;
  }

  .compound-value {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 4px;
  }

  .compound-note {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.55;
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
    font-size: 13px;
    line-height: 1.35;
    border: 1px solid var(--line);
  }

  .pill.good {
    background: var(--goodBg);
    color: var(--green);
    border-color: rgba(31,61,43,0.16);
  }

  .pill.warn {
    background: var(--warnBg);
    color: #7b5a20;
    border-color: rgba(185,150,87,0.2);
  }

  .toggle-row {
    margin-top: 14px;
  }

  .raw-box {
    display: none;
    margin-top: 14px;
  }

  .raw-box.open {
    display: block;
  }

  pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.6;
    background: rgba(255,255,255,0.82);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px;
  }

  @media (max-width: 980px) {
    .hero-grid,
    .grid-4,
    .grid-3,
    .grid-2,
    .compound-grid {
      grid-template-columns: 1fr;
    }
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
      <div class="hero-grid">
        <div>
          <div class="eyebrow">ALEM Chemical Intelligence</div>
          <h1>${esc(data.product_name || "COA Report")}</h1>
          <div class="hero-summary">${esc(
            data.opening_statement || buildHeroSummary()
          )}</div>

          <div class="hero-meta">
            <span class="badge">Batch ${esc(data.batch_number || "Not reported")}</span>
            <span class="badge">${esc(data.product_type || "Product type not reported")}</span>
            <span class="badge">${esc(data.laboratory_name || "Lab not reported")}</span>
            <span class="badge">${esc(data.coa_report_date || "Date not reported")}</span>
          </div>
        </div>

        <div class="hero-score">
          <div class="hero-score-label">Intelligence readout</div>
          <div class="hero-score-value">${esc(
            data.overall_score || buildHeroSummary()
          )}</div>
          <div class="hero-score-note">
            ${esc(direction.note)}
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="grid-4">
        <div class="insight-card">
          <div class="insight-label">Potency</div>
          <div class="insight-value">${esc(potency.label)}</div>
          <div class="insight-note">${esc(potency.note)}</div>
        </div>

        <div class="insight-card">
          <div class="insight-label">Aroma density</div>
          <div class="insight-value">${esc(aroma.label)}</div>
          <div class="insight-note">${esc(aroma.note)}</div>
        </div>

        <div class="insight-card">
          <div class="insight-label">Direction</div>
          <div class="insight-value">${esc(direction.label)}</div>
          <div class="insight-note">${esc(direction.note)}</div>
        </div>

        <div class="insight-card">
          <div class="insight-label">COA completeness</div>
          <div class="insight-value">${esc(completeness.label)}</div>
          <div class="insight-note">${esc(completeness.note)}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="grid-3">
        ${renderMetric("THC Total", data.thc_total)}
        ${renderMetric("CBD Total", data.cbd_total)}
        ${renderMetric("Total Terpenes", data.total_terpenes)}
      </div>
    </section>

    <section class="section">
      <div class="grid-2">
        <div class="card">
          <h2 class="section-title">What stands out</h2>
          <ul class="standout-list">
            ${standoutBullets.map((item) => `<li>${esc(item)}</li>`).join("")}
          </ul>
        </div>

        <div class="card">
          <h2 class="section-title">Batch details</h2>
          <div class="detail-list">
            <div class="detail-item">
              <div class="detail-key">Batch number</div>
              <div class="detail-value">${esc(data.batch_number || "Not reported")}</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Report date</div>
              <div class="detail-value">${esc(data.coa_report_date || "Not reported")}</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Product type</div>
              <div class="detail-value">${esc(data.product_type || "Not reported")}</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Laboratory</div>
              <div class="detail-value">${esc(data.laboratory_name || "Not reported")}</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="grid-2">
        <div class="card">
          <h2 class="section-title">Cannabinoid profile</h2>
          <div class="section-sub">
            ${esc(
              hasMinorCannabinoids
                ? "Minor cannabinoid presence suggests added chemistry depth beyond THC alone."
                : "The visible cannabinoid story is driven mainly by the major compounds captured in the report."
            )}
          </div>
          ${renderCannabinoidCards(topCannabinoids)}
          <div style="margin-top:14px;" class="muted">
            <strong>Minor cannabinoids:</strong> ${esc(data.minor_cannabinoids || "Not reported")}
          </div>
        </div>

        <div class="card">
          <h2 class="section-title">Terpene fingerprint</h2>
          <div class="section-sub">
            ${esc(
              dominantTerpene
                ? `${dominantTerpene} leads the aromatic profile${secondTerpene ? `, supported by ${secondTerpene}` : ""}.`
                : "A dominant terpene could not be clearly identified."
            )}
          </div>
          ${renderTerpeneBars(topTerpenes)}
          <div style="margin-top:14px;" class="muted">
            <strong>Aromatic profile:</strong> ${esc(aromaticProfile)}
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="grid-2">
        <div class="card">
          <h2 class="section-title">Positive flags</h2>
          ${renderFlagPills(positiveFlags, "good", "No positive flags reported.")}
        </div>

        <div class="card">
          <h2 class="section-title">Warning flags</h2>
          ${renderFlagPills(warningFlags, "warn", "No warning flags reported.")}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="grid-2">
        <div class="card">
          <h2 class="section-title">Contaminant overview</h2>
          <div class="muted">${esc(data.contaminant_overview || "Not reported")}</div>
        </div>

        <div class="card">
          <h2 class="section-title">Lab quality summary</h2>
          <div class="muted">${esc(data.lab_quality_summary || "Not reported")}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Scientific references</h2>
        <div class="muted">${esc(data.scientific_references || "Not reported")}</div>

        <div class="toggle-row">
          <button class="btn" type="button" onclick="toggleRaw()">Toggle Raw JSON</button>
        </div>

        <div id="rawBox" class="raw-box">
          <pre>${rawJson}</pre>
        </div>
      </div>
    </section>
  </div>

  <script>
    function toggleRaw() {
      const box = document.getElementById("rawBox");
      box.classList.toggle("open");
    }
  </script>
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