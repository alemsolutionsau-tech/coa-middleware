require("dotenv").config();

const multer = require("multer");
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

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});

const app = express();

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/public/index.html");
});

app.use(express.static("public"));
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

  const raw = String(v).trim();
  if (!raw) return 0;

  // handles:
  // "24.1364 wt%"
  // "24.14%"
  // "ND"
  // "<LOQ"
  const lower = raw.toLowerCase();
  if (
    lower === "nd" ||
    lower === "n/d" ||
    lower.includes("not detected") ||
    lower.includes("<loq") ||
    lower.includes("< lod")
  ) {
    return 0;
  }

  const match = raw.match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : 0;
}

function classifyPotency(thc) {
  if (thc >= 28) {
    return {
      label: "Very high potency",
      short: "Very high",
      note: "This sits in the upper end of flower potency and is likely to feel strong, especially for lower-tolerance users."
    };
  }

  if (thc >= 24) {
    return {
      label: "High potency",
      short: "High",
      note: "This is a high-THC flower profile and is likely to feel stronger than average."
    };
  }

  if (thc >= 18) {
    return {
      label: "Moderate-high potency",
      short: "Moderate-high",
      note: "This sits in the moderate-to-high THC range and may suit users comfortable with THC."
    };
  }

  if (thc > 0) {
    return {
      label: "Moderate potency",
      short: "Moderate",
      note: "This appears more moderate in THC intensity than high-potency flower products."
    };
  }

  return {
    label: "Potency unclear",
    short: "Unknown",
    note: "Potency could not be confidently classified from the extracted data."
  };
}

function classifyTerpeneDensity(totalTerpenes) {
  if (totalTerpenes >= 3) {
    return {
      label: "High aromatic intensity",
      short: "High",
      note: "This suggests stronger aromatic expression than average."
    };
  }

  if (totalTerpenes >= 1.5) {
    return {
      label: "Moderate aromatic intensity",
      short: "Moderate",
      note: "This suggests a meaningful terpene presence without being extremely terpene-dense."
    };
  }

  if (totalTerpenes > 0) {
    return {
      label: "Light aromatic intensity",
      short: "Light",
      note: "This suggests lighter terpene expression."
    };
  }

  return {
    label: "Aromatic intensity unclear",
    short: "Unknown",
    note: "Terpene density could not be confidently classified."
  };
}

function inferEffectDirection(dominantTerpene = "") {
  const t = String(dominantTerpene || "").toLowerCase();

  if (t.includes("terpinolene")) {
    return {
      label: "Uplifting / mentally active leaning",
      note: "Terpinolene-led profiles often feel brighter, more mentally active, and less body-heavy than sedating chemotypes."
    };
  }

  if (t.includes("myrcene")) {
    return {
      label: "Calming / body-heavy leaning",
      note: "Myrcene-dominant profiles often lean more calming and body-centered."
    };
  }

  if (t.includes("limonene")) {
    return {
      label: "Bright / mood-forward leaning",
      note: "Limonene-forward profiles are often positioned as brighter and more mood-forward."
    };
  }

  if (t.includes("caryophyllene")) {
    return {
      label: "Balanced / spicy profile",
      note: "Caryophyllene-led profiles often read as balanced, warm, and structured."
    };
  }

  if (t.includes("pinene")) {
    return {
      label: "Clear / alertness leaning",
      note: "Pinene-heavy profiles are often interpreted as clearer and more alertness-oriented."
    };
  }

  return {
    label: "General chemistry-led profile",
    note: "A stronger directional effect readout would require clearer terpene dominance."
  };
}

function buildExecutiveBrief({
  thcNumber,
  cbdNumber,
  terpNumber,
  dominantTerpene,
  hasMinorCannabinoids
}) {
  const potency = classifyPotency(thcNumber);
  const terpeneDensity = classifyTerpeneDensity(terpNumber);
  const effectDirection = inferEffectDirection(dominantTerpene);

  const cbdText =
    cbdNumber > 0
      ? "Visible CBD may provide some modulation of THC intensity."
      : "CBD appears absent or negligible, so THC modulation from CBD is likely minimal.";

  const minorText = hasMinorCannabinoids
    ? "Visible minor cannabinoids add chemical depth beyond THC alone."
    : "Minor cannabinoid depth was not clearly captured in the structured data.";

  return {
    headline: `${potency.label}, ${dominantTerpene && dominantTerpene !== "Unknown" ? `${dominantTerpene}-dominant` : "chemistry-led"} flower with ${terpeneDensity.short.toLowerCase()} terpene intensity.`,
    bullets: [
      potency.note,
      effectDirection.note,
      cbdText,
      minorText
    ]
  };
}

function compareAgainstBenchmarks({
  thcNumber,
  terpNumber,
  topTerpenes,
  hasMinorCannabinoids,
  hasContaminants
}) {
  return [
    {
      label: "vs High-THC flower",
      value:
        thcNumber >= 24
          ? "Aligned"
          : thcNumber >= 18
          ? "Slightly lighter"
          : thcNumber > 0
          ? "Below category"
          : "Unknown"
    },
    {
      label: "vs aromatic premium flower",
      value:
        terpNumber >= 3
          ? "Aligned"
          : terpNumber >= 1.5
          ? "Moderate"
          : terpNumber > 0
          ? "Below category"
          : "Unknown"
    },
    {
      label: "vs differentiated chemotype",
      value:
        hasMinorCannabinoids || topTerpenes.length >= 4
          ? "More distinctive"
          : topTerpenes.length >= 2
          ? "Moderately distinctive"
          : "More standard"
    },
    {
      label: "vs data-rich COA",
      value:
        topTerpenes.length && hasContaminants
          ? "Strong"
          : topTerpenes.length || hasContaminants
          ? "Partial"
          : "Limited"
    }
  ];
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
    <script>
async function uploadCOA(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/upload-coa", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();

  if (result.success) {
    window.location.href = result.report_url;
  } else {
    alert("Error: " + result.error);
  }
}

document.getElementById("fileInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) uploadCOA(file);
});
</script>
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

  const thc = numberFromPercentString(data?.thc_total);
  const cbd = numberFromPercentString(data?.cbd_total);
  const terps = numberFromPercentString(data?.total_terpenes);

  const productName = data?.product_name || "Cannabis Product";
  const productType = data?.product_type || "Product type not reported";
  const batchNumber = data?.batch_number || "Not reported";
  const laboratoryName = data?.laboratory_name || "Lab not reported";
  const reportDate = data?.coa_report_date || "Date not reported";
  const subtitle =
    data?.opening_statement ||
    "Chemistry translated into practical intelligence from the uploaded COA.";

  const dominantTerpene = topTerpenes[0]?.name || "Unknown";
  const secondTerpene = topTerpenes[1]?.name || "";
  const thirdTerpene = topTerpenes[2]?.name || "";

  const hasMinor = !!String(data?.minor_cannabinoids || "").trim();
  const hasContaminants = !!String(data?.contaminant_overview || "").trim();

  const compareProduct = data?.compare_product && typeof data.compare_product === "object"
    ? data.compare_product
    : null;

  function classifyPotency(v) {
    if (v >= 28) return { label: "Very high potency", short: "Very high" };
    if (v >= 24) return { label: "High potency", short: "High" };
    if (v >= 18) return { label: "Moderate-high potency", short: "Moderate-high" };
    if (v > 0) return { label: "Moderate potency", short: "Moderate" };
    return { label: "Potency unclear", short: "Unknown" };
  }

  function classifyTerpeneDensity(v) {
    if (v >= 3) return { label: "High aromatic intensity", short: "High" };
    if (v >= 1.5) return { label: "Moderate aromatic intensity", short: "Moderate" };
    if (v > 0) return { label: "Light aromatic intensity", short: "Light" };
    return { label: "Aromatic intensity unclear", short: "Unknown" };
  }

  function inferEffectDirection(lead = "") {
    const t = String(lead || "").toLowerCase();

    if (t.includes("terpinolene")) {
      return {
        label: "Uplifting / mentally active leaning",
        note: "Terpinolene-led profiles often feel brighter, more active, and less body-heavy."
      };
    }
    if (t.includes("myrcene")) {
      return {
        label: "Calming / body-heavy leaning",
        note: "Myrcene-forward profiles often read as more body-centered and calming."
      };
    }
    if (t.includes("limonene")) {
      return {
        label: "Bright / mood-forward leaning",
        note: "Limonene-heavy profiles are often read as brighter and more mood-forward."
      };
    }
    if (t.includes("caryophyllene")) {
      return {
        label: "Balanced / structured leaning",
        note: "Caryophyllene-heavy profiles often feel balanced, warm, and structured."
      };
    }
    if (t.includes("pinene")) {
      return {
        label: "Clear / alertness leaning",
        note: "Pinene-forward profiles are often associated with clearer, more alert expression."
      };
    }

    return {
      label: "General chemistry-led profile",
      note: "A stronger directional read would require clearer terpene dominance."
    };
  }

  function getPaletteForTerpene(lead = "") {
    const t = String(lead || "").toLowerCase();

    if (t.includes("terpinolene")) {
      return {
        core: "#B7FF64",
        glow: "#6DFFB4",
        outer: "#8EE8FF",
        spark: "#F6FF8A",
        bg1: "#F5FFF0",
        bg2: "#ECFFF9"
      };
    }
    if (t.includes("myrcene")) {
      return {
        core: "#7E9F52",
        glow: "#B2B36D",
        outer: "#D8A65C",
        spark: "#EFE3B3",
        bg1: "#F8F7EF",
        bg2: "#FBF7ED"
      };
    }
    if (t.includes("limonene")) {
      return {
        core: "#FFD54F",
        glow: "#FFF176",
        outer: "#C8FF7A",
        spark: "#FFF8BF",
        bg1: "#FFFDF2",
        bg2: "#F9FFF0"
      };
    }
    if (t.includes("pinene")) {
      return {
        core: "#4DB6AC",
        glow: "#80CBC4",
        outer: "#81D4FA",
        spark: "#D9FFF8",
        bg1: "#F1FFFC",
        bg2: "#F2FBFF"
      };
    }
    if (t.includes("caryophyllene")) {
      return {
        core: "#C28A52",
        glow: "#D6B06A",
        outer: "#5C8A5A",
        spark: "#F8E7B8",
        bg1: "#FFF8F1",
        bg2: "#F6FBF2"
      };
    }

    return {
      core: "#7BC47F",
      glow: "#A5D6A7",
      outer: "#90CAF9",
      spark: "#E8F5E9",
      bg1: "#F7FBF7",
      bg2: "#F6FAFF"
    };
  }

  function buildPostHarvestInsights() {
    const freshness =
      terps >= 2.5
        ? {
            label: "Strong freshness signal",
            note: "Terpene retention appears strong, which is generally consistent with better post-harvest preservation."
          }
        : terps >= 1.5
        ? {
            label: "Moderate freshness signal",
            note: "Terpene retention appears reasonably preserved, suggesting the flower maintained aromatic expression through handling."
          }
        : terps > 0
        ? {
            label: "Light freshness signal",
            note: "Some aromatic expression remains, though terpene density is not strong enough to suggest standout preservation."
          }
        : {
            label: "Freshness signal limited",
            note: "Freshness cannot be meaningfully inferred because terpene density is unclear."
          };

    const cure =
      topTerpenes.length >= 3 && terps >= 1.5
        ? {
            label: "Cure signal: positive",
            note: "The profile does not look chemically flat; visible terpene layering suggests preserved aromatic complexity."
          }
        : topTerpenes.length >= 2
        ? {
            label: "Cure signal: moderate",
            note: "Some terpene layering is visible, but confidence remains limited."
          }
        : {
            label: "Cure signal: limited",
            note: "A stronger curing inference would require clearer terpene spread and additional post-harvest variables."
          };

    const stability =
      hasContaminants
        ? {
            label: "Stability signal: partial",
            note: "Some quality-state interpretation is possible, but true stability confidence would require moisture, water activity, oxidation, or repeat-batch data."
          }
        : {
            label: "Stability signal: limited",
            note: "Stability confidence is limited because no moisture, water activity, oxidation, or longitudinal batch data are available."
          };

    return { freshness, cure, stability };
  }

  function buildCultivationInsights() {
    const complexityScore =
      Math.min(
        10,
        topTerpenes.length +
          (hasMinor ? 2 : 0) +
          (thc >= 24 ? 2 : thc >= 18 ? 1 : 0)
      );

    const expression =
      thc >= 24 && terps >= 1.8
        ? "Strong chemotype expression"
        : thc >= 18 && terps >= 1.5
        ? "Good chemotype expression"
        : "Moderate chemotype expression";

    const preservation =
      terps >= 2
        ? "Aromatic preservation appears reasonably intact."
        : "Aromatic preservation is present but not especially strong.";

    const cultivationNote =
      hasMinor && topTerpenes.length >= 3
        ? "This reads more like a well-expressed profile than a flat potency-only flower."
        : "The profile shows useful structure, though cultivation-quality inference remains moderate.";

    return {
      complexityScore,
      expression,
      preservation,
      cultivationNote
    };
  }

  function buildLineageInsights() {
    const t = String(dominantTerpene || "").toLowerCase();

    if (t.includes("terpinolene")) {
      return {
        family: "Haze / Jack / Durban-type families",
        confidence: "Moderate",
        note: "Terpinolene-dominant profiles often cluster closer to uplifting Haze-leaning chemotypes than to sedating Kush patterns."
      };
    }

    if (t.includes("myrcene")) {
      return {
        family: "Kush / OG / indica-leaning families",
        confidence: "Moderate",
        note: "Myrcene-led chemistry often maps more closely to body-heavier or Kush-leaning profile families."
      };
    }

    if (t.includes("limonene")) {
      return {
        family: "Citrus / Gelato / Runtz-like hybrid families",
        confidence: "Moderate",
        note: "Limonene-heavy flowers often align more closely with brighter citrus-forward hybrid families."
      };
    }

    if (t.includes("pinene")) {
      return {
        family: "Pine-forward or clearer sativa-leaning families",
        confidence: "Low-moderate",
        note: "Pinene prominence can suggest a clearer and more alertness-oriented chemotype cluster."
      };
    }

    if (t.includes("caryophyllene")) {
      return {
        family: "Balanced hybrid / spice-forward families",
        confidence: "Low-moderate",
        note: "Caryophyllene-forward expressions can map toward structured, spicy hybrid families."
      };
    }

    return {
      family: "Unclear lineage cluster",
      confidence: "Low",
      note: "A stronger lineage read would require clearer terpene dominance and a richer comparison database."
    };
  }

  function buildExecutiveBrief() {
    const potency = classifyPotency(thc);
    const terpeneDensity = classifyTerpeneDensity(terps);
    const effectDirection = inferEffectDirection(dominantTerpene);

    const cbdText =
      cbd > 0
        ? "Visible CBD may provide some modulation of THC intensity."
        : "CBD appears absent or negligible, so THC modulation from CBD is likely minimal.";

    const minorText = hasMinor
      ? "Visible minor cannabinoids add chemical depth beyond THC alone."
      : "Minor cannabinoid depth was not clearly captured in the structured data.";

    return {
      headline: `${potency.label}, ${dominantTerpene && dominantTerpene !== "Unknown" ? `${dominantTerpene}-dominant` : "chemistry-led"} flower with ${terpeneDensity.short.toLowerCase()} terpene intensity.`,
      bullets: [
        potency.label === "Potency unclear"
          ? "Potency could not be confidently classified from the extracted data."
          : `Potency reads as ${potency.short.toLowerCase()} relative to common flower ranges.`,
        effectDirection.note,
        cbdText,
        minorText
      ]
    };
  }

  function compareAgainstBenchmarks() {
    return [
      {
        label: "vs High-THC flower",
        value:
          thc >= 24
            ? "Aligned"
            : thc >= 18
            ? "Slightly lighter"
            : thc > 0
            ? "Below category"
            : "Unknown"
      },
      {
        label: "vs aromatic premium flower",
        value:
          terps >= 3
            ? "Aligned"
            : terps >= 1.5
            ? "Moderate"
            : terps > 0
            ? "Below category"
            : "Unknown"
      },
      {
        label: "vs differentiated chemotype",
        value:
          hasMinor || topTerpenes.length >= 4
            ? "More distinctive"
            : topTerpenes.length >= 2
            ? "Moderately distinctive"
            : "More standard"
      },
      {
        label: "vs data-rich COA",
        value:
          topTerpenes.length && hasContaminants
            ? "Strong"
            : topTerpenes.length || hasContaminants
            ? "Partial"
            : "Limited"
      }
    ];
  }

  const potency = classifyPotency(thc);
  const terpeneDensity = classifyTerpeneDensity(terps);
  const effectDirection = inferEffectDirection(dominantTerpene);
  const executiveBrief = buildExecutiveBrief();
  const postHarvest = buildPostHarvestInsights();
  const cultivation = buildCultivationInsights();
  const lineage = buildLineageInsights();
  const compareCards = compareAgainstBenchmarks();
  const palette = getPaletteForTerpene(dominantTerpene);

  const overallScoreText = data?.overall_score || executiveBrief.headline;

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

  const thcBand =
    thc >= 24 ? "THC-H" : thc >= 18 ? "THC-M" : thc > 0 ? "THC-L" : "THC-?";
  const terpBand =
    terps >= 3 ? "TERP-H" : terps >= 1.5 ? "TERP-M" : terps > 0 ? "TERP-L" : "TERP-?";
  const cbdBand =
    cbd >= 5 ? "CBD-H" : cbd >= 1 ? "CBD-M" : cbd > 0 ? "CBD-L" : "CBD-LOW";

  const fingerprintTags = [
    thcBand,
    terpBand,
    cbdBand,
    `${String(dominantTerpene || "UNKNOWN").toUpperCase().replace(/[^A-Z0-9]+/g, "-")}-DOM`,
    hasMinor ? "MC+" : "MC-LIMITED",
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
        (hasContaminants ? 4 : 0) +
        (data?.lab_quality_summary ? 4 : 0)
    )
  );

  const cannabinoidVisibility =
    topCannabinoids.length ? "High" : data?.thc_total || data?.cbd_total ? "Moderate" : "Limited";
  const terpeneVisibility =
    topTerpenes.length ? "High" : data?.total_terpenes ? "Moderate" : "Limited";

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
      value: hasContaminants ? "Reported" : "Missing",
      className: hasContaminants ? "good" : "warn",
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
        (hasContaminants ? 2 : 0) +
        (data?.lab_quality_summary ? 2 : 0) +
        (data?.laboratory_name ? 1 : 0) +
        (data?.coa_report_date ? 1 : 0)
    )
  );

  const patientSafeSummary = [
    potency.label === "Potency unclear"
      ? "Potency could not be clearly determined from the extracted data."
      : `This appears to be a ${potency.short.toLowerCase()} THC product.`,
    dominantTerpene && dominantTerpene !== "Unknown"
      ? `${dominantTerpene} appears to be the dominant terpene in this sample.`
      : "A dominant terpene could not be clearly identified.",
    terpeneDensity.label === "Aromatic intensity unclear"
      ? "Terpene intensity could not be clearly assessed."
      : terpeneDensity.label + " is suggested by the terpene density.",
    warningFlags.length
      ? "Some parts of the report should be read cautiously because certain fields were limited or flagged."
      : "No major caution flags were detected in the structured output."
  ];

  const clinicianInsights = [
    cbd <= 1
      ? "Low CBD suggests limited direct THC modulation."
      : "Visible CBD may contribute some modulation of THC-dominant effects.",
    thc >= 24
      ? "High THC concentration warrants tolerance-aware prescribing considerations."
      : "THC potency appears less extreme than ultra-high THC flower products.",
    hasMinor
      ? "Minor cannabinoids are present and may add profile complexity."
      : "Minor cannabinoid depth was not clearly reported.",
    hasContaminants
      ? "Contaminant summary was reported and should be reviewed alongside the source COA."
      : "Contaminant interpretation is limited because no structured overview was extracted."
  ];

  const industryInsights = [
    dominantTerpene && dominantTerpene !== "Unknown"
      ? `${dominantTerpene}-led chemistry can support differentiated market positioning.`
      : "Terpene dominance is not strong enough to support a clear positioning angle.",
    terps >= 2.5
      ? "Stronger terpene density supports premium aromatic positioning."
      : "Moderate terpene density may require positioning around effect profile rather than aroma intensity alone.",
    hasMinor
      ? "Minor cannabinoids add product-story depth for brand and category differentiation."
      : "Limited visible minor cannabinoid depth may reduce differentiation narrative.",
    warningFlags.length
      ? "Commercial claims should be conservative until flagged issues are clarified."
      : "Structured output suggests cleaner messaging conditions for educational positioning."
  ];

  const enthusiastInsights = [
    aromaticProfile !== "Aromatic profile not available"
      ? `Top aromatic signals: ${aromaticProfile}.`
      : "Aromatic structure could not be confidently extracted.",
    effectDirection.label,
    thirdTerpene
      ? `Layered terpene architecture includes ${dominantTerpene}, ${secondTerpene}, and ${thirdTerpene}.`
      : secondTerpene
      ? `Leading terpene pairing includes ${dominantTerpene} and ${secondTerpene}.`
      : `Primary terpene signal centers on ${dominantTerpene}.`,
    hasMinor
      ? "Minor cannabinoids suggest added complexity beyond THC alone."
      : "Profile appears more driven by the major compounds captured in the report."
  ];

  const benchmarkCards = [
    {
      label: "Potency Position",
      value:
        thc >= 24
          ? "High-tier"
          : thc >= 18
          ? "Mid-high"
          : thc > 0
          ? "Moderate"
          : "Unknown",
    },
    {
      label: "Aroma Density",
      value:
        terps >= 3
          ? "High"
          : terps >= 1.5
          ? "Moderate"
          : terps > 0
          ? "Light"
          : "Unknown",
    },
    {
      label: "Differentiation",
      value:
        hasMinor && topTerpenes.length >= 3
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

  const similaritySignals = [
    dominantTerpene && dominantTerpene !== "Unknown"
      ? `Chemotype anchor: ${dominantTerpene}-dominant`
      : "Chemotype anchor unavailable",
    thc >= 24
      ? "Similarity cluster: higher-THC flower"
      : thc >= 18
      ? "Similarity cluster: standard THC flower"
      : "Similarity cluster: lighter potency flower",
    terps >= 2
      ? "Comparable set: aromatic-forward profiles"
      : "Comparable set: lighter terpene expression"
  ];

  const compareHtml = compareProduct
    ? `
      <section class="section">
        <div class="card">
          <h2 class="section-title">Live compare</h2>
          <div class="compare-live-grid">
            <div class="compare-live-card">
              <div class="compare-live-title">Current</div>
              <div class="compare-live-name">${esc(productName)}</div>
              <div class="compare-live-row"><span>THC</span><strong>${esc(data?.thc_total || "—")}</strong></div>
              <div class="compare-live-row"><span>Terpenes</span><strong>${esc(data?.total_terpenes || "—")}</strong></div>
              <div class="compare-live-row"><span>Lead terpene</span><strong>${esc(dominantTerpene)}</strong></div>
            </div>
            <div class="compare-live-card">
              <div class="compare-live-title">Compared product</div>
              <div class="compare-live-name">${esc(compareProduct?.product_name || "Unnamed")}</div>
              <div class="compare-live-row"><span>THC</span><strong>${esc(compareProduct?.thc_total || "—")}</strong></div>
              <div class="compare-live-row"><span>Terpenes</span><strong>${esc(compareProduct?.total_terpenes || "—")}</strong></div>
              <div class="compare-live-row"><span>Lead terpene</span><strong>${esc(compareProduct?.top_terpenes?.[0]?.name || "—")}</strong></div>
            </div>
          </div>
        </div>
      </section>
    `
    : "";

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
    --soul-core: ${palette.core};
    --soul-glow: ${palette.glow};
    --soul-outer: ${palette.outer};
    --soul-spark: ${palette.spark};
    --soul-bg1: ${palette.bg1};
    --soul-bg2: ${palette.bg2};
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
    max-width: 1400px;
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

  .section { margin-top: 22px; }

  .section-title {
    margin: 0 0 14px;
    font-size: 30px;
    color: var(--text);
    letter-spacing: -0.03em;
  }

  .section-title.small { font-size: 24px; }

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

  .card-soft { background: var(--panel-2); }

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

  .top-metric.thc { background: linear-gradient(180deg, #f2f6f3, #ffffff); }
  .top-metric.cbd { background: linear-gradient(180deg, #faf6ec, #ffffff); }
  .top-metric.terps { background: linear-gradient(180deg, #eef5f3, #ffffff); }

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

  .bar-row + .bar-row { margin-top: 14px; }

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

  .terpene-big strong { color: var(--text); }

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

  .check-list { display: grid; gap: 10px; }

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

  .insight-card { min-height: 100%; }

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

  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

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
    grid-template-columns: 0.92fr 1.08fr;
    gap: 18px;
    align-items: stretch;
  }

  .soul-shell {
    position: relative;
    min-height: 420px;
    border-radius: 28px;
    border: 1px solid var(--line);
    overflow: hidden;
    background:
      radial-gradient(circle at 15% 15%, rgba(255,255,255,0.9), transparent 26%),
      linear-gradient(180deg, var(--soul-bg1), var(--soul-bg2));
  }

  .soul-head {
    position: absolute;
    left: 18px;
    top: 16px;
    z-index: 3;
    padding: 10px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(0,0,0,0.06);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--green);
    backdrop-filter: blur(8px);
  }

  .soul-meta {
    position: absolute;
    right: 16px;
    top: 16px;
    z-index: 3;
    display: grid;
    gap: 8px;
  }

  .soul-chip {
    padding: 8px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(0,0,0,0.06);
    font-size: 12px;
    color: var(--muted);
    font-weight: 700;
  }

  .soul-canvas-wrap {
    position: absolute;
    inset: 0;
  }

  #chemicalSoulCanvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  .soul-caption {
    position: absolute;
    left: 20px;
    bottom: 18px;
    right: 20px;
    z-index: 3;
    display: grid;
    gap: 8px;
  }

  .soul-caption-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #223127;
  }

  .soul-caption-copy {
    max-width: 560px;
    color: #465046;
    line-height: 1.7;
    font-size: 14px;
  }

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

  .benchmark-grid,
  .compare-grid,
  .post-grid,
  .lineage-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }

  .benchmark-card,
  .compare-card,
  .post-card,
  .lineage-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px;
    background: #fff;
  }

  .benchmark-label,
  .compare-label,
  .post-label,
  .lineage-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .benchmark-value,
  .compare-value,
  .post-value,
  .lineage-value {
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 6px;
  }

  .post-note,
  .lineage-note {
    font-size: 13px;
    line-height: 1.6;
    color: #5f5a53;
  }

  .compare-live-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .compare-live-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    background: #fff;
    padding: 16px;
  }

  .compare-live-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .compare-live-name {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 14px;
    color: var(--text);
    letter-spacing: -0.02em;
  }

  .compare-live-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--line);
    font-size: 14px;
  }

  .compare-live-row strong { color: var(--text); }

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

  @media (max-width: 1150px) {
    .grid-2,
    .reg-grid,
    .terpene-layout,
    .insight-grid,
    .fingerprint-hero,
    .compare-live-grid {
      grid-template-columns: 1fr;
    }

    .benchmark-grid,
    .compare-grid,
    .post-grid,
    .lineage-grid {
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

    .hero h1 { font-size: 38px; }
  }

  @media (max-width: 700px) {
    .shell {
      padding: 16px 14px 34px;
    }

    .hero {
      padding: 24px 20px 22px;
      border-radius: 24px;
    }

    .hero h1 { font-size: 31px; }

    .benchmark-grid,
    .compare-grid,
    .post-grid,
    .lineage-grid,
    .reg-status-cards {
      grid-template-columns: 1fr;
    }

    .top-metric-value { font-size: 28px; }
    .soul-shell { min-height: 360px; }
  }

  @media print {
    .action-bar { display: none; }
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
          <div class="brand-mark">ALEM COA Intelligence Interface V4</div>
          <h1>${esc(productName)}</h1>
          <div class="hero-subtitle">${esc(subtitle)}</div>

          <div class="hero-badges">
            <span class="badge">Batch ${esc(batchNumber)}</span>
            <span class="badge">${esc(productType)}</span>
            <span class="badge">${esc(laboratoryName)}</span>
            <span class="badge">${esc(reportDate)}</span>
            <span class="badge">${esc(potency.label)}</span>
            <span class="badge">Confidence ${esc(`${dataConfidenceScore}%`)}</span>
            <span class="badge status-badge ${statusClass}">${esc(regulatoryStatus)}</span>
          </div>
        </div>

        <div class="hero-score">
          <div class="hero-score-label">AI Chemical Brief</div>
          <div class="hero-score-value">${esc(overallScoreText)}</div>
          <div class="hero-score-note">
            ${esc(executiveBrief.headline)}
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Chemical Soul</h2>
        <div class="fingerprint-hero">
          <div class="soul-shell">
            <div class="soul-head">Chemical Soul Engine</div>
            <div class="soul-meta">
              <div class="soul-chip">${esc(dominantTerpene)} lead</div>
              <div class="soul-chip">${esc(potency.short)} potency</div>
              <div class="soul-chip">${esc(terpeneDensity.short)} aroma</div>
            </div>
            <div class="soul-canvas-wrap">
              <canvas id="chemicalSoulCanvas" width="800" height="520"></canvas>
            </div>
            <div class="soul-caption">
              <div class="soul-caption-title">${esc(dominantTerpene && dominantTerpene !== "Unknown" ? `${dominantTerpene}-led living fingerprint` : "Living chemical fingerprint")}</div>
              <div class="soul-caption-copy">
                Abstract chemotype bloom generated from potency, terpene density, minor-cannabinoid depth, diversity, dominance skew, and confidence.
              </div>
            </div>
          </div>

          <div>
            <div class="section-sub">This visual translates the product’s chemical structure into an algorithmic identity — a stylized chemotype galaxy rather than a generic chart.</div>
            <div class="fingerprint-tags">
              ${fingerprintTags.map((tag) => `<span class="fingerprint-tag">${esc(tag)}</span>`).join("")}
            </div>

            <div class="card card-soft" style="margin-top:16px;">
              <div class="insight-label">Similarity signals</div>
              ${renderPillList(similaritySignals, "Similarity analysis pending")}
            </div>

            <div class="card card-soft" style="margin-top:16px;">
              <div class="insight-label">Lineage / genetics / strain matching</div>
              <div class="copy"><strong>Likely family:</strong> ${esc(lineage.family)}</div>
              <div class="copy" style="margin-top:8px;"><strong>Confidence:</strong> ${esc(lineage.confidence)}</div>
              <div class="copy" style="margin-top:8px;">${esc(lineage.note)}</div>
            </div>
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
        <h2 class="section-title">Potency intelligence</h2>
        <div class="copy"><strong>Potency class:</strong> ${esc(potency.label)}</div>
        <div class="copy" style="margin-top:8px;"><strong>Total THC:</strong> ${esc(data?.thc_total || "Not reported")}</div>
        <div class="copy" style="margin-top:8px;"><strong>CBD:</strong> ${esc(data?.cbd_total || "Not reported")}</div>
        <div class="copy" style="margin-top:8px;"><strong>Interpretation:</strong> ${esc(classifyPotency(thc).label === "Potency unclear" ? "Potency could not be confidently classified." : `This sample reads as ${potency.short.toLowerCase()} relative to common flower potency ranges.`)}</div>
      </div>
    </section>

    <section class="section">
      <div class="card card-soft">
        <h2 class="section-title">AI chemical brief</h2>
        <ul class="insight-list">
          ${executiveBrief.bullets.map((x) => `<li>${esc(x)}</li>`).join("")}
        </ul>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Post-Harvest Intelligence</h2>
        <div class="post-grid">
          <div class="post-card">
            <div class="post-label">Freshness</div>
            <div class="post-value">${esc(postHarvest.freshness.label)}</div>
            <div class="post-note">${esc(postHarvest.freshness.note)}</div>
          </div>
          <div class="post-card">
            <div class="post-label">Curing</div>
            <div class="post-value">${esc(postHarvest.cure.label)}</div>
            <div class="post-note">${esc(postHarvest.cure.note)}</div>
          </div>
          <div class="post-card">
            <div class="post-label">Stability</div>
            <div class="post-value">${esc(postHarvest.stability.label)}</div>
            <div class="post-note">${esc(postHarvest.stability.note)}</div>
          </div>
          <div class="post-card">
            <div class="post-label">Aroma preservation</div>
            <div class="post-value">${esc(terpeneDensity.label)}</div>
            <div class="post-note">${esc(terpeneDensity.label === "Aromatic intensity unclear" ? "Aroma preservation is hard to assess from the current data." : "Aromatic intensity contributes to the post-harvest preservation readout.")}</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card">
        <h2 class="section-title">Cultivation Expression</h2>
        <div class="lineage-grid">
          <div class="lineage-card">
            <div class="lineage-label">Expression</div>
            <div class="lineage-value">${esc(cultivation.expression)}</div>
            <div class="lineage-note">${esc(cultivation.cultivationNote)}</div>
          </div>
          <div class="lineage-card">
            <div class="lineage-label">Complexity score</div>
            <div class="lineage-value">${esc(`${cultivation.complexityScore}/10`)}</div>
            <div class="lineage-note">Chemical depth estimated from terpene diversity, potency, and minor-cannabinoid visibility.</div>
          </div>
          <div class="lineage-card">
            <div class="lineage-label">Preservation</div>
            <div class="lineage-value">${esc(cultivation.preservation)}</div>
            <div class="lineage-note">Aroma and structure suggest how well the profile has been preserved into the final sample.</div>
          </div>
          <div class="lineage-card">
            <div class="lineage-label">Effect direction</div>
            <div class="lineage-value">${esc(effectDirection.label)}</div>
            <div class="lineage-note">${esc(effectDirection.note)}</div>
          </div>
        </div>
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
        <h2 class="section-title">Benchmarking engine</h2>
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

    ${compareHtml}

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
            ${renderBar("THC", data?.thc_total || "0", thc * 3, "")}
            ${renderBar("CBD", data?.cbd_total || "0", cbd * 10, "gold")}
            ${renderBar("Total terpenes", data?.total_terpenes || "0", terps * 20, "teal")}

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

    <section class="future-box section">
      <div class="future-title">Next intelligence layer</div>
      <div class="copy">
        This V4 page is ready for true database-backed 3D clustering, batch-to-batch compare, live nearest-neighbor profile matching, strain family probability scoring, and marketplace-grade fingerprint search.
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

      const canvas = document.getElementById("chemicalSoulCanvas");
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      const cx = width / 2;
      const cy = height / 2;

      const palette = {
        core: getComputedStyle(document.documentElement).getPropertyValue("--soul-core").trim(),
        glow: getComputedStyle(document.documentElement).getPropertyValue("--soul-glow").trim(),
        outer: getComputedStyle(document.documentElement).getPropertyValue("--soul-outer").trim(),
        spark: getComputedStyle(document.documentElement).getPropertyValue("--soul-spark").trim()
      };

      const potency = ${JSON.stringify(Math.max(0.5, Math.min(10, thc / 3)))};
      const aroma = ${JSON.stringify(Math.max(0.5, Math.min(10, terps * 2.2)))};
      const complexity = ${JSON.stringify(Math.max(2, Math.min(12, topTerpenes.length + (hasMinor ? 2 : 0) + 2)))};
      const confidence = ${JSON.stringify(dataConfidenceScore / 100)};
      const dominance = ${JSON.stringify(topTerpenes.length > 1
        ? Math.min(1, Math.max(0.1, Math.abs(
            numberFromPercentString(topTerpenes[0]?.value) - numberFromPercentString(topTerpenes[1]?.value)
          ) + 0.15))
        : 0.55)};
      const orbitalDots = ${JSON.stringify(Math.max(8, Math.min(28, topCannabinoids.length + topTerpenes.length + (hasMinor ? 6 : 2))))};

      const particles = Array.from({ length: orbitalDots }).map((_, i) => ({
        angle: (Math.PI * 2 * i) / orbitalDots,
        radius: 80 + (i % 7) * 12 + complexity * 2,
        size: 1.6 + (i % 4) * 0.8,
        speed: 0.001 + (i % 5) * 0.0009 + dominance * 0.0009,
        alpha: 0.25 + ((i % 5) * 0.08),
        wobble: 4 + (i % 3) * 3
      }));

      function drawBlob(time) {
        const petals = Math.max(6, Math.min(18, complexity));
        const baseR = 60 + potency * 7;
        const variance = 18 + aroma * 4;
        const rot = time * (0.00018 + dominance * 0.0002);

        ctx.beginPath();

        for (let i = 0; i <= petals * 2; i++) {
          const a = (Math.PI * 2 * i) / (petals * 2) + rot;
          const mod = i % 2 === 0 ? 1 : 0.65 + dominance * 0.35;
          const r = baseR + variance * mod + Math.sin(time * 0.001 + i) * 5;
          const x = cx + Math.cos(a) * r;
          const y = cy + Math.sin(a) * r * (0.82 + confidence * 0.22);

          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }

        ctx.closePath();

        const grad = ctx.createRadialGradient(cx, cy, 18, cx, cy, baseR + variance + 30);
        grad.addColorStop(0, palette.core);
        grad.addColorStop(0.45, palette.glow);
        grad.addColorStop(0.85, palette.outer);
        grad.addColorStop(1, "rgba(255,255,255,0)");

        ctx.fillStyle = grad;
        ctx.globalAlpha = 0.78;
        ctx.fill();

        ctx.lineWidth = 1.2;
        ctx.strokeStyle = palette.spark;
        ctx.globalAlpha = 0.32;
        ctx.stroke();
      }

      function drawCore(time) {
        const pulse = Math.sin(time * 0.002) * 6;
        const r = 30 + potency * 3 + pulse;

        const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, r + 36);
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.18, palette.spark);
        grad.addColorStop(0.45, palette.core);
        grad.addColorStop(0.8, palette.glow);
        grad.addColorStop(1, "rgba(255,255,255,0)");

        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        ctx.fillStyle = grad;
        ctx.arc(cx, cy, r + 36, 0, Math.PI * 2);
        ctx.fill();
      }

      function drawOrbitals(time) {
        particles.forEach((p, idx) => {
          const ang = p.angle + time * p.speed;
          const wobble = Math.sin(time * 0.002 + idx) * p.wobble;
          const rr = p.radius + wobble;
          const x = cx + Math.cos(ang) * rr;
          const y = cy + Math.sin(ang) * rr * 0.72;

          ctx.beginPath();
          ctx.fillStyle = idx % 3 === 0 ? palette.spark : idx % 2 === 0 ? palette.outer : palette.glow;
          ctx.globalAlpha = p.alpha;
          ctx.arc(x, y, p.size, 0, Math.PI * 2);
          ctx.fill();

          if (idx % 5 === 0) {
            ctx.beginPath();
            ctx.strokeStyle = palette.outer;
            ctx.globalAlpha = 0.08;
            ctx.moveTo(cx, cy);
            ctx.lineTo(x, y);
            ctx.stroke();
          }
        });
      }

      function drawGrid() {
        ctx.globalAlpha = 0.07;
        ctx.strokeStyle = "#506050";
        ctx.lineWidth = 1;

        for (let x = 0; x <= width; x += 40) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }

        for (let y = 0; y <= height; y += 40) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }

        ctx.globalAlpha = 0.11;
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, height);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(width, cy);
        ctx.stroke();
      }

      function drawAxesLabels() {
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = "#5f6a5f";
        ctx.font = "12px Inter, Arial";

        ctx.fillText("Potency →", width - 90, cy - 8);
        ctx.fillText("Aroma ↑", cx + 10, 18);
        ctx.fillText("Complexity depth", 18, height - 16);
      }

      function animate(time) {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawBlob(time);
        drawCore(time);
        drawOrbitals(time);
        drawAxesLabels();
        requestAnimationFrame(animate);
      }

      requestAnimationFrame(animate);
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
    pdf_url: pdfUrl,
  };
}

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

app.post("/upload-coa", upload.single("file"), async (req, res) => {
  try {
    console.log("📥 Upload received");

    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const fileBuffer = req.file.buffer;

    // -----------------------------
    // 1. OCR WITH AZURE
    // -----------------------------
    console.log("🔍 Sending to Azure OCR...");

    const poller = await azureClient.beginAnalyzeDocument(
      "prebuilt-document",
      fileBuffer
    );

    const result = await poller.pollUntilDone();

    const ocrText = result?.content || "";

    if (!ocrText) {
      throw new Error("OCR returned empty content");
    }

    console.log("✅ OCR complete");

    const trimmedText = ocrText.slice(0, MAX_OCR_CHARS_FOR_OPENAI);

    // -----------------------------
    // 2. OPENAI PARSE
    // -----------------------------
    console.log("🧠 Sending to OpenAI...");

    const aiResponse = await openai.responses.create({
      model: OPENAI_MODEL,
      input: [
        {
          role: "system",
          content: extractionPrompt,
        },
        {
          role: "user",
          content: trimmedText,
        },
      ],
      max_output_tokens: 4000,
    });

    const rawText =
      aiResponse.output_text ||
      (aiResponse.output || [])
        .map((item) =>
          (item.content || []).map((p) => p.text || "").join("")
        )
        .join("");

    let parsed;

    try {
      parsed = JSON.parse(rawText);
    } catch (err) {
      console.error("❌ JSON PARSE ERROR:", rawText);
      throw new Error("OpenAI did not return valid JSON");
    }

    console.log("✅ AI parsing complete");

    // -----------------------------
    // 3. SAVE TO SUPABASE
    // -----------------------------
    const { data, error } = await supabase
      .from("coa_ai_reports")
      .insert([
        {
          file_name: req.file.originalname,
          report_json: parsed,
        },
      ])
      .select()
      .single();

    if (error) {
      console.error(error);
      throw new Error("Supabase insert failed");
    }

    const documentId = data.id;

    // -----------------------------
    // 4. RETURN REPORT URL
    // -----------------------------
    const reportUrl = buildReportUrl(req, documentId);

    console.log("🚀 Done:", reportUrl);

    res.json({
      success: true,
      report_url: reportUrl,
      id: documentId,
    });
  } catch (err) {
    console.error("❌ Upload pipeline error:", err.message);

    res.status(500).json({
      success: false,
      error: err.message,
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

const { data, error } = await supabase
  .from("coa_ai_reports")
  .insert([
    {
      report_json: parsedJson, // 🔥 THIS is your main data
      overall_score: parsedJson.overall_score || null,
      report_confidence_score: parsedJson.report_confidence_score || null,
      chemotype_identity: parsedJson.chemotype_identity || null,
      chemotype_descriptor: parsedJson.chemotype_descriptor || null,
      fingerprint_id: parsedJson.fingerprint_id || null,
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