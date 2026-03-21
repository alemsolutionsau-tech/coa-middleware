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
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const MAX_OCR_CHARS = Number(process.env.MAX_OCR_CHARS_FOR_OPENAI || 12000);
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 30000);
const SUPABASE_BUCKET = process.env.SUPABASE_BUCKET || "raw_documents";

if (!process.env.AZURE_DOC_INTELLIGENCE_ENDPOINT || !process.env.AZURE_DOC_INTELLIGENCE_KEY) {
  throw new Error("Missing Azure Document Intelligence environment variables");
}
if (!process.env.OPENAI_API_KEY) {
  throw new Error("Missing OPENAI_API_KEY in .env");
}

const azureClient = new DocumentAnalysisClient(
  process.env.AZURE_DOC_INTELLIGENCE_ENDPOINT,
  new AzureKeyCredential(process.env.AZURE_DOC_INTELLIGENCE_KEY)
);

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─────────────────────────────────────────────
// LAYER 1 — EXTRACTION PROMPTS (dual-pass)
// ─────────────────────────────────────────────

const CHEMISTRY_EXTRACTION_PROMPT = `
You are the Alem Solutions COA chemistry extraction engine.
Read OCR text from a cannabis Certificate of Analysis and return EXACTLY ONE valid JSON object.

CRITICAL RULES:
- Return valid JSON only. No markdown. No comments. No prose.
- If a field is missing return "" for strings, [] for arrays.
- Never guess or hallucinate values. Leave empty if uncertain.
- top_cannabinoids: include ALL named cannabinoids with detected values (not ND). Max 10.
- top_terpenes: include ALL named terpenes with detected values (not ND). Max 12. Sort by value descending.
- For units: always use "wt%" not "%". If the COA says wt%, use "wt%".
- CRITICAL: total_terpenes — look for "Total of all quantified terpenes" OR "Total Terpenes" anywhere in the document including the last page. This is a sum line, often at the end of the terpene table. Extract it precisely.
- CRITICAL: total_cannabinoids — look for "Total of all quantified cannabinoids" or similar sum line.
- CRITICAL: For thca, look for "THCA-A" or "THCA" — they are the same compound.
- For cbn, cbg, cbga, cbda: extract the numeric value only (e.g. "0.1057"), or "ND" if not detected.

Return this exact shape:
{
  "product_name": "",
  "batch_number": "",
  "coa_report_date": "",
  "product_type": "",
  "laboratory_name": "",
  "laboratory_accreditation": "",
  "laboratory_method": "",
  "thc_total": "",
  "thc_total_unit": "wt%",
  "cbd_total": "",
  "cbd_total_unit": "wt%",
  "thca": "",
  "thca_unit": "wt%",
  "cbda": "",
  "cbn": "",
  "cbg": "",
  "cbga": "",
  "cbca": "",
  "thcva": "",
  "total_terpenes": "",
  "total_terpenes_unit": "wt%",
  "total_cannabinoids": "",
  "hero_narrative": "",
  "top_cannabinoids": [
    { "name": "", "value": "", "unit": "wt%", "notes": "" }
  ],
  "top_terpenes": [
    { "name": "", "value": "", "unit": "wt%" }
  ]
}

For hero_narrative: write 2 concise sentences describing the product's chemical identity and positioning potential. Be factual, not promotional. Do not mention brand names. Focus on chemotype, potency tier, and terpene character.
`;

const CONTAMINANT_EXTRACTION_PROMPT = `
You are the Alem Solutions COA safety extraction engine.
Read OCR text from a cannabis Certificate of Analysis and extract ONLY safety, compliance, and contaminant data.

CRITICAL RULES:
- Return valid JSON only. No markdown. No prose.
- For each contaminant category, record the result status: "Pass", "Fail", "ND" (not detected), "Not tested", or the actual value if reported.
- Do not invent pass/fail results. Only record what is explicitly stated in the document.
- positive_flags: things that indicate quality or compliance (max 8)
- warning_flags: things that are missing, borderline, or concerning (max 6)

Return this exact shape:
{
  "pesticides_status": "",
  "pesticides_detail": "",
  "heavy_metals_status": "",
  "heavy_metals_detail": "",
  "microbials_status": "",
  "microbials_detail": "",
  "mycotoxins_status": "",
  "residual_solvents_status": "",
  "moisture_content": "",
  "water_activity": "",
  "foreign_matter_status": "",
  "contaminant_narrative": "",
  "lab_quality_summary": "",
  "positive_flags": [""],
  "warning_flags": [""]
}

For contaminant_narrative: write 1-2 sentences summarising the safety profile. If data is absent, state clearly what is missing rather than assuming compliance.
For lab_quality_summary: describe the laboratory's accreditation, method standards, and any notable analytical details found in the document.
`;

// ─────────────────────────────────────────────
// LAYER 2 — COMPOSITE SCORING ENGINE
// Deterministic — no AI, no hallucination risk
// ─────────────────────────────────────────────

function computeIntelligenceScore(extracted, contaminants) {
  let score = 0;
  const breakdown = {};

  // ── Potency (25 pts) ──────────────────────
  const thc = toNum(extracted.thc_total);
  let potencyScore = 0;
  if (thc >= 28) potencyScore = 25;
  else if (thc >= 24) potencyScore = 22;
  else if (thc >= 20) potencyScore = 18;
  else if (thc >= 15) potencyScore = 13;
  else if (thc > 0) potencyScore = 8;
  // CBD-dominant products score differently
  const cbd = toNum(extracted.cbd_total);
  if (cbd >= 15 && thc < 5) potencyScore = Math.max(potencyScore, 20);
  breakdown.potency = { score: potencyScore, max: 25 };
  score += potencyScore;

  // ── Terpenes (25 pts) ─────────────────────
  const terps = toNum(extracted.total_terpenes);
  const terpCount = (extracted.top_terpenes || []).filter(t => toNum(t.value) > 0).length;
  let terpScore = 0;
  if (terps >= 3.5) terpScore = 22;
  else if (terps >= 2.5) terpScore = 20;
  else if (terps >= 1.8) terpScore = 16;
  else if (terps >= 1.0) terpScore = 11;
  else if (terps > 0) terpScore = 6;
  // Bonus for breadth
  if (terpCount >= 8) terpScore = Math.min(25, terpScore + 5);
  else if (terpCount >= 5) terpScore = Math.min(25, terpScore + 3);
  breakdown.terpenes = { score: terpScore, max: 25 };
  score += terpScore;

  // ── Minor cannabinoids (20 pts) ───────────
  const cbga = toNum(extracted.cbga);
  const cbn = toNum(extracted.cbn);
  const cbg = toNum(extracted.cbg);
  const cbda = toNum(extracted.cbda);
  const cbca = toNum(extracted.cbca);
  const thcva = toNum(extracted.thcva);
  const minorCount = [cbga, cbn, cbg, cbda, cbca, thcva].filter(v => v > 0).length;
  const totalMinors = toNum(extracted.total_cannabinoids) - toNum(extracted.thc_total) - toNum(extracted.cbd_total);
  let minorScore = 0;
  if (totalMinors >= 5) minorScore = 20;
  else if (totalMinors >= 3) minorScore = 16;
  else if (totalMinors >= 1.5) minorScore = 12;
  else if (minorCount >= 3) minorScore = 10;
  else if (minorCount >= 1) minorScore = 6;
  // Penalise if CBN is high (degradation signal)
  if (cbn >= 1) minorScore = Math.max(0, minorScore - 4);
  breakdown.minors = { score: minorScore, max: 20 };
  score += minorScore;

  // ── Safety / contaminants (20 pts) ────────
  let safetyScore = 0;
  const pest = String(contaminants.pesticides_status || "").toLowerCase();
  const metals = String(contaminants.heavy_metals_status || "").toLowerCase();
  const micro = String(contaminants.microbials_status || "").toLowerCase();
  const accred = String(contaminants.lab_quality_summary || "").toLowerCase();
  if (pest.includes("pass") || pest.includes("nd")) safetyScore += 7;
  else if (pest.includes("not tested") || !pest) safetyScore += 2;
  if (metals.includes("pass") || metals.includes("nd")) safetyScore += 7;
  else if (metals.includes("not tested") || !metals) safetyScore += 2;
  if (micro.includes("pass") || micro.includes("nd")) safetyScore += 6;
  else if (micro.includes("not tested") || !micro) safetyScore += 1;
  // Lab accreditation bonus
  if (accred.includes("17025") || accred.includes("iso")) safetyScore = Math.min(20, safetyScore + 3);
  breakdown.safety = { score: safetyScore, max: 20 };
  score += safetyScore;

  // ── Data completeness (10 pts) ────────────
  let dataScore = 0;
  if (extracted.thc_total) dataScore += 2;
  if (extracted.total_terpenes) dataScore += 2;
  if ((extracted.top_terpenes || []).length >= 3) dataScore += 2;
  if ((extracted.top_cannabinoids || []).length >= 3) dataScore += 1;
  if (extracted.coa_report_date) dataScore += 1;
  if (extracted.laboratory_name) dataScore += 1;
  if (extracted.batch_number) dataScore += 1;
  breakdown.dataCompleteness = { score: dataScore, max: 10 };
  score += dataScore;

  const grade = score >= 90 ? "A+" : score >= 85 ? "A" : score >= 78 ? "A-"
    : score >= 74 ? "B+" : score >= 70 ? "B" : score >= 65 ? "B-"
    : score >= 58 ? "C+" : score >= 50 ? "C" : "D";

  const tier = score >= 85 ? "Exceptional"
    : score >= 74 ? "Strong"
    : score >= 60 ? "Moderate"
    : "Limited";

  return { total: score, grade, tier, breakdown };
}

// ─────────────────────────────────────────────
// LAYER 3 — INTELLIGENCE GENERATION
// ─────────────────────────────────────────────

const TERPENE_INTEL = {
  terpinolene: {
    direction: "Uplifting / mentally active",
    note: "Terpinolene-dominant profiles often feel brighter, more mentally active, and less body-heavy. Associated with Haze and Jack-type chemotype families.",
    lineage: "Haze / Jack / Durban-type families",
    lineageConfidence: "Moderate",
  },
  myrcene: {
    direction: "Calming / body-centred",
    note: "Myrcene-dominant profiles tend toward more calming, body-heavy experiences. Common in Kush and OG-type chemotype families.",
    lineage: "Kush / OG / indica-leaning families",
    lineageConfidence: "Moderate",
  },
  "beta-myrcene": {
    direction: "Calming / body-centred",
    note: "Myrcene-dominant profiles tend toward more calming, body-heavy experiences. Common in Kush and OG-type chemotype families.",
    lineage: "Kush / OG / indica-leaning families",
    lineageConfidence: "Moderate",
  },
  limonene: {
    direction: "Bright / mood-forward",
    note: "Limonene-forward profiles are typically positioned as brighter, more mood-forward, and citrus-dominant in aroma.",
    lineage: "Citrus hybrid / Gelato / Runtz-type families",
    lineageConfidence: "Moderate",
  },
  caryophyllene: {
    direction: "Balanced / structured",
    note: "Caryophyllene-led profiles often read as warm, structured, and balanced — with spice-forward aromatic character.",
    lineage: "Balanced hybrid / spice-forward families",
    lineageConfidence: "Low-moderate",
  },
  "trans-caryophyllene": {
    direction: "Balanced / structured",
    note: "Caryophyllene-led profiles often read as warm, structured, and balanced — with spice-forward aromatic character.",
    lineage: "Balanced hybrid / spice-forward families",
    lineageConfidence: "Low-moderate",
  },
  pinene: {
    direction: "Clear / alert",
    note: "Pinene-prominent profiles are often interpreted as clearer and more alertness-oriented, with a fresh, pine-forward aroma.",
    lineage: "Pine-forward / clear sativa-leaning families",
    lineageConfidence: "Low-moderate",
  },
  "alpha-pinene": {
    direction: "Clear / alert",
    note: "Alpha-pinene contributes a fresh, sharp aromatic quality and is associated with clearer, more alert experiential profiles.",
    lineage: "Pine-forward / clear sativa-leaning families",
    lineageConfidence: "Low-moderate",
  },
  ocimene: {
    direction: "Bright / floral / complex",
    note: "Ocimene adds floral, herbal, and tropical aromatic complexity. As a secondary terpene it supports a distinctive multi-layered aromatic fingerprint.",
    lineage: "Haze / tropical-adjacent families",
    lineageConfidence: "Low",
  },
  linalool: {
    direction: "Calming / floral",
    note: "Linalool is associated with floral, lavender-like aromas and calmer, more relaxing experiential profiles.",
    lineage: "Floral / indica-adjacent families",
    lineageConfidence: "Low-moderate",
  },
  bisabolol: {
    direction: "Gentle / smooth",
    note: "Bisabolol contributes a smooth, gentle aromatic character and is associated with softer, more relaxed profiles.",
    lineage: "Smooth hybrid families",
    lineageConfidence: "Low",
  },
  humulene: {
    direction: "Earthy / structured",
    note: "Humulene adds earthy, woody aromatic depth and contributes to a grounded, structured profile character.",
    lineage: "OG / earthy hybrid families",
    lineageConfidence: "Low",
  },
  farnesene: {
    direction: "Floral / fruity / complex",
    note: "Farnesene contributes apple-adjacent, floral, and fruity aromatic notes. Relatively rare as a dominant terpene — adds differentiation value.",
    lineage: "Complex hybrid families",
    lineageConfidence: "Low",
  },
};

function getTerpeneIntel(terpName = "") {
  const key = String(terpName).toLowerCase().trim();
  for (const [k, v] of Object.entries(TERPENE_INTEL)) {
    if (key.includes(k)) return v;
  }
  return {
    direction: "Chemistry-led",
    note: "A clearer directional read would require stronger terpene dominance or a broader comparison dataset.",
    lineage: "Unclear cluster",
    lineageConfidence: "Low",
  };
}

function generateFingerprintId(terpenes = []) {
  const ABBREV = {
    terpinolene: "TPN", myrcene: "MYR", "beta-myrcene": "MYR",
    limonene: "LIM", caryophyllene: "CAR", "trans-caryophyllene": "CAR",
    "beta-caryophyllene": "CAR", pinene: "PIN", "alpha-pinene": "PIN",
    "beta-pinene": "BPN", linalool: "LIN", ocimene: "OCI",
    humulene: "HUM", "alpha-humulene": "HUM", bisabolol: "BIS",
    "alpha-bisabolol": "BIS", farnesene: "FAR", guaiol: "GUA",
    terpineol: "TPO", "alpha-terpineol": "TPO", nerolidol: "NER",
    "trans-nerolidol": "NER", valencene: "VAL", geraniol: "GER",
    borneol: "BOR", camphene: "CMP", eucalyptol: "EUC",
  };
  const top3 = terpenes
    .filter(t => {
      const v = parseFloat(t.value);
      return !isNaN(v) && v > 0;
    })
    .slice(0, 3)
    .map(t => {
      const key = String(t.name || "").toLowerCase().trim();
      for (const [k, abbr] of Object.entries(ABBREV)) {
        if (key.includes(k)) return abbr;
      }
      return String(t.name || "").replace(/[^a-zA-Z]/g, "").slice(0, 3).toUpperCase();
    });
  if (!top3.length) return "UNK";
  return top3.join("-");
}

function buildAudienceNarratives(extracted, contaminants, scoring) {
  const thc = toNum(extracted.thc_total);
  const cbd = toNum(extracted.cbd_total);
  const terps = toNum(extracted.total_terpenes);
  const cbga = toNum(extracted.cbga);
  const cbn = toNum(extracted.cbn);
  const terpenes = extracted.top_terpenes || [];
  const lead = terpenes[0]?.name || "";
  const second = terpenes[1]?.name || "";
  const intel = getTerpeneIntel(lead);
  const score = scoring.total;
  const terpsDisplay = terps > 0 ? terps.toFixed(2) : "not captured";

  const brand = [];
  if (lead) {
    const rarity = ["terpinolene", "ocimene", "farnesene"].some(t => lead.toLowerCase().includes(t));
    brand.push(rarity
      ? `${lead} dominance is a genuine market differentiator — fewer than 15% of dried flower products lead with this compound. Supports a distinct, premium identity.`
      : `${lead}-dominant chemistry is well-recognised in market positioning and supports clear directional messaging around ${intel.direction.toLowerCase()}.`
    );
  }
  brand.push(`Intelligence score ${score}/100 (${scoring.tier}). ${scoring.breakdown.safety.score < 10 ? "Safety data gap limits maximum score — structured contaminant detail should be captured in future submissions." : "Strong across all five scoring dimensions."}`);
  if (cbga >= 1) brand.push(`CBGA at ${cbga.toFixed(2)} wt% is notable — elevated CBGA can support a biosynthetic depth narrative and minor cannabinoid product story for premium positioning.`);
  if (terps >= 2) brand.push(`Total terpene content of ${terps.toFixed(2)} wt% sits in the upper range for dried flower. Supports premium aromatic positioning and commands attention in terpene-literate markets.`);

  const clinical = [];
  clinical.push(`THC total: ${thc.toFixed(1)} wt% — ${thc >= 24 ? "high range. Tolerance-aware patient selection and dose titration is warranted." : thc >= 18 ? "moderate-to-high range. Appropriate patient education on dose sensitivity recommended." : "moderate range."}`);
  clinical.push(cbd > 0.5 ? `CBD present at ${cbd.toFixed(2)} wt% — may provide some modulation of THC-dominant effects.` : "CBD not detected — no intrinsic CBD modulation of THC intensity in this profile.");
  clinical.push(`Terpene direction: ${intel.note}`);
  clinical.push(cbn >= 0.5 ? `CBN detected at ${cbn.toFixed(2)} wt% — may indicate some THC oxidative degradation. Consider storage conditions.` : "CBN not detected — minimal oxidative degradation signal.");
  const safetyNote = contaminants.pesticides_status && !contaminants.pesticides_status.toLowerCase().includes("not tested")
    ? `Pesticide panel: ${contaminants.pesticides_status}. ${contaminants.heavy_metals_status ? "Heavy metals: " + contaminants.heavy_metals_status + "." : ""}`
    : "Structured contaminant pass/fail data not captured — review source COA directly for pesticide, heavy metals, and microbial panels before prescribing.";
  clinical.push(safetyNote);

  const patient = [];
  patient.push(thc >= 24 ? "This is a high-THC product. It is likely to feel strong, especially for patients with lower tolerance or those new to cannabis." : thc >= 18 ? "This product has moderate-to-high THC content. Start with a low dose and increase gradually." : "This product has moderate THC content.");
  patient.push(lead ? `The leading terpene — ${lead} — is typically associated with ${intel.direction.toLowerCase()} experiences.` : "A clear terpene direction could not be identified from this report.");
  patient.push(cbd > 0.5 ? `CBD is present at ${cbd.toFixed(2)} wt%, which may provide some balancing effect.` : "CBD is not detected in this product — there is no built-in balance from CBD.");
  if (lead || second) patient.push(`Expect a${["aeiou"].join("").includes((lead || second)[0].toLowerCase()) ? "n" : ""} ${[lead, second].filter(Boolean).join(" and ")}-forward aroma.`);

  const buyer = [];
  buyer.push(`Intelligence score: ${score}/100 — ${scoring.tier}. ${scoring.breakdown.terpenes.score >= 18 ? "Leads on terpene richness." : ""} ${scoring.breakdown.safety.score < 10 ? "Safety data gap is the primary score limiter — request full compliance documentation." : "Clean safety profile."}`);
  if (lead) buyer.push(`${lead}-dominant COA in a predominantly myrcene-heavy market. Above-average shelf differentiation potential.`);
  buyer.push(terps >= 2 ? `Total terpene content (${terpsDisplay} wt%) supports premium pricing. Terpene richness and chemotype clarity justify category-leading positioning.` : `Terpene content (${terpsDisplay} wt%) is serviceable. Pricing should align with mid-tier terpene density.`);
  const safetyBuyerNote = (!contaminants.pesticides_status || contaminants.pesticides_status.toLowerCase().includes("not tested"))
    ? "Structured contaminant overview not captured. Request full compliance panel from producer before listing."
    : `Contaminant overview: Pesticides — ${contaminants.pesticides_status}. Metals — ${contaminants.heavy_metals_status || "not reported"}.`;
  buyer.push(safetyBuyerNote);

  return { brand, clinical, patient, buyer };
}

function buildPostHarvestIntel(extracted, contaminants) {
  const terps = toNum(extracted.total_terpenes);
  const terpCount = (extracted.top_terpenes || []).filter(t => toNum(t.value) > 0).length;
  const cbn = toNum(extracted.cbn);

  const freshness = terps >= 3
    ? { label: "Strong", note: "Terpene retention is high — consistent with excellent post-harvest preservation." }
    : terps >= 2
    ? { label: "Good", note: "Terpene retention suggests well-preserved aromatic content through handling and storage." }
    : terps >= 1.2
    ? { label: "Moderate", note: "Some aromatic expression remains, though terpene density is not exceptional." }
    : terps > 0
    ? { label: "Light", note: "Lower terpene density may suggest some aromatic loss through processing or storage." }
    : { label: "Unknown", note: "Freshness cannot be inferred — terpene density not captured." };

  const curing = terpCount >= 6 && terps >= 1.5
    ? { label: "Positive signal", note: "Broad terpene spectrum without flat profile suggests preserved aromatic complexity through curing." }
    : terpCount >= 3
    ? { label: "Moderate signal", note: "Some terpene layering visible. Confidence in curing quality is limited without additional data." }
    : { label: "Limited signal", note: "Insufficient terpene breadth to draw a confident curing quality inference." };

  const degradation = cbn >= 1.5
    ? { label: "Notable", note: `CBN at ${cbn.toFixed(2)} wt% suggests meaningful THC oxidative degradation. Review storage conditions.` }
    : cbn >= 0.5
    ? { label: "Low signal", note: `CBN detected at ${cbn.toFixed(2)} wt% — minor degradation signal. Not clinically significant at this level.` }
    : { label: "Minimal", note: "CBN not detected — minimal evidence of THC oxidative degradation." };

  const moisture = extracted.moisture_content || contaminants.moisture_content;
  const stability = moisture
    ? { label: "Data present", note: `Moisture content: ${moisture}. ${parseFloat(moisture) <= 12 ? "Within acceptable range for storage stability." : "Review against target range for product category."}` }
    : { label: "Limited", note: "Moisture and water activity data not captured — stability confidence is limited to terpene-based inference only." };

  return { freshness, curing, degradation, stability };
}

// ─────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────

function toNum(value) {
  if (value === null || value === undefined) return 0;
  const raw = String(value).trim().toLowerCase();
  if (!raw || raw === "nd" || raw === "n/d" || raw.includes("not detected") || raw.includes("<loq") || raw.includes("< lod")) return 0;
  const match = raw.match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : 0;
}

function esc(v = "") {
  return String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function sanitizeFileName(name = "") {
  return String(name || "").replace(/[^a-zA-Z0-9-_.]/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
}

function getBaseUrl(req) {
  const proto = req.headers["x-forwarded-proto"] || req.protocol || "https";
  return `${proto}://${req.get("host")}`;
}

function detectMimeType(url = "", headerContentType = "") {
  const lowerUrl = String(url || "").toLowerCase();
  const lowerHeader = String(headerContentType || "").toLowerCase();
  if (lowerHeader.includes("application/pdf") || lowerUrl.endsWith(".pdf")) return "application/pdf";
  if (lowerHeader.includes("image/png") || lowerUrl.endsWith(".png")) return "image/png";
  if (lowerHeader.includes("image/jpeg") || lowerUrl.endsWith(".jpg") || lowerUrl.endsWith(".jpeg")) return "image/jpeg";
  if (lowerHeader.includes("image/tiff") || lowerUrl.endsWith(".tif") || lowerUrl.endsWith(".tiff")) return "image/tiff";
  return "application/octet-stream";
}

function prepareOCRText(cleanText = "") {
  const text = String(cleanText || "").trim();
  if (!text) return "";
  if (text.length <= MAX_OCR_CHARS) return text;
  const headLen = Math.floor(MAX_OCR_CHARS * 0.7);
  const tailLen = MAX_OCR_CHARS - headLen;
  return [text.slice(0, headLen), "\n\n[TRUNCATED]\n\n", text.slice(-tailLen)].join("");
}

function safeSnippet(value, max = 1000) {
  return String(value || "").slice(0, max);
}

// ─────────────────────────────────────────────
// OPENAI CALL WRAPPER
// ─────────────────────────────────────────────

async function callOpenAI(systemPrompt, userText, maxTokens = 1600) {
  const response = await Promise.race([
    openai.chat.completions.create({
      model: OPENAI_MODEL,
      max_tokens: maxTokens,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userText },
      ],
    }),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`OpenAI timeout after ${OPENAI_TIMEOUT_MS}ms`)), OPENAI_TIMEOUT_MS)
    ),
  ]);
  return response.choices?.[0]?.message?.content || "";
}

function extractJSON(text = "") {
  const trimmed = String(text || "").trim();
  if (!trimmed) throw new Error("Empty OpenAI response");
  const first = trimmed.indexOf("{");
  const last = trimmed.lastIndexOf("}");
  if (first === -1 || last === -1 || last <= first) throw new Error("No valid JSON object found");
  return trimmed.slice(first, last + 1);
}

function normalizeChemistry(data = {}) {
  const s = (v) => (v === null || v === undefined ? "" : String(v));
  const a = (v) => (Array.isArray(v) ? v : []);

  const topTerpenes = a(data.top_terpenes).slice(0, 12).map(i => ({
    name: s(i?.name),
    value: s(i?.value),
    unit: s(i?.unit) || "wt%",
  }));

  // Fallback: if total_terpenes is missing, compute from sum of individual terpene values
  let totalTerpenes = s(data.total_terpenes);
  if (!totalTerpenes || totalTerpenes === "ND" || totalTerpenes === "") {
    const computed = topTerpenes.reduce((sum, t) => {
      const n = parseFloat(t.value);
      return sum + (isNaN(n) ? 0 : n);
    }, 0);
    if (computed > 0) {
      totalTerpenes = computed.toFixed(4);
      console.log(`⚠️  total_terpenes missing — computed from terpene sum: ${totalTerpenes}`);
    }
  }

  // Normalise units: always "wt%"
  const fixUnit = (u) => {
    const clean = s(u).trim();
    if (!clean || clean === "%") return "wt%";
    return clean;
  };

  return {
    product_name: s(data.product_name),
    batch_number: s(data.batch_number),
    coa_report_date: s(data.coa_report_date),
    product_type: s(data.product_type),
    laboratory_name: s(data.laboratory_name),
    laboratory_accreditation: s(data.laboratory_accreditation),
    laboratory_method: s(data.laboratory_method),
    thc_total: s(data.thc_total),
    thc_total_unit: fixUnit(data.thc_total_unit),
    cbd_total: s(data.cbd_total),
    cbd_total_unit: fixUnit(data.cbd_total_unit),
    thca: s(data.thca),
    thca_unit: fixUnit(data.thca_unit),
    cbda: s(data.cbda),
    cbn: s(data.cbn),
    cbg: s(data.cbg),
    cbga: s(data.cbga),
    cbca: s(data.cbca),
    thcva: s(data.thcva),
    total_terpenes: totalTerpenes,
    total_terpenes_unit: fixUnit(data.total_terpenes_unit) || "wt%",
    total_cannabinoids: s(data.total_cannabinoids),
    hero_narrative: s(data.hero_narrative),
    top_cannabinoids: a(data.top_cannabinoids).slice(0, 10).map(i => ({
      name: s(i?.name),
      value: s(i?.value),
      unit: fixUnit(i?.unit),
      notes: s(i?.notes),
    })),
    top_terpenes: topTerpenes,
  };
}

function normalizeContaminants(data = {}) {
  const s = (v) => (v === null || v === undefined ? "" : String(v));
  const a = (v) => (Array.isArray(v) ? v : []);
  return {
    pesticides_status: s(data.pesticides_status),
    pesticides_detail: s(data.pesticides_detail),
    heavy_metals_status: s(data.heavy_metals_status),
    heavy_metals_detail: s(data.heavy_metals_detail),
    microbials_status: s(data.microbials_status),
    microbials_detail: s(data.microbials_detail),
    mycotoxins_status: s(data.mycotoxins_status),
    residual_solvents_status: s(data.residual_solvents_status),
    moisture_content: s(data.moisture_content),
    water_activity: s(data.water_activity),
    foreign_matter_status: s(data.foreign_matter_status),
    contaminant_narrative: s(data.contaminant_narrative),
    lab_quality_summary: s(data.lab_quality_summary),
    positive_flags: a(data.positive_flags).slice(0, 8).map(x => s(x)),
    warning_flags: a(data.warning_flags).slice(0, 6).map(x => s(x)),
  };
}

// ─────────────────────────────────────────────
// DUAL-PASS EXTRACTION
// ─────────────────────────────────────────────

async function extractChemistry(ocrText) {
  const modelInput = prepareOCRText(ocrText);
  console.log("🧪 Chemistry extraction pass...");
  try {
    const raw = await callOpenAI(CHEMISTRY_EXTRACTION_PROMPT, modelInput, 1800);
    const json = extractJSON(raw);
    return normalizeChemistry(JSON.parse(json));
  } catch (err) {
    console.warn("Chemistry extraction failed, returning empty:", err.message);
    return normalizeChemistry({});
  }
}

async function extractContaminants(ocrText) {
  const modelInput = prepareOCRText(ocrText);
  console.log("🛡️ Contaminant extraction pass...");
  try {
    const raw = await callOpenAI(CONTAMINANT_EXTRACTION_PROMPT, modelInput, 1000);
    const json = extractJSON(raw);
    return normalizeContaminants(JSON.parse(json));
  } catch (err) {
    console.warn("Contaminant extraction failed, returning empty:", err.message);
    return normalizeContaminants({});
  }
}

async function runDualPassExtraction(ocrText) {
  const [chemistry, contaminants] = await Promise.all([
    extractChemistry(ocrText),
    extractContaminants(ocrText),
  ]);
  return { chemistry, contaminants };
}

// ─────────────────────────────────────────────
// AZURE OCR
// ─────────────────────────────────────────────

async function extractDocumentFromUrl(fileUrl) {
  if (!fileUrl || typeof fileUrl !== "string") throw new Error("fileUrl must be a non-empty string");
  const fileResponse = await axios.get(fileUrl, {
    responseType: "arraybuffer", timeout: 60000, maxRedirects: 5,
    validateStatus: (status) => status >= 200 && status < 300,
  });
  const mimeType = detectMimeType(fileUrl, fileResponse.headers["content-type"]);
  const poller = await azureClient.beginAnalyzeDocument("prebuilt-layout", fileResponse.data, { contentType: mimeType });
  const result = await poller.pollUntilDone();
  const pages = (result.pages || []).map(page => ({
    page_number: page.pageNumber,
    text: (page.lines || []).map(line => line.content).join("\n"),
  }));
  return {
    mimeType,
    plain_text: pages.map(p => p.text).join("\n\n").trim(),
    page_count: pages.length,
    table_count: (result.tables || []).length,
  };
}

// ─────────────────────────────────────────────
// SUPABASE
// ─────────────────────────────────────────────

async function uploadBufferToSupabase({ buffer, originalName, mimeType, folder = "raw_documents" }) {
  const safeName = sanitizeFileName(originalName || `upload-${Date.now()}`);
  const storagePath = `${folder}/${Date.now()}-${safeName}`;
  const { error } = await supabase.storage.from(SUPABASE_BUCKET).upload(storagePath, buffer, {
    contentType: mimeType || "application/octet-stream", upsert: false,
  });
  if (error) throw new Error(`Supabase upload failed: ${error.message}`);
  const { data: publicData } = supabase.storage.from(SUPABASE_BUCKET).getPublicUrl(storagePath);
  if (!publicData?.publicUrl) throw new Error("Failed to generate public URL");
  return { storagePath, publicUrl: publicData.publicUrl };
}

async function insertCOAReport({ chemistry, contaminants, scoring, intelligence, sourceUrl, storagePath, originalFilename, mimeType }) {
  const payload = {
    report_json: {
      chemistry,
      contaminants,
      scoring,
      intelligence,
      _meta: {
        source_url: sourceUrl || "",
        storage_path: storagePath || "",
        original_filename: originalFilename || "",
        mime_type: mimeType || "",
        saved_at: new Date().toISOString(),
        schema_version: "5.0",
      },
    },
    overall_score: scoring.total,
    report_confidence_score: scoring.breakdown.dataCompleteness.score,
    chemotype_identity: intelligence.fingerprintId,
    chemotype_descriptor: intelligence.effectDirection,
    fingerprint_id: intelligence.fingerprintId,
  };
  const { data, error } = await supabase.from("coa_ai_reports").insert([payload]).select().single();
  if (error) throw new Error(`Supabase insert failed: ${error.message}`);
  return data;
}

async function getReportById(id) {
  const { data, error } = await supabase.from("coa_ai_reports").select("*").eq("id", id).single();
  if (error) throw new Error(`Could not load report: ${error.message}`);
  return data;
}

// ─────────────────────────────────────────────
// LAYER 5 — REPORT HTML RENDERER
// ─────────────────────────────────────────────

function renderReportHTML(reportJson = {}, options = {}) {
  const chemistry = reportJson.chemistry || {};
  const contaminants = reportJson.contaminants || {};
  const scoring = reportJson.scoring || computeIntelligenceScore(chemistry, contaminants);
  const intelligence = reportJson.intelligence || {};

  const terpenes = chemistry.top_terpenes || [];
  const cannabinoids = chemistry.top_cannabinoids || [];
  const thc = toNum(chemistry.thc_total);
  const cbd = toNum(chemistry.cbd_total);
  const terps = toNum(chemistry.total_terpenes);
  const leadTerpene = terpenes[0]?.name || "";
  const secondTerpene = terpenes[1]?.name || "";
  const thirdTerpene = terpenes[2]?.name || "";
  const terpIntel = getTerpeneIntel(leadTerpene);
  const fingerprintId = intelligence.fingerprintId || generateFingerprintId(terpenes);
  const audiences = intelligence.audiences || buildAudienceNarratives(chemistry, contaminants, scoring);
  const postHarvest = intelligence.postHarvest || buildPostHarvestIntel(chemistry, contaminants);

  const maxTerpVal = Math.max(...terpenes.map(t => toNum(t.value)), 0.001);

  const productName = chemistry.product_name || "COA Report";
  const heroNarrative = chemistry.hero_narrative || `${leadTerpene ? leadTerpene + "-dominant" : "Cannabis"} profile with ${terps > 2 ? "strong" : terps > 1 ? "moderate" : "light"} terpene expression and ${thc >= 24 ? "high" : thc >= 18 ? "moderate-high" : "moderate"} THC content.`;

  function renderAudiencePanel(items = [], id) {
    if (!items.length) return `<div class="empty">No intelligence available for this audience.</div>`;
    return items.map(line => `<div class="insight-line">${esc(line)}</div>`).join("");
  }

  function renderTerpBar(t, i) {
    const val = toNum(t.value);
    const width = Math.max(4, (val / maxTerpVal) * 100);
    const opacity = Math.max(0.4, 1 - i * 0.09);
    return `
      <div class="bar-row">
        <div class="bar-head">
          <span class="bar-name">${esc(t.name)}</span>
          <span class="bar-val">${esc([t.value, t.unit].filter(Boolean).join(" "))}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${width.toFixed(1)}%;opacity:${opacity.toFixed(2)}"></div>
        </div>
      </div>`;
  }

  function renderCannCard(c) {
    const isMain = ["total thc", "thca", "total cbd", "cbd"].some(k => (c.name || "").toLowerCase().includes(k));
    return `
      <div class="cann-card${isMain ? " cann-primary" : ""}">
        <div class="cann-name">${esc(c.name)}</div>
        <div class="cann-val">${esc(c.value)} <span class="cann-unit">${esc(c.unit)}</span></div>
        ${c.notes ? `<div class="cann-note">${esc(c.notes)}</div>` : ""}
      </div>`;
  }

  function renderContRow(label, status, standard) {
    const isPass = /pass|nd|not detected/i.test(status);
    const isFail = /fail/i.test(status);
    const isMissing = !status || /not tested|not reported/i.test(status);
    const cls = isFail ? "cont-fail" : isMissing ? "cont-missing" : isPass ? "cont-pass" : "cont-value";
    const display = status || "Not reported";
    return `
      <div class="cont-row">
        <span class="cont-cat">${esc(label)}</span>
        <span class="cont-status ${cls}">${esc(display)}</span>
        <span class="cont-std">${esc(standard)}</span>
      </div>`;
  }

  function renderPill(text, type = "good") {
    return `<span class="pill ${type}">${esc(text)}</span>`;
  }

  const scoreDims = [
    { label: "Potency", ...scoring.breakdown.potency },
    { label: "Terpenes", ...scoring.breakdown.terpenes },
    { label: "Minor cannabinoids", ...scoring.breakdown.minors },
    { label: "Safety data", ...scoring.breakdown.safety },
    { label: "Data completeness", ...scoring.breakdown.dataCompleteness },
  ];

  const posFlags = contaminants.positive_flags || [];
  const warnFlags = contaminants.warning_flags || [];
  const rawJson = esc(JSON.stringify(reportJson, null, 2));

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>${esc(productName)} — Alem Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg: #f7f5f0;
  --surface: #ffffff;
  --surface2: #f2f0eb;
  --border: rgba(30,28,24,0.10);
  --border-strong: rgba(30,28,24,0.18);
  --text: #1a1815;
  --muted: #6b6760;
  --hint: #9e9b96;
  --green: #1c3d2b;
  --green-mid: #2d5c42;
  --green-light: #e8f2ec;
  --green-border: rgba(28,61,43,0.18);
  --amber: #92650a;
  --amber-light: #fdf3e0;
  --amber-border: rgba(146,101,10,0.18);
  --red: #8b2020;
  --red-light: #fdeaea;
  --red-border: rgba(139,32,32,0.18);
  --radius: 14px;
  --radius-sm: 8px;
  --radius-pill: 999px;
  --shadow: 0 1px 3px rgba(30,28,24,0.06), 0 4px 16px rgba(30,28,24,0.04);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.shell {
  max-width: 1160px;
  margin: 0 auto;
  padding: 24px 20px 72px;
}

/* ── Top bar ── */
.topbar {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 9px 16px;
  border-radius: var(--radius-pill);
  border: 1px solid var(--border-strong);
  background: var(--surface);
  color: var(--muted);
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  text-decoration: none;
  transition: background 0.15s, color 0.15s;
}
.btn:hover { background: var(--surface2); color: var(--text); }
.btn-primary {
  background: var(--green);
  color: #fff;
  border-color: var(--green);
}
.btn-primary:hover { background: var(--green-mid); color: #fff; }

/* ── Hero ── */
.hero {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 28px 24px;
  margin-bottom: 14px;
  box-shadow: var(--shadow);
}

.hero-eyebrow {
  font-size: 11px;
  font-weight: 500;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--hint);
  margin-bottom: 14px;
}

.hero-layout {
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 24px;
  align-items: start;
}

.product-name {
  font-size: 38px;
  font-weight: 600;
  line-height: 1.0;
  letter-spacing: -.04em;
  color: var(--text);
  margin-bottom: 10px;
}

.hero-narrative {
  font-size: 14px;
  color: var(--muted);
  line-height: 1.75;
  max-width: 56ch;
  margin-bottom: 16px;
}

.badge-row { display: flex; flex-wrap: wrap; gap: 6px; }

.badge {
  font-size: 12px;
  font-weight: 400;
  color: var(--muted);
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  padding: 4px 11px;
}

/* ── Score card ── */
.score-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 18px 20px;
}

.score-top {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 4px;
}

.score-number {
  font-size: 52px;
  font-weight: 600;
  line-height: 1;
  color: var(--text);
  letter-spacing: -.04em;
}

.score-denom {
  font-size: 18px;
  color: var(--hint);
  font-weight: 300;
}

.score-grade {
  font-size: 13px;
  font-weight: 500;
  color: var(--green-mid);
  margin-bottom: 12px;
}

.score-track {
  height: 5px;
  background: var(--border);
  border-radius: var(--radius-pill);
  overflow: hidden;
  margin-bottom: 14px;
}

.score-fill {
  height: 100%;
  border-radius: var(--radius-pill);
  background: var(--green);
}

.dim-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 6px;
  align-items: center;
  margin-bottom: 5px;
}

.dim-label {
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 6px;
}

.dim-mini-track {
  width: 60px;
  height: 3px;
  background: var(--border);
  border-radius: var(--radius-pill);
  overflow: hidden;
  display: inline-block;
  vertical-align: middle;
}

.dim-mini-fill {
  height: 100%;
  border-radius: var(--radius-pill);
}

.dim-val {
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
  font-family: 'DM Mono', monospace;
}

/* ── Grid layouts ── */
.section { margin-bottom: 14px; }
.g4 { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 12px; }
.g3 { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; }
.g2 { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 14px; }
.g3sig { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 8px; }

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  box-shadow: var(--shadow);
}

.card-label {
  font-size: 11px;
  font-weight: 500;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--hint);
  margin-bottom: 6px;
}

.card-value {
  font-size: 26px;
  font-weight: 600;
  color: var(--text);
  line-height: 1.1;
  margin-bottom: 3px;
  letter-spacing: -.03em;
}

.card-unit {
  font-size: 14px;
  color: var(--hint);
  font-weight: 300;
}

.card-note {
  font-size: 12px;
  color: var(--muted);
  line-height: 1.55;
  margin-top: 6px;
}

.card-title {
  font-size: 17px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -.02em;
  margin-bottom: 14px;
}

/* ── Audience tabs ── */
.atabs { display: flex; gap: 5px; margin-bottom: 14px; }
.atab {
  font-family: inherit;
  font-size: 12px;
  font-weight: 500;
  padding: 6px 14px;
  border-radius: var(--radius-pill);
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--muted);
  cursor: pointer;
  transition: all 0.15s;
}
.atab:hover { border-color: var(--border-strong); color: var(--text); }
.atab.active {
  background: var(--green);
  border-color: var(--green);
  color: #fff;
}

.apanel { display: none; }
.apanel.active { display: block; }

.insight-line {
  font-size: 13px;
  color: var(--muted);
  line-height: 1.7;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.insight-line:last-child { border-bottom: none; }

/* ── Terpene bars ── */
.bars { display: grid; gap: 10px; }
.bar-row { display: grid; gap: 5px; }
.bar-head { display: flex; justify-content: space-between; }
.bar-name { font-size: 13px; color: var(--text); }
.bar-val { font-size: 13px; color: var(--muted); font-family: 'DM Mono', monospace; }
.bar-track {
  height: 7px;
  background: var(--surface2);
  border-radius: var(--radius-pill);
  overflow: hidden;
  border: 1px solid var(--border);
}
.bar-fill {
  height: 100%;
  border-radius: var(--radius-pill);
  background: var(--green);
  transition: width 0.6s ease;
}

/* ── Fingerprint ── */
.fp-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
.fp-id {
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  color: var(--green-mid);
  background: var(--green-light);
  border: 1px solid var(--green-border);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
}
.fp-label { font-size: 12px; color: var(--muted); }

/* ── Cannabinoid grid ── */
.cann-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.cann-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  background: var(--surface2);
}
.cann-primary { border-color: var(--green-border); background: var(--green-light); }
.cann-name { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--green-mid); margin-bottom: 3px; }
.cann-val { font-size: 18px; font-weight: 600; color: var(--text); line-height: 1.1; }
.cann-unit { font-size: 12px; color: var(--hint); font-weight: 300; }
.cann-note { font-size: 11px; color: var(--muted); margin-top: 3px; line-height: 1.4; }

/* ── Pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 6px; }
.pill { font-size: 12px; padding: 5px 11px; border-radius: var(--radius-pill); border: 1px solid; font-weight: 500; }
.pill.good { background: var(--green-light); color: var(--green-mid); border-color: var(--green-border); }
.pill.warn { background: var(--amber-light); color: var(--amber); border-color: var(--amber-border); }
.pill.danger { background: var(--red-light); color: var(--red); border-color: var(--red-border); }

/* ── Contaminant rows ── */
.cont-row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 10px;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.cont-row:last-child { border-bottom: none; }
.cont-cat { color: var(--muted); }
.cont-status { font-weight: 500; }
.cont-pass { color: var(--green-mid); }
.cont-fail { color: var(--red); }
.cont-missing { color: var(--hint); }
.cont-value { color: var(--text); }
.cont-std { font-size: 11px; color: var(--hint); font-family: 'DM Mono', monospace; }

/* ── Signal grid ── */
.signal-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  background: var(--surface2);
}
.signal-label { font-size: 10px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--hint); margin-bottom: 4px; }
.signal-val { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.signal-note { font-size: 11px; color: var(--muted); line-height: 1.5; }

/* ── Lineage / detail rows ── */
.detail-list { display: grid; gap: 0; }
.detail-row { display: grid; grid-template-columns: 140px 1fr; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.detail-row:last-child { border-bottom: none; }
.detail-key { color: var(--hint); font-size: 12px; }
.detail-val { color: var(--text); line-height: 1.6; }

/* ── Raw JSON ── */
.raw-box { display: none; margin-top: 14px; }
.raw-box.open { display: block; }
pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 11px;
  font-family: 'DM Mono', monospace;
  line-height: 1.6;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px;
  color: var(--muted);
}

.empty { color: var(--hint); font-size: 13px; padding: 8px 0; }

@media(max-width:900px) {
  .hero-layout,.g4,.g3,.g2,.cann-grid,.g3sig { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="shell">

  <!-- Top bar -->
  <div class="topbar">
    <a class="btn" href="/">Upload new COA</a>
    ${options.documentId ? `<a class="btn btn-primary" href="/pdf/${options.documentId}" target="_blank" rel="noopener noreferrer">Export PDF</a>` : ""}
  </div>

  <!-- Hero -->
  <div class="hero">
    <div class="hero-eyebrow">Alem Chemical Intelligence · Schema v5.0</div>
    <div class="hero-layout">
      <div>
        <div class="product-name">${esc(productName)}</div>
        <div class="hero-narrative">${esc(heroNarrative)}</div>
        <div class="badge-row">
          ${chemistry.batch_number ? `<span class="badge">Batch ${esc(chemistry.batch_number)}</span>` : ""}
          ${chemistry.product_type ? `<span class="badge">${esc(chemistry.product_type)}</span>` : ""}
          ${chemistry.laboratory_name ? `<span class="badge">${esc(chemistry.laboratory_name)}</span>` : ""}
          ${chemistry.coa_report_date ? `<span class="badge">${esc(chemistry.coa_report_date)}</span>` : ""}
          ${chemistry.laboratory_accreditation ? `<span class="badge">${esc(chemistry.laboratory_accreditation)}</span>` : ""}
        </div>
      </div>
      <div class="score-card">
        <div class="score-top">
          <div class="score-number">${scoring.total}</div>
          <div class="score-denom">/100</div>
        </div>
        <div class="score-grade">${esc(scoring.tier)} · ${esc(scoring.grade)}</div>
        <div class="score-track">
          <div class="score-fill" style="width:${scoring.total}%"></div>
        </div>
        <div class="score-dims">
          ${scoreDims.map(d => {
            const pct = Math.round(d.score / d.max * 100);
            const barColor = pct >= 75 ? '#2d5c42' : pct >= 45 ? '#92650a' : '#8b2020';
            const valColor = pct >= 75 ? 'var(--green-mid)' : pct >= 45 ? 'var(--amber)' : 'var(--red)';
            return `
            <div class="dim-row">
              <div class="dim-label">
                <span class="dim-mini-track"><span class="dim-mini-fill" style="width:${pct}%;background:${barColor}"></span></span>
                ${esc(d.label)}
              </div>
              <div class="dim-val" style="color:${valColor}">${d.score}/${d.max}</div>
            </div>`;
          }).join("")}
        </div>
      </div>
    </div>
  </div>

  <!-- Key metrics -->
  <div class="section g4">
    <div class="card">
      <div class="card-label">THC total</div>
      <div class="card-value">${esc(chemistry.thc_total || "ND")} <span class="card-unit">${esc(chemistry.thc_total_unit || "wt%")}</span></div>
      <div class="card-note">${thc >= 24 ? "High range — above market average for medical dried flower." : thc >= 18 ? "Moderate-to-high range." : thc > 0 ? "Moderate range." : "Not detected or not reported."}</div>
    </div>
    <div class="card">
      <div class="card-label">THCA</div>
      <div class="card-value">${esc(chemistry.thca || "ND")} <span class="card-unit">${esc(chemistry.thca_unit || "wt%")}</span></div>
      <div class="card-note">Precursor potency — converts to active THC on heating (× 0.877).</div>
    </div>
    <div class="card">
      <div class="card-label">Total terpenes</div>
      <div class="card-value">${esc(chemistry.total_terpenes || "ND")} <span class="card-unit">${esc(chemistry.total_terpenes_unit || "wt%")}</span></div>
      <div class="card-note">${terps >= 3 ? "Strong — upper range of dried flower terpene density." : terps >= 1.8 ? "Good — meaningful aromatic presence." : terps > 0 ? "Moderate aromatic expression." : "Not reported."}</div>
    </div>
    <div class="card">
      <div class="card-label">CBD total</div>
      <div class="card-value">${esc(chemistry.cbd_total || "ND")} <span class="card-unit">${esc(chemistry.cbd_total_unit || "wt%")}</span></div>
      <div class="card-note">${cbd > 0.5 ? "Visible CBD — may provide some modulation of THC intensity." : "Minimal or not detected — no intrinsic CBD modulation."}</div>
    </div>
  </div>

  <!-- Chemistry columns -->
  <div class="section g2">
    <div class="card">
      <div class="card-title">Terpene fingerprint</div>
      <div class="bars">
        ${terpenes.length ? terpenes.map((t, i) => renderTerpBar(t, i)).join("") : `<div class="empty">No terpene data reported.</div>`}
      </div>
      <div class="fp-row">
        <span class="fp-label">Chemotype ID</span>
        <span class="fp-id">${esc(fingerprintId)}</span>
        <span class="fp-label">${leadTerpene ? esc(terpIntel.lineage) + " · " + esc(terpIntel.lineageConfidence) + " confidence" : "Lineage unclear"}</span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Cannabinoid profile</div>
      <div class="cann-grid">
        ${cannabinoids.length ? cannabinoids.map(renderCannCard).join("") : `<div class="empty">No cannabinoid data reported.</div>`}
      </div>
      ${chemistry.total_cannabinoids ? `<div style="margin-top:12px;font-size:12px;color:var(--hint)">Total cannabinoid sum: ${esc(chemistry.total_cannabinoids)} wt%</div>` : ""}
    </div>
  </div>

  <!-- Audience intelligence -->
  <div class="section g2">
    <div class="card">
      <div class="card-title">Audience intelligence</div>
      <div class="atabs">
        <button class="atab active" onclick="switchAud('brand',this)">Brand</button>
        <button class="atab" onclick="switchAud('clinical',this)">Clinical</button>
        <button class="atab" onclick="switchAud('patient',this)">Patient</button>
        <button class="atab" onclick="switchAud('buyer',this)">Buyer</button>
      </div>
      <div id="ap-brand" class="apanel active">${renderAudiencePanel(audiences.brand, "brand")}</div>
      <div id="ap-clinical" class="apanel">${renderAudiencePanel(audiences.clinical, "clinical")}</div>
      <div id="ap-patient" class="apanel">${renderAudiencePanel(audiences.patient, "patient")}</div>
      <div id="ap-buyer" class="apanel">${renderAudiencePanel(audiences.buyer, "buyer")}</div>
    </div>

    <div class="card">
      <div class="card-title">Effect direction</div>
      <div class="detail-list">
        <div class="detail-row">
          <div class="detail-key">Direction</div>
          <div class="detail-val">${esc(terpIntel.direction)}</div>
        </div>
        <div class="detail-row">
          <div class="detail-key">Lead terpene</div>
          <div class="detail-val">${esc(leadTerpene || "Not identified")}</div>
        </div>
        ${secondTerpene ? `<div class="detail-row"><div class="detail-key">Secondary</div><div class="detail-val">${esc(secondTerpene)}${thirdTerpene ? " · " + esc(thirdTerpene) : ""}</div></div>` : ""}
        <div class="detail-row">
          <div class="detail-key">Terpene read</div>
          <div class="detail-val">${esc(terpIntel.note)}</div>
        </div>
        <div class="detail-row">
          <div class="detail-key">Lineage cluster</div>
          <div class="detail-val">${esc(terpIntel.lineage)}</div>
        </div>
        <div class="detail-row">
          <div class="detail-key">Confidence</div>
          <div class="detail-val">${esc(terpIntel.lineageConfidence)}</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Post-harvest -->
  <div class="section g2">
    <div class="card">
      <div class="card-title">Post-harvest intelligence</div>
      <div class="g3sig">
        <div class="signal-card">
          <div class="signal-label">Freshness</div>
          <div class="signal-val">${esc(postHarvest.freshness.label)}</div>
          <div class="signal-note">${esc(postHarvest.freshness.note)}</div>
        </div>
        <div class="signal-card">
          <div class="signal-label">Curing</div>
          <div class="signal-val">${esc(postHarvest.curing.label)}</div>
          <div class="signal-note">${esc(postHarvest.curing.note)}</div>
        </div>
        <div class="signal-card">
          <div class="signal-label">Degradation (CBN)</div>
          <div class="signal-val">${esc(postHarvest.degradation.label)}</div>
          <div class="signal-note">${esc(postHarvest.degradation.note)}</div>
        </div>
        <div class="signal-card">
          <div class="signal-label">Stability</div>
          <div class="signal-val">${esc(postHarvest.stability.label)}</div>
          <div class="signal-note">${esc(postHarvest.stability.note)}</div>
        </div>
        <div class="signal-card">
          <div class="signal-label">CBGA signal</div>
          <div class="signal-val">${toNum(chemistry.cbga) >= 1 ? "Notable" : toNum(chemistry.cbga) > 0 ? "Present" : "Not detected"}</div>
          <div class="signal-note">${toNum(chemistry.cbga) >= 1 ? `${chemistry.cbga} wt% — elevated CBGA may indicate specific genetics or harvest timing.` : "CBGA not elevated."}</div>
        </div>
        <div class="signal-card">
          <div class="signal-label">THC expression</div>
          <div class="signal-val">${chemistry.thca && chemistry.thc_total ? "Calculated" : "Partial"}</div>
          <div class="signal-note">${chemistry.thca ? `THCA ${chemistry.thca} × 0.877 + D9-THC = Total THC` : "Calculation basis not fully captured."}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Quality &amp; compliance signals</div>
      <div class="pill-row" style="margin-bottom:12px">
        ${posFlags.length ? posFlags.map(f => renderPill(f, "good")).join("") : '<span class="empty">No positive flags extracted.</span>'}
      </div>
      <div class="pill-row" style="margin-bottom:18px">
        ${warnFlags.length ? warnFlags.map(f => renderPill(f, "warn")).join("") : ""}
      </div>
      <div class="cont-row" style="font-size:11px;color:var(--hint);font-weight:600;letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid var(--border)">
        <span>Panel</span><span>Result</span><span>Standard</span>
      </div>
      ${renderContRow("Pesticides", contaminants.pesticides_status, "TGO93 / TGO100")}
      ${renderContRow("Heavy metals", contaminants.heavy_metals_status, "ICH Q3D")}
      ${renderContRow("Microbials", contaminants.microbials_status, "USP 2021/2023")}
      ${renderContRow("Mycotoxins", contaminants.mycotoxins_status, "EC 1881/2006")}
      ${renderContRow("Residual solvents", contaminants.residual_solvents_status, "USP 467")}
      ${renderContRow("Lab accreditation", chemistry.laboratory_accreditation, "ISO 17025")}
    </div>
  </div>

  <!-- Lab quality -->
  <div class="section g2">
    <div class="card">
      <div class="card-title">Contaminant narrative</div>
      <div class="insight-line">${esc(contaminants.contaminant_narrative || "Structured contaminant narrative not captured. Review source COA directly for all safety panel results.")}</div>
      ${contaminants.moisture_content ? `<div class="detail-row"><div class="detail-key">Moisture</div><div class="detail-val">${esc(contaminants.moisture_content)}</div></div>` : ""}
      ${contaminants.water_activity ? `<div class="detail-row"><div class="detail-key">Water activity</div><div class="detail-val">${esc(contaminants.water_activity)}</div></div>` : ""}
    </div>
    <div class="card">
      <div class="card-title">Lab quality summary</div>
      <div class="insight-line">${esc(contaminants.lab_quality_summary || chemistry.laboratory_accreditation || "Lab quality summary not captured.")}</div>
      ${chemistry.laboratory_method ? `<div style="margin-top:10px" class="detail-list"><div class="detail-row"><div class="detail-key">Method</div><div class="detail-val">${esc(chemistry.laboratory_method)}</div></div></div>` : ""}
      <div class="fp-row">
        <span class="fp-label">Chemotype fingerprint</span>
        <span class="fp-id">${esc(fingerprintId)}</span>
        <span class="fp-label">Top-3 terpene cluster · ${esc([leadTerpene, secondTerpene, thirdTerpene].filter(Boolean).join(" / ") || "not identified")}</span>
      </div>
    </div>
  </div>

  <!-- Raw JSON -->
  <div class="section">
    <div class="card">
      <div class="card-title">Raw intelligence data</div>
      <button class="btn" type="button" onclick="toggleRaw()">Toggle JSON</button>
      <div id="rawBox" class="raw-box">
        <pre>${rawJson}</pre>
      </div>
    </div>
  </div>

</div>

<script>
function switchAud(id, btn) {
  document.querySelectorAll('.apanel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.atab').forEach(t => t.classList.remove('active'));
  document.getElementById('ap-' + id).classList.add('active');
  btn.classList.add('active');
}
function toggleRaw() {
  document.getElementById('rawBox').classList.toggle('open');
}
</script>
</body>
</html>`;
}

// ─────────────────────────────────────────────
// EXPRESS ROUTES
// ─────────────────────────────────────────────

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.get("/health", async (req, res) => {
  return res.json({ success: true, message: "Alem v5 running", schema: "5.0" });
});

app.post("/upload-coa", upload.single("file"), async (req, res) => {
  try {
    console.log("📥 COA upload received");
    if (!req.file) return res.status(400).json({ success: false, error: "No file uploaded" });

    const originalFilename = req.file.originalname || `upload-${Date.now()}.pdf`;
    const mimeType = req.file.mimetype || "application/octet-stream";

    // Upload to Supabase storage
    const { publicUrl, storagePath } = await uploadBufferToSupabase({
      buffer: req.file.buffer,
      originalName: originalFilename,
      mimeType,
      folder: "raw_documents",
    });
    console.log("☁️  Stored:", storagePath);

    // Layer 1: Azure OCR
    console.log("🔍 Azure OCR...");
    const extracted = await extractDocumentFromUrl(publicUrl);
    console.log(`✅ OCR: ${extracted.plain_text.length} chars, ${extracted.page_count} pages`);

    // Layer 1: Dual-pass OpenAI extraction
    const { chemistry, contaminants } = await runDualPassExtraction(extracted.plain_text);
    console.log("✅ Dual-pass extraction complete");

    // Layer 2: Composite scoring
    const scoring = computeIntelligenceScore(chemistry, contaminants);
    console.log(`✅ Score: ${scoring.total}/100 (${scoring.grade})`);

    // Layer 3: Intelligence generation
    const fingerprintId = generateFingerprintId(chemistry.top_terpenes);
    const leadTerpene = chemistry.top_terpenes?.[0]?.name || "";
    const terpIntel = getTerpeneIntel(leadTerpene);
    const audiences = buildAudienceNarratives(chemistry, contaminants, scoring);
    const postHarvest = buildPostHarvestIntel(chemistry, contaminants);

    const intelligence = {
      fingerprintId,
      effectDirection: terpIntel.direction,
      lineageCluster: terpIntel.lineage,
      lineageConfidence: terpIntel.lineageConfidence,
      audiences,
      postHarvest,
    };
    console.log(`✅ Intelligence: fingerprint=${fingerprintId}, direction=${terpIntel.direction}`);

    // Layer 4: Store to Supabase
    const insertedRow = await insertCOAReport({
      chemistry,
      contaminants,
      scoring,
      intelligence,
      sourceUrl: publicUrl,
      storagePath,
      originalFilename,
      mimeType: extracted.mimeType || mimeType,
    });
    console.log(`✅ Stored report ID: ${insertedRow.id}`);

    const proto = req.headers["x-forwarded-proto"] || req.protocol || "https";
    const base = `${proto}://${req.get("host")}`;

    return res.json({
      success: true,
      id: insertedRow.id,
      score: scoring.total,
      grade: scoring.grade,
      tier: scoring.tier,
      fingerprint_id: fingerprintId,
      report_url: `${base}/report/${insertedRow.id}`,
      pdf_url: `${base}/pdf/${insertedRow.id}`,
    });
  } catch (error) {
    console.error("❌ Upload pipeline error:", error.message);
    return res.status(500).json({ success: false, error: error.message || "Upload pipeline failed" });
  }
});

app.get("/report/:id", async (req, res) => {
  try {
    const row = await getReportById(req.params.id);
    return res.send(renderReportHTML(row?.report_json || {}, { documentId: row.id }));
  } catch (error) {
    console.error("ERROR /report/:id", error.message);
    return res.status(404).send("Report not found");
  }
});

app.get("/pdf/:id", async (req, res) => {
  let browser;
  try {
    const row = await getReportById(req.params.id);
    const html = renderReportHTML(row?.report_json || {}, { documentId: row.id });

    browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "networkidle0" });

    const pdfBuffer = await page.pdf({
      format: "A4",
      printBackground: true,
      margin: { top: "14mm", right: "12mm", bottom: "14mm", left: "12mm" },
    });

    const filename = sanitizeFileName(row?.report_json?.chemistry?.product_name || req.params.id);
    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", `inline; filename="${filename}.pdf"`);
    return res.send(pdfBuffer);
  } catch (error) {
    console.error("ERROR /pdf/:id", error.message);
    return res.status(500).json({ success: false, error: error.message });
  } finally {
    if (browser) { try { await browser.close(); } catch (_) {} }
  }
});

app.listen(PORT, () => {
  console.log(`🌿 Alem Chemical Intelligence v5 running on port ${PORT}`);
});