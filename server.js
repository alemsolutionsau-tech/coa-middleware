require("dotenv").config();

const renderReportHTMLV7  = require('./renderReportHTML');
const renderReportPDFDoc  = require('./renderReportPDF');
const { fetchBenchmark }        = require('./bqBenchmark');
const { fetchStrainIntelligence }  = require('./strainIntelligence');
const { buildScientificEvidence }  = require('./services/scientificLayer');

const path = require("path");
const express = require("express");
const cors = require("cors");
const rateLimit = require("express-rate-limit");
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
app.set("trust proxy", 1); // Render.com sits behind a proxy — needed for rate-limit IP detection

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

const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(",").map(o => o.trim()).filter(Boolean)
  : [];

app.use(cors({
  origin: (origin, cb) => {
    // Allow same-origin / server-side requests (no Origin header)
    if (!origin) return cb(null, true);
    if (ALLOWED_ORIGINS.length === 0 || ALLOWED_ORIGINS.includes(origin)) return cb(null, true);
    cb(new Error(`CORS: origin ${origin} not allowed`));
  },
}));

const uploadLimiter = rateLimit({
  windowMs: 60 * 1000,       // 1 minute window
  max: Number(process.env.UPLOAD_RATE_LIMIT || 10),
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, error: "Too many requests — please wait before uploading again." },
});

function requireApiKey(req, res, next) {
  const apiKey = process.env.UPLOAD_API_KEY;
  if (!apiKey) return next(); // no key configured → open (dev mode)
  const provided = req.headers["x-api-key"] || req.query.api_key;
  if (provided !== apiKey) {
    return res.status(401).json({ success: false, error: "Invalid or missing API key." });
  }
  next();
}

app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const MAX_OCR_CHARS = Number(process.env.MAX_OCR_CHARS_FOR_OPENAI || 80000); // gpt-4o handles 128k tokens; 80k chars ≈ 20k tokens — fits any COA
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 120000); // 2 min timeout for large docs
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
// LAYER 1 — EXTRACTION PROMPTS (dual-pass, maximally thorough)
// ─────────────────────────────────────────────

const CHEMISTRY_EXTRACTION_PROMPT = `
You are the Alem Solutions COA chemistry extraction engine — a highly precise analytical instrument.
Read ALL OCR text from a cannabis Certificate of Analysis and return EXACTLY ONE valid JSON object.

CRITICAL RULES:
- Return valid JSON only. No markdown. No comments. No prose.
- If a field is missing return "" for strings, [] for arrays, "ND" for undetected analytes.
- NEVER guess or hallucinate values. Leave empty if genuinely uncertain.
- NEVER truncate arrays. Extract EVERY single compound listed — no limits.
- For units: always use "wt%" not "%". If the COA says wt%, use "wt%".

UNIT CONVERSION — CRITICAL:
- If values are reported in mg/g, convert to wt% by dividing by 10. Example: 249.19 mg/g = 24.919 wt%.
- If values are reported as % (without "wt"), treat as wt%.
- If values are in both units, always use wt% (or convert mg/g ÷ 10).
- Do NOT convert ppm, ppb values — leave those as-is.

PRODUCT TYPE DETECTION:
- product_type: detect from the COA. Common values: "Dried Flower", "Concentrate", "Distillate", "Live Resin", "Rosin", "Hash", "Kief", "Tincture", "Edible", "Capsule", "Topical", "Oil", "Vape Cartridge", "Shatter", "Wax", "Budder", "Extract"
- If concentrate/extract, THC may be 60-95 wt% — this is normal, do NOT cap or modify.

CANNABINOIDS — EXTRACT ALL:
- top_cannabinoids: include EVERY cannabinoid row in the table, whether detected or ND. No limit.
- Capture BOTH anhydrous and as-received values if present. Prefer anhydrous (higher accuracy) for total_thc and total_cbd.
- Name normalisations: THCA-A → "THCA", Δ9-THC → "D9-THC", delta-9-THC → "D9-THC", D8-THC → "D8-THC", Δ8-THC → "D8-THC", CBC → "CBC", CBCA → "CBCA"
- If only "Total THC" is reported (no breakdown), put the value in thc_total and leave thca/d9thc empty.
- total_cannabinoids: look for "Total of all quantified cannabinoids", "Total Cannabinoids", or sum line. Use anhydrous if two exist.

TERPENES — EXTRACT ALL WITHOUT EXCEPTION:
- top_terpenes: include EVERY single terpene row that has a numeric value (wt%) above 0.
- Include BLQ entries as "BLQ" in the value field.
- Do NOT cap or truncate. If there are 25 terpenes detected, return all 25.
- Sort by value descending (highest first), BLQ entries at the bottom.
- Name normalisations: β-Myrcene → "Beta-Myrcene", β-Caryophyllene → "Trans-Caryophyllene", β-Pinene → "Beta-Pinene", D-Limonene → "(R)-(+)-Limonene", d-Limonene → "(R)-(+)-Limonene", α-Pinene → "Alpha-Pinene", α-Humulene → "Alpha-Humulene", α-Bisabolol → "Alpha-Bisabolol", α-Terpineol → "Alpha-Terpineol"
- total_terpenes: look for "Total of all quantified terpenes", "Total Terpenes", or sum line. Compute from sum ONLY if no total line exists.
- If terpenes are reported in mg/g, convert to wt% (÷ 10).

MINOR CANNABINOIDS — extract each individually:
- thca: THCA value (anhydrous preferred)
- cbda: CBDA value
- cbn: CBN free-form value (not total CBN formula result)
- cbna: CBNA value
- cbg: CBG value (free form, not acid)
- cbga: CBGA value
- cbca: CBCA value
- thcva: THCVA or THCV-A value
- d8thc: D8-THC if present
- cbc: CBC if present

FLAVONOIDS — EXTRACT ALL (if panel present):
- flavonoids: array of { name, value, unit } for every flavonoid with a numeric value.
- total_flavonoids: "Total of all quantified flavonoids" value if present.

PHYSICAL / QUALITY DATA:
- moisture_content: Loss on Drying % value (numeric only, e.g. "7.2854")
- water_activity: Water Activity aw value (numeric only, e.g. "0.5531")
- foreign_matter: result of Foreign Matter / Visual Inspection / Olfactory test

IDENTIFICATION TESTS:
- hptlc_result: result of HPTLC identification (e.g. "THC-dominant. Conforms")
- macroscopic_result: "Complies" or actual result
- microscopic_result: "Complies" or actual result

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
  "thc_total_anhydrous": "",
  "cbd_total": "",
  "cbd_total_unit": "wt%",
  "thca": "",
  "thca_unit": "wt%",
  "d9thc": "",
  "cbda": "",
  "cbn": "",
  "cbna": "",
  "cbg": "",
  "cbga": "",
  "cbca": "",
  "cbc": "",
  "thcva": "",
  "d8thc": "",
  "total_terpenes": "",
  "total_terpenes_unit": "wt%",
  "total_cannabinoids": "",
  "total_flavonoids": "",
  "moisture_content": "",
  "water_activity": "",
  "foreign_matter": "",
  "hptlc_result": "",
  "macroscopic_result": "",
  "microscopic_result": "",
  "hero_narrative": "",
  "top_cannabinoids": [
    { "name": "", "value": "", "unit": "wt%", "notes": "" }
  ],
  "top_terpenes": [
    { "name": "", "value": "", "unit": "wt%" }
  ],
  "flavonoids": [
    { "name": "", "value": "", "unit": "wt%" }
  ]
}

For hero_narrative: write 2 precise, clinical sentences about this specific product's chemistry. State the chemotype identity (dominant terpene + potency tier), then describe what makes this profile distinctive. Use factual chemical language. Do NOT use phrases like "making it suitable for", "seeking strong effects", "unique aromatic character", or any consumer-facing promotional language. Focus on: what the chemistry IS, not what it does to users.
`;

const CONTAMINANT_EXTRACTION_PROMPT = `
You are the Alem Solutions COA safety extraction engine — maximally thorough.
Read ALL OCR text from a cannabis Certificate of Analysis and extract EVERY safety, compliance, and contaminant data point.

CRITICAL RULES:
- Return valid JSON only. No markdown. No prose.
- NEVER invent or assume results. Only record what is explicitly stated.
- Accept ANY result format: "Pass", "PASS", "Fail", "FAIL", "ND", "N/D", "None Detected", "Not Detected", "BLQ", "Below LOQ", "<LOQ", "Absent", "Absent in 1g", "Absent in 10g", "<10 CFU/g", numeric ppm/ppb values, or "Not Tested".
- For overall status fields: if all individual results in a category are ND/BLQ/Absent/Pass → set status to "ND" or "Pass". If any fail → "Fail".
- positive_flags: things indicating quality or compliance (no limit)
- warning_flags: things missing, borderline, or concerning (no limit)

PESTICIDES — CRITICAL (may use any of these standards):
- EP 2.8.13 (European), TGO93/TGO100 (Australian), Health Canada (Canadian), California/Oregon/state-specific panels, or custom lab panels
- pesticides_status: derive from results — "ND" if all ND, "Pass" if all pass, "Fail" if any detected above RL
- pesticides_method: method reference found in document
- pesticides_compound_count: count of compounds in the panel (look for the total number tested)
- pesticides_detail: brief summary

HEAVY METALS — CRITICAL (may use any of these):
- ICH Q3D (US), Ph. Eur. 2.4.27 ICP-MS (EU), Health Canada, TGO101 (AU)
- For each metal record EXACT result text: "BLQ", "ND", "<0.1 ppm", "0.023 ppm", etc.
- heavy_metals_status: "Pass"/"ND"/"BLQ" if all within limits, "Fail" if any exceeded
- arsenic_result, cadmium_result, lead_result, mercury_result

MICROBIALS — CRITICAL (may use any of these):
- USP <2021>/<2023>, EP 2.6.12, ISO 21149, Health Canada, or state-specific
- Accept results like: "Absent in 10g", "Not Detected", "<10 CFU/g", "Pass", numeric CFU/g values
- microbials_status: overall — "Pass"/"ND"/"Absent" if all clear
- Individual: yeast_mold, total_aerobic, bile_tolerant_gram_negative, salmonella, s_aureus, p_aeruginosa, e_coli

MYCOTOXINS:
- Accept EU, US, or Canadian limits
- mycotoxins_status: "ND" if all not detected, "Pass" if all pass
- aflatoxin_b1, aflatoxin_b2, aflatoxin_g1, aflatoxin_g2: values in ppb
- sum_aflatoxins, ochratoxin_a

RESIDUAL SOLVENTS:
- May be present for extracts/concentrates — USP <467>, EP 2.4.24, or state panels
- residual_solvents_status: "Pass", "ND", "Not tested"

PHYSICAL / STABILITY:
- moisture_content: ANY of — "Loss on Drying", "Moisture Content", "Water Content" — numeric value only (e.g. "7.2854")
- water_activity: "Water Activity", "Aw" value — numeric only (e.g. "0.5531")
- foreign_matter_status: "Foreign Matter", "Visual Inspection", "Olfactory" result

LAB ACCREDITATION — look for any of:
- iso_17025: ISO 17025:2017, ISO/IEC 17025, A2LA, PJLA, Perry Johnson Labs
- scc_accredited: SCC, Standards Council of Canada
- Other accreditations: A2LA, NELAP, TNI, DAkkS, UKAS, NATA — put in lab_accreditation_body

Return this exact shape:
{
  "pesticides_status": "",
  "pesticides_detail": "",
  "pesticides_method": "",
  "pesticides_compound_count": "",
  "heavy_metals_status": "",
  "heavy_metals_detail": "",
  "heavy_metals_method": "",
  "arsenic_result": "",
  "cadmium_result": "",
  "lead_result": "",
  "mercury_result": "",
  "microbials_status": "",
  "microbials_detail": "",
  "microbials_method": "",
  "yeast_mold": "",
  "total_aerobic": "",
  "bile_tolerant_gram_negative": "",
  "salmonella": "",
  "s_aureus": "",
  "p_aeruginosa": "",
  "e_coli": "",
  "mycotoxins_status": "",
  "mycotoxins_detail": "",
  "mycotoxins_method": "",
  "aflatoxin_b1": "",
  "aflatoxin_b2": "",
  "aflatoxin_g1": "",
  "aflatoxin_g2": "",
  "sum_aflatoxins": "",
  "ochratoxin_a": "",
  "residual_solvents_status": "",
  "moisture_content": "",
  "water_activity": "",
  "foreign_matter_status": "",
  "iso_17025": false,
  "scc_accredited": false,
  "lab_accreditation_body": "",
  "contaminant_narrative": "",
  "lab_quality_summary": "",
  "positive_flags": [""],
  "warning_flags": [""]
}

For contaminant_narrative: 2-3 sentences summarising the complete safety profile. State clearly what is tested and what is missing.
For lab_quality_summary: describe accreditation, method standards, and notable analytical details.
`;

// ─────────────────────────────────────────────
// LAYER 2 — COMPOSITE SCORING ENGINE
// ─────────────────────────────────────────────

function computeIntelligenceScore(extracted, contaminants) {
  let score = 0;
  const breakdown = {};

  // Detect product type to apply correct scoring thresholds
  const productType = String(extracted.product_type || "").toLowerCase();
  const isConcentrate = /concentrat|distillat|resin|rosin|hash|kief|shatter|wax|budder|extract|oil|tincture|vape|cartridge/i.test(productType);
  const isCBD = toNum(extracted.cbd_total) > toNum(extracted.thc_total) * 2;

  const thc = toNum(extracted.thc_total);
  const cbd = toNum(extracted.cbd_total);
  let potencyScore = 0;

  if (isConcentrate) {
    // Concentrate potency thresholds (THC typically 50-95%)
    if (thc >= 80) potencyScore = 25;
    else if (thc >= 70) potencyScore = 22;
    else if (thc >= 60) potencyScore = 18;
    else if (thc >= 50) potencyScore = 13;
    else if (thc > 0) potencyScore = 8;
  } else if (isCBD) {
    // CBD-dominant product
    if (cbd >= 15) potencyScore = 25;
    else if (cbd >= 10) potencyScore = 20;
    else if (cbd >= 5) potencyScore = 15;
    else if (cbd > 0) potencyScore = 8;
  } else {
    // Flower (default)
    if (thc >= 28) potencyScore = 25;
    else if (thc >= 24) potencyScore = 22;
    else if (thc >= 20) potencyScore = 18;
    else if (thc >= 15) potencyScore = 13;
    else if (thc > 0) potencyScore = 8;
    if (cbd >= 15 && thc < 5) potencyScore = Math.max(potencyScore, 20);
  }
  breakdown.potency = { score: potencyScore, max: 25, productType: isConcentrate ? "concentrate" : isCBD ? "CBD" : "flower" };
  score += potencyScore;

  const terps = toNum(extracted.total_terpenes);
  const terpCount = (extracted.top_terpenes || []).filter(t => toNum(t.value) > 0).length;
  let terpScore = 0;
  if (terps >= 3.5) terpScore = 22;
  else if (terps >= 2.5) terpScore = 20;
  else if (terps >= 1.8) terpScore = 16;
  else if (terps >= 1.0) terpScore = 11;
  else if (terps > 0) terpScore = 6;
  if (terpCount >= 10) terpScore = Math.min(25, terpScore + 5);
  else if (terpCount >= 6) terpScore = Math.min(25, terpScore + 3);
  else if (terpCount >= 4) terpScore = Math.min(25, terpScore + 1);
  breakdown.terpenes = { score: terpScore, max: 25 };
  score += terpScore;

  const cbga = toNum(extracted.cbga);
  const cbn = toNum(extracted.cbn);
  const cbg = toNum(extracted.cbg);
  const cbda = toNum(extracted.cbda);
  const cbca = toNum(extracted.cbca);
  const thcva = toNum(extracted.thcva);
  const cbna = toNum(extracted.cbna);
  const minorCount = [cbga, cbn, cbg, cbda, cbca, thcva, cbna].filter(v => v > 0).length;
  const totalMinors = toNum(extracted.total_cannabinoids) - toNum(extracted.thc_total) - toNum(extracted.cbd_total);
  let minorScore = 0;
  if (totalMinors >= 5) minorScore = 20;
  else if (totalMinors >= 3) minorScore = 16;
  else if (totalMinors >= 1.5) minorScore = 12;
  else if (minorCount >= 3) minorScore = 10;
  else if (minorCount >= 1) minorScore = 6;
  if (cbn >= 1) minorScore = Math.max(0, minorScore - 4);
  breakdown.minors = { score: minorScore, max: 20 };
  score += minorScore;

  let safetyScore = 0;
  const pest = String(contaminants.pesticides_status || "").toLowerCase();
  const metals = String(contaminants.heavy_metals_status || "").toLowerCase();
  const micro = String(contaminants.microbials_status || "").toLowerCase();
  const myco = String(contaminants.mycotoxins_status || "").toLowerCase();
  const iso = contaminants.iso_17025 === true || /17025|iso/i.test(contaminants.lab_quality_summary || "");
  const scc = contaminants.scc_accredited === true || /scc/i.test(contaminants.positive_flags?.join(" ") || "");
  if (pest.includes("pass") || pest.includes("nd")) safetyScore += 6;
  else if (pest.includes("not tested") || !pest) safetyScore += 1;
  if (metals.includes("pass") || metals.includes("nd") || metals.includes("blq")) safetyScore += 5;
  else if (metals.includes("not tested") || !metals) safetyScore += 1;
  if (micro.includes("pass") || micro.includes("nd") || micro.includes("absent")) safetyScore += 5;
  else if (micro.includes("not tested") || !micro) safetyScore += 1;
  if (myco.includes("pass") || myco.includes("nd")) safetyScore += 2;
  if (iso) safetyScore = Math.min(20, safetyScore + 2);
  if (scc) safetyScore = Math.min(20, safetyScore + 2);
  // Bonus for complete safety panel
  const panelCount = [pest, metals, micro, myco].filter(s => s && !s.includes("not tested")).length;
  if (panelCount === 4) safetyScore = Math.min(20, safetyScore + 2);
  breakdown.safety = { score: safetyScore, max: 20 };
  score += safetyScore;

  let dataScore = 0;
  if (extracted.thc_total) dataScore += 2;
  if (extracted.total_terpenes) dataScore += 2;
  if ((extracted.top_terpenes || []).filter(t => toNum(t.value) > 0).length >= 5) dataScore += 2;
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
  terpinolene: { direction: "Uplifting / mentally active", note: "Terpinolene-dominant profiles often feel brighter, more mentally active, and less body-heavy. Associated with Haze and Jack-type chemotype families.", lineage: "Haze / Jack / Durban-type families", lineageConfidence: "Moderate" },
  myrcene: { direction: "Calming / body-centred", note: "Myrcene-dominant profiles tend toward more calming, body-heavy experiences. Common in Kush and OG-type chemotype families.", lineage: "Kush / OG / indica-leaning families", lineageConfidence: "Moderate" },
  "beta-myrcene": { direction: "Calming / body-centred", note: "Myrcene-dominant profiles tend toward more calming, body-heavy experiences. Common in Kush and OG-type chemotype families.", lineage: "Kush / OG / indica-leaning families", lineageConfidence: "Moderate" },
  limonene: { direction: "Bright / mood-forward", note: "Limonene-forward profiles are typically positioned as brighter, more mood-forward, and citrus-dominant in aroma.", lineage: "Citrus hybrid / Gelato / Runtz-type families", lineageConfidence: "Moderate" },
  caryophyllene: { direction: "Balanced / structured", note: "Caryophyllene-led profiles often read as warm, structured, and balanced — with spice-forward aromatic character.", lineage: "Balanced hybrid / spice-forward families", lineageConfidence: "Low-moderate" },
  "trans-caryophyllene": { direction: "Balanced / structured", note: "Caryophyllene-led profiles often read as warm, structured, and balanced — with spice-forward aromatic character.", lineage: "Balanced hybrid / spice-forward families", lineageConfidence: "Low-moderate" },
  pinene: { direction: "Clear / alert", note: "Pinene-prominent profiles are often interpreted as clearer and more alertness-oriented, with a fresh, pine-forward aroma.", lineage: "Pine-forward / clear sativa-leaning families", lineageConfidence: "Low-moderate" },
  "alpha-pinene": { direction: "Clear / alert", note: "Alpha-pinene contributes a fresh, sharp aromatic quality and is associated with clearer, more alert experiential profiles.", lineage: "Pine-forward / clear sativa-leaning families", lineageConfidence: "Low-moderate" },
  ocimene: { direction: "Bright / floral / complex", note: "Ocimene adds floral, herbal, and tropical aromatic complexity. As a secondary terpene it supports a distinctive multi-layered aromatic fingerprint.", lineage: "Haze / tropical-adjacent families", lineageConfidence: "Low" },
  linalool: { direction: "Calming / floral", note: "Linalool is associated with floral, lavender-like aromas and calmer, more relaxing experiential profiles.", lineage: "Floral / indica-adjacent families", lineageConfidence: "Low-moderate" },
  bisabolol: { direction: "Gentle / smooth", note: "Bisabolol contributes a smooth, gentle aromatic character and is associated with softer, more relaxed profiles.", lineage: "Smooth hybrid families", lineageConfidence: "Low" },
  humulene: { direction: "Earthy / structured", note: "Humulene adds earthy, woody aromatic depth and contributes to a grounded, structured profile character.", lineage: "OG / earthy hybrid families", lineageConfidence: "Low" },
  farnesene: { direction: "Floral / fruity / complex", note: "Farnesene contributes apple-adjacent, floral, and fruity aromatic notes. Relatively rare as a dominant terpene — adds differentiation value.", lineage: "Complex hybrid families", lineageConfidence: "Low" },
};

const TERPENE_EDUCATION = {
  "Terpinolene": { aroma: "Fresh, piney, floral, slightly herbal — bright and clean.", therapeutic: "Antioxidant properties noted in preclinical studies. Mild anxiolytic signals. Found in Jack Herer, Durban Poison, Ghost Train Haze. Fewer than 15% of dried flower products are Terpinolene-dominant — a genuine market differentiator." },
  "β-Myrcene": { aroma: "Earthy, musky, fruity — ripe mango or hops. The most common cannabis terpene.", therapeutic: "May contribute sedating, relaxing effects at higher concentrations. Anti-inflammatory and analgesic properties in preclinical data." },
  "Beta-Myrcene": { aroma: "Earthy, musky, fruity — ripe mango or hops.", therapeutic: "Anti-inflammatory and analgesic properties in preclinical data. At low concentrations (0.04 wt%) its aromatic contribution is subtle but present as part of the overall entourage." },
  "Ocimene": { aroma: "Sweet, herbal, woody — fresh basil and tarragon. Marker of Haze-type genetics.", therapeutic: "Antifungal and antiviral properties noted in vitro. Adds to the fresh, bright aromatic character of the profile." },
  "Alpha-Pinene": { aroma: "Crisp pine, fresh forest air — also found in rosemary and eucalyptus.", therapeutic: "Bronchodilator in preclinical models. May counteract short-term memory effects sometimes associated with THC. Studied for anti-inflammatory and antimicrobial properties." },
  "Beta-Pinene": { aroma: "Piney, green, fresh — slightly more herbal than Alpha-Pinene.", therapeutic: "Mild antiseptic properties. Works synergistically with Alpha-Pinene. Adds to the overall freshness of the aromatic profile." },
  "Trans-Caryophyllene": { aroma: "Spicy, peppery, woody — the same compound that gives black pepper its heat.", therapeutic: "The only terpene confirmed to act as a CB2 cannabinoid receptor agonist. Studied for anti-inflammatory, analgesic, and anxiolytic effects. A unique pharmacological pathway distinct from classical terpene activity." },
  "Farnesene": { aroma: "Green apple, woody, floral. Found in apple skin, green tea. Marker of well-preserved plant material.", therapeutic: "Anti-inflammatory properties in preclinical research. Its presence suggests minimal terpene degradation post-harvest — a freshness and quality signal." },
  "Alpha-Humulene": { aroma: "Woody, earthy, spicy — like hops. An isomer of Beta-Caryophyllene.", therapeutic: "Anti-inflammatory, antibacterial, and appetite-suppressing properties in preclinical research. Often works synergistically with Caryophyllene." },
  "Linalool": { aroma: "Floral, lavender-like with a subtle citrus note.", therapeutic: "Associated with calming, relaxing experiential profiles. Anti-anxiety and analgesic properties studied in preclinical models." },
  "Alpha-Bisabolol": { aroma: "Delicate floral, honey-like, slightly sweet. Widely used in cosmetics for skin-soothing.", therapeutic: "Well-documented anti-inflammatory and wound-healing properties. May enhance absorption of other compounds." },
  "Guaiol": { aroma: "Piney, floral, rose-like with a woody undertone. A sesquiterpenoid alcohol — less volatile than most terpenes.", therapeutic: "Antimicrobial and anti-inflammatory properties in preclinical models. Its presence at detectable levels suggests careful post-harvest preservation." },
  "Alpha-Terpineol": { aroma: "Lilac-like, floral, slightly citrusy.", therapeutic: "Sedative and antimicrobial properties noted in preclinical studies. Adds floral refinement to the overall aromatic signature." },
  "Caryophyllene oxide": { aroma: "Woody, spicy — the oxidised form of Caryophyllene. The compound canine units detect in cannabis.", therapeutic: "Antifungal properties. Marker of terpene oxidation; its presence alongside Trans-Caryophyllene is normal." },
  "Camphene": { aroma: "Damp woodlands, fir needles, herbal.", therapeutic: "Antioxidant properties and potential cardiovascular benefits in preclinical research. Minor aromatic contributor." },
  "Fenchone": { aroma: "Minty, camphor-like, slightly herbal.", therapeutic: "Occurs in fennel and some cannabis cultivars. Minor aromatic contributor with antimicrobial properties." },
  "Borneol": { aroma: "Minty camphor, herbal, slightly woody.", therapeutic: "Traditional medicine uses for pain and inflammation. Minor contributor at trace quantities." },
  "Limonene": { aroma: "Bright citrus — lemon, orange.", therapeutic: "Mood-elevating properties noted in early clinical studies. Potent antifungal and antibacterial." },
};

const CANNABINOID_EDUCATION = {
  "THCA": { title: "THCA — Tetrahydrocannabinolic Acid", body: "The raw, non-psychoactive precursor to THC. Converts to psychoactive THC through decarboxylation (heat). Multiply by 0.877 to estimate active THC yield.", therapeutic: "Emerging research suggests THCA itself may have anti-inflammatory and neuroprotective properties without psychoactivity." },
  "D9-THC": { title: "D9-THC — Delta-9-Tetrahydrocannabinol", body: "The primary psychoactive cannabinoid. Already-converted active THC present before any heating.", therapeutic: "Binds to CB1 receptors producing analgesic, antiemetic, and appetite-stimulating effects." },
  "CBNA": { title: "CBNA — Cannabinolic Acid", body: "The acid precursor to CBN. Formed through oxidative degradation of THCA. Its presence indicates some degree of THC degradation or age.", therapeutic: "Converts to CBN on heating. A minor degradation marker. Present here at low levels, indicating generally fresh material." },
  "CBGA": { title: "CBGA — Cannabigerolic Acid (The Mother Cannabinoid)", body: "The biosynthetic precursor to ALL other cannabinoids. Elevated CBGA suggests early harvest or genetics with slower enzymatic conversion.", therapeutic: "CBGA is under active research for anti-inflammatory, antibacterial, and metabolic effects." },
  "CBCA": { title: "CBCA — Cannabichromenic Acid", body: "Precursor to CBC. Non-psychoactive minor cannabinoid.", therapeutic: "Early research suggests CBC may contribute anti-inflammatory, antidepressant, and neurogenesis-promoting effects." },
  "THCVA": { title: "THCVA — Tetrahydrocannabivarinic Acid", body: "A varin-type cannabinoid. Associated with specific regional genetics.", therapeutic: "THCV may act as a CB1 antagonist at low doses. Studied for blood sugar regulation." },
  "CBG": { title: "CBG — Cannabigerol", body: "Non-psychoactive. Binds to both CB1 and CB2 receptors.", therapeutic: "Studied for antibacterial, neuroprotective, and anti-inflammatory properties." },
  "CBN": { title: "CBN — Cannabinol", body: "Degradation product of THC. Forms when THC oxidises.", therapeutic: "Mildly psychoactive. High CBN signals age or poor storage." },
  "CBD": { title: "CBD — Cannabidiol", body: "The second most researched cannabinoid. Non-psychoactive.", therapeutic: "Extensive research into anxiolytic, anticonvulsant, anti-inflammatory, and neuroprotective properties." },
  "CBC": { title: "CBC — Cannabichromene", body: "Non-psychoactive minor cannabinoid. The active form of CBCA.", therapeutic: "Studied for anti-inflammatory, antidepressant, and potential neurogenesis-promoting effects. One of the 'big six' cannabinoids of research interest." },
};

// Normalise terpene name for consistent lookups — handles Greek letters, variant spellings
function normaliseTerpeneName(name = "") {
  return String(name)
    .replace(/β-|β /gi, "Beta-")
    .replace(/α-|α /gi, "Alpha-")
    .replace(/Δ-|Δ /gi, "Delta-")
    .replace(/\(R\)-\(\+\)-/gi, "")
    .replace(/\(R\)-/gi, "")
    .replace(/\(S\)-/gi, "")
    .replace(/d-Limonene/gi, "Limonene")
    .replace(/D-Limonene/gi, "Limonene")
    .replace(/trans-/gi, "Trans-")
    .replace(/cis-/gi, "Cis-")
    .trim();
}

function getTerpeneIntel(terpName = "") {
  const normalised = normaliseTerpeneName(terpName);
  const key = normalised.toLowerCase().trim();
  for (const [k, v] of Object.entries(TERPENE_INTEL)) {
    if (key.includes(k)) return v;
  }
  return { direction: "Chemistry-led", note: "A clearer directional read would require stronger terpene dominance or a broader comparison dataset.", lineage: "Unclear cluster", lineageConfidence: "Low" };
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
    fenchone: "FEN", caryophyllene_oxide: "COX",
  };
  const top3 = terpenes.filter(t => { const v = parseFloat(t.value); return !isNaN(v) && v > 0; }).slice(0, 3)
    .map(t => {
      const key = String(t.name || "").toLowerCase().trim();
      for (const [k, abbr] of Object.entries(ABBREV)) { if (key.includes(k)) return abbr; }
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
  const cbna = toNum(extracted.cbna);
  const terpenes = extracted.top_terpenes || [];
  const lead = terpenes[0]?.name || "";
  const second = terpenes[1]?.name || "";
  const intel = getTerpeneIntel(lead);
  const score = scoring.total ?? 0;
  const terpsDisplay = terps > 0 ? terps.toFixed(2) : "not captured";
  const terpCount = terpenes.filter(t => toNum(t.value) > 0).length;

  const safetyScore = (scoring.breakdown?.safety?.score) ?? 0;
  const terpScore = (scoring.breakdown?.terpenes?.score) ?? 0;

  const brand = [];
  if (lead) {
    const rarity = ["terpinolene", "ocimene", "farnesene", "guaiol"].some(t => lead.toLowerCase().includes(t));
    brand.push(rarity
      ? `${lead} dominance is a genuine market differentiator — fewer than 15% of dried flower products lead with this compound. Supports a distinct, premium identity.`
      : `${lead}-dominant chemistry is well-recognised in market positioning and supports clear directional messaging around ${intel.direction.toLowerCase()}.`
    );
  }
  brand.push(`Intelligence score ${score}/100 (${scoring.tier || "—"}). ${safetyScore >= 15 ? "Excellent full-panel safety compliance — all tested contaminants below reporting limits." : safetyScore >= 10 ? "Strong safety compliance with complete testing panel." : "Safety data gap limits maximum score."}`);
  if (cbga >= 0.5) brand.push(`CBGA at ${cbga > 0 ? cbga.toFixed(4) : "trace"} wt% is notable — elevated CBGA supports a biosynthetic depth narrative for premium positioning.`);
  if (terps >= 2) brand.push(`Total terpene content of ${terps.toFixed(3)} wt% across ${terpCount} quantified compounds sits in the upper range for dried flower. Supports premium aromatic positioning.`);
  if ((extracted.flavonoids || []).length > 0) brand.push(`Full flavonoid panel detected — ${extracted.flavonoids.length} compounds quantified including cannabis-specific Cannflavins. A rare analytical data point that supports a phytochemical complexity narrative.`);

  const clinical = [];
  clinical.push(`THC total (anhydrous): ${extracted.thc_total_anhydrous || extracted.thc_total || "—"} wt% — ${thc >= 24 ? "high range. Tolerance-aware patient selection and dose titration is warranted." : thc >= 18 ? "moderate-to-high range." : "moderate range."}`);
  clinical.push(cbd > 0.5 ? `CBD present at ${cbd.toFixed(2)} wt% — may provide some modulation of THC-dominant effects.` : "CBD not detected — no intrinsic CBD modulation of THC intensity in this profile.");
  clinical.push(`Terpene direction: ${intel.note}`);
  if (cbna > 0) clinical.push(`CBNA detected at ${cbna.toFixed(4)} wt% — low-level degradation marker. Not clinically significant at this level.`);
  clinical.push(cbn >= 0.5 ? `CBN detected at ${cbn.toFixed(2)} wt% — may indicate some THC oxidative degradation. Consider storage conditions.` : "CBN not detected (free form) — minimal oxidative degradation signal.");
  const pestStatus = contaminants.pesticides_status;
  const metalStatus = contaminants.heavy_metals_status;
  const microStatus = contaminants.microbials_status;
  const mycoStatus = contaminants.mycotoxins_status;
  if (pestStatus && !pestStatus.toLowerCase().includes("not tested")) {
    clinical.push(`Full contaminant compliance: Pesticides — ${pestStatus}. Metals — ${metalStatus || "—"}. Microbials — ${microStatus || "—"}. Mycotoxins — ${mycoStatus || "—"}.`);
  } else {
    clinical.push("Structured contaminant pass/fail data not captured — review source COA directly before prescribing.");
  }

  const patient = [];
  patient.push(thc >= 24 ? "This is a high-THC product. It is likely to feel strong, especially for patients with lower tolerance or those new to cannabis." : thc >= 18 ? "This product has moderate-to-high THC content. Start with a low dose and increase gradually." : "This product has moderate THC content.");
  patient.push(lead ? `The leading terpene — ${lead} — is typically associated with ${intel.direction.toLowerCase()} experiences.` : "A clear terpene direction could not be identified from this report.");
  patient.push(cbd > 0.5 ? `CBD is present at ${cbd.toFixed(2)} wt%, which may provide some balancing effect.` : "CBD is not detected in this product — there is no built-in balance from CBD.");
  const aromaLead = (lead || second || "").trim();
  if (aromaLead) {
    const firstChar = aromaLead[0]?.toLowerCase() || "";
    const article = ["a","e","i","o","u"].includes(firstChar) ? "an" : "a";
    patient.push(`Expect ${article} ${[lead, second].filter(Boolean).join(" and ")}-forward aroma.`);
  }

  const buyer = [];
  buyer.push(`Intelligence score: ${score}/100 — ${scoring.tier || "—"}. ${terpScore >= 18 ? "Leads on terpene richness." : ""} ${safetyScore >= 15 ? "Full safety panel confirmed — clean compliance across all tested categories." : "Safety data gap is the primary score limiter."}`);
  if (lead) buyer.push(`${lead}-dominant COA. ${terpCount} terpenes quantified with a total of ${terpsDisplay} wt% — ${terps >= 2.5 ? "above-average shelf differentiation potential." : "serviceable terpene density."}`);
  buyer.push(terps >= 2 ? `Total terpene content (${terpsDisplay} wt%) supports premium pricing. Terpene richness and chemotype clarity justify category-leading positioning.` : `Terpene content (${terpsDisplay} wt%) is serviceable. Pricing should align with mid-tier terpene density.`);
  if (contaminants.pesticides_status && !contaminants.pesticides_status.toLowerCase().includes("not tested")) {
    buyer.push(`Full compliance: Pesticides (${contaminants.pesticides_compound_count || "comprehensive"} compounds) — ${contaminants.pesticides_status}. Metals — ${contaminants.heavy_metals_status || "—"}. Microbials — ${contaminants.microbials_status || "—"}. Mycotoxins — ${contaminants.mycotoxins_status || "—"}.`);
  } else {
    buyer.push("Structured contaminant overview not captured. Request full compliance panel from producer before listing.");
  }

  return { brand, clinical, patient, buyer };
}

function buildPostHarvestIntel(extracted, contaminants) {
  const terps = toNum(extracted.total_terpenes);
  const terpCount = (extracted.top_terpenes || []).filter(t => toNum(t.value) > 0).length;
  const cbn = toNum(extracted.cbn);
  const cbna = toNum(extracted.cbna);
  const moisture = extracted.moisture_content || contaminants.moisture_content;
  const waterActivity = extracted.water_activity || contaminants.water_activity;

  const freshness = terps >= 3 ? { label: "Strong", note: "Terpene retention is high — consistent with excellent post-harvest preservation." }
    : terps >= 2 ? { label: "Good", note: "Terpene retention suggests well-preserved aromatic content through handling and storage." }
    : terps >= 1.2 ? { label: "Moderate", note: "Some aromatic expression remains, though terpene density is not exceptional." }
    : terps > 0 ? { label: "Light", note: "Lower terpene density may suggest some aromatic loss through processing or storage." }
    : { label: "Unknown", note: "Freshness cannot be inferred — terpene density not captured." };

  const curing = terpCount >= 8 && terps >= 1.5 ? { label: "Positive signal", note: "Broad terpene spectrum without flat profile suggests preserved aromatic complexity through curing." }
    : terpCount >= 5 ? { label: "Moderate signal", note: "Good terpene breadth visible. Confidence in curing quality is moderate." }
    : terpCount >= 3 ? { label: "Moderate signal", note: "Some terpene layering visible. Confidence limited without additional data." }
    : { label: "Limited signal", note: "Insufficient terpene breadth to draw a confident curing quality inference." };

  const degradation = (cbn >= 1.5 || cbna >= 0.5) ? { label: "Notable", note: `Degradation markers elevated — review storage conditions.` }
    : (cbn >= 0.3 || cbna >= 0.1) ? { label: "Low signal", note: `Minor degradation markers present (CBN free: ${cbn > 0 ? cbn.toFixed(4) : "ND"} wt%, CBNA: ${cbna > 0 ? cbna.toFixed(4) : "ND"} wt%). Not clinically significant.` }
    : { label: "Minimal", note: "CBN and CBNA at minimal levels — low evidence of THC oxidative degradation." };

  const stabilityParts = [];
  if (moisture) stabilityParts.push(`Moisture: ${moisture}% (${parseFloat(moisture) <= 12 ? "within acceptable range" : "review target range"})`);
  if (waterActivity) stabilityParts.push(`Water activity: ${waterActivity} aw (${parseFloat(waterActivity) <= 0.65 ? "stable — below mould proliferation threshold" : "review — elevated risk"})`);

  const stability = stabilityParts.length > 0
    ? { label: "Data present", note: stabilityParts.join(". ") + "." }
    : { label: "Limited", note: "Moisture and water activity data not captured — stability confidence limited to terpene-based inference only." };

  return { freshness, curing, degradation, stability };
}

// ─────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────

function toNum(value) {
  if (value === null || value === undefined) return 0;
  const raw = String(value).trim().toLowerCase();
  if (!raw || raw === "nd" || raw === "n/d" || raw.includes("not detected") || raw.includes("<loq") || raw.includes("< lod") || raw === "blq" || raw === "absent") return 0;
  const match = raw.match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : 0;
}

function esc(v = "") {
  return String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function sanitizeFileName(name = "") {
  return String(name || "").replace(/[^a-zA-Z0-9-_.]/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
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

// ─────────────────────────────────────────────
// OPENAI CALL WRAPPER
// ─────────────────────────────────────────────

async function callOpenAI(systemPrompt, userText, maxTokens = 2400) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), OPENAI_TIMEOUT_MS);
  try {
    const response = await openai.chat.completions.create(
      {
        model: OPENAI_MODEL,
        max_tokens: maxTokens,
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userText },
        ],
      },
      { signal: controller.signal }
    );
    return response.choices?.[0]?.message?.content || "";
  } catch (err) {
    if (err.name === "AbortError" || controller.signal.aborted) {
      throw new Error(`OpenAI timeout after ${OPENAI_TIMEOUT_MS}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function extractJSON(text = "") {
  const trimmed = String(text || "").trim();
  if (!trimmed) throw new Error("Empty OpenAI response");
  const first = trimmed.indexOf("{");
  if (first === -1) throw new Error("No JSON object found in response");
  const last = trimmed.lastIndexOf("}");
  // If closing brace missing (truncated), attempt to repair
  if (last === -1 || last <= first) {
    console.warn("⚠️  JSON appears truncated — attempting repair");
    const partial = trimmed.slice(first);
    let braces = 0, brackets = 0, inStr = false, escape = false;
    for (const ch of partial) {
      if (escape) { escape = false; continue; }
      if (ch === "\\" && inStr) { escape = true; continue; }
      if (ch === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (ch === "{") braces++;
      if (ch === "}") braces--;
      if (ch === "[") brackets++;
      if (ch === "]") brackets--;
    }
    let repaired = partial;
    if (inStr) repaired += '"';
    for (let i = 0; i < brackets; i++) repaired += "]";
    for (let i = 0; i < braces; i++) repaired += "}";
    try { JSON.parse(repaired); console.warn("⚠️  JSON repair successful"); return repaired; }
    catch (_) { throw new Error("JSON truncated and repair failed"); }
  }
  return trimmed.slice(first, last + 1);
}

function normalizeChemistry(data = {}) {
  const s = (v) => (v === null || v === undefined ? "" : String(v));
  const a = (v) => (Array.isArray(v) ? v : []);
  // No cap on terpenes — return ALL
  const topTerpenes = a(data.top_terpenes).map(i => ({ name: s(i?.name), value: s(i?.value), unit: s(i?.unit) || "wt%" }));
  let totalTerpenes = s(data.total_terpenes);
  if (!totalTerpenes || totalTerpenes === "ND" || totalTerpenes === "") {
    const computed = topTerpenes.reduce((sum, t) => { const n = parseFloat(t.value); return sum + (isNaN(n) ? 0 : n); }, 0);
    if (computed > 0) { totalTerpenes = computed.toFixed(4); console.log(`⚠️  total_terpenes missing — computed from sum: ${totalTerpenes}`); }
  }
  const fixUnit = (u) => { const clean = s(u).trim(); if (!clean || clean === "%") return "wt%"; return clean; };
  const normCannName = (name) => {
    const n = s(name).trim();
    if (/^thca-a$/i.test(n)) return "THCA";
    if (/^cbda-a$/i.test(n)) return "CBDA";
    if (/^d9-?thc$/i.test(n)) return "D9-THC";
    if (/^d8-?thc$/i.test(n)) return "D8-THC";
    if (/^cbna$/i.test(n)) return "CBNA";
    return n;
  };
  const flavonoids = a(data.flavonoids).map(i => ({ name: s(i?.name), value: s(i?.value), unit: s(i?.unit) || "wt%" }));
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
    thc_total_anhydrous: s(data.thc_total_anhydrous),
    cbd_total: s(data.cbd_total),
    cbd_total_unit: fixUnit(data.cbd_total_unit),
    thca: s(data.thca),
    thca_unit: fixUnit(data.thca_unit),
    d9thc: s(data.d9thc),
    cbda: s(data.cbda),
    cbn: s(data.cbn),
    cbna: s(data.cbna),
    cbg: s(data.cbg),
    cbga: s(data.cbga),
    cbca: s(data.cbca),
    cbc: s(data.cbc),
    thcva: s(data.thcva),
    d8thc: s(data.d8thc),
    total_terpenes: totalTerpenes,
    total_terpenes_unit: fixUnit(data.total_terpenes_unit) || "wt%",
    total_cannabinoids: s(data.total_cannabinoids),
    total_flavonoids: s(data.total_flavonoids),
    moisture_content: s(data.moisture_content),
    water_activity: s(data.water_activity),
    foreign_matter: s(data.foreign_matter),
    hptlc_result: s(data.hptlc_result),
    macroscopic_result: s(data.macroscopic_result),
    microscopic_result: s(data.microscopic_result),
    hero_narrative: s(data.hero_narrative),
    top_cannabinoids: a(data.top_cannabinoids).map(i => ({
      name: normCannName(i?.name),
      value: s(i?.value),
      unit: fixUnit(i?.unit),
      notes: s(i?.notes)
    })),
    top_terpenes: topTerpenes,
    flavonoids: flavonoids,
  };
}

function normalizeContaminants(data = {}) {
  const s = (v) => (v === null || v === undefined ? "" : String(v));
  const a = (v) => (Array.isArray(v) ? v : []);
  const b = (v) => (v === true || String(v).toLowerCase() === "true");
  return {
    pesticides_status: s(data.pesticides_status),
    pesticides_detail: s(data.pesticides_detail),
    pesticides_method: s(data.pesticides_method),
    pesticides_compound_count: s(data.pesticides_compound_count),
    heavy_metals_status: s(data.heavy_metals_status),
    heavy_metals_detail: s(data.heavy_metals_detail),
    heavy_metals_method: s(data.heavy_metals_method),
    arsenic_result: s(data.arsenic_result),
    cadmium_result: s(data.cadmium_result),
    lead_result: s(data.lead_result),
    mercury_result: s(data.mercury_result),
    microbials_status: s(data.microbials_status),
    microbials_detail: s(data.microbials_detail),
    microbials_method: s(data.microbials_method),
    yeast_mold: s(data.yeast_mold),
    total_aerobic: s(data.total_aerobic),
    bile_tolerant_gram_negative: s(data.bile_tolerant_gram_negative),
    salmonella: s(data.salmonella),
    s_aureus: s(data.s_aureus),
    p_aeruginosa: s(data.p_aeruginosa),
    e_coli: s(data.e_coli),
    mycotoxins_status: s(data.mycotoxins_status),
    mycotoxins_detail: s(data.mycotoxins_detail),
    mycotoxins_method: s(data.mycotoxins_method),
    aflatoxin_b1: s(data.aflatoxin_b1),
    aflatoxin_b2: s(data.aflatoxin_b2),
    aflatoxin_g1: s(data.aflatoxin_g1),
    aflatoxin_g2: s(data.aflatoxin_g2),
    sum_aflatoxins: s(data.sum_aflatoxins),
    ochratoxin_a: s(data.ochratoxin_a),
    residual_solvents_status: s(data.residual_solvents_status),
    moisture_content: s(data.moisture_content),
    water_activity: s(data.water_activity),
    foreign_matter_status: s(data.foreign_matter_status),
    iso_17025: b(data.iso_17025),
    scc_accredited: b(data.scc_accredited),
    lab_accreditation_body: s(data.lab_accreditation_body),
    contaminant_narrative: s(data.contaminant_narrative),
    lab_quality_summary: s(data.lab_quality_summary),
    positive_flags: a(data.positive_flags).map(x => s(x)).filter(Boolean),
    warning_flags: a(data.warning_flags).map(x => s(x)).filter(Boolean),
  };
}

// ─────────────────────────────────────────────
// DUAL-PASS EXTRACTION
// ─────────────────────────────────────────────

async function extractChemistry(ocrText) {
  // Text is already pre-sliced by runDualPassExtraction — use directly
  const modelInput = String(ocrText || "").trim();
  if (!modelInput) return normalizeChemistry({});
  console.log(`🧪 Chemistry extraction pass... (${modelInput.length} chars)`);
  try {
    const raw = await callOpenAI(CHEMISTRY_EXTRACTION_PROMPT, modelInput, 3200);
    const json = extractJSON(raw);
    const parsed = JSON.parse(json);
    console.log(`   → Terpenes extracted: ${(parsed.top_terpenes || []).length}`);
    console.log(`   → Cannabinoids extracted: ${(parsed.top_cannabinoids || []).length}`);
    console.log(`   → Flavonoids extracted: ${(parsed.flavonoids || []).length}`);
    console.log(`   → Total terpenes: ${parsed.total_terpenes}`);
    console.log(`   → Moisture: ${parsed.moisture_content}, Water activity: ${parsed.water_activity}`);
    return normalizeChemistry(parsed);
  } catch (err) {
    console.warn("Chemistry extraction failed, returning empty:", err.message);
    return normalizeChemistry({});
  }
}

async function extractContaminants(ocrText) {
  const modelInput = String(ocrText || "").trim();
  if (!modelInput) return normalizeContaminants({});
  console.log(`🛡️ Contaminant extraction pass... (${modelInput.length} chars)`);
  console.log(`   → Preview: ${modelInput.slice(0, 400).replace(/\n/g, " ")}`);
  try {
    const raw = await callOpenAI(CONTAMINANT_EXTRACTION_PROMPT, modelInput, 3000);
    console.log(`   → Raw response: ${raw.length} chars`);
    const json = extractJSON(raw);
    const parsed = JSON.parse(json);
    console.log(`   → Pesticides: "${parsed.pesticides_status}" (${parsed.pesticides_compound_count || "?"} compounds)`);
    console.log(`   → Metals: "${parsed.heavy_metals_status}" | As:${parsed.arsenic_result} Cd:${parsed.cadmium_result} Pb:${parsed.lead_result} Hg:${parsed.mercury_result}`);
    console.log(`   → Microbials: "${parsed.microbials_status}" | Yeast:${parsed.yeast_mold} Salmonella:${parsed.salmonella}`);
    console.log(`   → Mycotoxins: "${parsed.mycotoxins_status}" | B1:${parsed.aflatoxin_b1} OTA:${parsed.ochratoxin_a}`);
    console.log(`   → Water activity: "${parsed.water_activity}" | Moisture: "${parsed.moisture_content}"`);
    console.log(`   → ISO 17025: ${parsed.iso_17025} | SCC: ${parsed.scc_accredited}`);
    return normalizeContaminants(parsed);
  } catch (err) {
    console.error("❌ Contaminant extraction FAILED:", err.message);
    return normalizeContaminants({});
  }
}

async function runDualPassExtraction(ocrText, pages = []) {
  const fullText = String(ocrText || "").trim();

  // Simple, reliable approach:
  // Azure OCR extracts ALL text from ALL pages.
  // We send the FULL text to both passes and let GPT find what it needs.
  // gpt-4o supports 128k token context — a full 20-page COA is only ~15-20k tokens.
  // No splitting, no routing, no missed data.

  let textToSend = fullText;

  // Only truncate if truly enormous (shouldn't happen for any real COA)
  if (textToSend.length > MAX_OCR_CHARS) {
    console.log(`📄 COA text ${fullText.length} chars — truncating to ${MAX_OCR_CHARS}`);
    textToSend = fullText.slice(0, MAX_OCR_CHARS);
  }

  console.log(`📄 Full OCR text: ${textToSend.length} chars across ${pages.length || "?"} pages — sending complete text to both passes`);

  const [chemistry, contaminants] = await Promise.all([
    extractChemistry(textToSend),
    extractContaminants(textToSend),
  ]);
  return { chemistry, contaminants };
}

// ─────────────────────────────────────────────
// AZURE OCR
// ─────────────────────────────────────────────

async function extractDocumentFromBuffer(buffer, mimeType) {
  if (!buffer) throw new Error("No buffer provided");
  console.log(`   → Sending ${buffer.length} bytes directly to Azure OCR (${mimeType})`);
  const poller = await azureClient.beginAnalyzeDocument("prebuilt-layout", buffer, { contentType: mimeType || "application/pdf" });
  const result = await poller.pollUntilDone();
  const pages = (result.pages || []).map(page => ({
    page_number: page.pageNumber,
    text: (page.lines || []).map(line => line.content).join("\n")
  }));
  const plain_text = pages.map(p => `[PAGE ${p.page_number}]\n${p.text}`).join("\n\n").trim();
  console.log(`   → OCR pages: ${pages.length}, total chars: ${plain_text.length}`);
  pages.forEach(p => console.log(`   → Page ${p.page_number}: ${p.text.length} chars`));
  return { mimeType, plain_text, page_count: pages.length, table_count: (result.tables || []).length, pages };
}

// Keep URL-based version as fallback only
async function extractDocumentFromUrl(fileUrl) {
  if (!fileUrl || typeof fileUrl !== "string") throw new Error("fileUrl must be a non-empty string");
  const fileResponse = await axios.get(fileUrl, { responseType: "arraybuffer", timeout: 60000, maxRedirects: 5, validateStatus: (status) => status >= 200 && status < 300 });
  const mimeType = detectMimeType(fileUrl, fileResponse.headers["content-type"]);
  return extractDocumentFromBuffer(Buffer.from(fileResponse.data), mimeType);
}

// ─────────────────────────────────────────────
// SUPABASE
// ─────────────────────────────────────────────

async function uploadBufferToSupabase({ buffer, originalName, mimeType, folder = "raw_documents" }) {
  const safeName = sanitizeFileName(originalName || `upload-${Date.now()}`);
  const storagePath = `${folder}/${Date.now()}-${safeName}`;
  const { error } = await supabase.storage.from(SUPABASE_BUCKET).upload(storagePath, buffer, { contentType: mimeType || "application/octet-stream", upsert: false });
  if (error) throw new Error(`Supabase upload failed: ${error.message}`);
  const { data: publicData } = supabase.storage.from(SUPABASE_BUCKET).getPublicUrl(storagePath);
  if (!publicData?.publicUrl) throw new Error("Failed to generate public URL");
  return { storagePath, publicUrl: publicData.publicUrl };
}

async function insertCOAReport({ chemistry, contaminants, scoring, intelligence, sourceUrl, storagePath, originalFilename, mimeType }) {
  const payload = {
    report_json: { chemistry, contaminants, scoring, intelligence, _meta: { source_url: sourceUrl || "", storage_path: storagePath || "", original_filename: originalFilename || "", mime_type: mimeType || "", saved_at: new Date().toISOString(), schema_version: "6.0" } },
    overall_score: scoring.total, report_confidence_score: (scoring.breakdown?.dataCompleteness?.score) ?? 0,
    chemotype_identity: intelligence.fingerprintId, chemotype_descriptor: intelligence.effectDirection, fingerprint_id: intelligence.fingerprintId,
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
// LAYER 5 — REPORT HTML RENDERER (v7 — Full Chemistry + Full Safety)
// ─────────────────────────────────────────────

function renderReportHTML(reportJson = {}, options = {}) {
  const chemistry = reportJson.chemistry || {};
  const contaminants = reportJson.contaminants || {};
  const scoring = reportJson.scoring || computeIntelligenceScore(chemistry, contaminants);
  const intelligence = reportJson.intelligence || {};
  const benchmark          = options.benchmark          || null;
  const strainIntel        = options.strainIntel        || intelligence.strainIntel        || null;
  const scientificEvidence = options.scientificEvidence || intelligence.scientificEvidence || null;
  const sciLoading         = options.sciLoading         || false;

  const terpenes = chemistry.top_terpenes || [];
  const cannabinoids = chemistry.top_cannabinoids || [];
  const flavonoids = chemistry.flavonoids || [];
  const thc = toNum(chemistry.thc_total);
  const cbd = toNum(chemistry.cbd_total);
  const terps = toNum(chemistry.total_terpenes);
  const cbga = toNum(chemistry.cbga);
  const cbn = toNum(chemistry.cbn);
  const cbna = toNum(chemistry.cbna);
  const moisture = chemistry.moisture_content || contaminants.moisture_content || "";
  const waterActivity = chemistry.water_activity || contaminants.water_activity || "";
  const leadTerpene = terpenes[0]?.name || "";
  const secondTerpene = terpenes[1]?.name || "";
  const thirdTerpene = terpenes[2]?.name || "";
  const terpIntel = getTerpeneIntel(leadTerpene);
  const fingerprintId = intelligence.fingerprintId || generateFingerprintId(terpenes);

  // Safely build audiences and postHarvest — guard against old DB records with wrong shape
  let audiences;
  try {
    const rawAud = intelligence.audiences;
    audiences = (rawAud && Array.isArray(rawAud.brand)) ? rawAud : buildAudienceNarratives(chemistry, contaminants, scoring);
  } catch (_) {
    audiences = buildAudienceNarratives(chemistry, contaminants, scoring);
  }
  // Ensure all four panels are always arrays
  audiences.brand    = Array.isArray(audiences.brand)    ? audiences.brand    : [];
  audiences.clinical = Array.isArray(audiences.clinical) ? audiences.clinical : [];
  audiences.patient  = Array.isArray(audiences.patient)  ? audiences.patient  : [];
  audiences.buyer    = Array.isArray(audiences.buyer)    ? audiences.buyer    : [];

  let postHarvest;
  try {
    const rawPH = intelligence.postHarvest;
    postHarvest = (rawPH && rawPH.freshness && rawPH.curing && rawPH.degradation && rawPH.stability)
      ? rawPH : buildPostHarvestIntel(chemistry, contaminants);
  } catch (_) {
    postHarvest = buildPostHarvestIntel(chemistry, contaminants);
  }
  // Ensure all four signals always have label+note
  const safePH = {
    freshness:   postHarvest.freshness   || { label: "Unknown", note: "Data not available." },
    curing:      postHarvest.curing      || { label: "Unknown", note: "Data not available." },
    degradation: postHarvest.degradation || { label: "Unknown", note: "Data not available." },
    stability:   postHarvest.stability   || { label: "Unknown", note: "Data not available." },
  };

  const activeTerps = terpenes.filter(t => toNum(t.value) > 0);
  const blqTerps = terpenes.filter(t => String(t.value).toLowerCase() === "blq");
  const maxTerpVal = activeTerps.length > 0 ? Math.max(...activeTerps.map(t => toNum(t.value)), 0.001) : 0.001;
  const productName = chemistry.product_name || "COA Report";
  const heroNarrative = chemistry.hero_narrative || `${leadTerpene ? leadTerpene + "-dominant" : "Cannabis"} profile with ${terps > 2 ? "strong" : terps > 1 ? "moderate" : "light"} terpene expression.`;

  const safeTotal = scoring.total ?? 0;
  const safeGrade = scoring.grade || "—";
  const safeTier = scoring.tier || "—";

  const ringCirc = 351.86;
  const ringOffset = ((100 - safeTotal) / 100 * ringCirc).toFixed(1);

  // Status helpers
  const pestPass = /pass|nd|not detected/i.test(contaminants.pesticides_status || "");
  const metalsPass = /pass|nd|not detected|blq/i.test(contaminants.heavy_metals_status || "");
  const microPass = /pass|nd|not detected|absent/i.test(contaminants.microbials_status || "");
  const mycoPass = /pass|nd|not detected/i.test(contaminants.mycotoxins_status || "");
  const isoPass = contaminants.iso_17025 === true || /17025/i.test(chemistry.laboratory_accreditation || "");
  const sccPass = contaminants.scc_accredited === true;
  const hasMoisture = !!(moisture && moisture !== "");
  const hasWaterActivity = !!(waterActivity && waterActivity !== "");
  const hasForeignMatter = !!(contaminants.foreign_matter_status && contaminants.foreign_matter_status !== "");
  const hasFlavonoids = flavonoids.length > 0;

  function infoIcon(content, variant = "") {
    return `<span class="info-icon${variant ? " " + variant : ""}" tabindex="0">ⓘ${content}</span>`;
  }

  function sbRow(label, fillClass, widthPct, delay, scoreStr, tooltipTitle, ttBody, ttHow, ttCoa) {
    return `
    <div class="sb-row">
      <div class="sb-lbl-wrap">
        <span class="sb-lbl">${esc(label)}</span>
        ${infoIcon(`<span class="tooltip"><strong>${esc(tooltipTitle)}</strong><span class="tt-body">${esc(ttBody)}</span><span class="tt-how"><strong>How we scored it:</strong> ${esc(ttHow)}</span><span class="tt-coa"><strong>Check on your COA:</strong> ${esc(ttCoa)}</span></span>`)}
      </div>
      <div class="sb-track"><div class="sb-fill ${fillClass}" style="width:${widthPct}%;animation-delay:${delay}s"></div></div>
      <div class="sb-num">${esc(scoreStr)}</div>
    </div>`;
  }

  function terpBar(t, i) {
    if (!t || !t.name) return "";
    const val = toNum(t.value);
    const isBlq = String(t.value || "").toLowerCase() === "blq";
    const width = isBlq ? 2 : Math.max(3, maxTerpVal > 0 ? (val / maxTerpVal) * 100 : 0);
    const isLead = i === 0;
    // Try exact name first, then normalised name for education lookup
    const edu = TERPENE_EDUCATION[t.name] || TERPENE_EDUCATION[normaliseTerpeneName(t.name)] || null;
    const icon = edu
      ? infoIcon(`<span class="tooltip"><strong>${esc(t.name)}</strong><span class="tt-body">${esc("Aroma: " + edu.aroma)}</span><span class="tt-coa">${esc(edu.therapeutic)}</span></span>`, "tt-right row-info")
      : "";
    return `
    <div class="tb">
      <div class="tb-row">
        <span class="tb-name${isLead ? " lead" : ""}${isBlq ? " blq" : ""}">${esc(t.name)}${icon}</span>
        <span class="tb-pct${isBlq ? " blq" : ""}">${isBlq ? "BLQ" : esc(String(t.value || "")) + " " + esc(t.unit || "wt%")}</span>
      </div>
      <div class="tb-track"><div class="tb-fill${isLead ? " lead" : ""}${isBlq ? " blq" : ""}" style="width:${width.toFixed(1)}%;animation-delay:${(i * 0.04 + 0.04).toFixed(2)}s"></div></div>
    </div>`;
  }

  function cannCell(name, value, unit, note, isHighlight = false) {
    if (!name) return "";
    const edu = CANNABINOID_EDUCATION[String(name)] || null;
    const icon = edu
      ? infoIcon(`<span class="tooltip"><strong>${esc(edu.title)}</strong><span class="tt-body">${esc(edu.body)}</span><span class="tt-coa">${esc(edu.therapeutic)}</span></span>`, "cann-info")
      : "";
    const safeVal = String(value ?? "");
    const isNd = !safeVal || safeVal === "ND" || safeVal === "" || toNum(safeVal) === 0;
    return `
    <div class="cann-cell${isHighlight ? " hl" : ""}${isNd ? " nd-cell" : ""}">
      ${icon}
      <div class="cann-lbl">${esc(name)}</div>
      <div class="cann-val${isNd ? " nd" : ""}">${isNd ? "<span class='nd-tag'>ND</span>" : esc(safeVal) + "<span class='cann-unit'> " + esc(unit || "wt%") + "</span>"}</div>
      ${note ? `<div class="cann-note">${esc(String(note))}</div>` : ""}
    </div>`;
  }

  function phRow(dotClass, label, note, ttTitle, ttBody, ttCoa) {
    const icon = infoIcon(`<span class="tooltip"><strong>${esc(ttTitle)}</strong><span class="tt-body">${esc(ttBody)}</span><span class="tt-coa"><strong>Signal source:</strong> ${esc(ttCoa)}</span></span>`, "ph-info tt-right");
    return `
    <div class="ph-row">
      <div class="ph-dot ${dotClass}"></div>
      <div>
        <div class="ph-lbl" style="display:flex;align-items:center;gap:5px;">${esc(label)}${icon}</div>
        <div class="ph-note">${esc(note)}</div>
      </div>
    </div>`;
  }

  function sfRow(label, dotClass, statusText, pass) {
    return `
    <div class="sf-row">
      <div class="sf-dot ${pass ? "pass" : "nt"}"></div>
      <div class="sf-name">${esc(label)}</div>
      <div class="sf-val${pass ? " pass" : ""}">${esc(statusText || (pass ? "✓ Confirmed" : "not tested"))}</div>
    </div>`;
  }

  function metalRow(metal, result) {
    const safeResult = String(result ?? "");
    const isNd = !safeResult || safeResult === "ND" || safeResult === "" || safeResult.toLowerCase().includes("nd");
    const isBlq = safeResult.toLowerCase().includes("blq");
    const isPass = isNd || isBlq;
    return `<div class="metal-row">
      <div class="sf-dot ${isPass ? "pass" : "nt"}"></div>
      <div class="metal-name">${esc(metal)}</div>
      <div class="metal-val ${isPass ? "pass" : "warn"}">${esc(safeResult || "—")}</div>
    </div>`;
  }

  function microbialRow(pathogen, result) {
    const safeResult = String(result ?? "");
    const isPass = !safeResult || /absent|nd|not detected|< 10/i.test(safeResult);
    return `<div class="metal-row">
      <div class="sf-dot ${isPass ? "pass" : "nt"}"></div>
      <div class="metal-name">${esc(pathogen)}</div>
      <div class="metal-val ${isPass ? "pass" : "warn"}">${esc(safeResult || "—")}</div>
    </div>`;
  }

  function mycoRow(compound, result) {
    const safeResult = String(result ?? "");
    const isNd = !safeResult || safeResult.toLowerCase() === "nd" || safeResult === "0";
    return `<div class="metal-row">
      <div class="sf-dot ${isNd ? "pass" : "nt"}"></div>
      <div class="metal-name">${esc(compound)}</div>
      <div class="metal-val ${isNd ? "pass" : "warn"}">${esc(safeResult || "ND")}</div>
    </div>`;
  }

  function flavonoidBar(f, i) {
    if (!f || !f.name) return "";
    const val = toNum(f.value);
    const flavNums = flavonoids.map(x => toNum(x.value)).filter(v => v > 0);
    const maxVal = flavNums.length > 0 ? Math.max(...flavNums) : 0.001;
    const width = maxVal > 0 ? Math.max(3, (val / maxVal) * 100) : 3;
    return `<div class="flav-row">
      <div class="flav-name">${esc(f.name)}</div>
      <div class="flav-track"><div class="flav-fill" style="width:${width.toFixed(1)}%;animation-delay:${(i*0.04).toFixed(2)}s"></div></div>
      <div class="flav-val">${esc(String(f.value ?? ""))} ${esc(f.unit || "wt%")}</div>
    </div>`;
  }

  function audiencePanel(items = []) {
    if (!items.length) return `<div class="ins-row"><div class="ins-arr"></div><span class="ins-text">No intelligence available.</span></div>`;
    return items.map(line => `<div class="ins-row"><div class="ins-arr"></div><span class="ins-text">${esc(line)}</span></div>`).join("");
  }

  const safeBreakdown = scoring.breakdown || {};
  const safePotency = safeBreakdown.potency || { score: 0, max: 25 };
  const safeTerpenes = safeBreakdown.terpenes || { score: 0, max: 25 };
  const safeMinors = safeBreakdown.minors || { score: 0, max: 20 };
  const safeSafety = safeBreakdown.safety || { score: 0, max: 20 };
  const safeData = safeBreakdown.dataCompleteness || { score: 0, max: 10 };

  const potencyPct = safePotency.max > 0 ? Math.round(safePotency.score / safePotency.max * 100) : 0;
  const terpPct = safeTerpenes.max > 0 ? Math.round(safeTerpenes.score / safeTerpenes.max * 100) : 0;
  const minorPct = safeMinors.max > 0 ? Math.round(safeMinors.score / safeMinors.max * 100) : 0;
  const safetyPct = safeSafety.max > 0 ? Math.round(safeSafety.score / safeSafety.max * 100) : 0;
  const dataPct = safeData.max > 0 ? Math.round(safeData.score / safeData.max * 100) : 0;

  const terpCount = activeTerps.length;
  const blqCount = blqTerps.length;

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${esc(productName)} — Alem Chemical Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,300;1,400;1,600&family=Space+Mono:wght@400;700&family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --alem-dark:   #0d2d3e;
  --alem-mid:    #1a4a62;
  --alem-accent: #8ecfb0;
  --alem-tint:   #eef6fb;
  --alem-wash:   #f4f8fb;
  --white:       #ffffff;
  --off:         #fafcfe;
  --t-dark:      #0d2d3e;
  --t-body:      #2a3d4a;
  --t-mid:       #4a6070;
  --t-light:     #8aa0b0;
  --t-faint:     #b8c8d4;
  --border:      #ccdde8;
  --border-l:    #e2eef5;
  --gold:        #9a7822;
  --gold-l:      #c09a38;
  --gold-tint:   #fef8ec;
  --gold-bord:   #e8d8a8;
  --warn-bg:     #fef6f0;
  --warn-text:   #8a4818;
  --warn-bord:   #f0cdb0;
  --pass-green:  #2a7a5c;
  --sh:          rgba(13,45,62,.09);
}
body { background:var(--alem-wash); font-family:'Nunito',sans-serif; font-size:13.5px; line-height:1.6; color:var(--t-body); min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:48px 16px 72px; }
@keyframes rise { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes grow { from{transform:scaleX(0)} to{transform:scaleX(1)} }

.rcard { width:100%; max-width:620px; background:var(--white); border:1px solid var(--border); box-shadow:0 2px 20px var(--sh),0 8px 40px var(--sh); animation:rise .55s cubic-bezier(.16,1,.3,1) both; }

/* ── NAV ── */
.nav { background:var(--alem-dark); padding:16px 28px; display:flex; justify-content:space-between; align-items:center; }
.nav-right { text-align:right; }
.nav-date { font-size:8px; font-weight:500; letter-spacing:2px; text-transform:uppercase; color:rgba(255,255,255,.38); }
.nav-schema { font-family:'Space Mono',monospace; font-size:7.5px; color:rgba(255,255,255,.22); margin-top:3px; letter-spacing:1px; }

/* ── HERO ── */
.hero { background:var(--white); padding:32px 28px 28px; border-bottom:1px solid var(--border-l); }
.hero-top { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; }
.hero-left { flex:1; }
.hero-eye { font-size:7.5px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:var(--t-light); margin-bottom:10px; }
.hero-name { font-family:'EB Garamond',Georgia,serif; font-size:46px; font-weight:600; color:var(--alem-dark); line-height:.95; letter-spacing:-.5px; }
.hero-name-sub { font-family:'EB Garamond',Georgia,serif; font-size:24px; font-weight:400; font-style:italic; color:var(--t-light); display:block; margin-top:4px; }
.hero-meta { font-size:9.5px; font-weight:400; letter-spacing:1px; color:var(--t-light); margin-top:10px; }
.hero-narrative { font-family:'Nunito',sans-serif; font-style:italic; font-size:14.5px; font-weight:300; line-height:1.65; color:var(--t-mid); margin-top:14px; max-width:320px; }

/* ── SCORE RING ── */
.score-block { flex-shrink:0; text-align:center; }
.score-ring { width:130px; height:130px; position:relative; }
.score-ring svg { transform:rotate(-90deg); }
.score-inner { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.score-n { font-family:'Nunito',sans-serif; font-size:52px; font-weight:800; color:var(--alem-dark); line-height:1; }
.score-d { font-family:'Space Mono',monospace; font-size:11px; color:var(--t-light); }
.score-grade { font-size:11px; font-weight:700; letter-spacing:2px; color:var(--alem-dark); margin-top:10px; text-transform:uppercase; }
.score-tier { font-size:9px; font-weight:300; letter-spacing:2px; color:var(--t-light); text-transform:uppercase; margin-top:3px; }

/* ── SCORE BREAKDOWN ── */
.score-bd { background:var(--alem-tint); padding:20px 28px 24px; border-bottom:1px solid var(--border); }
.sb-row { display:grid; grid-template-columns:1fr 1fr 56px; align-items:center; gap:14px; padding:8px 0; border-top:1px solid var(--border-l); }
.sb-row:first-child { border-top:none; }
.sb-lbl-wrap { display:flex; align-items:center; gap:6px; }
.sb-lbl { font-size:9px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--t-mid); }
.sb-track { height:8px; background:var(--border); border-radius:4px; overflow:hidden; }
.sb-fill { height:100%; border-radius:4px; transform-origin:left; animation:grow .8s cubic-bezier(.16,1,.3,1) both; }
.f-g { background:var(--alem-mid); }
.f-o { background:var(--gold); }
.f-r { background:#c07858; }
.sb-num { font-family:'Space Mono',monospace; font-size:10px; font-weight:700; color:var(--t-mid); text-align:right; }

/* ── INFO ICON + TOOLTIP ── */
.info-icon { position:relative; display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px; background:var(--border); color:var(--t-mid); border-radius:50%; font-size:10px; font-style:normal; cursor:help; flex-shrink:0; transition:background .15s,color .15s; line-height:1; font-family:'Nunito',sans-serif; font-weight:800; user-select:none; vertical-align:middle; }
.info-icon:hover,.info-icon:focus { background:var(--alem-dark); color:var(--white); outline:none; }
.tooltip { display:none; position:absolute; left:22px; top:50%; transform:translateY(-50%); width:290px; background:var(--white); border:1px solid var(--border); border-radius:6px; padding:14px 16px; box-shadow:0 8px 32px rgba(13,45,62,.14),0 2px 8px rgba(13,45,62,.08); z-index:999; pointer-events:none; font-size:11px; font-weight:400; color:var(--t-body); line-height:1.6; font-family:'Nunito',sans-serif; text-transform:none; letter-spacing:0; }
.tooltip strong { display:block; font-size:11.5px; font-weight:800; color:var(--alem-dark); margin-bottom:7px; letter-spacing:0; }
.tooltip .tt-body { display:block; margin-bottom:9px; color:var(--t-body); }
.tooltip .tt-how,.tooltip .tt-coa { display:block; font-size:10.5px; color:var(--t-mid); margin-top:7px; padding-top:7px; border-top:1px solid var(--border-l); line-height:1.55; }
.tooltip .tt-how strong,.tooltip .tt-coa strong { display:inline; font-size:10.5px; font-weight:700; color:var(--alem-dark); margin-bottom:0; }
.tooltip::before { content:''; position:absolute; left:-6px; top:50%; transform:translateY(-50%); width:0; height:0; border-top:6px solid transparent; border-bottom:6px solid transparent; border-right:6px solid var(--border); }
.tooltip::after { content:''; position:absolute; left:-5px; top:50%; transform:translateY(-50%); width:0; height:0; border-top:5px solid transparent; border-bottom:5px solid transparent; border-right:5px solid var(--white); }
.info-icon:hover .tooltip,.info-icon:focus .tooltip { display:block; }
.tt-right .tooltip { left:22px; right:auto; top:50%; transform:translateY(-50%); }
.tt-right .tooltip::before { left:-6px; right:auto; border-left:none; border-right:6px solid var(--border); border-top:6px solid transparent; border-bottom:6px solid transparent; top:50%; transform:translateY(-50%); }
.tt-right .tooltip::after { left:-5px; right:auto; border-left:none; border-right:5px solid var(--white); border-top:5px solid transparent; border-bottom:5px solid transparent; top:50%; transform:translateY(-50%); }
.sec-info { vertical-align:middle; margin-left:6px; }
.row-info { margin-left:4px; flex-shrink:0; }
.cann-info { position:absolute; top:8px; right:8px; }
.cann-info .tooltip { width:260px; right:22px; left:auto; top:0; transform:none; }
.cann-info .tooltip::before { right:-6px; left:auto; border-right:none; border-left:6px solid var(--border); border-top:6px solid transparent; border-bottom:6px solid transparent; top:12px; transform:none; }
.cann-info .tooltip::after { right:-5px; left:auto; border-right:none; border-left:5px solid var(--white); border-top:5px solid transparent; border-bottom:5px solid transparent; top:13px; transform:none; }
.ph-info .tooltip { width:260px; left:22px; top:50%; transform:translateY(-50%); right:auto; }
.ph-info .tooltip::before { left:-6px; border-right:6px solid var(--border); border-left:none; border-top:6px solid transparent; border-bottom:6px solid transparent; top:50%; transform:translateY(-50%); right:auto; }
.ph-info .tooltip::after { left:-5px; border-right:5px solid var(--white); border-left:none; border-top:5px solid transparent; border-bottom:5px solid transparent; top:50%; transform:translateY(-50%); right:auto; }
.m-info { margin-top:5px; display:inline-flex; }
.m-info .tooltip { width:240px; left:50%; transform:translateX(-50%); top:calc(100% + 10px); right:auto; }
.m-info .tooltip::before { left:50%; transform:translateX(-50%); top:-6px; right:auto; border-top:none; border-bottom:6px solid var(--border); border-left:6px solid transparent; border-right:6px solid transparent; }
.m-info .tooltip::after { left:50%; transform:translateX(-50%); top:-5px; right:auto; border-top:none; border-bottom:5px solid var(--white); border-left:5px solid transparent; border-right:5px solid transparent; }

/* ── CHEMOTYPE ── */
.ct-band { background:var(--white); border-bottom:1px solid var(--border-l); border-top:1px solid var(--border-l); padding:12px 28px; display:flex; align-items:center; }
.ct-code { font-family:'Space Mono',monospace; font-size:13px; font-weight:700; letter-spacing:3px; color:var(--alem-dark); display:flex; align-items:center; gap:6px; }
.ct-sep { width:1px; height:28px; background:var(--border); margin:0 18px; }
.ct-lbl { font-size:7px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--t-light); }
.ct-lin { font-family:'Nunito',sans-serif; font-style:italic; font-size:13px; font-weight:400; color:var(--t-body); margin-top:1px; }
.ct-right { margin-left:auto; display:flex; flex-direction:column; align-items:flex-end; gap:4px; }
.ct-effect { font-size:8px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:var(--alem-mid); background:var(--alem-tint); border:1px solid var(--border); padding:3px 10px; border-radius:20px; display:flex; align-items:center; gap:5px; }
.ct-conf { font-size:7px; letter-spacing:1px; color:var(--t-faint); text-transform:uppercase; display:flex; align-items:center; gap:4px; }

/* ── METRICS ── */
.metrics { display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid var(--border-l); }
.m-cell { padding:18px 12px; text-align:center; border-right:1px solid var(--border-l); position:relative; }
.m-cell:last-child { border-right:none; }
.m-val { font-family:'Nunito',sans-serif; font-size:28px; font-weight:800; color:var(--alem-dark); line-height:1; }
.m-unit { font-family:'Nunito',sans-serif; font-size:10px; color:var(--t-light); }
.m-lbl { font-size:7px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--t-light); margin-top:5px; }
.m-ctx { font-size:8px; font-weight:500; color:var(--alem-mid); margin-top:4px; }
.m-ctx.nd { color:var(--t-faint); font-style:italic; }

/* ── SUMMARY ── */
.summary { padding:20px 28px; background:var(--alem-tint); border-bottom:1px solid var(--border-l); border-left:3px solid var(--alem-accent); }
.summary-lbl { font-size:7.5px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--alem-mid); margin-bottom:9px; }
.summary-body { font-family:'EB Garamond',Georgia,serif; font-size:15px; font-weight:400; line-height:1.7; color:var(--t-dark); }

/* ── SECTIONS ── */
.sec { padding:24px 28px; border-bottom:1px solid var(--border-l); }
.sec-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:18px; }
.sec-title { font-size:8px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--t-light); }
.sec-badge { font-family:'Space Mono',monospace; font-size:8px; color:var(--alem-mid); background:var(--alem-tint); padding:2px 10px; border-radius:20px; }

/* ── TERPENES ── */
.terp-intro { font-size:12px; font-weight:400; color:var(--t-mid); background:var(--off); border:1px solid var(--border-l); padding:10px 14px; margin-bottom:16px; line-height:1.55; }
.terp-intro strong { font-weight:700; color:var(--alem-dark); }
.terp-wrap { display:flex; gap:20px; align-items:flex-start; }
.t-bars { flex:1; display:flex; flex-direction:column; gap:7px; }
.tb-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:3px; }
.tb-name { font-size:9px; font-weight:400; color:var(--t-body); display:flex; align-items:center; gap:4px; }
.tb-name.lead { font-weight:700; color:var(--alem-dark); }
.tb-name.blq { color:var(--t-faint); font-style:italic; }
.tb-pct { font-family:'Space Mono',monospace; font-size:9px; color:var(--alem-mid); }
.tb-pct.blq { color:var(--t-faint); font-size:8px; }
.tb-track { height:3px; background:var(--border-l); border-radius:2px; overflow:hidden; }
.tb-fill { height:100%; background:var(--alem-mid); border-radius:2px; transform-origin:left; animation:grow .8s cubic-bezier(.16,1,.3,1) both; opacity:.75; }
.tb-fill.lead { background:var(--alem-dark); opacity:1; }
.tb-fill.blq { background:var(--border); opacity:.5; }
.terp-footer { margin-top:12px; padding:8px 12px; background:var(--off); border:1px solid var(--border-l); display:flex; justify-content:space-between; align-items:center; }
.terp-footer-txt { font-size:9px; color:var(--t-mid); }
.terp-footer-val { font-family:'Space Mono',monospace; font-size:10px; font-weight:700; color:var(--alem-dark); }

/* ── CANNABINOIDS ── */
.cann-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.cann-cell { padding:12px 13px; background:var(--off); border:1px solid var(--border-l); position:relative; }
.cann-cell.hl { background:var(--gold-tint); border-color:var(--gold-bord); }
.cann-cell.nd-cell { opacity:.65; }
.cann-lbl { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:1px; color:var(--t-mid); margin-bottom:5px; }
.cann-cell.hl .cann-lbl { color:var(--gold); }
.cann-val { font-family:'Nunito',sans-serif; font-size:20px; font-weight:800; color:var(--alem-dark); line-height:1; }
.cann-val.nd { font-size:12px; color:var(--t-faint); }
.nd-tag { font-family:'Space Mono',monospace; font-size:10px; color:var(--t-faint); }
.cann-cell.hl .cann-val { color:var(--gold); }
.cann-unit { font-family:'Nunito',sans-serif; font-size:9px; color:var(--t-light); }
.cann-note { font-size:8px; font-weight:300; color:var(--t-light); margin-top:5px; line-height:1.4; }
.cann-cell.hl .cann-note { color:rgba(154,120,34,.7); }
.cann-total { margin-top:10px; padding:11px 16px; background:var(--alem-tint); border:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
.ct-lbl2 { font-size:8px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--t-mid); }
.ct-val2 { font-family:'Nunito',sans-serif; font-size:20px; font-weight:800; color:var(--alem-dark); }

/* ── ANHYDROUS BADGE ── */
.anhydrous-note { margin-top:8px; padding:7px 12px; background:var(--off); border:1px solid var(--border-l); font-size:9px; color:var(--t-mid); display:flex; align-items:center; gap:8px; }
.anhydrous-note strong { color:var(--alem-dark); font-weight:700; }

/* ── ENTOURAGE BANNER ── */
.entourage { margin-top:14px; padding:11px 14px; background:linear-gradient(135deg,var(--alem-tint),#f0f8ff); border:1px solid var(--border); border-left:3px solid var(--alem-accent); display:flex; align-items:flex-start; gap:10px; }
.entourage-icon { font-size:16px; flex-shrink:0; margin-top:1px; }
.entourage-body { font-size:10.5px; color:var(--t-body); line-height:1.55; }
.entourage-body strong { font-weight:700; color:var(--alem-dark); }

/* ── FLAVONOIDS ── */
.flav-grid { display:flex; flex-direction:column; gap:6px; }
.flav-row { display:grid; grid-template-columns:160px 1fr 80px; align-items:center; gap:10px; }
.flav-name { font-size:10px; color:var(--t-body); }
.flav-track { height:4px; background:var(--border-l); border-radius:2px; overflow:hidden; }
.flav-fill { height:100%; background:#7a9e8a; border-radius:2px; transform-origin:left; animation:grow .8s cubic-bezier(.16,1,.3,1) both; opacity:.8; }
.flav-val { font-family:'Space Mono',monospace; font-size:8.5px; color:var(--t-mid); text-align:right; }
.flav-total { margin-top:10px; padding:8px 12px; background:var(--off); border:1px solid var(--border-l); display:flex; justify-content:space-between; font-size:9px; }
.flav-total-val { font-family:'Space Mono',monospace; font-weight:700; color:var(--alem-dark); }

/* ── TWO COL ── */
.two-col { display:grid; grid-template-columns:1fr 1fr; }
.two-col .sec:first-child { border-right:1px solid var(--border-l); }
.eff-pill { display:inline-flex; align-items:center; gap:7px; background:var(--alem-tint); border:1px solid var(--border); padding:6px 14px; border-radius:30px; margin-bottom:14px; }
.eff-icon { font-size:13px; }
.eff-dir { font-family:'Nunito',sans-serif; font-style:italic; font-size:14px; font-weight:600; color:var(--alem-dark); }
.eff-body { font-size:10px; font-weight:400; color:var(--t-mid); line-height:1.7; }
.eff-body strong { font-weight:600; color:var(--t-body); }
.ph-list { display:flex; flex-direction:column; gap:11px; }
.ph-row { display:flex; gap:10px; align-items:flex-start; }
.ph-dot { width:8px; height:8px; border-radius:50%; margin-top:4px; flex-shrink:0; }
.ph-dot.good { background:var(--alem-accent); }
.ph-dot.mild { background:var(--border); border:1px solid var(--alem-accent); }
.ph-lbl { font-size:8px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; color:var(--t-body); }
.ph-note { font-size:10px; font-weight:300; color:var(--t-mid); line-height:1.45; }

/* ── AUDIENCE TABS ── */
.audience { padding:24px 28px; border-bottom:1px solid var(--border-l); }
.tab-row { display:flex; gap:6px; margin-bottom:16px; flex-wrap:wrap; }
.tab-btn { font-family:'Nunito',sans-serif; font-size:8px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; padding:6px 14px; border:1px solid var(--border); background:transparent; color:var(--t-light); cursor:pointer; border-radius:30px; transition:all .15s; display:flex; align-items:center; gap:5px; }
.tab-btn.active { background:var(--alem-dark); border-color:var(--alem-dark); color:var(--white); }
.tab-btn:hover:not(.active) { border-color:var(--alem-accent); color:var(--alem-dark); }
.tab-icon { font-size:10px; }
.tab-panel { display:none; }
.tab-panel.active { display:block; }
.ins-list { display:flex; flex-direction:column; gap:10px; }
.ins-row { display:flex; gap:11px; align-items:flex-start; }
.ins-arr { width:6px; height:6px; border-right:2px solid var(--alem-accent); border-bottom:2px solid var(--alem-accent); transform:rotate(-45deg); margin-top:6px; flex-shrink:0; }
.ins-text { font-size:12px; font-weight:400; color:var(--t-body); line-height:1.65; }

/* ── SAFETY ── */
.safety-section { padding:24px 28px; border-bottom:1px solid var(--border-l); }
.safety-section .sec-head { margin-bottom:18px; }
.safety-panels { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.safety-panel { border:1px solid var(--border-l); background:var(--off); }
.sp-head { padding:10px 14px; background:var(--alem-tint); border-bottom:1px solid var(--border-l); display:flex; justify-content:space-between; align-items:center; }
.sp-title { font-size:8px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--t-mid); }
.sp-status { font-size:8px; font-weight:700; }
.sp-status.pass { color:var(--pass-green); }
.sp-status.nt { color:var(--t-faint); font-style:italic; }
.sp-body { padding:10px 14px; display:flex; flex-direction:column; gap:5px; }
.metal-row { display:flex; align-items:center; gap:8px; }
.metal-name { font-size:9px; color:var(--t-body); flex:1; }
.metal-val { font-family:'Space Mono',monospace; font-size:8px; }
.metal-val.pass { color:var(--pass-green); }
.metal-val.warn { color:#c07850; }
.sf-row { display:flex; align-items:center; gap:9px; padding:6px 0; border-bottom:1px solid var(--border-l); }
.sf-row:last-child { border-bottom:none; }
.sf-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.sf-dot.pass { background:var(--alem-accent); }
.sf-dot.nt { background:#ddc8b4; border:1px solid #c8a888; }
.sf-name { font-size:9px; font-weight:500; color:var(--t-body); flex:1; }
.sf-val { font-family:'Space Mono',monospace; font-size:8px; color:var(--t-faint); font-style:italic; }
.sf-val.pass { color:var(--pass-green); font-style:normal; font-weight:700; }
.safety-summary-grid { margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.safety-note { margin-top:14px; padding:12px 14px; background:var(--warn-bg); border-left:3px solid var(--warn-bord); font-size:11px; font-weight:400; color:var(--warn-text); line-height:1.6; }
.safety-note strong { font-weight:700; }
.safety-pass-banner { margin-top:14px; padding:12px 14px; background:#f0faf5; border-left:3px solid var(--alem-accent); font-size:11px; color:#1a5a3a; line-height:1.6; }
.safety-pass-banner strong { font-weight:700; color:var(--pass-green); }

/* ── METHOD TABLE ── */
.method-strip { margin-top:14px; padding:10px 14px; background:var(--off); border:1px solid var(--border-l); }
.method-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid var(--border-l); font-size:9px; }
.method-row:last-child { border-bottom:none; }
.method-label { color:var(--t-mid); font-weight:600; }
.method-val { font-family:'Space Mono',monospace; color:var(--t-body); font-size:8px; }

/* ── LAB STRIP ── */
.lab-strip { display:flex; justify-content:space-between; align-items:center; padding:14px 28px; background:var(--alem-tint); border-bottom:1px solid var(--border-l); }
.lab-cell { text-align:center; }
.lab-val { font-size:10px; font-weight:600; color:var(--alem-dark); letter-spacing:.3px; }
.lab-lbl { font-size:7px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--t-light); margin-top:2px; display:flex; align-items:center; justify-content:center; gap:4px; }

/* ── FOOTER ── */
.footer { background:var(--alem-dark); padding:20px 28px; display:flex; justify-content:space-between; align-items:center; }
.footer-url { font-size:8px; font-weight:300; letter-spacing:1.5px; color:rgba(255,255,255,.35); margin-top:6px; }
.footer-tagline { font-family:'Nunito',sans-serif; font-style:italic; font-size:13px; font-weight:300; color:rgba(255,255,255,.5); text-align:right; line-height:1.6; }

/* ── ACTION BUTTONS ── */
.ractions { width:100%; max-width:620px; margin-top:14px; display:flex; gap:10px; animation:rise .55s cubic-bezier(.16,1,.3,1) .2s both; }
.rbtn { flex:1; padding:14px 20px; font-family:'Nunito',sans-serif; font-size:8.5px; font-weight:700; letter-spacing:2px; text-transform:uppercase; cursor:pointer; transition:all .15s; border-radius:30px; text-decoration:none; display:inline-block; text-align:center; }
.rbtn-p { background:var(--alem-dark); border:1px solid var(--alem-dark); color:var(--white); }
.rbtn-p:hover { background:var(--alem-mid); }
.rbtn-s { background:var(--white); border:1px solid var(--border); color:var(--t-mid); }
.rbtn-s:hover { border-color:var(--alem-accent); color:var(--alem-dark); }
.page-foot { margin-top:20px; font-size:8px; font-weight:500; letter-spacing:2px; text-transform:uppercase; color:var(--t-faint); text-align:center; animation:rise .55s ease .35s both; }
.page-foot a { color:var(--alem-mid); text-decoration:none; }
/* ── Market Intelligence (Benchmark) ── */
.bm-wrap { margin-top:20px; border-radius:8px; overflow:hidden; border:1px solid #c8dce8; }
.bm-head { background:linear-gradient(135deg,#0d2d3e 0%,#14425e 100%); padding:14px 20px; display:flex; justify-content:space-between; align-items:flex-start; }
.bm-head-left {}
.bm-head-title { font-size:9px; font-weight:700; letter-spacing:3px; text-transform:uppercase; color:#8ecfb0; }
.bm-head-sub { font-size:10px; color:rgba(255,255,255,0.5); margin-top:2px; }
.bm-head-right { text-align:right; flex-shrink:0; }
.bm-head-lib { font-family:'Space Mono',monospace; font-size:16px; font-weight:700; color:#fff; line-height:1; }
.bm-head-lib-label { font-size:8px; letter-spacing:1.5px; text-transform:uppercase; color:rgba(255,255,255,0.4); margin-top:2px; }
.bm-filter-strip { background:#f0f6fa; border-bottom:1px solid #dce8f0; padding:7px 20px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.bm-filter-label { font-size:8px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:var(--t-faint); }
.bm-filter-pill { font-size:9px; font-weight:700; background:#0d2d3e; color:#fff; border-radius:20px; padding:2px 9px; }
.bm-filter-count { font-size:9px; color:var(--t-mid); margin-left:auto; }
.bm-stats { display:grid; grid-template-columns:1fr 1fr; background:#fff; }
.bm-stat { padding:16px 20px; }
.bm-stat + .bm-stat { border-left:1px solid var(--border-l); }
.bm-stat-lbl { font-size:8px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--t-faint); margin-bottom:6px; }
.bm-stat-num { font-family:'Space Mono',monospace; font-size:22px; font-weight:700; color:var(--alem-dark); line-height:1; }
.bm-stat-num-unit { font-size:11px; font-weight:400; }
.bm-track-row { display:flex; align-items:center; gap:8px; margin:8px 0; }
.bm-track { flex:1; height:6px; background:#eef3f6; border-radius:4px; overflow:hidden; }
.bm-fill { height:100%; border-radius:4px; background:#2a8a60; }
.bm-fill-mid { background:var(--gold); }
.bm-fill-low { background:#c8d8e4; }
.bm-rank-pill { font-size:9px; font-weight:700; padding:2px 8px; border-radius:20px; white-space:nowrap; }
.bm-rank-top { background:#e0f4ec; color:#1a6645; }
.bm-rank-mid { background:#fdf3de; color:#8a6100; }
.bm-rank-low { background:#f0f4f6; color:var(--t-mid); }
.bm-stat-vs { font-size:9px; color:var(--t-faint); display:flex; align-items:baseline; gap:4px; flex-wrap:wrap; }
.bm-stat-med { font-weight:600; color:var(--t-mid); }
.bm-stat-up { color:#1a6645; font-weight:700; }
.bm-stat-dn { color:#8a6100; font-weight:700; }
.bm-diff-head { padding:12px 20px 6px; background:#fafaf8; border-top:1px solid var(--border-l); display:flex; align-items:center; gap:8px; }
.bm-diff-head-title { font-size:8px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--t-faint); }
.bm-diff-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(155px,1fr)); gap:0; background:#fafaf8; }
.bm-diff-card { padding:12px 16px 14px; border-top:1px solid var(--border-l); border-right:1px solid var(--border-l); }
.bm-diff-card:last-child { border-right:none; }
.bm-diff-icon { font-size:15px; margin-bottom:5px; }
.bm-diff-name { font-size:10px; font-weight:700; color:var(--alem-dark); margin-bottom:2px; }
.bm-diff-badge { display:inline-block; font-size:8px; font-weight:700; padding:1px 6px; border-radius:10px; margin-bottom:4px; }
.bm-diff-note { font-size:9px; color:var(--t-mid); line-height:1.5; }
.bm-diff-empty { padding:14px 20px; font-size:10px; color:var(--t-faint); font-style:italic; background:#fafaf8; border-top:1px solid var(--border-l); }
.bm-thresholds { background:#fff; border-top:1px solid var(--border-l); padding:10px 20px; display:flex; gap:20px; flex-wrap:wrap; font-size:9px; color:var(--t-faint); }
.bm-thresholds strong { color:var(--t-mid); }
/* ── Strain Intelligence ── */
.si-sec { background:var(--white); border:1px solid var(--border-l); border-radius:8px; padding:18px 20px; margin-top:20px; }
.si-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:14px; }
.si-title { font-size:9px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--alem-dark); }
.si-sub { font-size:9px; color:var(--t-faint); }
/* Match card */
.si-match-card { display:flex; align-items:flex-start; gap:16px; padding:14px 16px; background:var(--alem-tint); border-radius:6px; border:1px solid var(--border); }
.si-match-score { flex-shrink:0; text-align:center; }
.si-match-pct { font-size:28px; font-weight:900; color:var(--alem-dark); font-family:'Space Mono',monospace; line-height:1; }
.si-match-lbl { font-size:8px; letter-spacing:2px; text-transform:uppercase; color:var(--t-light); margin-top:2px; }
.si-match-body { flex:1; }
.si-match-name { font-size:15px; font-weight:700; color:var(--alem-dark); margin-bottom:4px; }
.si-match-meta { display:flex; gap:12px; flex-wrap:wrap; font-size:10px; color:var(--t-mid); }
.si-match-meta span { display:flex; align-items:center; gap:4px; }
.si-cluster-pill { display:inline-block; font-size:8px; font-weight:700; letter-spacing:1px; text-transform:uppercase; padding:2px 8px; border-radius:20px; background:var(--alem-mid); color:#fff; margin-top:6px; }
/* Strain rows (substitutes + complementary) */
.si-rows { display:flex; flex-direction:column; gap:8px; }
.si-row { display:flex; align-items:center; gap:10px; padding:9px 12px; background:var(--off); border-radius:6px; border:1px solid var(--border-l); }
.si-row-name { flex:1; font-size:12px; font-weight:600; color:var(--alem-dark); }
.si-row-pct { flex-shrink:0; font-size:11px; font-weight:700; font-family:'Space Mono',monospace; color:var(--alem-mid); width:42px; text-align:right; }
.si-row-bar-wrap { width:80px; flex-shrink:0; }
.si-row-bar { height:5px; background:var(--border-l); border-radius:3px; overflow:hidden; }
.si-row-bar-fill { height:100%; border-radius:3px; background:var(--alem-accent); }
.si-row-meta { flex-shrink:0; font-size:9px; color:var(--t-mid); text-align:right; min-width:90px; }
.si-row-cluster { flex-shrink:0; font-size:8px; font-weight:700; letter-spacing:1px; text-transform:uppercase; padding:2px 7px; border-radius:20px; background:var(--alem-tint); color:var(--alem-mid); }
.si-divider { border:none; border-top:1px solid var(--border-l); margin:14px 0; }
.si-empty { font-size:10px; color:var(--t-faint); text-align:center; padding:12px; }
/* ── Scientific Evidence ── */
.sci-wrap { margin-top:20px; }
.sci-masthead { background:var(--alem-dark); border-radius:8px 8px 0 0; padding:16px 20px; display:flex; justify-content:space-between; align-items:flex-start; }
.sci-masthead-left {}
.sci-masthead-title { font-size:11px; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:#8ecfb0; }
.sci-masthead-sub { font-size:10px; color:rgba(255,255,255,0.5); margin-top:3px; line-height:1.5; }
.sci-masthead-right { text-align:right; flex-shrink:0; }
.sci-masthead-count { font-family:'Space Mono',monospace; font-size:22px; font-weight:700; color:#fff; line-height:1; }
.sci-masthead-count-label { font-size:8px; letter-spacing:1.5px; text-transform:uppercase; color:rgba(255,255,255,0.4); margin-top:2px; }
.sci-plain-english { background:#f4f8fb; border:1px solid #dce8f0; border-top:none; padding:16px 20px; font-size:11px; color:var(--alem-dark); line-height:1.75; }
.sci-pe-title { font-size:9px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--t-mid); margin-bottom:10px; }
.sci-body { border:1px solid #e0ddd6; border-top:none; border-radius:0 0 8px 8px; background:var(--white); overflow:hidden; }
/* Accordion panels */
.sci-panel { border-bottom:1px solid #f0ece6; }
.sci-panel:last-child { border-bottom:none; }
.sci-panel-summary { padding:13px 18px; cursor:pointer; list-style:none; display:flex; align-items:center; gap:10px; user-select:none; }
.sci-panel-summary::-webkit-details-marker { display:none; }
.sci-panel-icon { font-size:14px; flex-shrink:0; }
.sci-panel-name { flex:1; font-size:12px; font-weight:600; color:var(--alem-dark); }
.sci-panel-areas { display:flex; gap:4px; flex-wrap:wrap; }
.sci-area-tag { font-size:8px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; padding:2px 6px; border-radius:3px; background:var(--alem-tint); color:var(--alem-mid); }
.sci-panel-chevron { font-size:11px; color:var(--t-faint); flex-shrink:0; transition:transform .2s; }
details.sci-panel[open] .sci-panel-chevron { transform:rotate(90deg); }
/* Mechanism note */
.sci-mechanism { background:#f7f9fb; border-left:3px solid var(--alem-accent); padding:9px 14px; margin:0 18px 12px; border-radius:0 4px 4px 0; font-size:10px; color:var(--t-mid); line-height:1.6; font-style:italic; }
/* Article cards */
.sci-articles { padding:0 18px 14px; display:flex; flex-direction:column; gap:10px; }
.sci-card { background:var(--off); border:1px solid #eceae4; border-radius:6px; padding:12px 14px; }
.sci-card-top { display:flex; align-items:flex-start; gap:10px; }
.sci-card-badge { flex-shrink:0; margin-top:2px; }
.sci-badge { font-size:8px; font-weight:700; letter-spacing:0.8px; text-transform:uppercase; padding:2px 7px; border-radius:3px; white-space:nowrap; }
.sci-badge-gold    { background:#fef3c7; color:#92400e; }
.sci-badge-green   { background:#d1fae5; color:#065f46; }
.sci-badge-blue    { background:#dbeafe; color:#1e3a8a; }
.sci-badge-grey    { background:#f3f4f6; color:#374151; }
.sci-badge-faint   { background:#f9fafb; color:#9ca3af; }
.sci-badge-default { background:var(--alem-tint); color:var(--alem-mid); }
.sci-card-body { flex:1; min-width:0; }
.sci-card-title { font-size:12px; font-weight:600; line-height:1.45; margin-bottom:4px; }
.sci-card-title a { color:var(--alem-mid); text-decoration:none; }
.sci-card-title a:hover { color:#16855c; text-decoration:underline; }
.sci-card-meta { font-family:'Space Mono',monospace; font-size:9px; color:var(--t-light); line-height:1.6; }
.sci-card-footer { display:flex; align-items:center; gap:8px; margin-top:7px; flex-wrap:wrap; }
.sci-rel-tag { background:#eef7f3; color:#1a6645; font-family:'Space Mono',monospace; font-size:9px; padding:2px 7px; border-radius:3px; }
.sci-pmid { font-family:'Space Mono',monospace; font-size:8px; color:#c4c0b8; }
/* Empty + footer */
.sci-empty { padding:12px 18px 14px; font-size:10px; color:var(--t-faint); font-style:italic; }
.sci-disclaimer { padding:12px 18px; border-top:1px solid #f0ece6; font-size:9px; color:var(--t-faint); line-height:1.6; font-style:italic; }
.sci-loading { background:var(--white); border:1px solid #e8e4da; border-radius:8px; padding:18px; font-size:10px; color:var(--t-faint); text-align:center; }
</style>
</head>
<body>

<div class="rcard">

  <!-- NAV -->
  <div class="nav">
    <div>
      <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png"
           alt="Alem — Beyond Healthcare Solutions"
           style="height:52px;display:block;"
           onerror="this.style.display='none';this.nextElementSibling.style.display='block'">
      <div style="display:none;color:#fff;font-family:'Nunito',sans-serif;font-weight:800;font-size:17px;letter-spacing:5px;">ALEM</div>
    </div>
    <div class="nav-right">
      <div class="nav-date">${esc(chemistry.coa_report_date || "")}</div>
      <div class="nav-schema">Schema v6.0 · ${esc(chemistry.laboratory_accreditation || "COA Report")}</div>
    </div>
  </div>

  <!-- HERO -->
  <div class="hero">
    <div class="hero-top">
      <div class="hero-left">
        <div class="hero-eye">Certificate of Analysis · Chemical Intelligence Report</div>
        <div class="hero-name">
          ${esc(productName)}
          ${chemistry.product_type ? `<span class="hero-name-sub">(${esc(chemistry.product_type)})</span>` : ""}
        </div>
        <div class="hero-meta">${[chemistry.product_type, chemistry.batch_number ? "Batch " + chemistry.batch_number : "", chemistry.laboratory_name].filter(Boolean).map(esc).join(" &nbsp;·&nbsp; ")}</div>
        <div class="hero-narrative">${esc(heroNarrative)}</div>
      </div>
      <div class="score-block">
        <div class="score-ring">
          <svg width="130" height="130" viewBox="0 0 130 130">
            <circle cx="65" cy="65" r="56" fill="none" stroke="var(--border)" stroke-width="7"/>
            <circle cx="65" cy="65" r="56" fill="none" stroke="#8ecfb0" stroke-width="7"
              stroke-dasharray="${ringCirc}" stroke-dashoffset="${ringOffset}"
              stroke-linecap="round"/>
          </svg>
          <div class="score-inner">
            <div class="score-n">${safeTotal}</div>
            <div class="score-d">/100</div>
          </div>
        </div>
        <div class="score-grade">${esc(safeGrade)} &nbsp; Grade</div>
        <div class="score-tier">${esc(safeTier)}</div>
      </div>
    </div>
  </div>

  <!-- SCORE BREAKDOWN -->
  <div class="score-bd">
    ${sbRow("Potency", potencyPct >= 75 ? "f-g" : potencyPct >= 45 ? "f-o" : "f-r", potencyPct, 0.06, `${safePotency.score} / ${safePotency.max}`,
      `Potency Score: ${safePotency.score}/${safePotency.max}`,
      `THC Total of ${chemistry.thc_total || "N/A"} wt% (anhydrous: ${chemistry.thc_total_anhydrous || "N/A"} wt%). THCA at ${chemistry.thca || "N/A"} wt% is the primary precursor — multiply by 0.877 to estimate post-decarboxylation THC.`,
      `Max potency score (25pts) requires THC total above 28 wt%. Score tiers: ≥28=25, ≥24=22, ≥20=18, ≥15=13, >0=8.`,
      `Look for "Total THC" or calculate: (THCA × 0.877) + D9-THC. Two values may appear: anhydrous (moisture-corrected) and as-received.`)}

    ${sbRow("Terpenes", terpPct >= 75 ? "f-g" : terpPct >= 45 ? "f-o" : "f-r", terpPct, 0.12, `${safeTerpenes.score} / ${safeTerpenes.max}`,
      `Terpene Score: ${safeTerpenes.score}/${safeTerpenes.max}`,
      `Total terpene content of ${chemistry.total_terpenes || "N/A"} wt% with ${terpCount} compounds above LOQ detected${blqCount > 0 ? ` and ${blqCount} at BLQ` : ""}. Score improves with both density (wt%) and breadth (number of compounds).`,
      `Tiers: ≥3.5 wt%=22, ≥2.5=20, ≥1.8=16, ≥1.0=11. Breadth bonus: +5 for ≥10 compounds, +3 for ≥6.`,
      `Find the "Terpene Analysis" section. The total is usually listed as "Total of all quantified terpenes".`)}

    ${sbRow("Minor Cannabinoids", minorPct >= 75 ? "f-g" : minorPct >= 45 ? "f-o" : "f-r", minorPct, 0.18, `${safeMinors.score} / ${safeMinors.max}`,
      `Minor Cannabinoid Score: ${safeMinors.score}/${safeMinors.max}`,
      `Detected minors: ${[
        cbna > 0 ? `CBNA: ${cbna.toFixed(4)} wt%` : "",
        toNum(chemistry.cbg) > 0 ? `CBG: ${toNum(chemistry.cbg).toFixed(4)} wt%` : "",
        toNum(chemistry.cbca) > 0 ? `CBCA: ${toNum(chemistry.cbca).toFixed(4)} wt%` : "",
        toNum(chemistry.thcva) > 0 ? `THCVA: ${toNum(chemistry.thcva).toFixed(4)} wt%` : "",
      ].filter(Boolean).join(", ") || "None beyond THCA and CBN"}. Minor cannabinoids contribute to the entourage effect.`,
      `Score based on sum of minors (total - THC - CBD) and count. CBN ≥1% applies a 4-point degradation penalty.`,
      `Look for all rows beyond THCA/D9-THC/CBD/CBDA in the cannabinoid table.`)}

    ${sbRow("Safety Data", safetyPct >= 75 ? "f-g" : safetyPct >= 45 ? "f-o" : "f-r", safetyPct, 0.24, `${safeSafety.score} / ${safeSafety.max}`,
      `Safety Score: ${safeSafety.score}/${safeSafety.max}`,
      `Pesticides: ${contaminants.pesticides_status || "Not tested"} · Metals: ${contaminants.heavy_metals_status || "Not tested"} · Microbials: ${contaminants.microbials_status || "Not tested"} · Mycotoxins: ${contaminants.mycotoxins_status || "Not tested"}. ISO 17025: ${isoPass ? "Confirmed" : "Not confirmed"}. SCC: ${sccPass ? "Confirmed" : "Not confirmed"}.`,
      `6pts pesticides, 5pts metals, 5pts microbials, 2pts mycotoxins. +2 ISO 17025, +2 SCC, +2 full panel bonus.`,
      `Look for: Pesticide Analysis, Heavy Metals, Microbiological, and Mycotoxin sections. All four panels present = maximum safety score potential.`)}

    ${sbRow("Data Completeness", dataPct >= 75 ? "f-g" : dataPct >= 45 ? "f-o" : "f-r", dataPct, 0.30, `${safeData.score} / ${safeData.max}`,
      `Data Completeness: ${safeData.score}/${safeData.max}`,
      `Checks 10 mandatory fields: THC total (2pts), total terpenes (2pts), ≥5 terpenes reported (2pts), ≥3 cannabinoids (1pt), report date (1pt), lab name (1pt), batch number (1pt).`,
      `A complete submission scores all 10 points. Missing fields reduce completeness.`,
      `A complete COA includes: lab name, accreditation, batch ID, report date, product type, cannabinoid table, terpene table, and method references.`)}
  </div>

  <!-- CHEMOTYPE BAND -->
  <div class="ct-band">
    <div class="ct-code">
      ${esc(fingerprintId)}
      ${infoIcon(`<span class="tooltip"><strong>Chemotype Fingerprint: ${esc(fingerprintId)}</strong><span class="tt-body">The top-3 dominant terpenes in order of concentration. This fingerprint clusters products by chemical identity.</span><span class="tt-coa"><strong>Derived from:</strong> We rank all reported terpenes by wt% and take the top 3. Each terpene is abbreviated (CAR = Trans-Caryophyllene, LIM = Limonene, FAR = Farnesene, HUM = Alpha-Humulene, etc.).</span></span>`, "tt-right")}
    </div>
    <div class="ct-sep"></div>
    <div>
      <div class="ct-lbl">Chemotype fingerprint</div>
      <div class="ct-lin">${esc(terpIntel.lineage)}</div>
    </div>
    <div class="ct-right">
      <div class="ct-effect">
        ☀ &nbsp;${esc(terpIntel.direction)}
        ${infoIcon(`<span class="tooltip"><strong>Effect Direction: ${esc(terpIntel.direction)}</strong><span class="tt-body">Inferred from the terpene chemotype using observational data — not from clinical trials on this specific batch. Individual responses vary significantly.</span><span class="tt-coa"><strong>Important:</strong> This is a population-level directional signal. Always consult a healthcare professional for personalised guidance.</span></span>`, "tt-right")}
      </div>
      <div class="ct-conf">
        ${esc(terpIntel.lineageConfidence)} confidence
        ${infoIcon(`<span class="tooltip"><strong>Lineage Confidence: ${esc(terpIntel.lineageConfidence)}</strong><span class="tt-body">Confidence reflects how closely this chemotype matches known genetic lineages in our reference database.</span><span class="tt-coa"><strong>How to improve it:</strong> Submit multiple COA batches from the same cultivar. Consistent chemotypes across batches increase lineage confidence.</span></span>`, "tt-right")}
      </div>
    </div>
  </div>

  <!-- KEY METRICS -->
  <div class="metrics">
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.thc_total || "ND")}<span class="m-unit">${chemistry.thc_total && chemistry.thc_total !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">THC Total</div>
      <div class="m-ctx">${thc >= 24 ? "↑ above avg." : thc >= 18 ? "moderate-high" : thc > 0 ? "moderate" : "not detected"}</div>
      ${infoIcon(`<span class="tooltip"><strong>THC Total: ${esc(chemistry.thc_total || "ND")} wt% (anhydrous: ${esc(chemistry.thc_total_anhydrous || "—")} wt%)</strong><span class="tt-body">As-received value shown. Anhydrous (moisture-corrected) value: ${esc(chemistry.thc_total_anhydrous || "—")} wt%. Market average for premium dried flower is 18–22 wt%.</span><span class="tt-coa"><strong>Check on your COA:</strong> THC Total = (THCA × 0.877) + D9-THC. Anhydrous values use moisture-corrected weights.</span></span>`, "m-info")}
    </div>
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.thca || "ND")}<span class="m-unit">${chemistry.thca && chemistry.thca !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">THCA</div>
      <div class="m-ctx">×0.877 on heat</div>
      ${infoIcon(`<span class="tooltip"><strong>THCA: ${esc(chemistry.thca || "ND")} wt%</strong><span class="tt-body">Raw, non-psychoactive acid form. Decarboxylates to active THC when heated. Multiply THCA × 0.877 to estimate active THC produced.</span><span class="tt-coa"><strong>Check on your COA:</strong> Listed as "THCA" or "THCA-A". Typically the largest value in the cannabinoid profile for potent flower.</span></span>`, "m-info")}
    </div>
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.total_terpenes || "ND")}<span class="m-unit">${chemistry.total_terpenes && chemistry.total_terpenes !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">Terpenes</div>
      <div class="m-ctx">${terps >= 3 ? "strong" : terps >= 2 ? "top 20%" : terps > 0 ? "moderate" : "not reported"}</div>
      ${infoIcon(`<span class="tooltip"><strong>Total Terpenes: ${esc(chemistry.total_terpenes || "ND")} wt% · ${terpCount} compounds</strong><span class="tt-body">Total of all quantified terpenes. ${terpCount} compounds detected above LOQ${blqCount > 0 ? `, ${blqCount} at BLQ` : ""}. Above 2.0 wt% is considered aromatic and complex. Above 3.0 wt% is exceptional.</span><span class="tt-coa"><strong>Check on your COA:</strong> Look for "Total of all quantified terpenes" at the end of the terpene table.</span></span>`, "m-info")}
    </div>
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.cbd_total || "ND")}<span class="m-unit">${chemistry.cbd_total && chemistry.cbd_total !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">CBD Total</div>
      <div class="m-ctx ${cbd < 0.5 ? "nd" : ""}">${cbd >= 0.5 ? "modulating" : "not detected"}</div>
      ${infoIcon(`<span class="tooltip"><strong>CBD Total: ${esc(chemistry.cbd_total || "ND")} wt%</strong><span class="tt-body">CBD may modulate the intensity of THC effects when both are present. In this product, CBD is ${cbd >= 0.5 ? `present at ${chemistry.cbd_total} wt%.` : "absent — meaning no built-in CBD buffer."}</span><span class="tt-coa"><strong>Check on your COA:</strong> CBD Total = (CBDA × 0.877) + CBD. If both read ND, CBD is absent.</span></span>`, "m-info")}
    </div>
  </div>

  <!-- SUMMARY -->
  <div class="summary">
    <div class="summary-lbl">chemical intelligence summary</div>
    <div class="summary-body">${esc(heroNarrative)}</div>
  </div>

  <!-- MARKET BENCHMARK -->
  ${(() => {
    if (!benchmark) return "";
    const { n, formFactorLabel, thcPercentile, terpPercentile, medianThc, medianTerp, p90Thc, p90Terp, supplierCount } = benchmark;
    const thcVal  = parseFloat(chemistry.thc_total)      || 0;
    const cbdVal  = parseFloat(chemistry.cbd_total)       || 0;
    const terpVal = parseFloat(chemistry.total_terpenes) || 0;
    const thcTier = thcVal >= 28 ? "Very High THC" : thcVal >= 22 ? "High THC" : thcVal >= 15 ? "Mid THC" : "Low THC";
    const cbdTier = cbdVal >= 5  ? "High CBD" : cbdVal >= 1 ? "CBD-Present" : "Low CBD";

    // ── Percentile helpers ────────────────────────────────────────────────
    function rankPill(pct) {
      const top = Math.max(1, 100 - pct);
      const cls = top <= 20 ? "bm-rank-top" : top <= 50 ? "bm-rank-mid" : "bm-rank-low";
      return `<span class="bm-rank-pill ${cls}">Top ${top}%</span>`;
    }
    function fillCls(pct) {
      const top = 100 - pct;
      return top <= 20 ? "" : top <= 50 ? "bm-fill-mid" : "bm-fill-low";
    }
    function deltaHtml(actual, median) {
      if (!median || !actual) return "";
      const d = ((actual - median) / median * 100).toFixed(1);
      if (d > 0) return `<span class="bm-stat-up">↑${d}% vs median</span>`;
      if (d < 0) return `<span class="bm-stat-dn">↓${Math.abs(d)}% vs median</span>`;
      return "";
    }

    // ── Rare terpene detection ────────────────────────────────────────────
    const RARE_TERPS = {
      "alpha-bisabolol":{"label":"α-Bisabolol","note":"Anti-inflammatory & skin-penetrating — found in <15% of COAs"},
      "bisabolol":      {"label":"α-Bisabolol","note":"Anti-inflammatory & skin-penetrating — found in <15% of COAs"},
      "terpinolene":    {"label":"Terpinolene","note":"Jack Herer lineage marker — uplifting, antifungal, found in ~20% of COAs"},
      "ocimene":        {"label":"Ocimene","note":"Sweet & herbaceous — antiviral properties, found in <20% of COAs"},
      "nerolidol":      {"label":"Nerolidol","note":"Deep sedative terpene — enhances transdermal absorption, found in <12% of COAs"},
      "guaiol":         {"label":"Guaiol","note":"Piney, woody — anti-inflammatory, found in <10% of COAs"},
      "geraniol":       {"label":"Geraniol","note":"Rose-like floral — neuroprotective, found in <10% of COAs"},
      "valencene":      {"label":"Valencene","note":"Sweet citrus — extremely rare in cannabis, found in <5% of COAs"},
      "sabinene":       {"label":"Sabinene","note":"Spicy & woody — antioxidant, found in <12% of COAs"},
      "eucalyptol":     {"label":"Eucalyptol","note":"Minty & cooling — bronchodilatory, found in <10% of COAs"},
      "1,8-cineole":    {"label":"Eucalyptol","note":"Minty & cooling — bronchodilatory, found in <10% of COAs"},
      "camphene":       {"label":"Camphene","note":"Earthy & musky — antioxidant, found in <15% of COAs"},
      "delta-3-carene": {"label":"δ-3-Carene","note":"Sweet & earthy — bone metabolism research, found in <15% of COAs"},
      "phytol":         {"label":"Phytol","note":"Sedating terpene alcohol — antioxidant, found in <15% of COAs"},
      "fenchol":        {"label":"Fenchol","note":"Pine & lime — antimicrobial, found in <12% of COAs"},
    };
    const rareTerpsFound = (chemistry.top_terpenes || [])
      .filter(t => parseFloat(t.value) >= 0.05 && RARE_TERPS[String(t.name||"").toLowerCase().trim()])
      .map(t => ({ ...RARE_TERPS[String(t.name||"").toLowerCase().trim()], pct: parseFloat(t.value).toFixed(3) }))
      .slice(0, 2);

    // ── Minor cannabinoid detection ───────────────────────────────────────
    const MINOR_CANN = {
      cbg:  {label:"CBG",name:"Cannabigerol",note:"The 'Mother Cannabinoid'",sub:"Neuroprotective · Antibacterial · Anti-inflammatory",bg:"#e0f4ec",fg:"#1a6645"},
      cbga: {label:"CBGA",name:"Cannabigerolic Acid",note:"CBG precursor — converts on heat",sub:"Neuroprotective precursor",bg:"#e8f4ee",fg:"#1a6645"},
      cbc:  {label:"CBC",name:"Cannabichromene",note:"Third-most studied cannabinoid",sub:"Neurogenesis · Mood · Anti-inflammatory",bg:"#edf2fc",fg:"#2c52b0"},
      cbca: {label:"CBCA",name:"Cannabichromenic Acid",note:"CBC precursor",sub:"Anti-inflammatory precursor",bg:"#edf2fc",fg:"#2c52b0"},
      cbn:  {label:"CBN",name:"Cannabinol",note:"Craft curing indicator — THC oxidation product",sub:"Sedating · Analgesic · Appetite",bg:"#fdf3de",fg:"#8a6100"},
      cbna: {label:"CBNA",name:"Cannabinolic Acid",note:"CBN precursor",sub:"Sedating precursor",bg:"#fdf3de",fg:"#8a6100"},
      thcv: {label:"THCV",name:"Tetrahydrocannabivarin",note:"The 'Sports Car' cannabinoid — rare",sub:"Fast-acting · Appetite suppressant · Energising",bg:"#f5e8fc",fg:"#6b1a8a"},
      thcva:{label:"THCVA",name:"THCV Acid",note:"THCV precursor",sub:"Rare — converts on heat",bg:"#f5e8fc",fg:"#6b1a8a"},
    };
    const minorCanns = Object.entries(MINOR_CANN)
      .map(([key, info]) => {
        const v = parseFloat(chemistry[key]) || 0;
        return v >= 0.05 ? { ...info, pct: v.toFixed(3) } : null;
      })
      .filter(Boolean)
      .slice(0, 2);

    // ── Terpene complexity ────────────────────────────────────────────────
    const terpCount2 = (chemistry.top_terpenes || []).filter(t => parseFloat(t.value) >= 0.05).length;
    const complexityLabel = terpCount2 >= 20 ? ["Exceptional","Top 3% for complexity","#e0f4ec","#1a6645"]
                          : terpCount2 >= 15 ? ["High Complexity","Top 10% for profile depth","#edf2fc","#2c52b0"]
                          : terpCount2 >= 10 ? ["Rich Profile","Above-average diversity","#fdf3de","#8a6100"]
                          : ["Standard Profile","Typical terpene diversity","#f0f4f6","#5a7080"];

    // ── Flavonoids ────────────────────────────────────────────────────────
    const flavsDetected = (chemistry.flavonoids || []).filter(f => f.value && String(f.value) !== "ND" && parseFloat(f.value) > 0).length;

    // ── Differentiator cards ──────────────────────────────────────────────
    const diffCards = [];
    for (const rt of rareTerpsFound) {
      diffCards.push(`<div class="bm-diff-card">
        <div class="bm-diff-icon">✦</div>
        <div class="bm-diff-name">${esc(rt.label)}</div>
        <div><span class="bm-diff-badge" style="background:#fdf3de;color:#8a6100;">Rare Terpene · ${rt.pct}%</span></div>
        <div class="bm-diff-note">${esc(rt.note)}</div>
      </div>`);
    }
    for (const mc of minorCanns) {
      diffCards.push(`<div class="bm-diff-card">
        <div class="bm-diff-icon">◆</div>
        <div class="bm-diff-name">${esc(mc.label)} <span style="font-weight:400;font-size:9px;color:var(--t-mid)">${mc.pct}%</span></div>
        <div><span class="bm-diff-badge" style="background:${mc.bg};color:${mc.fg};">${esc(mc.name)}</span></div>
        <div class="bm-diff-note">${esc(mc.note)}<br><span style="font-size:8px;color:var(--t-faint)">${esc(mc.sub)}</span></div>
      </div>`);
    }
    diffCards.push(`<div class="bm-diff-card">
      <div class="bm-diff-icon">◈</div>
      <div class="bm-diff-name">${terpCount2} Terpenes Detected</div>
      <div><span class="bm-diff-badge" style="background:${complexityLabel[2]};color:${complexityLabel[3]};">${esc(complexityLabel[0])}</span></div>
      <div class="bm-diff-note">${esc(complexityLabel[1])}</div>
    </div>`);
    if (flavsDetected > 0) {
      diffCards.push(`<div class="bm-diff-card">
        <div class="bm-diff-icon">❋</div>
        <div class="bm-diff-name">${flavsDetected} Flavonoid${flavsDetected > 1 ? "s" : ""} Detected</div>
        <div><span class="bm-diff-badge" style="background:#fce8f0;color:#8a1a4a;">Flavonoids Present</span></div>
        <div class="bm-diff-note">Cannaflavins &amp; plant polyphenols — anti-inflammatory, potentiate cannabinoid activity</div>
      </div>`);
    }

    return `<div class="bm-wrap">
      <div class="bm-head">
        <div class="bm-head-left">
          <div class="bm-head-title">Market Intelligence</div>
          <div class="bm-head-sub">Benchmarked against our 600K+ COA library · last 2 years${supplierCount ? " · " + supplierCount.toLocaleString() + " licensed suppliers" : ""}</div>
        </div>
        <div class="bm-head-right">
          <div class="bm-head-lib">600K<span style="font-size:11px;font-weight:400">+</span></div>
          <div class="bm-head-lib-label">COAs analysed</div>
        </div>
      </div>
      <div class="bm-filter-strip">
        <span class="bm-filter-label">Peer group</span>
        <span class="bm-filter-pill">${esc(formFactorLabel)}</span>
        <span class="bm-filter-pill">${thcTier}</span>
        <span class="bm-filter-pill">${cbdTier}</span>
        <span class="bm-filter-count">${n.toLocaleString()} matching COAs</span>
      </div>
      <div class="bm-stats">
        <div class="bm-stat">
          <div class="bm-stat-lbl">THC Potency Rank</div>
          <div class="bm-stat-num">${esc(String(thcVal))}<span class="bm-stat-num-unit">%</span></div>
          <div class="bm-track-row">
            <div class="bm-track"><div class="bm-fill ${fillCls(thcPercentile)}" style="width:${thcPercentile}%"></div></div>
            ${rankPill(thcPercentile)}
          </div>
          <div class="bm-stat-vs">
            <span class="bm-stat-med">${medianThc != null ? medianThc + "% market median" : "—"}</span>
            ${deltaHtml(thcVal, medianThc)}
          </div>
        </div>
        ${terpPercentile != null ? `<div class="bm-stat">
          <div class="bm-stat-lbl">Terpene Richness Rank</div>
          <div class="bm-stat-num">${esc(String(terpVal))}<span class="bm-stat-num-unit">%</span></div>
          <div class="bm-track-row">
            <div class="bm-track"><div class="bm-fill ${fillCls(terpPercentile)}" style="width:${terpPercentile}%"></div></div>
            ${rankPill(terpPercentile)}
          </div>
          <div class="bm-stat-vs">
            <span class="bm-stat-med">${medianTerp != null ? medianTerp + "% market median" : "—"}</span>
            ${deltaHtml(terpVal, medianTerp)}
          </div>
        </div>` : ""}
      </div>
      <div class="bm-diff-head">
        <div class="bm-diff-head-title">What makes this product stand out</div>
      </div>
      <div class="bm-diff-grid">${diffCards.join("")}</div>
      ${(p90Thc != null || p90Terp != null) ? `<div class="bm-thresholds">
        ${p90Thc  != null ? `<span>Top 10% THC: <strong>≥${p90Thc}%</strong></span>` : ""}
        ${p90Terp != null ? `<span>Top 10% terpenes: <strong>≥${p90Terp}%</strong></span>` : ""}
      </div>` : ""}
    </div>`;
  })()}

  <!-- SCIENTIFIC EVIDENCE (moved above terpene fingerprint) -->
  ${(() => {
    if (sciLoading) return `<div class="sci-loading">⏳ Scientific context is loading — refresh in a moment to view citations.</div>`;
    if (!scientificEvidence) return "";

    const { terpeneStudies=[], cannabinoidStudies=[], indicationStudies=[], strainFamilyStudies=[], safetyStudies=[], totalArticles=0, queriesRun=0 } = scientificEvidence;

    const BADGE_CLASS = { gold:"sci-badge-gold", green:"sci-badge-green", blue:"sci-badge-blue", grey:"sci-badge-grey", faint:"sci-badge-faint", default:"sci-badge-default" };
    const PANEL_ICONS = { terpene:"🧪", cannabinoid:"⚗️", indication:"🏥", strain:"🌿", safety:"🛡️" };

    function qualityBadge(a) {
      const q = a.quality || { label:"Research Article", color:"default" };
      return `<span class="sci-badge ${BADGE_CLASS[q.color]||'sci-badge-default'}">${esc(q.label)}</span>`;
    }

    function articleCard(a) {
      const meta = [a.authors, a.journal, a.year].filter(Boolean).map(esc).join(" · ");
      return `
      <div class="sci-card">
        <div class="sci-card-top">
          <div class="sci-card-badge">${qualityBadge(a)}</div>
          <div class="sci-card-body">
            <div class="sci-card-title"><a href="${esc(a.url)}" target="_blank" rel="noopener noreferrer">${esc(a.title)}</a></div>
            <div class="sci-card-meta">${meta}</div>
            <div class="sci-card-footer">
              ${a.relevanceHint ? `<span class="sci-rel-tag">${esc(a.relevanceHint)}</span>` : ""}
              <span class="sci-pmid">PMID&nbsp;${esc(a.pmid)}</span>
            </div>
          </div>
        </div>
      </div>`;
    }

    function panel(icon, title, articles, mechanismNote, therapeuticAreas=[], openByDefault=false) {
      const count = articles.length;
      const areaHtml = therapeuticAreas.map(a => `<span class="sci-area-tag">${esc(a)}</span>`).join("");
      return `
      <details class="sci-panel"${openByDefault ? " open" : ""}>
        <summary class="sci-panel-summary">
          <span class="sci-panel-icon">${icon}</span>
          <span class="sci-panel-name">${esc(title)} <span style="font-weight:400;color:var(--t-mid);font-size:10px;">(${count} ${count===1?"study":"studies"})</span></span>
          <span class="sci-panel-areas">${areaHtml}</span>
          <span class="sci-panel-chevron">›</span>
        </summary>
        ${mechanismNote ? `<div class="sci-mechanism">${esc(mechanismNote)}</div>` : ""}
        ${count > 0
          ? `<div class="sci-articles">${articles.map(articleCard).join("")}</div>`
          : `<div class="sci-empty">No high-quality studies retrieved for this query. This may reflect limited research on this specific compound combination.</div>`
        }
      </details>`;
    }

    const terpPanels = terpeneStudies
      .filter(g => g.articles.length > 0)
      .map((g, i) => panel(PANEL_ICONS.terpene, `${g.terpene} — Pharmacology & Clinical Evidence`, g.articles, g.mechanismNote, g.therapeuticAreas || [], i === 0))
      .join("");

    const allPanels = [
      terpPanels,
      cannabinoidStudies.length  ? panel(PANEL_ICONS.cannabinoid, "Cannabinoid Interactions & Entourage Effect", cannabinoidStudies, null, ["Cannabinoid Science"])  : "",
      indicationStudies.length   ? panel(PANEL_ICONS.indication,  "Therapeutic Indication Research",            indicationStudies,  null, ["Clinical Evidence"])    : "",
      strainFamilyStudies.length ? panel(PANEL_ICONS.strain,      "Strain Family Research",                     strainFamilyStudies,null, ["Strain Science"])        : "",
      safetyStudies.length       ? panel(PANEL_ICONS.safety,      "Safety, Tolerability & Drug Interactions",   safetyStudies,      null, ["Safety"])                : "",
    ].filter(Boolean).join("");

    if (!allPanels) return "";

    const allArts = [
      ...terpeneStudies.flatMap(g=>g.articles),
      ...cannabinoidStudies, ...indicationStudies, ...strainFamilyStudies, ...safetyStudies,
    ];
    const highQuality = allArts.filter(a => (a.quality?.tier||0) >= 3).length;

    // ── Plain-English summary ─────────────────────────────────────────────
    const activeTerpGroups = terpeneStudies.filter(g => g.articles.length > 0);
    const terpNames  = activeTerpGroups.map(g => g.terpene).join(", ");
    const allAreas   = [...new Set(activeTerpGroups.flatMap(g => g.therapeuticAreas || []))].slice(0, 4);
    const areasText  = allAreas.length
      ? allAreas.slice(0, -1).join(", ") + (allAreas.length > 1 ? " and " + allAreas[allAreas.length - 1] : allAreas[0])
      : "multiple therapeutic areas";

    const thcNum = parseFloat(chemistry.thc_total) || 0;
    const cbdNum = parseFloat(chemistry.cbd_total)  || 0;
    let cannabSummary;
    if (cbdNum >= 5) {
      cannabSummary = `The high CBD content (${cbdNum}%) may moderate psychoactive effects and independently contributes analgesic and anti-inflammatory activity, as supported by cannabinoid interaction studies below.`;
    } else if (cbdNum >= 1) {
      cannabSummary = `The THC:CBD ratio of approximately ${(thcNum / cbdNum).toFixed(0)}:1 suggests a balanced profile where CBD may soften the intensity of THC while extending the therapeutic window (see Entourage Effect research below).`;
    } else {
      cannabSummary = `At ${thcNum}% THC with minimal CBD, the effects are largely driven by THC and the terpene matrix — the combination of these compounds is studied as the "entourage effect."`;
    }

    let terpSummary = "";
    if (activeTerpGroups.length > 0) {
      const mechNotes = activeTerpGroups.filter(g => g.mechanismNote).map(g => `<strong>${g.terpene}</strong> — ${g.mechanismNote}`).join(" ");
      terpSummary = `<p style="margin:0 0 8px;">The dominant terpenes in this profile — <strong>${terpNames}</strong> — have been independently studied for their roles in <strong>${areasText}</strong>. ${mechNotes}</p>`;
    }

    const plainEnglish = `
    <div class="sci-plain-english">
      <div class="sci-pe-title">What the research says about this product</div>
      ${terpSummary}
      <p style="margin:0">${cannabSummary} The ${totalArticles} studies below are automatically matched to this specific chemical fingerprint — expand each section to read the full citations.</p>
    </div>`;

    return `
    <div class="sci-wrap">
      <div class="sci-masthead">
        <div class="sci-masthead-left">
          <div class="sci-masthead-title">Scientific Evidence Layer</div>
          <div class="sci-masthead-sub">Peer-reviewed literature auto-matched to this chemical profile<br>Source: PubMed / NCBI National Library of Medicine · ${queriesRun} queries executed</div>
        </div>
        <div class="sci-masthead-right">
          <div class="sci-masthead-count">${totalArticles}</div>
          <div class="sci-masthead-count-label">citations${highQuality > 0 ? `\n${highQuality} clinical/review` : ""}</div>
        </div>
      </div>
      ${plainEnglish}
      <div class="sci-body">
        ${allPanels}
        <div class="sci-disclaimer">Evidence summaries are provided for informational purposes only and do not constitute medical advice. Citations are automatically retrieved and matched based on chemical profile similarity. Study quality indicators (Meta-Analysis, Systematic Review, Clinical Trial, Preclinical) are inferred from publication titles. Always consult current clinical guidelines and a qualified healthcare professional.</div>
      </div>
    </div>`;
  })()}

  <!-- TERPENE FINGERPRINT -->
  <div class="sec">
    <div class="sec-head">
      <div class="sec-title">terpene fingerprint ${infoIcon(`<span class="tooltip"><strong>What are Terpenes?</strong><span class="tt-body">Terpenes are aromatic compounds produced by the cannabis plant. They shape scent, flavour, and may modulate the therapeutic effects of cannabinoids via the Entourage Effect.</span><span class="tt-coa"><strong>Entourage Effect:</strong> Terpenes and cannabinoids may work synergistically — the combination produces different outcomes than either compound alone.</span></span>`, "sec-info")}</div>
      <div class="sec-badge">${esc(chemistry.total_terpenes || "—")} wt% · ${terpCount} detected${blqCount > 0 ? ` · ${blqCount} BLQ` : ""}</div>
    </div>
    ${leadTerpene ? `<div class="terp-intro"><strong>${esc(leadTerpene)}-dominant.</strong> ${esc(TERPENE_EDUCATION[leadTerpene]?.aroma || terpIntel.note)}</div>` : ""}
    <div class="terp-wrap">
      <!-- Radar SVG -->
      <svg width="150" height="150" viewBox="0 0 150 150" style="flex-shrink:0">
        <defs>
          <radialGradient id="rg${esc(options.documentId || "x")}" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#0d2d3e" stop-opacity=".18"/>
            <stop offset="100%" stop-color="#1a4a62" stop-opacity=".02"/>
          </radialGradient>
        </defs>
        <polygon points="75,10 132,44 132,106 75,140 18,106 18,44" fill="none" stroke="var(--border-l)" stroke-width="1"/>
        <polygon points="75,30 112,56 112,94 75,120 38,94 38,56" fill="none" stroke="var(--border-l)" stroke-width="1"/>
        <polygon points="75,50 92,68 92,82 75,100 58,82 58,68" fill="none" stroke="var(--border-l)" stroke-width="1"/>
        <line x1="75" y1="10" x2="75" y2="140" stroke="var(--border-l)" stroke-width=".8"/>
        <line x1="18" y1="44" x2="132" y2="106" stroke="var(--border-l)" stroke-width=".8"/>
        <line x1="132" y1="44" x2="18" y2="106" stroke="var(--border-l)" stroke-width=".8"/>
        ${activeTerps.slice(0, 6).length >= 3 ? (() => {
          const pts = activeTerps.slice(0, 6).map((t, i) => {
            const v = toNum(t.value);
            const frac = Math.max(0.1, Math.min(1, v / maxTerpVal));
            const angles = [270, 30, 90, 150, 210, 330].map(a => a * Math.PI / 180);
            const angle = angles[i] || (i * 60 * Math.PI / 180);
            const r = 10 + frac * 55;
            return `${(75 + r * Math.cos(angle)).toFixed(1)},${(75 + r * Math.sin(angle)).toFixed(1)}`;
          });
          return `<polygon points="${pts.join(" ")}" fill="url(#rg${esc(options.documentId || "x")})" stroke="var(--alem-dark)" stroke-width="1.5" stroke-linejoin="round"/>
        ${pts.map((p, i) => `<circle cx="${p.split(",")[0]}" cy="${p.split(",")[1]}" r="${i === 0 ? 4 : 2.5}" fill="var(--alem-dark)" opacity="${(1 - i * 0.12).toFixed(2)}"/>`).join("\n        ")}`;
        })() : ""}
        <text x="75"  y="7"   text-anchor="middle" fill="var(--t-light)" font-size="7" font-family="Space Mono,monospace">${esc(activeTerps[0]?.name?.slice(0,4).toUpperCase() || "T1")}</text>
        <text x="138" y="47"  text-anchor="start"  fill="var(--t-light)" font-size="7" font-family="Space Mono,monospace">${esc(activeTerps[1]?.name?.slice(0,4).toUpperCase() || "T2")}</text>
        <text x="138" y="110" text-anchor="start"  fill="var(--t-light)" font-size="7" font-family="Space Mono,monospace">${esc(activeTerps[2]?.name?.slice(0,4).toUpperCase() || "T3")}</text>
        <text x="75"  y="148" text-anchor="middle" fill="var(--t-light)" font-size="7" font-family="Space Mono,monospace">${esc(activeTerps[3]?.name?.slice(0,4).toUpperCase() || "T4")}</text>
        <text x="12"  y="110" text-anchor="end"    fill="var(--t-light)" font-size="7" font-family="Space Mono,monospace">${esc(activeTerps[4]?.name?.slice(0,4).toUpperCase() || "T5")}</text>
        <text x="12"  y="47"  text-anchor="end"    fill="var(--t-light)" font-size="7" font-family="Space Mono,monospace">${esc(activeTerps[5]?.name?.slice(0,4).toUpperCase() || "T6")}</text>
      </svg>
      <div class="t-bars">
        ${terpenes.length ? terpenes.map((t, i) => terpBar(t, i)).join("") : "<div style='color:var(--t-faint);font-size:11px;'>No terpene data reported.</div>"}
      </div>
    </div>
    <div class="terp-footer">
      <span class="terp-footer-txt">Total of all quantified terpenes</span>
      <span class="terp-footer-val">${esc(chemistry.total_terpenes || "—")} wt%</span>
    </div>
  </div>

  <!-- CANNABINOID PROFILE -->
  <div class="sec">
    <div class="sec-head">
      <div class="sec-title">cannabinoid profile ${infoIcon(`<span class="tooltip"><strong>The Entourage Effect</strong><span class="tt-body">Cannabinoids and terpenes work synergistically — producing different therapeutic outcomes together than in isolation. THC alone differs from THC alongside CBNA, and a rich terpene profile.</span><span class="tt-coa"><strong>Clinical implication:</strong> A product with multiple detected minor cannabinoids produces more nuanced effects. The breadth of minors here is a positive signal for entourage complexity.</span></span>`, "sec-info")}</div>
      <div class="sec-badge">${esc(chemistry.total_cannabinoids || "—")} wt% total</div>
    </div>
    <div class="cann-grid">
      ${cannabinoids.length ? cannabinoids.map(c => {
        const isHL = ["CBGA"].includes(c.name) && toNum(c.value) > 0;
        const noteMap = {
          "THCA": "Precursor · ×0.877 on heat",
          "D9-THC": "Active form",
          "CBNA": "Degradation marker (acid)",
          "CBN": "Degradation marker",
          "CBGA": "Mother cannabinoid",
        };
        return cannCell(c.name, c.value, c.unit, c.notes || noteMap[c.name] || "", isHL);
      }).join("") : "<div style='color:var(--t-faint);'>No cannabinoid data reported.</div>"}
    </div>
    ${chemistry.total_cannabinoids ? `<div class="cann-total"><span class="ct-lbl2">Total Cannabinoid Sum</span><span class="ct-val2">${esc(chemistry.total_cannabinoids)} wt%</span></div>` : ""}
    ${chemistry.thc_total_anhydrous && chemistry.thc_total_anhydrous !== chemistry.thc_total ? `
    <div class="anhydrous-note">
      <span>⚗</span>
      <span><strong>Anhydrous (moisture-corrected) THC Total:</strong> ${esc(chemistry.thc_total_anhydrous)} wt% · As-received: ${esc(chemistry.thc_total)} wt% · Moisture: ${esc(chemistry.moisture_content || "—")}%</span>
    </div>` : ""}
    <div class="entourage">
      <span class="entourage-icon">⬡</span>
      <div class="entourage-body">
        <strong>Entourage Effect signal: ${cannabinoids.filter(c => toNum(c.value) > 0).length >= 5 ? "Strong" : cannabinoids.filter(c => toNum(c.value) > 0).length >= 3 ? "Moderate" : "Limited"}.</strong>
        With ${cannabinoids.filter(c => toNum(c.value) > 0).length} quantified cannabinoids and ${terpCount} terpenes, this profile has biochemical complexity associated with full-spectrum, synergistic activity.
        ${infoIcon(`<span class="tooltip"><strong>Why Entourage Matters</strong><span class="tt-body">The entourage effect describes how whole-plant cannabis extracts produce effects greater than the sum of their individual parts. THC embedded in a complex cannabinoid-terpene matrix behaves differently to THC in isolation.</span><span class="tt-coa"><strong>Clinical implication:</strong> Patients and clinicians should consider the full profile, not just THC%. A product with rich minor cannabinoids and terpenes will behave differently to a THC-dominant, terpene-poor product of the same potency.</span></span>`, "tt-right")}
      </div>
    </div>
  </div>

  ${hasFlavonoids ? `
  <!-- FLAVONOIDS -->
  <div class="sec">
    <div class="sec-head">
      <div class="sec-title">flavonoid profile ${infoIcon(`<span class="tooltip"><strong>Cannabis Flavonoids</strong><span class="tt-body">Flavonoids are polyphenolic compounds present in cannabis. Cannabis-specific flavonoids (Cannflavins A, B, C) are unique to the Cannabis genus and have attracted significant research interest.</span><span class="tt-coa"><strong>Significance:</strong> Cannflavin A and B have been shown to inhibit prostaglandin E2 production — potentially 30x more potent than aspirin in preclinical models. Their presence is a marker of phytochemical richness and analytical thoroughness.</span></span>`, "sec-info")}</div>
      <div class="sec-badge">${esc(chemistry.total_flavonoids || flavonoids.length + " compounds")}</div>
    </div>
    <div class="flav-grid">
      ${flavonoids.filter(f => toNum(f.value) > 0).map((f, i) => flavonoidBar(f, i)).join("")}
    </div>
    ${chemistry.total_flavonoids ? `<div class="flav-total"><span style="font-size:8px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--t-mid);">Total Quantified Flavonoids</span><span class="flav-total-val">${esc(chemistry.total_flavonoids)} wt%</span></div>` : ""}
  </div>` : ""}

  <!-- EFFECT + POST HARVEST -->
  <div class="two-col">
    <div class="sec" style="border-bottom:1px solid var(--border-l)">
      <div class="sec-title" style="margin-bottom:14px">effect direction ${infoIcon(`<span class="tooltip"><strong>How Effect Direction is Inferred</strong><span class="tt-body">Direction is derived from the chemotype fingerprint using observational data correlating terpene dominance to user-reported experiences. Not based on clinical trials.</span><span class="tt-coa"><strong>Limitations:</strong> Individual responses vary widely. Tolerance, consumption method, dose, and physiology all influence the experience. This is directional, not predictive.</span></span>`, "sec-info")}</div>
      <div class="eff-pill"><span class="eff-icon">☀</span><span class="eff-dir">${esc(terpIntel.direction)}</span></div>
      <div class="eff-body">
        <strong>Lead:</strong> ${esc(leadTerpene || "Not identified")}<br>
        ${secondTerpene ? `<strong>2nd:</strong> ${esc(secondTerpene)}${thirdTerpene ? " · <strong>3rd:</strong> " + esc(thirdTerpene) : ""}<br>` : ""}
        <br>${esc(terpIntel.note)}
      </div>
    </div>
    <div class="sec" style="border-bottom:1px solid var(--border-l)">
      <div class="sec-title" style="margin-bottom:14px">post-harvest signals</div>
      <div class="ph-list">
        ${phRow(terps >= 2 ? "good" : "mild", `Freshness — ${safePH.freshness.label}`, safePH.freshness.note,
          "Freshness Signal",
          "Inferred from total terpene content. Terpenes degrade with heat, light, and time. High terpene retention suggests good post-harvest handling.",
          "Compare terpene content against producer's historical batches.")}
        ${phRow(terpenes.length >= 6 ? "good" : "mild", `Curing — ${safePH.curing.label}`, safePH.curing.note,
          "Curing Quality Signal",
          "A broad, intact terpene spectrum suggests gentle, appropriate curing conditions.",
          "If only low-volatility terpenes survive, it suggests aggressive drying.")}
        ${phRow(cbn < 0.3 && cbna < 0.1 ? "good" : "mild", `Degradation — ${safePH.degradation.label}`, safePH.degradation.note,
          "THC Oxidative Degradation (CBN/CBNA)",
          "CBN and CBNA form when THC oxidises. They are the primary markers of cannabis age and storage quality.",
          "Find CBN and CBNA in the cannabinoid table. Values above 0.3 wt% CBN may indicate significant age or poor storage.")}
        ${phRow(hasMoisture || hasWaterActivity ? "good" : "mild", `Stability — ${safePH.stability.label}`, safePH.stability.note,
          "Physical Stability Data",
          "Moisture content and water activity measure stability against mould and microbial growth. Water activity below 0.65 aw is considered microbiologically stable.",
          "Find 'Loss on Drying' (moisture) and 'Water Activity Analysis' in the COA. Both should be reported for complete stability characterisation.")}
      </div>
    </div>
  </div>

  <!-- AUDIENCE INTELLIGENCE -->
  <div class="audience">
    <div class="sec-head" style="margin-bottom:14px">
      <div class="sec-title">audience intelligence</div>
    </div>
    <div class="tab-row">
      <button class="tab-btn active" onclick="showTab('brand',this)"><span class="tab-icon">◈</span>Brand</button>
      <button class="tab-btn" onclick="showTab('clinical',this)"><span class="tab-icon">✚</span>Clinical</button>
      <button class="tab-btn" onclick="showTab('patient',this)"><span class="tab-icon">◯</span>Patient</button>
      <button class="tab-btn" onclick="showTab('buyer',this)"><span class="tab-icon">◇</span>Buyer</button>
    </div>
    <div id="ap-brand" class="tab-panel active"><div class="ins-list">${audiencePanel(audiences.brand)}</div></div>
    <div id="ap-clinical" class="tab-panel"><div class="ins-list">${audiencePanel(audiences.clinical)}</div></div>
    <div id="ap-patient" class="tab-panel"><div class="ins-list">${audiencePanel(audiences.patient)}</div></div>
    <div id="ap-buyer" class="tab-panel"><div class="ins-list">${audiencePanel(audiences.buyer)}</div></div>
  </div>

  <!-- FULL QUALITY & SAFETY -->
  <div class="safety-section">
    <div class="sec-head">
      <div class="sec-title">quality &amp; safety analysis ${infoIcon(`<span class="tooltip"><strong>What Safety Panels Mean</strong><span class="tt-body">A complete cannabis COA includes chemical profiling AND safety screening. The chemical profile tells you what IS in the product. Safety panels tell you what should NOT be there.</span><span class="tt-coa"><strong>Key panels:</strong> Pesticides (EP 2.8.13), Heavy Metals (Ph. Eur. 2.4.27 ICP-MS), Microbials (EP 2.6.12), Mycotoxins (Ph. Eur. 2.8.18/2.8.22). If absent, request them before prescribing or listing.</span></span>`, "sec-info")}</div>
    </div>

    <div class="safety-panels">

      <!-- PESTICIDES -->
      <div class="safety-panel">
        <div class="sp-head">
          <span class="sp-title">Pesticides</span>
          <span class="sp-status ${pestPass ? "pass" : "nt"}">${pestPass ? "✓ " + esc(contaminants.pesticides_status || "ND") : "Not tested"}</span>
        </div>
        <div class="sp-body" style="padding:10px 14px;">
          <div style="font-size:9px;color:var(--t-mid);line-height:1.6;">
            ${contaminants.pesticides_method ? `<div style="margin-bottom:6px;"><strong style="color:var(--alem-dark);">Method:</strong> ${esc(contaminants.pesticides_method)}</div>` : ""}
            ${contaminants.pesticides_compound_count ? `<div><strong style="color:var(--alem-dark);">Compounds tested:</strong> ${esc(contaminants.pesticides_compound_count)}</div>` : ""}
            ${contaminants.pesticides_detail ? `<div style="margin-top:6px;">${esc(contaminants.pesticides_detail)}</div>` : ""}
            ${!contaminants.pesticides_status ? `<div style="color:var(--t-faint);font-style:italic;">Pesticide panel not found in this COA. Request from producer.</div>` : ""}
          </div>
        </div>
      </div>

      <!-- HEAVY METALS -->
      <div class="safety-panel">
        <div class="sp-head">
          <span class="sp-title">Heavy Metals</span>
          <span class="sp-status ${metalsPass ? "pass" : "nt"}">${metalsPass ? "✓ " + esc(contaminants.heavy_metals_status || "BLQ/ND") : (contaminants.heavy_metals_status || "Not tested")}</span>
        </div>
        <div class="sp-body">
          ${contaminants.arsenic_result ? metalRow("Arsenic (As)", contaminants.arsenic_result) : ""}
          ${contaminants.cadmium_result ? metalRow("Cadmium (Cd)", contaminants.cadmium_result) : ""}
          ${contaminants.lead_result ? metalRow("Lead (Pb)", contaminants.lead_result) : ""}
          ${contaminants.mercury_result ? metalRow("Mercury (Hg)", contaminants.mercury_result) : ""}
          ${!contaminants.arsenic_result && !contaminants.cadmium_result ? `<div style="font-size:9px;color:var(--t-faint);font-style:italic;padding:4px 0;">No individual metal results captured.</div>` : ""}
          ${contaminants.heavy_metals_method ? `<div style="font-size:8px;color:var(--t-light);margin-top:6px;border-top:1px solid var(--border-l);padding-top:6px;">${esc(contaminants.heavy_metals_method)}</div>` : ""}
        </div>
      </div>

      <!-- MICROBIALS -->
      <div class="safety-panel">
        <div class="sp-head">
          <span class="sp-title">Microbials</span>
          <span class="sp-status ${microPass ? "pass" : "nt"}">${microPass ? "✓ " + esc(contaminants.microbials_status || "Absent") : (contaminants.microbials_status || "Not tested")}</span>
        </div>
        <div class="sp-body">
          ${contaminants.yeast_mold ? microbialRow("Yeast & Mold", contaminants.yeast_mold) : ""}
          ${contaminants.total_aerobic ? microbialRow("Total Aerobic", contaminants.total_aerobic) : ""}
          ${contaminants.bile_tolerant_gram_negative ? microbialRow("Bile-Tolerant Gram-Neg", contaminants.bile_tolerant_gram_negative) : ""}
          ${contaminants.salmonella ? microbialRow("Salmonella", contaminants.salmonella) : ""}
          ${contaminants.s_aureus ? microbialRow("S. aureus", contaminants.s_aureus) : ""}
          ${contaminants.p_aeruginosa ? microbialRow("P. aeruginosa", contaminants.p_aeruginosa) : ""}
          ${contaminants.e_coli ? microbialRow("E. coli", contaminants.e_coli) : ""}
          ${!contaminants.yeast_mold && !contaminants.salmonella ? `<div style="font-size:9px;color:var(--t-faint);font-style:italic;padding:4px 0;">No individual microbial results captured.</div>` : ""}
          ${contaminants.microbials_method ? `<div style="font-size:8px;color:var(--t-light);margin-top:6px;border-top:1px solid var(--border-l);padding-top:6px;">${esc(contaminants.microbials_method)}</div>` : ""}
        </div>
      </div>

      <!-- MYCOTOXINS -->
      <div class="safety-panel">
        <div class="sp-head">
          <span class="sp-title">Mycotoxins</span>
          <span class="sp-status ${mycoPass ? "pass" : "nt"}">${mycoPass ? "✓ " + esc(contaminants.mycotoxins_status || "ND") : (contaminants.mycotoxins_status || "Not tested")}</span>
        </div>
        <div class="sp-body">
          ${contaminants.aflatoxin_b1 ? mycoRow("Aflatoxin B1", contaminants.aflatoxin_b1) : ""}
          ${contaminants.aflatoxin_b2 ? mycoRow("Aflatoxin B2", contaminants.aflatoxin_b2) : ""}
          ${contaminants.aflatoxin_g1 ? mycoRow("Aflatoxin G1", contaminants.aflatoxin_g1) : ""}
          ${contaminants.aflatoxin_g2 ? mycoRow("Aflatoxin G2", contaminants.aflatoxin_g2) : ""}
          ${contaminants.sum_aflatoxins !== undefined && contaminants.sum_aflatoxins !== "" ? mycoRow("Sum Aflatoxins", contaminants.sum_aflatoxins) : ""}
          ${contaminants.ochratoxin_a ? mycoRow("Ochratoxin A", contaminants.ochratoxin_a) : ""}
          ${!contaminants.aflatoxin_b1 && !contaminants.ochratoxin_a ? `<div style="font-size:9px;color:var(--t-faint);font-style:italic;padding:4px 0;">No individual mycotoxin results captured.</div>` : ""}
          ${contaminants.mycotoxins_method ? `<div style="font-size:8px;color:var(--t-light);margin-top:6px;border-top:1px solid var(--border-l);padding-top:6px;">${esc(contaminants.mycotoxins_method)}</div>` : ""}
        </div>
      </div>
    </div>

    <!-- PHYSICAL / STABILITY SUMMARY -->
    <div class="safety-summary-grid">
      ${sfRow("ISO 17025:2017 Accreditation", "pass", isoPass ? "✓ Confirmed" : "Not confirmed in document", isoPass)}
      ${sfRow("SCC Accredited (Standards Council of Canada)", "pass", sccPass ? "✓ Confirmed" : "Not confirmed in document", sccPass)}
      ${sfRow("Residual Solvents", /pass|nd/i.test(contaminants.residual_solvents_status || "") ? "pass" : "nt", /pass|nd/i.test(contaminants.residual_solvents_status || "") ? "✓ Pass" : (contaminants.residual_solvents_status || "not tested"), /pass|nd/i.test(contaminants.residual_solvents_status || ""))}
      ${sfRow("Foreign Matter", /none detected|pass|nd/i.test(contaminants.foreign_matter_status || "") ? "pass" : "nt", esc(contaminants.foreign_matter_status || chemistry.foreign_matter || "not reported"), /none detected|pass|nd/i.test(contaminants.foreign_matter_status || chemistry.foreign_matter || ""))}
      ${sfRow(`Moisture Content${moisture ? ": " + moisture + "%" : ""}`, hasMoisture ? "pass" : "nt", hasMoisture ? `${moisture}% (EP 2.2.32 Vacuum Oven)` : "not reported", hasMoisture)}
      ${sfRow(`Water Activity${waterActivity ? ": " + waterActivity + " aw" : ""}`, hasWaterActivity ? "pass" : "nt", hasWaterActivity ? `${waterActivity} aw` : "not reported", hasWaterActivity)}
    </div>

    ${chemistry.hptlc_result || chemistry.macroscopic_result ? `
    <div class="method-strip">
      <div style="font-size:7.5px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--t-light);margin-bottom:8px;">Botanical Identification Tests</div>
      ${chemistry.hptlc_result ? `<div class="method-row"><span class="method-label">HPTLC (EP Monograph)</span><span class="method-val">${esc(chemistry.hptlc_result)}</span></div>` : ""}
      ${chemistry.macroscopic_result ? `<div class="method-row"><span class="method-label">Macroscopic (EP 2.8.2)</span><span class="method-val">${esc(chemistry.macroscopic_result)}</span></div>` : ""}
      ${chemistry.microscopic_result ? `<div class="method-row"><span class="method-label">Microscopic (EP 2.8.2)</span><span class="method-val">${esc(chemistry.microscopic_result)}</span></div>` : ""}
    </div>` : ""}

    ${(pestPass && metalsPass && microPass && mycoPass) ? `
    <div class="safety-pass-banner">
      <strong>✓ Full Compliance Panel — All Categories Clear.</strong> This COA includes a comprehensive safety screening across pesticides, heavy metals, microbials, and mycotoxins. All tested analytes are at or below reporting limits. ${isoPass ? "Testing conducted under ISO 17025:2017 accreditation." : ""} ${sccPass ? "SCC accredited." : ""}
    </div>` : (!contaminants.pesticides_status && !contaminants.heavy_metals_status) ? `
    <div class="safety-note"><strong>Incomplete safety panel:</strong> ${esc(contaminants.contaminant_narrative || "Contaminant panels were not captured from this COA. Request the full compliance documentation from the producer before clinical prescription or commercial listing.")}</div>` : ""}
  </div>

  <!-- LAB STRIP -->
  <div class="lab-strip">
    <div class="lab-cell">
      <div class="lab-val">${esc(chemistry.laboratory_name || "—")}</div>
      <div class="lab-lbl">Laboratory ${infoIcon(`<span class="tooltip"><strong>${esc(chemistry.laboratory_name || "Laboratory")}</strong><span class="tt-body">${esc(contaminants.lab_quality_summary || (chemistry.laboratory_accreditation ? `Accredited under ${chemistry.laboratory_accreditation}.` : "Laboratory accreditation details not captured."))} ISO 17025 is the international gold standard for testing laboratories.</span><span class="tt-coa"><strong>Why it matters:</strong> Not all cannabis labs hold ISO 17025. Accreditation means results are traceable to international measurement standards.</span></span>`, "tt-right")}</div>
    </div>
    <div class="lab-cell">
      <div class="lab-val">${esc(chemistry.laboratory_method || "—")}</div>
      <div class="lab-lbl">Method</div>
    </div>
    <div class="lab-cell">
      <div class="lab-val">${esc(chemistry.coa_report_date || "—")}</div>
      <div class="lab-lbl">Report Date</div>
    </div>
    <div class="lab-cell">
      <div class="lab-val">${esc(chemistry.laboratory_accreditation || "—")}</div>
      <div class="lab-lbl">Accreditation</div>
    </div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <div>
      <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png"
           alt="Alem" style="height:48px;display:block;"
           onerror="this.style.display='none'">
      <div class="footer-url">alem.solutions &nbsp;·&nbsp; Free. No account. No catch.</div>
    </div>
    <div class="footer-tagline">Upload your COA.<br>Understand your chemistry.</div>
  </div>

</div>

<!-- ACTION BUTTONS -->
<div class="ractions">
  <a class="rbtn rbtn-p" href="/" onclick="window.location='/'; return false;" style="cursor:pointer;">↗ &nbsp; Analyse New COA</a>
  ${options.documentId ? `<a class="rbtn rbtn-s" href="/pdf/${esc(options.documentId)}" target="_blank">⬇ &nbsp; Export PDF</a>` : ""}
</div>
<p class="page-foot">Powered by <a href="https://alem.solutions">alem.solutions</a> &nbsp;·&nbsp; Chemical Intelligence</p>

<script>
function showTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('ap-' + id).classList.add('active');
  btn.classList.add('active');
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
  return res.json({ success: true, message: "Alem v6 running", schema: "6.0" });
});

// ─────────────────────────────────────────────
// MULTI-FILE UPLOAD ROUTE
// ─────────────────────────────────────────────

app.post("/upload-coa-multi", requireApiKey, uploadLimiter, upload.array("files", 10), async (req, res) => {
  try {
    console.log(`📥 Multi-file COA upload: ${(req.files || []).length} file(s)`);
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ success: false, error: "No files uploaded" });
    }

    // Upload to Supabase for storage, but OCR from buffer directly
    let uploadResults = [];
    try {
      uploadResults = await Promise.all(
        req.files.map((file, idx) =>
          uploadBufferToSupabase({
            buffer: file.buffer,
            originalName: file.originalname || `upload-${Date.now()}-${idx}`,
            mimeType: file.mimetype,
            folder: "raw_documents",
          })
        )
      );
    } catch (uploadErr) {
      console.error("❌ Supabase upload failed:", uploadErr.message);
      return res.status(500).json({ success: false, error: "Storage upload failed: " + uploadErr.message });
    }
    console.log(`☁️  Stored ${uploadResults.length} file(s)`);

    // Send buffers DIRECTLY to Azure — avoids partial reads from URL fetching
    console.log("🔍 Azure OCR — processing all files directly from buffer...");
    const ocrResults = await Promise.all(
      req.files.map((file) => extractDocumentFromBuffer(file.buffer, file.mimetype))
    );

    const combinedText = ocrResults
      .map((r, i) => "[FILE " + (i + 1) + "]\n" + r.plain_text)
      .join("\n\n---\n\n")
      .trim();

    const totalPages = ocrResults.reduce((sum, r) => sum + r.page_count, 0);
    console.log(`✅ OCR complete: ${combinedText.length} chars, ${totalPages} total pages across ${req.files.length} file(s)`);

    // Merge pages from all files with renumbered page numbers for smart splitting
    const allPages = ocrResults.flatMap((r, fileIdx) =>
      (r.pages || []).map(p => ({ ...p, page_number: p.page_number + (fileIdx > 0 ? ocrResults.slice(0, fileIdx).reduce((s, x) => s + x.page_count, 0) : 0) }))
    );

    const { chemistry, contaminants } = await runDualPassExtraction(combinedText, allPages);
    console.log("✅ Dual-pass extraction complete");

    const scoring = computeIntelligenceScore(chemistry, contaminants);
    console.log(`✅ Score: ${scoring.total}/100 (${scoring.grade})`);

    const fingerprintId = generateFingerprintId(chemistry.top_terpenes);
    const leadTerpene   = chemistry.top_terpenes?.[0]?.name || "";
    const terpIntel     = getTerpeneIntel(leadTerpene);
    const audiences     = buildAudienceNarratives(chemistry, contaminants, scoring);
    const postHarvest   = buildPostHarvestIntel(chemistry, contaminants);
    const intelligence  = { fingerprintId, effectDirection: terpIntel.direction, lineageCluster: terpIntel.lineage, lineageConfidence: terpIntel.lineageConfidence, audiences, postHarvest };

    // Start scientific evidence fetch early (non-blocking)
    const sciPromise = buildScientificEvidence({ chemistry, contaminants, intelligence }).catch(() => null);

    const strainIntel = await fetchStrainIntelligence(chemistry).catch(() => null);
    if (strainIntel) intelligence.strainIntel = strainIntel;
    if (strainIntel) console.log(`✅ Strain match: ${strainIntel.match?.strain_name} (${strainIntel.match?.similarity}%)`);

    // Await scientific evidence with 10s hard timeout
    const sciTimeout = new Promise(r => setTimeout(() => r(null), 10_000));
    const scientificEvidence = await Promise.race([sciPromise, sciTimeout]);
    if (scientificEvidence?.totalArticles > 0) {
      intelligence.scientificEvidence = scientificEvidence;
      console.log(`✅ Scientific evidence: ${scientificEvidence.totalArticles} articles in ${scientificEvidence.executionMs}ms`);
    } else {
      console.warn("⚠️  Scientific evidence timed out or returned empty — report rendered without citations");
    }

    const primaryFilename = req.files[0].originalname || `multi-upload-${Date.now()}`;
    const insertedRow = await insertCOAReport({
      chemistry, contaminants, scoring, intelligence,
      sourceUrl:        uploadResults[0].publicUrl,
      storagePath:      uploadResults[0].storagePath,
      originalFilename: primaryFilename,
      mimeType:         ocrResults[0].mimeType,
    });
    console.log(`✅ Stored report ID: ${insertedRow.id}`);

    const proto = req.headers["x-forwarded-proto"] || req.protocol || "https";
    const base  = `${proto}://${req.get("host")}`;

    return res.json({
      success:      true,
      id:           insertedRow.id,
      score:        scoring.total,
      grade:        scoring.grade,
      tier:         scoring.tier,
      fingerprint_id: fingerprintId,
      files_processed: req.files.length,
      report_url:   `${base}/report/${insertedRow.id}`,
      pdf_url:      `${base}/pdf/${insertedRow.id}`,
    });
  } catch (error) {
    console.error("❌ Multi-upload pipeline error:", error.message);
    // Clean up any files already uploaded to Supabase before the failure
    if (uploadResults && uploadResults.length > 0) {
      const paths = uploadResults.map(r => r.storagePath).filter(Boolean);
      if (paths.length > 0) {
        supabase.storage.from(SUPABASE_BUCKET).remove(paths).catch(e =>
          console.warn("⚠️  Supabase cleanup failed:", e.message)
        );
      }
    }
    return res.status(500).json({ success: false, error: error.message || "Upload pipeline failed" });
  }
});

app.post("/upload-coa", requireApiKey, uploadLimiter, upload.single("file"), async (req, res) => {
  try {
    console.log("📥 COA upload received");
    if (!req.file) return res.status(400).json({ success: false, error: "No file uploaded" });

    const originalFilename = req.file.originalname || `upload-${Date.now()}.pdf`;
    const mimeType = req.file.mimetype || "application/octet-stream";

    const { publicUrl, storagePath } = await uploadBufferToSupabase({ buffer: req.file.buffer, originalName: originalFilename, mimeType, folder: "raw_documents" });
    console.log("☁️  Stored:", storagePath);

    // Send buffer DIRECTLY to Azure — avoids partial reads from URL fetching
    console.log("🔍 Azure OCR (direct buffer)...");
    const extracted = await extractDocumentFromBuffer(req.file.buffer, mimeType);
    console.log(`✅ OCR: ${extracted.plain_text.length} chars, ${extracted.page_count} pages`);

    const { chemistry, contaminants } = await runDualPassExtraction(extracted.plain_text, extracted.pages || []);
    console.log("✅ Dual-pass extraction complete");

    const scoring = computeIntelligenceScore(chemistry, contaminants);
    console.log(`✅ Score: ${scoring.total}/100 (${scoring.grade})`);

    const fingerprintId = generateFingerprintId(chemistry.top_terpenes);
    const leadTerpene = chemistry.top_terpenes?.[0]?.name || "";
    const terpIntel = getTerpeneIntel(leadTerpene);
    const audiences = buildAudienceNarratives(chemistry, contaminants, scoring);
    const postHarvest = buildPostHarvestIntel(chemistry, contaminants);

    const intelligence = { fingerprintId, effectDirection: terpIntel.direction, lineageCluster: terpIntel.lineage, lineageConfidence: terpIntel.lineageConfidence, audiences, postHarvest };
    console.log(`✅ Intelligence: fingerprint=${fingerprintId}, direction=${terpIntel.direction}`);

    const insertedRow = await insertCOAReport({ chemistry, contaminants, scoring, intelligence, sourceUrl: publicUrl, storagePath, originalFilename, mimeType: extracted.mimeType || mimeType });
    console.log(`✅ Stored report ID: ${insertedRow.id}`);

    const proto = req.headers["x-forwarded-proto"] || req.protocol || "https";
    const base = `${proto}://${req.get("host")}`;

    return res.json({ success: true, id: insertedRow.id, score: scoring.total, grade: scoring.grade, tier: scoring.tier, fingerprint_id: fingerprintId, report_url: `${base}/report/${insertedRow.id}`, pdf_url: `${base}/pdf/${insertedRow.id}` });
  } catch (error) {
    console.error("❌ Upload pipeline error:", error.message);
    return res.status(500).json({ success: false, error: error.message || "Upload pipeline failed" });
  }
});

app.get("/report/:id", async (req, res) => {
  try {
    const row = await getReportById(req.params.id);
    if (!row) throw new Error("Row not found");
    const chemistry   = row.report_json?.chemistry || {};
    const storedIntel = row.report_json?.intelligence || {};

    const [benchmark, strainIntel] = await Promise.all([
      fetchBenchmark(chemistry).catch(err => { console.error("📊 [benchmark] threw:", err.message); return null; }),
      storedIntel.strainIntel
        ? Promise.resolve(storedIntel.strainIntel)
        : fetchStrainIntelligence(chemistry).catch(err => { console.error("🌿 [strain] threw:", err.message); return null; }),
    ]);
    console.log(`📊 [benchmark] result: ${benchmark ? `thc=${benchmark.thcPercentile}th pct, n=${benchmark.n}` : "null"}`);
    console.log(`🌿 [strain] result: ${strainIntel ? `match=${strainIntel.match?.strain_name} ${strainIntel.match?.similarity}% totalStrains=${strainIntel.totalStrains}` : "null"}`);
    console.log(`🧪 [chemistry] product_type="${chemistry.product_type}" thc_total="${chemistry.thc_total}" total_terpenes="${chemistry.total_terpenes}" top_terpenes_count=${(chemistry.top_terpenes||[]).length}`);

    // Scientific evidence: use stored copy, or fetch with 10s timeout
    let scientificEvidence = storedIntel.scientificEvidence || null;
    let sciLoading = false;
    if (!scientificEvidence) {
      const totalTerps = parseFloat(chemistry.total_terpenes) || 0;
      const topTerps   = (chemistry.top_terpenes || []).length;
      console.log(`🔬 [sci] report ${req.params.id} — totalTerps=${totalTerps} topTerps=${topTerps} thc=${chemistry.thc_total}`);
      const sciTimeout = new Promise(r => setTimeout(() => r("timeout"), 25_000));
      const result = await Promise.race([
        buildScientificEvidence({ chemistry, intelligence: storedIntel }).catch(err => {
          console.error("🔬 [sci] buildScientificEvidence threw:", err.message);
          return null;
        }),
        sciTimeout,
      ]);
      if (result === "timeout") {
        sciLoading = true;
        console.warn("🔬 [sci] timed out after 10s — rendering without citations");
      } else if (!result) {
        console.warn("🔬 [sci] returned null — no evidence rendered");
      } else {
        scientificEvidence = result;
        console.log(`🔬 [sci] ${result.totalArticles} articles in ${result.executionMs}ms — insufficient=${result.insufficient||false}`);
      }
    } else {
      console.log(`🔬 [sci] using stored evidence: ${scientificEvidence.totalArticles} articles`);
    }

    return res.send(renderReportHTML(row?.report_json || {}, { documentId: row.id, benchmark, strainIntel, scientificEvidence, sciLoading }));
  } catch (error) {
    console.error("ERROR /report/:id", error.message);
    return res.status(404).send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Report Not Found — Alem</title>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f4f8fb; font-family: 'Nunito', sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 24px; }
  .box { background: #fff; border: 1px solid #ccdde8; max-width: 480px; width: 100%; padding: 48px 40px; text-align: center; }
  .logo { font-size: 13px; font-weight: 800; letter-spacing: 6px; color: #0d2d3e; margin-bottom: 32px; }
  .code { font-size: 72px; font-weight: 800; color: #eef6fb; line-height: 1; }
  h1 { font-size: 22px; font-weight: 700; color: #0d2d3e; margin: 16px 0 10px; }
  p { font-size: 13px; color: #4a6070; line-height: 1.7; margin-bottom: 28px; }
  a { display: inline-block; background: #0d2d3e; color: #fff; padding: 12px 28px; font-size: 9px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; text-decoration: none; border-radius: 30px; }
  a:hover { background: #1a4a62; }
</style>
</head>
<body>
  <div class="box">
    <div class="logo">ALEM</div>
    <div class="code">404</div>
    <h1>Report not found</h1>
    <p>This report ID doesn't exist or may have been removed. Please check the URL or upload a new COA to generate a fresh report.</p>
    <a href="/">Upload a COA</a>
  </div>
</body>
</html>`);
  }
});

app.get("/pdf/:id", async (req, res) => {
  let browser;
  try {
    const row = await getReportById(req.params.id);
    const benchmark = await fetchBenchmark(row?.report_json?.chemistry || {}).catch(() => null);
    const html = renderReportPDFDoc(row?.report_json || {}, { documentId: row.id, benchmark });

    browser = await puppeteer.launch({ headless: true, args: ["--no-sandbox", "--disable-setuid-sandbox"] });
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "networkidle0" });
    const pdfBuffer = await page.pdf({ format: "A4", printBackground: true, margin: { top: "14mm", right: "12mm", bottom: "14mm", left: "12mm" } });

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

// ─────────────────────────────────────────────
// CATCH-ALL 404 + GLOBAL ERROR MIDDLEWARE
// ─────────────────────────────────────────────

app.use((req, res) => {
  res.status(404).json({ success: false, error: "Route not found" });
});

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  console.error("❌ Express error middleware:", err.message);
  if (err.type === "entity.too.large") {
    return res.status(413).json({ success: false, error: "File too large. Maximum size is 20MB." });
  }
  if (err.message && err.message.includes("Only PDF")) {
    return res.status(415).json({ success: false, error: err.message });
  }
  return res.status(500).json({ success: false, error: err.message || "Internal server error" });
});

// ─────────────────────────────────────────────
// GLOBAL ERROR SAFETY NET
// ─────────────────────────────────────────────

process.on("unhandledRejection", (reason, promise) => {
  console.error("⚠️  Unhandled Promise Rejection:", reason?.message || reason);
  // Do NOT exit — keep the server alive for other requests
});

process.on("uncaughtException", (err) => {
  console.error("⚠️  Uncaught Exception:", err.message);
  process.exit(1);
});

app.listen(PORT, () => {
  console.log(`🌿 Alem Chemical Intelligence v7 running on port ${PORT}`);
});
