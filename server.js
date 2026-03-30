
require("dotenv").config();

const path = require("path");
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const puppeteer = require("puppeteer");
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
const MAX_OCR_CHARS = Number(process.env.MAX_OCR_CHARS_FOR_OPENAI || 80000);
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 120000);
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

const CHEMISTRY_EXTRACTION_PROMPT = `
You are the Alem Solutions COA chemistry extraction engine.
Read OCR text from a cannabis Certificate of Analysis and return EXACTLY ONE valid JSON object.

Rules:
- Return valid JSON only.
- Never hallucinate.
- If unknown use "" or [].
- Convert mg/g to wt% by dividing by 10.
- Prefer anhydrous values when both are present.
- Extract ALL terpene rows with numeric values or BLQ.
- Extract ALL cannabinoid rows present.
- Keep unit as "wt%" whenever applicable.

Return:
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
  "top_cannabinoids": [{ "name": "", "value": "", "unit": "wt%", "notes": "" }],
  "top_terpenes": [{ "name": "", "value": "", "unit": "wt%" }],
  "flavonoids": [{ "name": "", "value": "", "unit": "wt%" }]
}
`;

const CONTAMINANT_EXTRACTION_PROMPT = `
You are the Alem Solutions COA safety extraction engine.
Read OCR text from a cannabis Certificate of Analysis and return EXACTLY ONE valid JSON object.

Rules:
- Return valid JSON only.
- Never hallucinate.
- If unknown use "" or [].
- Accept ND, BLQ, Pass, Absent, <10 CFU/g, numeric ppm/ppb values.

Return:
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
  "positive_flags": [],
  "warning_flags": []
}
`;

function esc(v = "") {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function s(v) {
  return v === null || v === undefined ? "" : String(v).trim();
}

function a(v) {
  return Array.isArray(v) ? v : [];
}

function toNum(v) {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  const n = Number(String(v ?? "").replace(/[^0-9.\-]/g, ""));
  return Number.isFinite(n) ? n : 0;
}

function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

function sanitizeFileName(name = "") {
  return String(name || "report")
    .replace(/[\/\\?%*:|"<>]/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 140);
}

function fixUnit(unit = "") {
  const raw = String(unit || "").trim().toLowerCase();
  if (!raw) return "wt%";
  if (raw === "%" || raw === "wt%" || raw === "w/w") return "wt%";
  return unit;
}

function normTerpName(name = "") {
  const raw = String(name || "").trim();
  const map = {
    "β-Myrcene": "Beta-Myrcene",
    "β-Caryophyllene": "Trans-Caryophyllene",
    "Beta-Caryophyllene": "Trans-Caryophyllene",
    "β-Pinene": "Beta-Pinene",
    "D-Limonene": "(R)-(+)-Limonene",
    "d-Limonene": "(R)-(+)-Limonene",
    "α-Pinene": "Alpha-Pinene",
    "α-Humulene": "Alpha-Humulene",
    "α-Bisabolol": "Alpha-Bisabolol",
    "α-Terpineol": "Alpha-Terpineol",
  };
  return map[raw] || raw;
}

function normCannName(name = "") {
  const raw = String(name || "").trim();
  const map = {
    "THCA-A": "THCA",
    "Δ9-THC": "D9-THC",
    "delta-9-THC": "D9-THC",
    "Δ8-THC": "D8-THC",
    "CBCA-A": "CBCA",
  };
  return map[raw] || raw;
}

function normalizeChemistry(data = {}) {
  const topTerpenes = a(data.top_terpenes)
    .map((i) => ({
      name: normTerpName(i?.name),
      value: s(i?.value),
      unit: fixUnit(i?.unit),
    }))
    .filter((i) => i.name);

  const flavonoids = a(data.flavonoids)
    .map((i) => ({
      name: s(i?.name),
      value: s(i?.value),
      unit: fixUnit(i?.unit),
    }))
    .filter((i) => i.name);

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
    total_terpenes: s(data.total_terpenes),
    total_terpenes_unit: fixUnit(data.total_terpenes_unit),
    total_cannabinoids: s(data.total_cannabinoids),
    total_flavonoids: s(data.total_flavonoids),
    moisture_content: s(data.moisture_content),
    water_activity: s(data.water_activity),
    foreign_matter: s(data.foreign_matter),
    hptlc_result: s(data.hptlc_result),
    macroscopic_result: s(data.macroscopic_result),
    microscopic_result: s(data.microscopic_result),
    hero_narrative: s(data.hero_narrative),
    top_cannabinoids: a(data.top_cannabinoids).map((i) => ({
      name: normCannName(i?.name),
      value: s(i?.value),
      unit: fixUnit(i?.unit),
      notes: s(i?.notes),
    })).filter((i) => i.name),
    top_terpenes: topTerpenes,
    flavonoids,
  };
}

function normalizeContaminants(data = {}) {
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
    iso_17025: data.iso_17025 === true || String(data.iso_17025).toLowerCase() === "true",
    scc_accredited: data.scc_accredited === true || String(data.scc_accredited).toLowerCase() === "true",
    lab_accreditation_body: s(data.lab_accreditation_body),
    contaminant_narrative: s(data.contaminant_narrative),
    lab_quality_summary: s(data.lab_quality_summary),
    positive_flags: a(data.positive_flags).map(s).filter(Boolean),
    warning_flags: a(data.warning_flags).map(s).filter(Boolean),
  };
}

function computeIntelligenceScore(extracted, contaminants) {
  let score = 0;
  const breakdown = {};
  const productType = String(extracted.product_type || "").toLowerCase();
  const isConcentrate = /concentrat|distillat|resin|rosin|hash|kief|shatter|wax|budder|extract|oil|tincture|vape|cartridge/i.test(productType);
  const isCBD = toNum(extracted.cbd_total) > toNum(extracted.thc_total) * 2;

  const thc = toNum(extracted.thc_total);
  const cbd = toNum(extracted.cbd_total);
  let potencyScore = 0;
  if (isConcentrate) {
    if (thc >= 80) potencyScore = 25;
    else if (thc >= 70) potencyScore = 22;
    else if (thc >= 60) potencyScore = 18;
    else if (thc >= 50) potencyScore = 13;
    else if (thc > 0) potencyScore = 8;
  } else if (isCBD) {
    if (cbd >= 15) potencyScore = 25;
    else if (cbd >= 10) potencyScore = 20;
    else if (cbd >= 5) potencyScore = 15;
    else if (cbd > 0) potencyScore = 8;
  } else {
    if (thc >= 28) potencyScore = 25;
    else if (thc >= 24) potencyScore = 22;
    else if (thc >= 20) potencyScore = 18;
    else if (thc >= 15) potencyScore = 13;
    else if (thc > 0) potencyScore = 8;
    if (cbd >= 15 && thc < 5) potencyScore = Math.max(potencyScore, 20);
  }
  breakdown.potency = { score: potencyScore, max: 25 };
  score += potencyScore;

  const terps = toNum(extracted.total_terpenes);
  const terpCount = a(extracted.top_terpenes).filter((t) => toNum(t.value) > 0).length;
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

  const minorValues = [
    toNum(extracted.cbga), toNum(extracted.cbn), toNum(extracted.cbg),
    toNum(extracted.cbda), toNum(extracted.cbca), toNum(extracted.thcva), toNum(extracted.cbna)
  ];
  const minorCount = minorValues.filter((v) => v > 0).length;
  const totalMinors = Math.max(0, toNum(extracted.total_cannabinoids) - toNum(extracted.thc_total) - toNum(extracted.cbd_total));
  let minorScore = 0;
  if (totalMinors >= 5) minorScore = 20;
  else if (totalMinors >= 3) minorScore = 16;
  else if (totalMinors >= 1.5) minorScore = 12;
  else if (minorCount >= 3) minorScore = 10;
  else if (minorCount >= 1) minorScore = 6;
  if (toNum(extracted.cbn) >= 1) minorScore = Math.max(0, minorScore - 4);
  breakdown.minors = { score: minorScore, max: 20 };
  score += minorScore;

  let safetyScore = 0;
  const pest = String(contaminants.pesticides_status || "").toLowerCase();
  const metals = String(contaminants.heavy_metals_status || "").toLowerCase();
  const micro = String(contaminants.microbials_status || "").toLowerCase();
  const myco = String(contaminants.mycotoxins_status || "").toLowerCase();
  if (pest.includes("pass") || pest.includes("nd")) safetyScore += 6;
  else if (!pest || pest.includes("not tested")) safetyScore += 1;
  if (metals.includes("pass") || metals.includes("nd") || metals.includes("blq")) safetyScore += 5;
  else if (!metals || metals.includes("not tested")) safetyScore += 1;
  if (micro.includes("pass") || micro.includes("nd") || micro.includes("absent")) safetyScore += 5;
  else if (!micro || micro.includes("not tested")) safetyScore += 1;
  if (myco.includes("pass") || myco.includes("nd")) safetyScore += 2;
  if (contaminants.iso_17025) safetyScore = Math.min(20, safetyScore + 2);
  if (contaminants.scc_accredited) safetyScore = Math.min(20, safetyScore + 2);
  breakdown.safety = { score: safetyScore, max: 20 };
  score += safetyScore;

  let dataScore = 0;
  if (extracted.thc_total) dataScore += 2;
  if (extracted.total_terpenes) dataScore += 2;
  if (a(extracted.top_terpenes).filter((t) => toNum(t.value) > 0).length >= 5) dataScore += 2;
  if (a(extracted.top_cannabinoids).length >= 3) dataScore += 1;
  if (extracted.coa_report_date) dataScore += 1;
  if (extracted.laboratory_name) dataScore += 1;
  if (extracted.batch_number) dataScore += 1;
  breakdown.dataCompleteness = { score: dataScore, max: 10 };
  score += dataScore;

  const grade = score >= 90 ? "A+" : score >= 85 ? "A" : score >= 78 ? "A-" :
    score >= 74 ? "B+" : score >= 70 ? "B" : score >= 65 ? "B-" :
    score >= 58 ? "C+" : score >= 50 ? "C" : "D";
  const tier = score >= 85 ? "Exceptional" : score >= 74 ? "Strong" : score >= 60 ? "Moderate" : "Limited";

  return { total: score, grade, tier, breakdown };
}

const TERPENE_INTEL = {
  "trans-caryophyllene": {
    direction: "Balanced / structured",
    note: "Caryophyllene-led profiles often read as warm, structured, and spice-forward.",
    lineage: "Balanced hybrid / spice-forward families",
    lineageConfidence: "Low-moderate",
  },
  "beta-myrcene": {
    direction: "Calming / body-centred",
    note: "Myrcene-dominant profiles often lean more body-heavy and calming.",
    lineage: "Kush / OG / indica-leaning families",
    lineageConfidence: "Moderate",
  },
  "limonene": {
    direction: "Bright / mood-forward",
    note: "Limonene-forward profiles often read brighter and more citrus-led.",
    lineage: "Citrus hybrid / Gelato / Runtz-type families",
    lineageConfidence: "Moderate",
  },
  "(r)-(+)-limonene": {
    direction: "Bright / mood-forward",
    note: "Limonene-forward profiles often read brighter and more citrus-led.",
    lineage: "Citrus hybrid / Gelato / Runtz-type families",
    lineageConfidence: "Moderate",
  },
  "alpha-pinene": {
    direction: "Clear / alert",
    note: "Pinene-forward profiles often read sharper, fresher, and more alertness-oriented.",
    lineage: "Pine-forward / clear sativa-leaning families",
    lineageConfidence: "Low-moderate",
  },
  "terpinolene": {
    direction: "Uplifting / mentally active",
    note: "Terpinolene-dominant profiles often feel brighter and more mentally active.",
    lineage: "Haze / Jack / Durban-type families",
    lineageConfidence: "Moderate",
  },
  "linalool": {
    direction: "Calming / floral",
    note: "Linalool contributes floral softness and a calmer directional interpretation.",
    lineage: "Floral / indica-adjacent families",
    lineageConfidence: "Low-moderate",
  },
  "farnesene": {
    direction: "Floral / fruity / complex",
    note: "Farnesene adds a rarer green-apple/floral edge and supports differentiation.",
    lineage: "Complex hybrid families",
    lineageConfidence: "Low",
  },
};

function getTerpeneIntel(name = "") {
  return TERPENE_INTEL[String(name || "").toLowerCase()] || {
    direction: "Balanced / mixed",
    note: "This profile reads as mixed chemistry without a strongly canonical dominant interpretation.",
    lineage: "Mixed chemotype families",
    lineageConfidence: "Low",
  };
}

function generateFingerprintId(terpenes = []) {
  const top = a(terpenes).slice(0, 3).map((t) => {
    const raw = String(t?.name || "").toLowerCase();
    if (raw.includes("caryophyllene")) return "CAR";
    if (raw.includes("limonene")) return "LIM";
    if (raw.includes("farnesene")) return "FAR";
    if (raw.includes("myrcene")) return "MYR";
    if (raw.includes("pinene")) return "PIN";
    if (raw.includes("linalool")) return "LIN";
    if (raw.includes("terpinolene")) return "TER";
    return raw.slice(0, 3).toUpperCase() || "—";
  });
  return top.filter(Boolean).join(" — ") || "—";
}

function buildAudienceNarratives(chemistry, contaminants, scoring) {
  const thc = toNum(chemistry.thc_total);
  const terps = toNum(chemistry.total_terpenes);
  const lead = chemistry.top_terpenes?.[0]?.name || "lead terpene";
  const safetyClear =
    /pass|nd/i.test(contaminants.pesticides_status || "") &&
    /pass|nd|blq/i.test(contaminants.heavy_metals_status || "") &&
    /pass|nd|absent/i.test(contaminants.microbials_status || "") &&
    /pass|nd/i.test(contaminants.mycotoxins_status || "");

  return {
    patient: [
      `This is a ${thc >= 24 ? "strong" : thc >= 18 ? "moderately strong" : "milder"} THC product with ${chemistry.cbd_total ? "some CBD present" : "no CBD detected"}, so start low and build slowly.`,
      `${lead} sits at the front of the terpene profile, which helps explain the product’s overall aromatic direction.`,
      `${safetyClear ? "The COA shows clear safety panels across the categories tested." : "Review the safety section carefully because not every panel is fully clear or complete."}`,
    ],
    clinical: [
      `${chemistry.thc_total || "—"} total THC with ${chemistry.cbd_total || "CBD ND"} indicates the cannabinoid balance is heavily THC-led.`,
      `${lead} is the dominant terpene and total terpenes are ${chemistry.total_terpenes || "not reported"} wt%, giving a broader pharmacological discussion than THC alone.`,
      `Score ${scoring.total}/${scoring.breakdown?.safety?.max ? 100 : 100} with ${scoring.grade} grade; safety data ${safetyClear ? "supports high-confidence interpretation" : "requires caution due to incompleteness"}.`,
    ],
    buyer: [
      `${scoring.total}/100 (${scoring.grade}) gives the batch strong proof points for premium positioning.`,
      `${terps >= 2.5 ? "High terpene richness" : "Moderate terpene richness"} helps differentiate the batch beyond headline THC alone.`,
      `${chemistry.flavonoids?.length ? "Flavonoid data adds extra differentiation value." : "No extra specialty panel is present, so the commercial story should stay focused on potency, terpene depth, and safety."}`,
    ],
    brand: [
      `${lead} gives the batch a clear fingerprint and more ownable chemistry narrative.`,
      `${safetyClear ? "Complete safety coverage supports confident sales communication." : "Safety coverage should be described carefully because some areas are incomplete."}`,
      `${terps >= 2.5 ? "Terpene richness is strong enough to support top-shelf storytelling." : "The story should lean more on balance and cleanliness than terpene extremity."}`,
    ],
  };
}

function buildPostHarvestIntel(chemistry, contaminants) {
  const terps = toNum(chemistry.total_terpenes);
  const cbn = toNum(chemistry.cbn);
  const cbna = toNum(chemistry.cbna);
  const water = toNum(chemistry.water_activity || contaminants.water_activity);

  return {
    freshness: {
      label: terps >= 2.5 ? "Strong" : terps >= 1.5 ? "Positive" : "Moderate",
      note: terps >= 2.5
        ? "Terpene retention is strong, which supports an interpretation of very good post-harvest preservation."
        : "Terpene retention is present but not extreme, suggesting acceptable preservation.",
    },
    curing: {
      label: a(chemistry.top_terpenes).length >= 8 ? "Positive signal" : "Mixed signal",
      note: a(chemistry.top_terpenes).length >= 8
        ? "A broad terpene spread suggests the profile avoided major flattening during drying and curing."
        : "A narrower terpene spread makes curing quality harder to interpret confidently.",
    },
    degradation: {
      label: cbn < 0.3 && cbna < 0.1 ? "Minimal" : "Present",
      note: cbn < 0.3 && cbna < 0.1
        ? "Low degradation markers support a fresher, better-preserved interpretation."
        : "CBN and/or CBNA suggest some oxidation or age-related change.",
    },
    stability: {
      label: water > 0 && water < 0.65 ? "Confirmed" : "Unknown",
      note: water > 0 && water < 0.65
        ? "Water activity is below the common microbial risk threshold for storage."
        : "Water activity is missing or not clearly inside a stable range.",
    },
  };
}

async function parseJSONFromText(prompt, text) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), OPENAI_TIMEOUT_MS);

  try {
    const response = await openai.responses.create({
      model: OPENAI_MODEL,
      temperature: 0,
      max_output_tokens: 5000,
      input: [
        { role: "system", content: prompt },
        { role: "user", content: text.slice(0, MAX_OCR_CHARS) },
      ],
      signal: controller.signal,
    });

    const raw = response.output_text || "";
    const cleaned = raw.trim().replace(/^```json/i, "").replace(/^```/, "").replace(/```$/, "").trim();
    return JSON.parse(cleaned);
  } finally {
    clearTimeout(timeout);
  }
}

async function extractOCRTextFromBuffer(buffer) {
  const poller = await azureClient.beginAnalyzeDocument("prebuilt-read", buffer);
  const result = await poller.pollUntilDone();
  const text = a(result.pages)
    .flatMap((page) => a(page.lines).map((line) => line.content))
    .join("\n");
  if (!text.trim()) throw new Error("OCR extraction returned empty text");
  return text;
}

async function extractCOAFromBuffer(buffer) {
  const ocrText = await extractOCRTextFromBuffer(buffer);
  const [chemRaw, contamRaw] = await Promise.all([
    parseJSONFromText(CHEMISTRY_EXTRACTION_PROMPT, ocrText),
    parseJSONFromText(CONTAMINANT_EXTRACTION_PROMPT, ocrText),
  ]);

  const chemistry = normalizeChemistry(chemRaw);
  const contaminants = normalizeContaminants(contamRaw);
  const scoring = computeIntelligenceScore(chemistry, contaminants);
  const terpIntel = getTerpeneIntel(chemistry.top_terpenes?.[0]?.name || "");
  const intelligence = {
    fingerprintId: generateFingerprintId(chemistry.top_terpenes),
    effectDirection: terpIntel.direction,
    lineageCluster: terpIntel.lineage,
    lineageConfidence: terpIntel.lineageConfidence,
    audiences: buildAudienceNarratives(chemistry, contaminants, scoring),
    postHarvest: buildPostHarvestIntel(chemistry, contaminants),
  };

  return { chemistry, contaminants, scoring, intelligence, ocrText };
}

async function uploadBufferToSupabase({ buffer, originalName, mimeType, folder = "raw_documents" }) {
  const safeName = sanitizeFileName(originalName || `upload-${Date.now()}`);
  const storagePath = `${folder}/${Date.now()}-${safeName}`;
  const { error } = await supabase.storage
    .from(SUPABASE_BUCKET)
    .upload(storagePath, buffer, {
      contentType: mimeType || "application/octet-stream",
      upsert: false,
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
        schema_version: "7.1-dynamic",
      },
    },
    overall_score: scoring.total,
    report_confidence_score: scoring.breakdown?.dataCompleteness?.score ?? 0,
    chemotype_identity: intelligence.fingerprintId,
    chemotype_descriptor: intelligence.effectDirection,
    fingerprint_id: intelligence.fingerprintId,
  };

  const { data, error } = await supabase
    .from("coa_ai_reports")
    .insert([payload])
    .select()
    .single();

  if (error) throw new Error(`Supabase insert failed: ${error.message}`);
  return data;
}

async function getReportById(id) {
  const { data, error } = await supabase
    .from("coa_ai_reports")
    .select("*")
    .eq("id", id)
    .single();
  if (error) throw new Error(`Could not load report: ${error.message}`);
  return data;
}

function renderMetricRow(label, value, width, color) {
  return `
    <div class="mini-metric">
      <div class="mini-metric-label">${esc(label)}</div>
      <div class="mini-track"><div class="mini-fill" style="width:${clamp(width, 0, 100)}%;background:${color};"></div></div>
      <div class="mini-value">${esc(value)}</div>
    </div>
  `;
}

function renderSafeCard(title, verdict, rows, method = "") {
  return `
    <div class="safe-card">
      <div class="safe-card-head">
        <div class="safe-card-title">${esc(title)}</div>
        <div class="safe-card-verdict">${esc(verdict)}</div>
      </div>
      <div class="safe-card-body">
        ${rows.map(([k, v]) => `
          <div class="safe-row"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>
        `).join("")}
      </div>
      ${method ? `<div class="safe-method">${esc(method)}</div>` : ""}
    </div>
  `;
}


function computeDataConfidence(chemistry = {}, contaminants = {}) {
  let points = 0;
  const max = 12;

  if (chemistry.product_name) points += 1;
  if (chemistry.batch_number) points += 1;
  if (chemistry.coa_report_date) points += 1;
  if (chemistry.thc_total) points += 1;
  if (chemistry.total_terpenes) points += 1;
  if (a(chemistry.top_terpenes).length >= 3) points += 1;
  if (a(chemistry.top_cannabinoids).length >= 3) points += 1;
  if (chemistry.water_activity || chemistry.moisture_content) points += 1;

  if (contaminants.pesticides_status) points += 1;
  if (contaminants.heavy_metals_status) points += 1;
  if (contaminants.microbials_status) points += 1;
  if (contaminants.mycotoxins_status) points += 1;

  const pct = Math.round((points / max) * 100);
  let label = "Low";
  if (pct >= 75) label = "High";
  else if (pct >= 45) label = "Medium";

  let tone = "var(--red)";
  if (label === "High") tone = "var(--green2)";
  else if (label === "Medium") tone = "var(--amber)";

  let summary = "This COA is sparse. Interpretation should be limited to the fields explicitly present.";
  if (label === "High") summary = "This COA is sufficiently complete for high-confidence interpretation.";
  else if (label === "Medium") summary = "This COA supports partial interpretation, but some conclusions remain limited.";

  return { points, max, pct, label, tone, summary };
}

function buildMissingDataInsights(chemistry = {}, contaminants = {}) {
  const missing = [];

  if (!chemistry.total_terpenes && !a(chemistry.top_terpenes).length) {
    missing.push({
      label: "Terpene panel missing",
      impact: "Effect direction and aromatic fingerprint cannot be interpreted confidently."
    });
  }

  if (!chemistry.total_cannabinoids && !a(chemistry.top_cannabinoids).length) {
    missing.push({
      label: "Cannabinoid detail missing",
      impact: "Minor cannabinoid richness and internal balance cannot be assessed."
    });
  }

  if (!contaminants.pesticides_status) {
    missing.push({
      label: "Pesticide data missing",
      impact: "Full contaminant clearance cannot be verified from this COA alone."
    });
  }

  if (!contaminants.heavy_metals_status) {
    missing.push({
      label: "Heavy metals data missing",
      impact: "Metal contamination risk cannot be ruled out from this document alone."
    });
  }

  if (!contaminants.microbials_status) {
    missing.push({
      label: "Microbial panel missing",
      impact: "Microbiological stability and pathogen absence cannot be verified."
    });
  }

  if (!contaminants.mycotoxins_status) {
    missing.push({
      label: "Mycotoxin data missing",
      impact: "Toxin-producing mold byproducts cannot be assessed."
    });
  }

  if (!chemistry.water_activity && !chemistry.moisture_content && !contaminants.water_activity && !contaminants.moisture_content) {
    missing.push({
      label: "Post-harvest stability data missing",
      impact: "Storage stability and mold-risk interpretation remain limited."
    });
  }

  return missing;
}

function buildPremiumBadges(chemistry = {}, contaminants = {}, scoring = {}) {
  const badges = [];
  const terpCount = a(chemistry.top_terpenes).filter((t) => toNum(t.value) > 0).length;
  const totalTerps = Number(chemistry.total_terpenes || 0);
  const flavs = a(chemistry.flavonoids).length;

  const fullSafety =
    !!contaminants.pesticides_status &&
    !!contaminants.heavy_metals_status &&
    !!contaminants.microbials_status &&
    !!contaminants.mycotoxins_status;

  if (fullSafety) badges.push({ label: "Complete Safety Panel", tone: "strong" });
  if (totalTerps >= 2.5 || terpCount >= 8) badges.push({ label: "Terpene Richness", tone: "strong" });
  if (Number(scoring?.breakdown?.minors?.score || 0) >= 12) badges.push({ label: "Minor Cannabinoid Depth", tone: "strong" });
  if (flavs > 0) badges.push({ label: "Rare Specialty Data", tone: "strong" });

  if (!fullSafety) badges.push({ label: "Incomplete COA", tone: "warning" });
  if (!chemistry.total_terpenes && !a(chemistry.top_terpenes).length) {
    badges.push({ label: "Interpretation Limited", tone: "warning" });
  }

  return badges;
}


function renderReportHTML(reportJson = {}, options = {}) {
  const chemistry = reportJson.chemistry || {};
  const contaminants = reportJson.contaminants || {};
  const scoring = reportJson.scoring || computeIntelligenceScore(chemistry, contaminants);
  const intelligence = reportJson.intelligence || {};

  const confidence = computeDataConfidence(chemistry, contaminants);
  const missingInsights = buildMissingDataInsights(chemistry, contaminants);
  const premiumBadges = buildPremiumBadges(chemistry, contaminants, scoring);

  const terpenes = a(chemistry.top_terpenes);
  const cannabinoids = a(chemistry.top_cannabinoids);
  const flavonoids = a(chemistry.flavonoids);

  const thc = toNum(chemistry.thc_total);
  const cbd = toNum(chemistry.cbd_total);
  const terps = toNum(chemistry.total_terpenes);
  const cbn = toNum(chemistry.cbn);
  const cbna = toNum(chemistry.cbna);
  const moisture = chemistry.moisture_content || contaminants.moisture_content || "";
  const waterActivity = chemistry.water_activity || contaminants.water_activity || "";

  const leadTerpene = terpenes[0]?.name || "";
  const secondTerpene = terpenes[1]?.name || "";
  const thirdTerpene = terpenes[2]?.name || "";

  const terpIntel = getTerpeneIntel(leadTerpene);
  const fingerprintId = intelligence.fingerprintId || generateFingerprintId(terpenes);

  let audiences;
  try {
    const rawAud = intelligence.audiences;
    audiences = rawAud && Array.isArray(rawAud.brand)
      ? rawAud
      : buildAudienceNarratives(chemistry, contaminants, scoring);
  } catch (_) {
    audiences = buildAudienceNarratives(chemistry, contaminants, scoring);
  }

  audiences.brand = Array.isArray(audiences.brand) ? audiences.brand : [];
  audiences.clinical = Array.isArray(audiences.clinical) ? audiences.clinical : [];
  audiences.patient = Array.isArray(audiences.patient) ? audiences.patient : [];
  audiences.buyer = Array.isArray(audiences.buyer) ? audiences.buyer : [];

  let postHarvest;
  try {
    const rawPH = intelligence.postHarvest;
    postHarvest = rawPH && rawPH.freshness && rawPH.curing && rawPH.degradation && rawPH.stability
      ? rawPH
      : buildPostHarvestIntel(chemistry, contaminants);
  } catch (_) {
    postHarvest = buildPostHarvestIntel(chemistry, contaminants);
  }

  const safePH = {
    freshness: postHarvest.freshness || { label: "Unknown", note: "Data not available." },
    curing: postHarvest.curing || { label: "Unknown", note: "Data not available." },
    degradation: postHarvest.degradation || { label: "Unknown", note: "Data not available." },
    stability: postHarvest.stability || { label: "Unknown", note: "Data not available." },
  };

  const activeTerps = terpenes.filter((t) => toNum(t.value) > 0);
  const maxTerpVal = activeTerps.length ? Math.max(...activeTerps.map((t) => toNum(t.value)), 0.001) : 0.001;

  const productName = chemistry.product_name || "COA Report";
  const heroNarrative = chemistry.hero_narrative ||
    `${leadTerpene ? `${leadTerpene}-dominant` : "Cannabis"} profile with ${terps >= 2.5 ? "strong" : terps >= 1 ? "moderate" : "light"} terpene expression.`;

  const safeTotal = scoring.total ?? 0;
  const safeGrade = scoring.grade || "—";
  const safeTier = scoring.tier || "—";
  const ringCirc = 377;
  const ringOffset = ((100 - clamp(safeTotal, 0, 100)) / 100 * ringCirc).toFixed(1);

  const pestPass = /pass|nd|not detected/i.test(contaminants.pesticides_status || "");
  const metalsPass = /pass|nd|not detected|blq/i.test(contaminants.heavy_metals_status || "");
  const microPass = /pass|nd|not detected|absent/i.test(contaminants.microbials_status || "");
  const mycoPass = /pass|nd|not detected/i.test(contaminants.mycotoxins_status || "");
  const safetyClear = pestPass && metalsPass && microPass && mycoPass;

  const minorTerps = activeTerps.slice(3);

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${esc(productName)} · Alem COA Intelligence</title>
  <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *,*::before,*::after{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{margin:0;background:#f7f5ef;color:#171714;font-family:"DM Sans",system-ui,sans-serif;line-height:1.6;-webkit-font-smoothing:antialiased}
    :root{
      --bg:#f7f5ef;--surface:#fff;--ink:#171714;--ink2:#34342f;--ink3:#616159;--ink4:#919188;
      --line:#e7e3d8;--line2:#d8d3c7;--green:#16855c;--green2:#0f6847;--greenBg:#eef8f3;
      --amber:#b97b17;--amberBg:#fdf7ec;--red:#b74242;--redBg:#fdf1f1;--shadow:0 10px 30px rgba(20,20,18,.06);
      --max:1180px;--pad:clamp(20px,4vw,56px);--serif:"Libre Baskerville",Georgia,serif;--mono:"DM Mono",monospace;
    }
    @keyframes growX{from{transform:scaleX(0)}to{transform:scaleX(1)}}
    .wrap{width:100%;max-width:calc(var(--max) + (var(--pad)*2));margin:0 auto;padding-left:var(--pad);padding-right:var(--pad)}
    .topbar{position:sticky;top:0;z-index:100;background:rgba(255,255,255,.9);backdrop-filter:blur(16px);border-bottom:1px solid rgba(231,227,216,.9)}
    .topbar-inner{min-height:58px;display:flex;align-items:center;gap:14px}
    .brand img{height:24px;display:block}
    .divider{width:1px;height:18px;background:var(--line)}
    .topbar-label{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink4)}
    .topbar-right{margin-left:auto;display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
    .pill{display:inline-flex;align-items:center;border-radius:999px;padding:7px 12px;border:1px solid var(--line);background:var(--surface);font-size:11px;color:var(--ink3);font-family:var(--mono)}
    .pill.score{background:var(--greenBg);border-color:#cbe8d9;color:var(--green2);font-weight:500}
    .hero{padding:clamp(28px,5vw,56px) 0 28px;border-bottom:1px solid var(--line);background:
      radial-gradient(circle at top right, rgba(22,133,92,.08), transparent 28%),
      linear-gradient(180deg, rgba(255,255,255,.75), rgba(255,255,255,.48))}
    .hero-grid{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:clamp(24px,4vw,56px);align-items:start}
    .eyebrow{display:inline-flex;align-items:center;gap:10px;margin-bottom:14px;font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--green);font-weight:700}
    .eyebrow::before{content:"";width:18px;height:1.5px;background:var(--green);border-radius:999px}
    .product-name{margin:0 0 8px;font-size:clamp(42px,8vw,84px);line-height:.92;letter-spacing:-.05em;color:var(--ink)}
    .product-type{font-size:18px;color:var(--ink4);font-weight:300;margin-bottom:24px}
    .hero-verdict{max-width:780px;margin-bottom:20px;font-family:var(--serif);font-size:clamp(20px,2.8vw,31px);line-height:1.5;color:var(--ink);font-style:italic}
    .hero-verdict em{font-style:normal;color:var(--green2);font-weight:700}
    .hero-verdict [data-aud]{display:none}.hero-verdict [data-aud].active{display:inline}
    .meta-row{margin-bottom:18px;font-size:11px;color:var(--ink4);letter-spacing:.04em;font-family:var(--mono)}
    .tag-row{display:flex;flex-wrap:wrap;gap:8px}
    .tag{display:inline-flex;align-items:center;min-height:30px;padding:6px 11px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.65);color:var(--ink3);font-size:11px;font-weight:500}
    .tag.strong{background:var(--greenBg);color:var(--green2);border-color:#c8e7d8}
.tag.warning{background:var(--amberBg);color:var(--amber);border-color:#ead4a6}
    .hero-meta-stack{display:flex;flex-direction:column;gap:12px;margin-top:18px}
    .confidence-chip{display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border:1px solid var(--line);border-radius:999px;background:var(--surface);width:fit-content}
    .confidence-chip strong{font-size:12px;letter-spacing:.08em;text-transform:uppercase}
    .confidence-chip span{font-size:12px;color:var(--ink3);font-family:var(--mono)}
    .score-panel{position:sticky;top:78px;background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:20px;padding:24px 22px;box-shadow:var(--shadow)}
    .score-ring{position:relative;width:146px;height:146px;margin:0 auto 18px}
    .score-ring svg{position:absolute;inset:0;transform:rotate(-90deg)}
    .score-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
    .score-value{font-size:52px;font-weight:700;line-height:1;letter-spacing:-.04em;color:var(--ink)}
    .score-max{font-size:12px;color:var(--ink4);font-family:var(--mono)}
    .score-grade{margin-top:3px;font-family:var(--serif);font-size:18px;color:var(--green2)}
    .score-label{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink4)}
    .mini-metrics{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--surface)}
    .mini-metric{display:grid;grid-template-columns:1fr 50px 42px;gap:10px;align-items:center;padding:11px 13px;border-bottom:1px solid var(--line);font-size:12px}
    .mini-metric:last-child{border-bottom:none}
    .mini-metric-label{color:var(--ink3)}
    .mini-track{height:3px;background:var(--line);border-radius:999px;overflow:hidden}
    .mini-fill{height:100%;border-radius:inherit;transform-origin:left;animation:growX .75s cubic-bezier(.16,1,.3,1) both}
    .mini-value{text-align:right;font-size:11px;color:var(--ink);font-family:var(--mono)}
    .score-note{margin-top:14px;font-size:11px;color:var(--ink4);text-align:center;line-height:1.65}
    .audience-bar{background:var(--surface);border-bottom:1px solid var(--line)}
    .audience-inner{min-height:62px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
    .audience-label{margin-right:8px;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--ink4)}
    .aud-btn{appearance:none;border:1px solid transparent;background:transparent;color:var(--ink3);border-radius:999px;padding:10px 14px;font-size:13px;font-weight:600;cursor:pointer;transition:.18s ease}
    .aud-btn:hover{background:#f5f3ed;color:var(--ink)} .aud-btn.active{background:var(--greenBg);color:var(--green2);border-color:#cae8da}
    .section{padding:clamp(28px,5vw,56px) 0;border-bottom:1px solid var(--line)}
    .section.alt{background:linear-gradient(180deg, rgba(255,255,255,.72), rgba(255,255,255,.96))}
    .section-head{margin-bottom:24px;max-width:760px}
    .section-head h2,.section-head h3{margin:0 0 10px;font-family:var(--serif);font-size:clamp(27px,3.3vw,40px);line-height:1.18;color:var(--ink);font-weight:400}
    .section-head p{margin:0;font-size:15px;color:var(--ink3);line-height:1.8}
    .summary-grid{display:grid;grid-template-columns:220px minmax(0,1fr);gap:18px;align-items:stretch}
    .summary-score,.card,.market-card,.intensity,.terp-card,.intel-shell,.panel-card,.harvest-card{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--shadow)}
    .summary-score{display:flex;flex-direction:column;justify-content:center;min-height:220px}
    .summary-score .small,.card-title,.market-label,.intensity-label,.safe-card-title,.harvest-label{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink4);margin-bottom:10px}
    .summary-score .big{font-size:58px;line-height:1;letter-spacing:-.05em;color:var(--ink);font-weight:700}
    .summary-score .sub{margin-top:6px;color:var(--green2);font-family:var(--serif);font-size:18px}
    .summary-main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
    .fingerprint{font-family:var(--mono);font-size:28px;color:var(--green2);letter-spacing:.08em;margin-bottom:10px}
    .muted{color:var(--ink3);font-size:13px;line-height:1.7}
    .bullets{display:grid;gap:12px}
    .bullet{display:grid;grid-template-columns:18px minmax(0,1fr);gap:10px;align-items:start;font-size:14px;color:var(--ink2)}
    .bullet-num{font-family:var(--mono);color:var(--green2);font-size:11px;padding-top:2px}
    .safety-banner{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:18px;align-items:center;background:var(--greenBg);border:1px solid #c7e7d7;border-left:4px solid var(--green);border-radius:0 18px 18px 0;padding:20px 22px;margin-bottom:18px}
    .safety-banner h3{margin:0 0 4px;font-size:18px;color:var(--ink)} .safety-banner p{margin:0;font-size:13px;color:var(--ink3)}
    .confidence{text-align:right;min-width:120px}.confidence strong{display:block;color:var(--green2);font-size:30px;line-height:1;font-family:var(--mono);font-weight:500}.confidence span{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--ink4)}
    .grid-4,.grid-3{display:grid;gap:14px}.grid-4{grid-template-columns:repeat(4,minmax(0,1fr))}.grid-3{grid-template-columns:repeat(3,minmax(0,1fr))}
    .safe-card{background:var(--surface);border:1px solid var(--line);border-radius:16px;overflow:hidden}
    .safe-card-head{padding:16px 16px 12px;border-bottom:1px solid var(--line)}
    .safe-card-verdict{font-size:16px;color:var(--green2);font-weight:700}
    .safe-card-body{padding:12px 16px;display:grid;gap:6px;font-size:12px;color:var(--ink3)}
    .safe-row{display:flex;justify-content:space-between;gap:10px}.safe-row strong{color:var(--ink);font-family:var(--mono);font-size:11px;font-weight:500}
    .safe-method{padding:10px 16px 14px;border-top:1px solid var(--line);font-family:var(--mono);font-size:10px;color:var(--ink4)}
    .callout{border-radius:18px;padding:18px 20px;border-left:4px solid;margin-bottom:18px}.callout.amber{background:var(--amberBg);border-color:var(--amber)}.callout.green{background:var(--greenBg);border-color:var(--green)}
    .callout strong{display:block;color:var(--ink);margin-bottom:5px;font-size:15px}.callout p{margin:0;color:var(--ink3);font-size:13px;line-height:1.7}
    .two-col{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start}
    .stat-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:18px;overflow:hidden;margin-bottom:18px}
    .stat{background:var(--surface);padding:22px 18px;text-align:center}.stat-value{font-family:var(--mono);font-size:clamp(26px,3vw,36px);line-height:1;color:var(--ink);margin-bottom:8px}.stat-label{font-size:10px;text-transform:uppercase;letter-spacing:.11em;color:var(--ink4)}
    .market-bar{position:relative;height:42px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#f6f3eb;margin-bottom:8px}
    .zone-low,.zone-mid,.zone-high{position:absolute;top:0;bottom:0}.zone-low{left:0;width:55%;background:rgba(145,145,136,.08)}.zone-mid{left:55%;width:20%;background:rgba(185,123,23,.09)}.zone-high{left:75%;right:0;background:rgba(22,133,92,.11)}
    .market-fill{position:absolute;left:0;top:0;bottom:0;width:${clamp((thc / 30) * 100, 0, 100)}%;background:linear-gradient(90deg,var(--green),var(--green2));transform-origin:left;animation:growX .9s cubic-bezier(.16,1,.3,1) both}
    .market-pin{position:absolute;left:${clamp((thc / 30) * 100, 0, 100)}%;top:50%;transform:translate(-50%,-50%);background:var(--green2);color:#fff;padding:4px 10px;border-radius:999px;font-size:10px;font-family:var(--mono);white-space:nowrap}
    .market-legend,.intensity-legend{display:flex;justify-content:space-between;gap:10px;font-size:9px;color:var(--ink4);font-family:var(--mono)}
    .intensity-scale{position:relative;height:6px;background:linear-gradient(90deg,#dcd7ca 0%, #ccb686 55%, #b97b17 78%, #8b4c15 100%);border-radius:999px;margin:16px 6px 8px}
    .intensity-marker{position:absolute;left:${clamp((thc / 30) * 100, 0, 100)}%;top:50%;width:14px;height:14px;border-radius:50%;transform:translate(-50%,-50%);background:#fff;border:2px solid var(--ink);box-shadow:0 2px 8px rgba(0,0,0,.12)}
    .cannabinoid-table{background:var(--surface);border:1px solid var(--line);border-radius:18px;overflow:hidden}
    .ct-row{display:grid;grid-template-columns:120px minmax(0,1fr) 90px;gap:12px;align-items:center;padding:12px 18px;border-bottom:1px solid var(--line)}
    .ct-row:last-child,.flav-row:last-child{border-bottom:none}.ct-row.head{background:#faf8f3;padding-top:10px;padding-bottom:10px;font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink4)}
    .ct-name{font-family:var(--mono);font-size:12px;color:var(--ink2)} .ct-bar,.flav-track,.top-terp-track,.mini-track{height:4px;background:var(--line);border-radius:999px;overflow:hidden}
    .ct-fill,.top-terp-fill,.flav-fill{height:100%;border-radius:inherit;transform-origin:left;animation:growX .8s cubic-bezier(.16,1,.3,1) both}
    .ct-value,.top-terp-value,.flav-value{text-align:right;font-family:var(--mono);font-size:12px;color:var(--ink3)}
    .terp-layout{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr);gap:18px;align-items:start}
    .terp-total{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin-bottom:18px}
    .terp-total-label{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink4);white-space:nowrap}.terp-total-track{height:7px;border-radius:999px;background:var(--line);overflow:hidden}
    .terp-total-fill{width:${clamp((terps / 4) * 100, 0, 100)}%;height:100%;background:linear-gradient(90deg,var(--green),var(--green2));border-radius:inherit;transform-origin:left;animation:growX 1s cubic-bezier(.16,1,.3,1) both}
    .terp-total-value{font-family:var(--mono);color:var(--ink);font-size:18px;white-space:nowrap}
    .top-terp-list{display:grid;gap:12px}.top-terp{border-bottom:1px solid var(--line);padding-bottom:12px}.top-terp:last-child{border-bottom:none;padding-bottom:0}
    .top-terp-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px}.top-terp-name{font-size:14px;font-weight:700;color:var(--ink)}.top-terp-note{font-size:12px;color:var(--ink4);font-style:italic}.top-terp-copy{font-size:12px;color:var(--ink3);line-height:1.7}
    .fingerprint-card{background:linear-gradient(180deg,#1d1d1a,#121210);color:#fff;border-radius:18px;padding:20px;box-shadow:var(--shadow);overflow:hidden;position:relative}
    .fingerprint-card::before{content:"";position:absolute;inset:0 0 auto 0;height:2px;background:linear-gradient(90deg,var(--green),#73d5ae)}
    .fingerprint-card .label{font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:rgba(255,255,255,.45);margin-bottom:8px}
    .fingerprint-card .value{font-family:var(--mono);font-size:26px;letter-spacing:.12em;color:#77ddb6;margin-bottom:10px}
    .fingerprint-card .copy{font-size:13px;color:rgba(255,255,255,.78);line-height:1.75}
    .expand-box{margin-top:14px;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#fbfaf7}
    .expand-toggle{width:100%;appearance:none;border:none;background:transparent;text-align:left;padding:14px 16px;font:inherit;color:var(--ink);cursor:pointer;display:flex;justify-content:space-between;gap:12px;align-items:center;font-weight:600}
    .expand-toggle span:last-child{color:var(--ink4);font-family:var(--mono);font-size:11px}
    .expand-content{display:none;padding:0 16px 16px;border-top:1px solid var(--line)}.expand-box.open .expand-content{display:block}
    .mini-list{display:grid;gap:10px;padding-top:14px}.mini-item{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:var(--ink3);border-bottom:1px dashed #e8e3d9;padding-bottom:8px}.mini-item:last-child{border-bottom:none;padding-bottom:0}
    .mini-item strong{color:var(--ink);font-weight:500}
    .intel-tabs{display:inline-flex;gap:4px;padding:4px;background:#f5f2ea;border:1px solid var(--line);border-radius:999px;margin-bottom:20px}
    .intel-tab{appearance:none;border:none;background:transparent;padding:9px 16px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:.04em;color:var(--ink4);cursor:pointer}
    .intel-tab.active{background:var(--green2);color:#fff}
    .intel-panel{display:none}.intel-panel.active{display:grid;gap:12px}
    .intel-row{display:grid;grid-template-columns:24px minmax(0,1fr);gap:12px;align-items:start;padding-bottom:12px;border-bottom:1px solid var(--line)}.intel-row:last-child{border-bottom:none;padding-bottom:0}
    .intel-row .num{font-family:var(--mono);font-size:11px;color:var(--green2);padding-top:2px}.intel-row .copy{font-size:14px;color:var(--ink2);line-height:1.8}
    .harvest-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.harvest-value{font-size:20px;color:var(--green2);font-weight:700;margin-bottom:7px}.harvest-copy{font-size:12px;color:var(--ink3);line-height:1.75}
    .advanced-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start}
    .flav-row{display:grid;grid-template-columns:120px minmax(0,1fr) 90px;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line)}
    .flav-name{font-size:12px;color:var(--ink2)} .flav-fill{background:linear-gradient(90deg, rgba(22,133,92,.32), rgba(22,133,92,.68))}
    .lab-strip{background:var(--surface);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
    .lab-grid{display:flex;gap:clamp(20px,3vw,44px);flex-wrap:wrap;align-items:center;padding-top:18px;padding-bottom:18px}
    .lab-item strong{display:block;font-size:13px;color:var(--ink)}.lab-item span{display:block;margin-top:3px;font-size:9px;text-transform:uppercase;letter-spacing:.11em;color:var(--ink4)}
    .lab-item.last{margin-left:auto}.actions{display:flex;gap:10px;flex-wrap:wrap;padding-top:16px;padding-bottom:16px}
    .btn{appearance:none;text-decoration:none;border-radius:999px;padding:11px 18px;font-size:12px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;display:inline-flex;align-items:center;justify-content:center}
    .btn.primary{background:var(--green2);color:#fff;border:1px solid var(--green2)} .btn.secondary{background:transparent;color:var(--ink3);border:1px solid var(--line2)}
    footer{background:var(--surface);padding:26px 0 34px}.footer-inner{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;flex-wrap:wrap}
    .footer-copy{font-size:11px;color:var(--ink4);margin-top:8px}.footer-line{font-family:var(--serif);font-size:15px;color:var(--ink4);line-height:1.8;text-align:right}
    @media(max-width:1100px){.hero-grid,.summary-grid,.two-col,.terp-layout,.advanced-grid{grid-template-columns:1fr}.score-panel{position:static}.grid-4,.harvest-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.summary-main{grid-template-columns:1fr}.lab-item.last{margin-left:0}}
    @media(max-width:720px){.grid-4,.grid-3,.harvest-grid{grid-template-columns:1fr}.stat-row{grid-template-columns:1fr}.ct-row,.flav-row{grid-template-columns:98px minmax(0,1fr) 72px;gap:10px;padding-left:14px;padding-right:14px}.topbar-label,.divider{display:none}.confidence{text-align:left}.footer-line{text-align:left}}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="wrap">
      <div class="topbar-inner">
        <div class="brand">
          <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png" alt="Alem">
        </div>
        <div class="divider"></div>
        <div class="topbar-label">COA Intelligence Report</div>
        <div class="topbar-right">
          <div class="pill">Schema v7.0</div>
          <div class="pill score">${esc(safeGrade)} · ${esc(String(safeTotal))} / 100</div>
        </div>
      </div>
    </div>
  </header>

  <section class="hero">
    <div class="wrap">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Certificate of Analysis</div>
          <h1 class="product-name">${esc(productName)}</h1>
          <div class="product-type">${esc(chemistry.product_type || "Cannabis Product")}</div>

          <div class="hero-verdict" id="heroVerdict">
            <span data-aud="patient" class="active">
              <em>${esc(chemistry.thc_total || "—")}% THC</em> with ${chemistry.cbd_total ? `${esc(chemistry.cbd_total)}% CBD` : "CBD not detected"} and a ${esc(leadTerpene || "lead terpene")} front profile. ${esc(audiences.patient[0] || heroNarrative)}
            </span>
            <span data-aud="clinician">
              <em>${esc(chemistry.thc_total || "—")}% THC</em>, ${chemistry.cbd_total ? `${esc(chemistry.cbd_total)}% CBD` : "CBD ND"}, ${esc(fingerprintId)} chemotype, and score ${esc(String(safeTotal))}/100.
            </span>
            <span data-aud="buyer">
              <em>${esc(String(safeTotal))}/100 ${esc(safeTier)}</em> with ${esc(chemistry.total_terpenes || "—")} wt% terpenes, ${esc(String(terpenes.length))} terpene rows, and ${safetyClear ? "clear safety panels" : "mixed safety completeness"}.
            </span>
          </div>

          <div class="meta-row">
            ${esc(chemistry.laboratory_name || "Laboratory not stated")} ·
            ${esc(chemistry.batch_number || "Batch n/a")} ·
            ${esc(chemistry.coa_report_date || "Date n/a")}
          </div>

          <div class="hero-meta-stack">
            <div class="confidence-chip">
              <strong style="color:${confidence.tone}">Data Confidence</strong>
              <span>${esc(confidence.label)} · ${esc(String(confidence.pct))}%</span>
            </div>

            <div class="tag-row">
              <span class="tag strong">${esc(fingerprintId)}</span>
              <span class="tag ${safetyClear ? "strong" : "warning"}">${safetyClear ? "Full safety compliance" : "Safety review required"}</span>
              ${chemistry.laboratory_accreditation ? `<span class="tag strong">${esc(chemistry.laboratory_accreditation)}</span>` : ""}
              ${leadTerpene ? `<span class="tag">${esc(leadTerpene)} dominant</span>` : ""}
              ${terpenes.length ? `<span class="tag">${esc(String(terpenes.length))} terpenes detected</span>` : ""}
              ${chemistry.total_terpenes ? `<span class="tag">${esc(chemistry.total_terpenes)} wt% terpenes</span>` : ""}
              ${chemistry.cbd_total ? `<span class="tag">CBD present</span>` : `<span class="tag">CBD ND</span>`}
              ${premiumBadges.map((b) => `<span class="tag ${b.tone === "strong" ? "strong" : "warning"}">${esc(b.label)}</span>`).join("")}
            </div>
          </div>
        </div>

        <aside class="score-panel">
          <div class="score-ring">
            <svg viewBox="0 0 146 146" aria-hidden="true">
              <defs>
                <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stop-color="#16855c"></stop>
                  <stop offset="100%" stop-color="#0f6847"></stop>
                </linearGradient>
              </defs>
              <circle cx="73" cy="73" r="60" fill="none" stroke="#e7e3d8" stroke-width="8"></circle>
              <circle cx="73" cy="73" r="60" fill="none" stroke="url(#ringGrad)" stroke-width="8" stroke-linecap="round"
                stroke-dasharray="${ringCirc}" stroke-dashoffset="${ringOffset}"></circle>
            </svg>
            <div class="score-center">
              <div class="score-value">${esc(String(safeTotal))}</div>
              <div class="score-max">/100</div>
              <div class="score-grade">${esc(safeGrade)}</div>
              <div class="score-label">${esc(safeTier)}</div>
            </div>
          </div>

          <div class="mini-metrics">
            ${renderMetricRow("Potency", `${scoring.breakdown?.potency?.score || 0}/25`, ((scoring.breakdown?.potency?.score || 0) / 25) * 100, "#b97b17")}
            ${renderMetricRow("Terpenes", `${scoring.breakdown?.terpenes?.score || 0}/25`, ((scoring.breakdown?.terpenes?.score || 0) / 25) * 100, "#16855c")}
            ${renderMetricRow("Minor Cann.", `${scoring.breakdown?.minors?.score || 0}/20`, ((scoring.breakdown?.minors?.score || 0) / 20) * 100, "#b97b17")}
            ${renderMetricRow("Safety", `${scoring.breakdown?.safety?.score || 0}/20`, ((scoring.breakdown?.safety?.score || 0) / 20) * 100, "#16855c")}
            ${renderMetricRow("Completeness", `${scoring.breakdown?.dataCompleteness?.score || 0}/10`, ((scoring.breakdown?.dataCompleteness?.score || 0) / 10) * 100, "#16855c")}
          </div>

          <div class="score-note">
            Scores the <strong>COA document</strong>, not the product. More complete data creates a higher score ceiling.
          </div>
        </aside>
      </div>
    </div>
  </section>

  <nav class="audience-bar">
    <div class="wrap">
      <div class="audience-inner">
        <div class="audience-label">View lens</div>
        <button class="aud-btn active" data-aud-btn="patient" type="button">Patient</button>
        <button class="aud-btn" data-aud-btn="clinician" type="button">Clinician</button>
        <button class="aud-btn" data-aud-btn="buyer" type="button">Buyer / Brand</button>
      </div>
    </div>
  </nav>

  <section class="section alt">
    <div class="wrap">
      <div class="section-head">
        <div class="eyebrow">Core summary</div>
        <h2>One-page understanding before the deep dive</h2>
        <p>
          This batch shows <strong>${esc(chemistry.thc_total || "—")} total THC</strong>,
          ${chemistry.cbd_total ? `<strong>${esc(chemistry.cbd_total)} CBD</strong>` : `<strong>no CBD detected</strong>`},
          ${chemistry.total_terpenes ? `<strong>${esc(chemistry.total_terpenes)} wt% terpenes</strong>` : `a <strong>reported terpene profile</strong>`},
          and <em>${safetyClear ? "clear safety coverage" : "mixed safety coverage"}</em>.
        </p>
      </div>

      <div class="summary-grid">
        <div class="summary-score">
          <div class="small">Intelligence score</div>
          <div class="big">${esc(String(safeTotal))}</div>
          <div class="sub">${esc(safeTier)} ${esc(safeGrade)}</div>
        </div>

        <div class="summary-main">
          <div class="card">
            <div class="card-title">Chemotype fingerprint</div>
            <div class="fingerprint">${esc(fingerprintId)}</div>
            <div class="muted">${esc(heroNarrative)}</div>
          </div>

          <div class="card">
            <div class="card-title">Three things that matter most</div>
            <div class="bullets">
              <div class="bullet"><div class="bullet-num">01</div><div><strong>${esc(chemistry.thc_total || "—")} total THC</strong> with ${chemistry.cbd_total ? `<strong>${esc(chemistry.cbd_total)} CBD</strong>` : `<strong>CBD not detected</strong>`} defines the cannabinoid balance.</div></div>
              <div class="bullet"><div class="bullet-num">02</div><div>${chemistry.total_terpenes ? `<strong>${esc(chemistry.total_terpenes)} wt% total terpenes</strong>` : `<strong>Terpene rows reported</strong>`} across <strong>${esc(String(terpenes.length))} compounds</strong>.</div></div>
              <div class="bullet"><div class="bullet-num">03</div><div><strong>${safetyClear ? "Safety panels read clear" : "Safety panels require closer review"}</strong> based on the contaminant data returned by the COA.</div></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="wrap">
      <div class="section-head">
        <div class="eyebrow">Data confidence</div>
        <h2>How complete is this COA?</h2>
        <p>${esc(confidence.summary)}</p>
      </div>

      <div class="two-col">
        <div class="card">
          <div class="card-title">Confidence level</div>
          <div style="font-size:34px;font-weight:700;color:${confidence.tone};margin-bottom:10px;">${esc(confidence.label)}</div>
          <div class="mini-track" style="height:8px;margin-bottom:10px;">
            <div class="mini-fill" style="width:${clamp(confidence.pct, 0, 100)}%;background:${confidence.tone};"></div>
          </div>
          <div class="muted">${esc(String(confidence.pct))}% of core report fields were present and interpretable.</div>
        </div>

        <div class="card">
          <div class="card-title">Missing data insights</div>
          <div class="bullets">
            ${missingInsights.length ? missingInsights.map((m, idx) => `
              <div class="bullet"><div class="bullet-num">${String(idx + 1).padStart(2, "0")}</div><div><strong>${esc(m.label)}</strong> — ${esc(m.impact)}</div></div>
            `).join("") : `<div class="muted">No major data gaps detected.</div>`}
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="wrap">
      <div class="section-head">
        <div class="eyebrow">Safety first</div>
        <h2>Fully tested across all major safety categories</h2>
        <p>
          This section consolidates pesticides, heavy metals, microbials, and mycotoxins into one readable view.
        </p>
      </div>

      <div class="safety-banner">
        <div>
          <h3>${safetyClear ? "All clear — full panel safety compliance" : "Safety panel present — review details below"}</h3>
          <p>${esc(contaminants.contaminant_narrative || "Use the panel cards below to interpret the exact safety data returned by the COA.")}</p>
        </div>
        <div class="confidence">
          <strong>${safetyClear ? "HIGH" : "MIXED"}</strong>
          <span>Confidence level</span>
        </div>
      </div>

      <div class="grid-4">
        ${renderSafeCard("Pesticides", contaminants.pesticides_status || "Not reported", [
          ["Compounds tested", contaminants.pesticides_compound_count || "—"],
          ["Summary", contaminants.pesticides_detail || "—"],
          ["Method", contaminants.pesticides_method || "—"],
        ], contaminants.pesticides_method)}

        ${renderSafeCard("Heavy Metals", contaminants.heavy_metals_status || "Not reported", [
          ["Arsenic", contaminants.arsenic_result || "—"],
          ["Cadmium", contaminants.cadmium_result || "—"],
          ["Lead", contaminants.lead_result || "—"],
          ["Mercury", contaminants.mercury_result || "—"],
        ], contaminants.heavy_metals_method)}

        ${renderSafeCard("Microbials", contaminants.microbials_status || "Not reported", [
          ["Yeast & Mold", contaminants.yeast_mold || "—"],
          ["Total Aerobic", contaminants.total_aerobic || "—"],
          ["Salmonella", contaminants.salmonella || "—"],
          ["E. coli", contaminants.e_coli || "—"],
        ], contaminants.microbials_method)}

        ${renderSafeCard("Mycotoxins", contaminants.mycotoxins_status || "Not reported", [
          ["Aflatoxin B1", contaminants.aflatoxin_b1 || "—"],
          ["Aflatoxin B2", contaminants.aflatoxin_b2 || "—"],
          ["Ochratoxin A", contaminants.ochratoxin_a || "—"],
          ["Water activity", waterActivity || "—"],
        ], contaminants.mycotoxins_method)}
      </div>
    </div>
  </section>

  <section class="section alt">
    <div class="wrap">
      <div class="section-head">
        <div class="eyebrow">Potency</div>
        <h2>Dynamic potency section driven by report data</h2>
        <p>
          This section is now fully dynamic. It reads the product actually stored in <code>report_json</code>,
          instead of a hard-coded strain or fixed potency narrative.
        </p>
      </div>

      <div class="callout amber">
        <strong>Intensity note</strong>
        <p data-role-copy="patient">${esc(audiences.patient[0] || "Start low and build slowly.")}</p>
        <p data-role-copy="clinician" style="display:none;">${esc(audiences.clinical[0] || "Review cannabinoid balance and titration needs.")}</p>
        <p data-role-copy="buyer" style="display:none;">${esc(audiences.buyer[0] || "Use potency in context with terpene depth and safety.")}</p>
      </div>

      <div class="two-col">
        <div>
          <div class="stat-row">
            <div class="stat"><div class="stat-value">${esc(chemistry.thc_total || "—")}${chemistry.thc_total ? "%" : ""}</div><div class="stat-label">Total THC</div></div>
            <div class="stat"><div class="stat-value">${esc(chemistry.cbd_total || "ND")}${chemistry.cbd_total ? "%" : ""}</div><div class="stat-label">CBD Total</div></div>
            <div class="stat"><div class="stat-value">${cbn < 0.3 && cbna < 0.1 ? "Fresh" : "Mixed"}</div><div class="stat-label">Degradation signal</div></div>
          </div>

          <div class="market-card">
            <div class="market-label">Market context · dried flower THC</div>
            <div class="market-bar">
              <div class="zone-low"></div><div class="zone-mid"></div><div class="zone-high"></div>
              <div class="market-fill"></div>
              <div class="market-pin">${esc(chemistry.thc_total || "—")}%</div>
            </div>
            <div class="market-legend"><span>0%</span><span>avg 18%</span><span>premium 22%</span><span>top tier</span><span>30%+</span></div>
          </div>
        </div>

        <div>
          <div class="intensity">
            <div class="intensity-label">Experience intensity scale</div>
            <div class="intensity-scale"><div class="intensity-marker"></div></div>
            <div class="intensity-legend"><span>Low</span><span>Moderate</span><span>High</span><span>Very high</span></div>
          </div>

          <div class="callout green" style="margin-top:18px;">
            <strong>Conversion insight</strong>
            <p>
              THCA: <strong>${esc(chemistry.thca || "—")}</strong> · D9-THC: <strong>${esc(chemistry.d9thc || "—")}</strong>.
              If the COA reports both, total THC reflects active potential after decarboxylation.
            </p>
          </div>
        </div>
      </div>

      <div style="margin-top:18px;">
        <div class="cannabinoid-table">
          <div class="ct-row head"><div>Compound</div><div>Relative abundance</div><div style="text-align:right;">wt%</div></div>
          ${cannabinoids.length ? cannabinoids.map((c, idx) => `
            <div class="ct-row">
              <div class="ct-name">${esc(c.name)}</div>
              <div class="ct-bar"><div class="ct-fill" style="width:${clamp((toNum(c.value) / Math.max(...cannabinoids.map(x => toNum(x.value)), 0.001)) * 100, 0, 100)}%;background:${idx === 0 ? "#16855c" : idx === 1 ? "#0f6847" : "#b97b17"};"></div></div>
              <div class="ct-value">${esc(c.value || "—")}</div>
            </div>
          `).join("") : `
            <div class="ct-row"><div class="ct-name">No cannabinoid rows stored</div><div class="ct-bar"></div><div class="ct-value">—</div></div>
          `}
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="wrap">
      <div class="section-head">
        <div class="eyebrow">Terpene architecture</div>
        <h2>Fully dynamic terpene section</h2>
        <p>
          The old static “Lucy in the Sky” content is gone. Everything below now reads directly from
          <strong>chemistry.top_terpenes</strong>, <strong>chemistry.total_terpenes</strong>, and the stored intelligence layer.
        </p>
      </div>

      <div class="terp-layout">
        <div class="terp-card">
          <div class="terp-total">
            <div class="terp-total-label">Total terpenes</div>
            <div class="terp-total-track"><div class="terp-total-fill"></div></div>
            <div class="terp-total-value">${esc(chemistry.total_terpenes || "—")} ${chemistry.total_terpenes ? "wt%" : ""}</div>
          </div>

          <div class="top-terp-list">
            ${activeTerps.slice(0, 3).map((t, idx) => `
              <div class="top-terp">
                <div class="top-terp-head">
                  <div>
                    <div class="top-terp-name">${esc(t.name)}</div>
                    <div class="top-terp-note">${idx === 0 ? "lead terpene" : idx === 1 ? "secondary terpene" : "tertiary terpene"}</div>
                  </div>
                  <div class="top-terp-value">${esc(t.value)} ${esc(t.unit || "wt%")}</div>
                </div>
                <div class="top-terp-track"><div class="top-terp-fill" style="width:${clamp((toNum(t.value) / maxTerpVal) * 100, 0, 100)}%;background:${idx === 0 ? "#16855c" : idx === 1 ? "#2fa5ad" : "#7389be"};"></div></div>
                <div class="top-terp-copy">
                  ${idx === 0 ? esc(terpIntel.note) : "Part of the broader chemotype architecture for this specific batch."}
                </div>
              </div>
            `).join("") || `<div class="top-terp"><div class="top-terp-copy">No terpene rows were stored in this report.</div></div>`}
          </div>

          ${minorTerps.length ? `
            <div class="expand-box" id="moreTerps">
              <button class="expand-toggle" type="button" onclick="toggleExpand('moreTerps')">
                <span>+${minorTerps.length} minor compounds</span>
                <span>Expand</span>
              </button>
              <div class="expand-content">
                <div class="mini-list">
                  ${minorTerps.map((t) => `
                    <div class="mini-item"><span><strong>${esc(t.name)}</strong></span><span>${esc(t.value)} ${esc(t.unit || "wt%")}</span></div>
                  `).join("")}
                </div>
              </div>
            </div>
          ` : ""}
        </div>

        <div style="display:grid;gap:18px;">
          <div class="fingerprint-card">
            <div class="label">Chemotype fingerprint</div>
            <div class="value">${esc(fingerprintId)}</div>
            <div class="copy">
              <strong>Lead:</strong> ${esc(leadTerpene || "Not identified")}<br>
              ${secondTerpene ? `<strong>2nd:</strong> ${esc(secondTerpene)}<br>` : ""}
              ${thirdTerpene ? `<strong>3rd:</strong> ${esc(thirdTerpene)}<br>` : ""}
              <br>${esc(terpIntel.note)}
            </div>
          </div>

          <div class="card">
            <div class="card-title">Effect direction</div>
            <div style="font-size:24px;font-weight:700;color:var(--green2);margin-bottom:8px;">${esc(terpIntel.direction)}</div>
            <div class="muted">
              ${esc(terpIntel.lineage)} · confidence ${esc(terpIntel.lineageConfidence)}
            </div>
          </div>

          <div class="card">
            <div class="card-title">Lab / quality note</div>
            <div class="muted">${esc(contaminants.lab_quality_summary || chemistry.laboratory_accreditation || "Use the metadata and safety panels to interpret lab quality and completeness.")}</div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="section alt">
    <div class="wrap">
      <div class="section-head">
        <div class="eyebrow">What this means</div>
        <h2>The same chemistry, explained for different people</h2>
        <p>This swaps dynamically between patient, clinician, and buyer narratives built from the stored data.</p>
      </div>

      <div class="intel-shell">
        <div class="intel-tabs">
          <button class="intel-tab active" type="button" data-intel-tab="patient">Patient</button>
          <button class="intel-tab" type="button" data-intel-tab="clinician">Clinician</button>
          <button class="intel-tab" type="button" data-intel-tab="buyer">Buyer / Brand</button>
        </div>

<div class="intel-panel" data-intel-panel="buyer">
  ${(audiences.buyer?.length ? audiences.buyer : (audiences.brand || [])).map((item, idx) => `
    <div class="intel-row"><div class="num">${String(idx + 1).padStart(2, "0")}</div><div class="copy">${esc(item)}</div></div>
  `).join("")}
</div>

        <div class="intel-panel" data-intel-panel="clinician">
          ${(audiences.clinical || []).map((item, idx) => `
            <div class="intel-row"><div class="num">${String(idx + 1).padStart(2, "0")}</div><div class="copy">${esc(item)}</div></div>
          `).join("")}
        </div>

        <div class="intel-panel" data-intel-panel="buyer">
          {(audiences.buyer?.length ? audiences.buyer : audiences.brand || []).map((item, idx) => `
            <div class="intel-row"><div class="num">${String(idx + 1).padStart(2, "0")}</div><div class="copy">${esc(item)}</div></div>
          `).join("")}
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="wrap">
      <div class="section-head">
        <div class="eyebrow">Post-harvest signals</div>
        <h2>What the chemistry suggests about handling and preservation</h2>
        <p>These are chemistry-based signals, not absolute claims.</p>
      </div>

      <div class="harvest-grid">
        <div class="harvest-card"><div class="harvest-label">Freshness</div><div class="harvest-value">${esc(safePH.freshness.label)}</div><div class="harvest-copy">${esc(safePH.freshness.note)}</div></div>
        <div class="harvest-card"><div class="harvest-label">Curing signal</div><div class="harvest-value">${esc(safePH.curing.label)}</div><div class="harvest-copy">${esc(safePH.curing.note)}</div></div>
        <div class="harvest-card"><div class="harvest-label">Degradation</div><div class="harvest-value">${esc(safePH.degradation.label)}</div><div class="harvest-copy">${esc(safePH.degradation.note)}</div></div>
        <div class="harvest-card"><div class="harvest-label">Stability</div><div class="harvest-value">${esc(safePH.stability.label)}</div><div class="harvest-copy">${esc(safePH.stability.note)}</div></div>
      </div>
    </div>
  </section>

  ${flavonoids.length ? `
    <section class="section alt" id="advancedSection">
      <div class="wrap">
        <div class="section-head">
          <div class="eyebrow">Advanced data</div>
          <h2>Rare or premium analytical layers</h2>
          <p>This section only appears when flavonoid or specialty panel data exists in the report.</p>
        </div>

        <div class="advanced-grid">
          <div class="panel-card">
            <div class="card-title">Flavonoid profile</div>
            <h3 style="margin:0 0 10px;font-family:var(--serif);font-size:24px;font-weight:400;">Flavonoid panel detected</h3>
            <div class="muted" style="margin-bottom:16px;">Total flavonoids: ${esc(chemistry.total_flavonoids || "—")} ${chemistry.total_flavonoids ? "wt%" : ""}</div>

            ${flavonoids.map((f) => `
              <div class="flav-row">
                <div class="flav-name">${esc(f.name)}</div>
                <div class="flav-track"><div class="flav-fill" style="width:${clamp((toNum(f.value) / Math.max(...flavonoids.map(x => toNum(x.value)), 0.001)) * 100, 0, 100)}%"></div></div>
                <div class="flav-value">${esc(f.value)} ${esc(f.unit || "wt%")}</div>
              </div>
            `).join("")}
          </div>

          <div class="panel-card">
            <div class="card-title">Why this matters</div>
            <div class="bullets">
              <div class="bullet"><div class="bullet-num">01</div><div>Rarer panel coverage makes the report feel more complete and commercially stronger.</div></div>
              <div class="bullet"><div class="bullet-num">02</div><div>Useful for brand differentiation when speaking to clinics, pharmacies, or buyers.</div></div>
              <div class="bullet"><div class="bullet-num">03</div><div>This block stays conditional and only appears when the data truly exists.</div></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  ` : ""}

  <div class="lab-strip">
    <div class="wrap">
      <div class="lab-grid">
        <div class="lab-item"><strong>${esc(chemistry.laboratory_name || "Laboratory not stated")}</strong><span>Laboratory</span></div>
        <div class="lab-item"><strong>${esc(chemistry.laboratory_accreditation || contaminants.lab_accreditation_body || "Accreditation not stated")}</strong><span>Accreditation</span></div>
        <div class="lab-item"><strong>${esc(chemistry.coa_report_date || "Date not stated")}</strong><span>Report date</span></div>
        <div class="lab-item"><strong>${esc(chemistry.batch_number || "Batch not stated")}</strong><span>Batch / Lot</span></div>
        <div class="lab-item last"><strong style="font-family:var(--mono);color:var(--green2);">Schema v7.0</strong><span>Alem Intelligence</span></div>
      </div>

      <div class="actions">
        <a href="/" class="btn primary">Analyse new COA</a>
        <a href="/pdf/${esc(options.documentId || "")}" class="btn secondary">Export PDF</a>
      </div>
    </div>
  </div>

  <footer>
    <div class="wrap">
      <div class="footer-inner">
        <div>
          <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png" alt="Alem" style="height:24px;display:block;">
          <div class="footer-copy">alem.solutions · COA Intelligence · Dynamic report renderer</div>
        </div>
        <div class="footer-line">Upload your COA.<br>Understand your chemistry.</div>
      </div>
    </div>
  </footer>

  <script>
    (function () {
      let currentAudience = "patient";
      const audienceButtons = document.querySelectorAll("[data-aud-btn]");
      const verdictSpans = document.querySelectorAll("#heroVerdict [data-aud]");
      const intelTabs = document.querySelectorAll("[data-intel-tab]");
      const intelPanels = document.querySelectorAll("[data-intel-panel]");
      const roleCopies = document.querySelectorAll("[data-role-copy]");

      function setAudience(audience) {
        currentAudience = audience;
        audienceButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.audBtn === audience));
        verdictSpans.forEach(span => span.classList.toggle("active", span.dataset.aud === audience));
        roleCopies.forEach(copy => copy.style.display = copy.dataset.roleCopy === audience ? "block" : "none");
        intelTabs.forEach(tab => tab.classList.toggle("active", tab.dataset.intelTab === audience));
        intelPanels.forEach(panel => panel.classList.toggle("active", panel.dataset.intelPanel === audience));
      }

      audienceButtons.forEach(btn => btn.addEventListener("click", function () { setAudience(this.dataset.audBtn); }));
      intelTabs.forEach(tab => tab.addEventListener("click", function () { setAudience(this.dataset.intelTab); }));

      window.toggleExpand = function (id) {
        const box = document.getElementById(id);
        if (!box) return;
        box.classList.toggle("open");
        const label = box.querySelector(".expand-toggle span:last-child");
        if (label) label.textContent = box.classList.contains("open") ? "Collapse" : "Expand";
      };

      setAudience(currentAudience);
    })();
  </script>
</body>
</html>`;
}

app.get("/", (req, res) => {
  return res.send(`<!DOCTYPE html>
  <html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Alem COA Intelligence</title>
  <style>body{font-family:Arial,sans-serif;background:#f7f5ef;padding:40px;line-height:1.5} .wrap{max-width:720px;margin:0 auto;background:#fff;border:1px solid #e7e3d8;padding:28px;border-radius:16px} h1{margin-top:0} code{background:#f1eee6;padding:2px 6px;border-radius:6px}</style>
  </head><body><div class="wrap"><h1>Alem COA Intelligence</h1><p>POST a file to <code>/upload</code> with field name <code>file</code> to create a report.</p></div></body></html>`);
});

app.get("/health", (req, res) => {
  return res.json({
    success: true,
    status: "ok",
    hasSupabaseUrl: !!process.env.SUPABASE_URL,
    hasSupabaseKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
    hasAzureEndpoint: !!process.env.AZURE_DOC_INTELLIGENCE_ENDPOINT,
    hasAzureKey: !!process.env.AZURE_DOC_INTELLIGENCE_KEY,
    hasOpenAIKey: !!process.env.OPENAI_API_KEY,
    bucket: SUPABASE_BUCKET,
  });
});

app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file?.buffer) throw new Error("No file uploaded");
    const mimeType = req.file.mimetype;
    const originalFilename = req.file.originalname;

    const extracted = await extractCOAFromBuffer(req.file.buffer);
    const { storagePath, publicUrl } = await uploadBufferToSupabase({
      buffer: req.file.buffer,
      originalName: originalFilename,
      mimeType,
      folder: "raw_documents",
    });

    const insertedRow = await insertCOAReport({
      chemistry: extracted.chemistry,
      contaminants: extracted.contaminants,
      scoring: extracted.scoring,
      intelligence: extracted.intelligence,
      sourceUrl: publicUrl,
      storagePath,
      originalFilename,
      mimeType,
    });

    const proto = req.headers["x-forwarded-proto"] || req.protocol || "https";
    const base = `${proto}://${req.get("host")}`;

    return res.json({
      success: true,
      id: insertedRow.id,
      score: extracted.scoring.total,
      grade: extracted.scoring.grade,
      tier: extracted.scoring.tier,
      fingerprint_id: extracted.intelligence.fingerprintId,
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
    if (!row) throw new Error("Row not found");
    return res.send(renderReportHTML(row?.report_json || {}, { documentId: row.id }));
  } catch (error) {
    console.error("ERROR /report/:id", error.message);
    return res.status(404).send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Report Not Found — Alem</title>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:#f4f8fb;font-family:'Nunito',sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:24px}
  .box{background:#fff;border:1px solid #ccdde8;max-width:480px;width:100%;padding:48px 40px;text-align:center}
  .logo{font-size:13px;font-weight:800;letter-spacing:6px;color:#0d2d3e;margin-bottom:32px}
  .code{font-size:72px;font-weight:800;color:#eef6fb;line-height:1}
  h1{font-size:22px;font-weight:700;color:#0d2d3e;margin:16px 0 10px}
  p{font-size:13px;color:#4a6070;line-height:1.7;margin-bottom:28px}
  a{display:inline-block;background:#0d2d3e;color:#fff;padding:12px 28px;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;text-decoration:none;border-radius:30px}
</style></head>
<body><div class="box"><div class="logo">ALEM</div><div class="code">404</div><h1>Report not found</h1><p>This report ID doesn't exist or may have been removed. Please check the URL or upload a new COA to generate a fresh report.</p><a href="/">Upload a COA</a></div></body></html>`);
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
    if (browser) {
      try { await browser.close(); } catch (_) {}
    }
  }
});

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

process.on("unhandledRejection", (reason) => {
  console.error("⚠️  Unhandled Promise Rejection:", reason?.message || reason);
});

process.on("uncaughtException", (err) => {
  console.error("⚠️  Uncaught Exception:", err.message);
});

app.listen(PORT, () => {
  console.log(`🌿 Alem Chemical Intelligence dynamic v7.1 running on port ${PORT}`);
});
