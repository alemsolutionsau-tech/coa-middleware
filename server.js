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

// ─────────────────────────────────────────────────────────────────────
// renderReportHTML — v8  (drop-in replacement for the function in server.js)
// Design: preview.html visual identity + graceful incomplete-COA handling
// ─────────────────────────────────────────────────────────────────────

function renderReportHTML(reportJson = {}, options = {}) {

  const chemistry    = reportJson.chemistry    || {};
  const contaminants = reportJson.contaminants || {};
  const scoring      = reportJson.scoring      || computeIntelligenceScore(chemistry, contaminants);
  const intelligence = reportJson.intelligence || {};

  const terpenes     = chemistry.top_terpenes    || [];
  const cannabinoids = chemistry.top_cannabinoids || [];
  const flavonoids   = chemistry.flavonoids       || [];

  const thc  = toNum(chemistry.thc_total);
  const cbd  = toNum(chemistry.cbd_total);
  const terps = toNum(chemistry.total_terpenes);
  const cbn  = toNum(chemistry.cbn);
  const cbna = toNum(chemistry.cbna);
  const moisture     = chemistry.moisture_content  || contaminants.moisture_content  || "";
  const waterActivity = chemistry.water_activity   || contaminants.water_activity    || "";

  const activeTerps  = terpenes.filter(t => toNum(t.value) > 0);
  const blqTerps     = terpenes.filter(t => String(t.value).toLowerCase() === "blq");
  const terpCount    = activeTerps.length;
  const blqCount     = blqTerps.length;
  const maxTerpVal   = activeTerps.length > 0 ? Math.max(...activeTerps.map(t => toNum(t.value)), 0.001) : 0.001;

  const leadTerpene   = terpenes[0]?.name || "";
  const secondTerpene = terpenes[1]?.name || "";
  const thirdTerpene  = terpenes[2]?.name || "";
  const terpIntel     = getTerpeneIntel(leadTerpene);
  const fingerprintId = intelligence.fingerprintId || generateFingerprintId(terpenes);

  const productName  = chemistry.product_name || "COA Report";
  const heroNarrative = chemistry.hero_narrative || "";

  const safeTotal = scoring.total ?? 0;
  const safeGrade = scoring.grade || "—";
  const safeTier  = scoring.tier  || "—";

  // Safely rebuild audiences / postHarvest
  let audiences;
  try {
    const rawAud = intelligence.audiences;
    audiences = (rawAud && Array.isArray(rawAud.brand)) ? rawAud : buildAudienceNarratives(chemistry, contaminants, scoring);
  } catch (_) { audiences = buildAudienceNarratives(chemistry, contaminants, scoring); }
  audiences.brand    = Array.isArray(audiences.brand)    ? audiences.brand    : [];
  audiences.clinical = Array.isArray(audiences.clinical) ? audiences.clinical : [];
  audiences.patient  = Array.isArray(audiences.patient)  ? audiences.patient  : [];
  audiences.buyer    = Array.isArray(audiences.buyer)    ? audiences.buyer    : [];

  let postHarvest;
  try {
    const rawPH = intelligence.postHarvest;
    postHarvest = (rawPH && rawPH.freshness) ? rawPH : buildPostHarvestIntel(chemistry, contaminants);
  } catch (_) { postHarvest = buildPostHarvestIntel(chemistry, contaminants); }

  // ── COMPLETENESS FLAGS ───────────────────────────────────────────────────
  const hasTHC            = thc > 0 || (chemistry.thc_total && chemistry.thc_total !== "ND" && chemistry.thc_total !== "");
  const hasTerpenes       = terps > 0 || terpCount > 0;
  const hasCannabinoids   = cannabinoids.length > 0;
  const hasSafety         = !!(contaminants.pesticides_status || contaminants.heavy_metals_status || contaminants.microbials_status || contaminants.mycotoxins_status);
  const hasPesticides     = !!(contaminants.pesticides_status && !/not tested/i.test(contaminants.pesticides_status));
  const hasMetals         = !!(contaminants.heavy_metals_status && !/not tested/i.test(contaminants.heavy_metals_status));
  const hasMicrobials     = !!(contaminants.microbials_status && !/not tested/i.test(contaminants.microbials_status));
  const hasMycotoxins     = !!(contaminants.mycotoxins_status && !/not tested/i.test(contaminants.mycotoxins_status));
  const hasAllSafety      = hasPesticides && hasMetals && hasMicrobials && hasMycotoxins;
  const hasFlavonoids     = flavonoids.length > 0;
  const hasMoisture       = !!(moisture && moisture !== "");
  const hasWaterActivity  = !!(waterActivity && waterActivity !== "");
  const hasBatchNum       = !!(chemistry.batch_number && chemistry.batch_number !== "");
  const hasLabName        = !!(chemistry.laboratory_name && chemistry.laboratory_name !== "");
  const hasDate           = !!(chemistry.coa_report_date && chemistry.coa_report_date !== "");
  const isoPass           = contaminants.iso_17025 === true || /17025/i.test(chemistry.laboratory_accreditation || "");
  const sccPass           = contaminants.scc_accredited === true;

  // Critical missing = things that severely limit the report's usefulness
  const criticalMissing = [];
  const minorMissing    = [];
  if (!hasTHC)       criticalMissing.push({ key: "thc",       label: "THC data",        msg: "Potency cannot be evaluated — no THC value was captured from this COA." });
  if (!hasTerpenes)  criticalMissing.push({ key: "terpenes",  label: "Terpene data",    msg: "Terpene profile not captured. Effect direction and chemotype fingerprint cannot be determined." });
  if (!hasSafety)    criticalMissing.push({ key: "safety",    label: "Safety panels",   msg: "No safety data was found in this COA. Pesticide, heavy metal, microbial, and mycotoxin results are all absent. Request the full compliance documentation from the producer before prescribing or listing this product." });
  if (!hasAllSafety && hasSafety) {
    const missing = [];
    if (!hasPesticides)  missing.push("Pesticides");
    if (!hasMetals)      missing.push("Heavy Metals");
    if (!hasMicrobials)  missing.push("Microbials");
    if (!hasMycotoxins)  missing.push("Mycotoxins");
    if (missing.length) minorMissing.push({ key: "partial_safety", label: missing.join(", "), msg: `Missing safety panels: ${missing.join(", ")}. Request these from the producer — incomplete safety data limits compliance confidence.` });
  }
  if (!hasBatchNum)  minorMissing.push({ key: "batch",  label: "Batch number", msg: "Batch number not identified on this COA." });
  if (!hasDate)      minorMissing.push({ key: "date",   label: "Report date",  msg: "Report date not captured." });
  if (!hasLabName)   minorMissing.push({ key: "lab",    label: "Lab name",     msg: "Laboratory name not identified." });

  const hasAnyCritical = criticalMissing.length > 0;

  // Safety pass/fail flags
  const pestPass  = /pass|nd|not detected/i.test(contaminants.pesticides_status  || "");
  const metalsPass = /pass|nd|not detected|blq/i.test(contaminants.heavy_metals_status || "");
  const microPass  = /pass|nd|not detected|absent/i.test(contaminants.microbials_status || "");
  const mycoPass   = /pass|nd|not detected/i.test(contaminants.mycotoxins_status || "");
  const allSafetyPass = pestPass && metalsPass && microPass && mycoPass && hasAllSafety;

  // Score breakdown
  const safeBreakdown = scoring.breakdown || {};
  const safePotency   = safeBreakdown.potency          || { score: 0, max: 25 };
  const safeTerpScore = safeBreakdown.terpenes         || { score: 0, max: 25 };
  const safeMinors    = safeBreakdown.minors           || { score: 0, max: 20 };
  const safeSafetyS   = safeBreakdown.safety           || { score: 0, max: 20 };
  const safeData      = safeBreakdown.dataCompleteness || { score: 0, max: 10 };
  const potencyPct    = safePotency.max  > 0 ? Math.round(safePotency.score  / safePotency.max  * 100) : 0;
  const terpPct       = safeTerpScore.max > 0 ? Math.round(safeTerpScore.score / safeTerpScore.max * 100) : 0;
  const minorPct      = safeMinors.max   > 0 ? Math.round(safeMinors.score   / safeMinors.max   * 100) : 0;
  const safetyPct     = safeSafetyS.max  > 0 ? Math.round(safeSafetyS.score  / safeSafetyS.max  * 100) : 0;
  const dataPct       = safeData.max     > 0 ? Math.round(safeData.score     / safeData.max     * 100) : 0;

  // ── HTML HELPER FUNCTIONS ────────────────────────────────────────────────

  function h(str = "") {
    return String(str ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }

  // Missing data placeholder block
  function missingBlock(message, severity = "minor") {
    const isMinor = severity === "minor";
    return `<div class="missing-block ${isMinor ? "missing-minor" : "missing-critical"}">
      <div class="missing-icon">${isMinor ? "○" : "!"}</div>
      <div class="missing-msg">${h(message)}</div>
    </div>`;
  }

  // Section with optional "not available" state
  function sectionUnavailable(title, eyebrow, reason) {
    return `
    <section class="section alt">
      <div class="wrap">
        <div class="section-head">
          <div class="eyebrow">${h(eyebrow)}</div>
          <h2>${h(title)}</h2>
        </div>
        <div class="unavailable-section">
          <div class="unavail-icon">○</div>
          <div class="unavail-body">
            <div class="unavail-title">Data not available</div>
            <div class="unavail-note">${h(reason)}</div>
          </div>
        </div>
      </div>
    </section>`;
  }

  // Audience intel rows
  function audiencePanel(items = []) {
    if (!items.length) return `<div class="intel-row"><div class="num">—</div><div class="copy">No intelligence available for this panel.</div></div>`;
    return items.map((line, i) => `<div class="intel-row"><div class="num">0${i+1}</div><div class="copy">${h(line)}</div></div>`).join("");
  }

  // Terpene bar row
  function terpBar(t, i) {
    if (!t?.name) return "";
    const val   = toNum(t.value);
    const isBlq = String(t.value || "").toLowerCase() === "blq";
    const width = isBlq ? 2 : Math.max(3, maxTerpVal > 0 ? (val / maxTerpVal) * 100 : 0);
    const isLead = i === 0;
    const edu = TERPENE_EDUCATION[t.name] || TERPENE_EDUCATION[normaliseTerpeneName(t.name)] || null;
    const aromaNote = edu ? `<span class="top-terp-aroma">${h(edu.aroma)}</span>` : "";
    const pharmNote = edu ? `<span class="top-terp-pharm">${h(edu.therapeutic)}</span>` : "";
    const isTopThree = i < 3;
    return isTopThree ? `
    <div class="top-terp">
      <div class="top-terp-head">
        <div>
          <div class="top-terp-name">${h(t.name)}</div>
          ${aromaNote}
        </div>
        <div class="top-terp-value">${isBlq ? "BLQ" : h(String(t.value||"")) + " wt%"}</div>
      </div>
      <div class="top-terp-track"><div class="top-terp-fill" style="width:${width.toFixed(1)}%;background:${isLead?"#16855c":i===1?"#2fa5ad":"#7389be"};animation-delay:${(i*0.1).toFixed(2)}s"></div></div>
      ${pharmNote ? `<div class="top-terp-copy">${pharmNote}</div>` : ""}
    </div>` : `
    <div class="mini-item">
      <span><strong>${h(t.name)}</strong>${edu ? ` · ${h(edu.aroma.split(".")[0].split("—")[0].trim())}` : ""}</span>
      <span>${isBlq ? "BLQ" : h(String(t.value||""))}</span>
    </div>`;
  }

  // Safety panel card
  function safetyCard(title, status, pass, present, rows, method) {
    const statusClass = present ? (pass ? "verdict-pass" : "verdict-fail") : "verdict-nt";
    const statusText  = present ? (pass ? status || "✓ Clear" : status || "⚠ Review") : "Not tested";
    return `
    <div class="safe-card">
      <div class="safe-card-head">
        <div class="safe-card-title">${h(title)}</div>
        <div class="safe-card-verdict ${statusClass}">${h(statusText)}</div>
      </div>
      <div class="safe-card-body">
        ${present ? rows : `<div class="safe-nt-note">Panel not found in this COA.<br>Request from producer before prescribing or listing.</div>`}
      </div>
      ${method ? `<div class="safe-method">${h(method)}</div>` : ""}
    </div>`;
  }

  function safeRow(label, val) {
    const v = String(val || "");
    const isGood = /nd|not detected|absent|pass|< 10|blq/i.test(v) || !v;
    return `<div class="safe-row"><span>${h(label)}</span><strong class="${isGood ? "ok" : "warn"}">${h(v || "—")}</strong></div>`;
  }

  function checkRow(label, value, pass, dimmed = false) {
    return `
    <div class="chk${dimmed ? " chk-dim" : ""}">
      <div class="chk-dot" style="background:${pass ? "var(--green)" : "var(--line-2)"}"></div>
      <div>${h(label)}</div>
      <div class="chk-val ${pass ? "" : "chk-nt"}">${h(value)}</div>
    </div>`;
  }

  // ── CSS ──────────────────────────────────────────────────────────────────

  const CSS = `
    *,*::before,*::after{box-sizing:border-box;}
    html{scroll-behavior:smooth;}
    body{margin:0;background:#f7f5ef;color:#171714;font-family:"DM Sans",system-ui,sans-serif;line-height:1.6;-webkit-font-smoothing:antialiased;}
    :root{
      --bg:#f7f5ef;--surface:#ffffff;--surface-2:#fcfbf8;
      --ink:#171714;--ink-2:#34342f;--ink-3:#616159;--ink-4:#919188;
      --line:#e7e3d8;--line-2:#d8d3c7;
      --green:#16855c;--green-2:#0f6847;--green-bg:#eef8f3;
      --amber:#b97b17;--amber-bg:#fdf7ec;
      --red:#b74242;--red-bg:#fdf1f1;
      --shadow:0 10px 30px rgba(20,20,18,0.06);
      --radius-xl:22px;--radius-lg:16px;--radius-md:12px;
      --max:1180px;--pad:clamp(20px,4vw,56px);
      --serif:"Libre Baskerville",Georgia,serif;
      --sans:"DM Sans",system-ui,sans-serif;
      --mono:"DM Mono",monospace;
    }
    @keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
    @keyframes growX{from{transform:scaleX(0)}to{transform:scaleX(1)}}
    @keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}
    .fade-up{animation:fadeUp .45s ease both;}
    .wrap{width:100%;max-width:calc(var(--max) + var(--pad)*2);margin:0 auto;padding-left:var(--pad);padding-right:var(--pad);}

    /* TOPBAR */
    .topbar{position:sticky;top:0;z-index:100;background:rgba(255,255,255,.88);backdrop-filter:blur(16px);border-bottom:1px solid rgba(231,227,216,.9);}
    .topbar-inner{min-height:58px;display:flex;align-items:center;gap:14px;}
    .brand img{height:24px;display:block;}
    .brand-fallback{display:none;font-weight:700;font-size:14px;color:var(--ink);}
    .divider{width:1px;height:18px;background:var(--line);flex-shrink:0;}
    .topbar-label{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-4);white-space:nowrap;}
    .topbar-right{margin-left:auto;display:flex;align-items:center;gap:10px;}
    .pill{display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:7px 12px;border:1px solid var(--line);background:var(--surface);font-size:11px;color:var(--ink-3);font-family:var(--mono);white-space:nowrap;}
    .pill.score{background:var(--green-bg);border-color:#cbe8d9;color:var(--green-2);font-weight:500;}
    .pill.incomplete{background:#fdf7ec;border-color:#f0d080;color:#8a5a00;}

    /* CRITICAL BANNER — shown when major data is missing */
    .critical-banner{background:#fdf1f1;border-bottom:3px solid var(--red);padding:14px 0;}
    .critical-banner-inner{display:flex;align-items:flex-start;gap:14px;}
    .cb-icon{width:32px;height:32px;border-radius:50%;background:var(--red);color:#fff;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:800;flex-shrink:0;}
    .cb-body{}
    .cb-title{font-size:14px;font-weight:700;color:var(--red);margin-bottom:4px;}
    .cb-items{display:flex;flex-direction:column;gap:4px;}
    .cb-item{font-size:13px;color:#7a2020;line-height:1.6;}
    .cb-item strong{color:var(--red);}

    /* MINOR NOTICE BANNER */
    .notice-banner{background:var(--amber-bg);border-bottom:1px solid #e8d080;padding:10px 0;}
    .notice-inner{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12px;color:#6a4800;}
    .notice-label{font-weight:700;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--amber);white-space:nowrap;}
    .notice-items{display:flex;gap:14px;flex-wrap:wrap;}
    .notice-item{display:flex;align-items:center;gap:5px;}
    .notice-item::before{content:"○";font-size:9px;}

    /* HERO */
    .hero{padding:clamp(28px,5vw,56px) 0 32px;border-bottom:1px solid var(--line);background:radial-gradient(circle at top right,rgba(22,133,92,.08),transparent 28%),linear-gradient(180deg,rgba(255,255,255,.75),rgba(255,255,255,.48));}
    .hero-grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:clamp(24px,4vw,56px);align-items:start;}
    .eyebrow{display:inline-flex;align-items:center;gap:10px;margin-bottom:14px;font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--green);font-weight:700;}
    .eyebrow::before{content:"";width:18px;height:1.5px;background:var(--green);border-radius:99px;}
    .product-name{margin:0 0 6px;font-family:var(--sans);font-size:clamp(38px,7vw,78px);line-height:.92;letter-spacing:-.05em;color:var(--ink);}
    .product-type{font-size:17px;color:var(--ink-4);font-weight:300;margin-bottom:22px;}
    .hero-verdict{max-width:680px;margin-bottom:18px;font-family:var(--serif);font-size:clamp(18px,2.4vw,26px);line-height:1.5;color:var(--ink);font-style:italic;}
    .hero-verdict em{font-style:normal;color:var(--green-2);font-weight:700;}
    .hero-verdict [data-aud]{display:none;}
    .hero-verdict [data-aud].active{display:inline;}
    .meta-row{margin-bottom:18px;font-size:11px;color:var(--ink-4);letter-spacing:.04em;font-family:var(--mono);}
    .tag-row{display:flex;flex-wrap:wrap;gap:8px;}
    .tag{display:inline-flex;align-items:center;min-height:30px;padding:5px 11px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.65);color:var(--ink-3);font-size:11px;font-weight:500;}
    .tag.strong{background:var(--green-bg);color:var(--green-2);border-color:#c8e7d8;}
    .tag.warn{background:var(--amber-bg);color:var(--amber);border-color:#e8d080;}

    /* SCORE PANEL */
    .score-panel{position:sticky;top:78px;background:rgba(255,255,255,.95);border:1px solid var(--line);border-radius:var(--radius-xl);padding:22px 20px;box-shadow:var(--shadow);}
    .score-ring{position:relative;width:180px;height:180px;margin:0 auto 18px;}
    .score-ring svg{position:absolute;inset:0;transform:rotate(-90deg);}
    .score-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0;}
    .score-value{font-size:52px;font-weight:700;line-height:1;letter-spacing:-.04em;color:var(--ink);font-family:var(--sans);}
    .score-max{font-size:11px;color:var(--ink-4);font-family:var(--mono);margin-top:1px;}
    .score-grade{font-family:var(--serif);font-size:17px;color:var(--green-2);font-style:italic;margin-top:4px;}
    .score-label{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-4);margin-top:2px;}
    .mini-metrics{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--surface);}
    .mini-metric{display:grid;grid-template-columns:1fr 48px 38px;gap:8px;align-items:center;padding:10px 12px;border-bottom:1px solid var(--line);font-size:11px;}
    .mini-metric:last-child{border-bottom:none;}
    .mini-metric-label{color:var(--ink-3);}
    .mini-track{height:3px;background:var(--line);border-radius:999px;overflow:hidden;}
    .mini-fill{height:100%;border-radius:inherit;transform-origin:left;animation:growX .75s cubic-bezier(.16,1,.3,1) both;}
    .mini-value{text-align:right;font-size:11px;color:var(--ink);font-family:var(--mono);}
    .score-note{margin-top:12px;font-size:11px;color:var(--ink-4);text-align:center;line-height:1.6;}
    .score-note strong{color:var(--ink-3);}
    /* Score capped notice */
    .score-cap-note{margin-top:10px;padding:8px 12px;background:var(--amber-bg);border:1px solid #e8d080;border-radius:8px;font-size:11px;color:#6a4800;line-height:1.55;text-align:left;}
    .score-cap-note strong{font-weight:700;}

    /* AUDIENCE BAR */
    .audience-bar{background:var(--surface);border-bottom:1px solid var(--line);}
    .audience-inner{min-height:56px;display:flex;align-items:center;gap:6px;flex-wrap:wrap;}
    .audience-label{margin-right:8px;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--ink-4);}
    .aud-btn{appearance:none;border:1px solid transparent;background:transparent;color:var(--ink-3);border-radius:999px;padding:9px 14px;font-size:13px;font-weight:600;cursor:pointer;transition:.18s ease;font-family:var(--sans);}
    .aud-btn:hover{background:#f5f3ed;color:var(--ink);}
    .aud-btn.active{background:var(--green-bg);color:var(--green-2);border-color:#cae8da;}

    /* SECTIONS */
    .section{padding:clamp(28px,5vw,56px) 0;border-bottom:1px solid var(--line);}
    .section.alt{background:linear-gradient(180deg,rgba(255,255,255,.72),rgba(255,255,255,.96));}
    .section-head{margin-bottom:24px;max-width:760px;}
    .section-head h2{margin:0 0 10px;font-family:var(--serif);font-size:clamp(24px,3.2vw,36px);line-height:1.18;color:var(--ink);font-weight:400;}
    .section-head p{margin:0;font-size:15px;color:var(--ink-3);line-height:1.8;}
    .section-head p strong{color:var(--ink);}
    .section-head p em{color:var(--green-2);font-style:normal;font-weight:600;}

    /* MISSING DATA INDICATORS */
    .missing-block{display:flex;align-items:flex-start;gap:10px;padding:14px 16px;border-radius:10px;margin-bottom:14px;}
    .missing-block.missing-critical{background:#fdf1f1;border:1px solid #f0b8b8;border-left:4px solid var(--red);}
    .missing-block.missing-minor{background:var(--amber-bg);border:1px solid #e8d080;border-left:4px solid var(--amber);}
    .missing-icon{font-size:16px;flex-shrink:0;margin-top:1px;}
    .missing-msg{font-size:13px;line-height:1.65;color:var(--ink-2);}
    .missing-critical .missing-msg{color:#7a2020;}
    .missing-minor .missing-msg{color:#6a4800;}

    /* UNAVAILABLE SECTION */
    .unavailable-section{display:flex;align-items:center;gap:16px;padding:28px 24px;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);border-left:4px solid var(--line-2);}
    .unavail-icon{font-size:28px;color:var(--ink-4);flex-shrink:0;}
    .unavail-title{font-size:16px;font-weight:600;color:var(--ink-3);margin-bottom:5px;}
    .unavail-note{font-size:13px;color:var(--ink-4);line-height:1.65;}

    /* CARDS */
    .card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-xl);padding:22px;box-shadow:var(--shadow);}
    .card-title{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink-4);margin-bottom:12px;}
    .muted{color:var(--ink-3);font-size:13px;line-height:1.7;}
    .bullets{display:grid;gap:12px;}
    .bullet{display:grid;grid-template-columns:18px minmax(0,1fr);gap:10px;align-items:start;font-size:14px;color:var(--ink-2);}
    .bullet-num{font-family:var(--mono);color:var(--green-2);font-size:11px;padding-top:2px;}

    /* SUMMARY GRID */
    .summary-grid{display:grid;grid-template-columns:200px minmax(0,1fr);gap:18px;align-items:stretch;}
    .summary-score{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-xl);padding:22px;box-shadow:var(--shadow);display:flex;flex-direction:column;justify-content:center;min-height:160px;}
    .summary-score .small{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink-4);margin-bottom:8px;}
    .summary-score .big{font-family:var(--sans);font-size:56px;line-height:1;letter-spacing:-.05em;color:var(--ink);font-weight:700;}
    .summary-score .big.dim{color:var(--ink-4);font-size:36px;}
    .summary-score .sub{margin-top:6px;color:var(--green-2);font-family:var(--serif);font-size:17px;font-style:italic;}
    .summary-main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;}
    .fingerprint{font-family:var(--mono);font-size:22px;color:var(--green-2);letter-spacing:.08em;margin-bottom:8px;}
    .fingerprint.dim{color:var(--ink-4);font-size:16px;}

    /* SAFETY BANNER */
    .safety-banner{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:18px;align-items:center;background:var(--green-bg);border:1px solid #c7e7d7;border-left:4px solid var(--green);border-radius:0 var(--radius-xl) var(--radius-xl) 0;padding:20px 22px;margin-bottom:20px;}
    .safety-banner h3{margin:0 0 4px;font-size:18px;color:var(--ink);}
    .safety-banner p{margin:0;font-size:13px;color:var(--ink-3);line-height:1.65;}
    .safety-banner.warn{background:var(--amber-bg);border-color:#e8d080;border-left-color:var(--amber);}
    .safety-banner.warn h3{color:var(--amber);}
    .confidence{text-align:right;min-width:90px;}
    .confidence strong{display:block;color:var(--green-2);font-size:26px;line-height:1;font-family:var(--mono);font-weight:500;}
    .confidence strong.warn{color:var(--amber);}
    .confidence span{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--ink-4);}

    /* SAFETY CARDS */
    .grid-4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;}
    .safe-card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);overflow:hidden;}
    .safe-card-head{padding:14px 16px 10px;border-bottom:1px solid var(--line);}
    .safe-card-title{font-size:9px;text-transform:uppercase;letter-spacing:.13em;color:var(--ink-4);margin-bottom:7px;}
    .safe-card-verdict{font-size:15px;font-weight:700;display:flex;align-items:center;gap:7px;}
    .safe-card-verdict.verdict-pass{color:var(--green-2);}
    .safe-card-verdict.verdict-pass::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2.5s ease-in-out infinite;flex-shrink:0;}
    .safe-card-verdict.verdict-fail{color:var(--red);}
    .safe-card-verdict.verdict-nt{color:var(--ink-4);font-style:italic;font-weight:400;font-size:13px;}
    .safe-card-body{padding:10px 16px;display:grid;gap:5px;font-size:12px;color:var(--ink-3);}
    .safe-row{display:flex;justify-content:space-between;gap:10px;}
    .safe-row strong.ok{color:var(--green-2);font-family:var(--mono);font-size:10px;font-weight:500;}
    .safe-row strong.warn{color:var(--red);font-family:var(--mono);font-size:10px;font-weight:700;}
    .safe-nt-note{font-size:11px;color:var(--ink-4);font-style:italic;line-height:1.65;padding:4px 0;}
    .safe-method{padding:8px 16px 12px;border-top:1px solid var(--line);font-family:var(--mono);font-size:9px;color:var(--ink-4);}

    /* CHECKS ROW */
    .checks-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:16px;}
    .chk{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 13px;font-size:12px;color:var(--ink-2);}
    .chk-dim{opacity:.55;}
    .chk-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
    .chk-val{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--green-2);font-weight:500;}
    .chk-nt{color:var(--ink-4);font-style:italic;font-weight:400;}

    /* CALLOUT */
    .callout{border-radius:var(--radius-lg);padding:18px 20px;border-left:4px solid;margin-bottom:18px;}
    .callout.amber{background:var(--amber-bg);border-color:var(--amber);}
    .callout.green{background:var(--green-bg);border-color:var(--green);}
    .callout.red{background:var(--red-bg);border-color:var(--red);}
    .callout.blue{background:#eef4ff;border-color:#285ca8;}
    .callout strong{display:block;color:var(--ink);margin-bottom:5px;font-size:15px;}
    .callout p{margin:0;color:var(--ink-3);font-size:13px;line-height:1.7;}

    /* POTENCY */
    .two-col{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start;}
    .stat-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:var(--radius-lg);overflow:hidden;margin-bottom:18px;}
    .stat{background:var(--surface);padding:20px 16px;text-align:center;}
    .stat-value{font-family:var(--mono);font-size:clamp(22px,3vw,32px);line-height:1;color:var(--ink);margin-bottom:6px;}
    .stat-value.nd{color:var(--ink-4);font-size:clamp(16px,2vw,22px);}
    .stat-label{font-size:10px;text-transform:uppercase;letter-spacing:.11em;color:var(--ink-4);}
    .market-card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);padding:20px;}
    .market-label{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:var(--ink-4);margin-bottom:12px;}
    .market-bar{position:relative;height:40px;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#f6f3eb;margin-bottom:7px;}
    .zone-low{position:absolute;top:0;bottom:0;left:0;width:55%;background:rgba(145,145,136,.07);}
    .zone-mid{position:absolute;top:0;bottom:0;left:55%;width:20%;background:rgba(185,123,23,.08);}
    .zone-high{position:absolute;top:0;bottom:0;left:75%;right:0;background:rgba(22,133,92,.1);}
    .market-fill{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,var(--green),var(--green-2));transform-origin:left;animation:growX .9s cubic-bezier(.16,1,.3,1) both;}
    .market-pin{position:absolute;top:50%;transform:translate(-50%,-50%);background:var(--green-2);color:#fff;padding:3px 9px;border-radius:999px;font-size:10px;font-family:var(--mono);white-space:nowrap;}
    .market-legend{display:flex;justify-content:space-between;font-size:9px;color:var(--ink-4);font-family:var(--mono);}
    .intensity{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);padding:18px 20px;}
    .intensity-label{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:var(--ink-4);margin-bottom:14px;}
    .intensity-scale{position:relative;height:6px;background:linear-gradient(90deg,#dcd7ca 0%,#ccb686 55%,#b97b17 78%,#8b4c15 100%);border-radius:999px;margin:16px 6px 8px;}
    .intensity-marker{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;transform:translate(-50%,-50%);background:#fff;border:2px solid var(--ink);box-shadow:0 2px 8px rgba(0,0,0,.12);}
    .intensity-legend{display:flex;justify-content:space-between;font-size:9px;color:var(--ink-4);font-family:var(--mono);}
    .cannabinoid-table{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);overflow:hidden;margin-top:18px;}
    .ct-row{display:grid;grid-template-columns:100px minmax(0,1fr) 82px;gap:12px;align-items:center;padding:12px 18px;border-bottom:1px solid var(--line);}
    .ct-row:last-child{border-bottom:none;}
    .ct-row.head{background:#faf8f3;padding-top:9px;padding-bottom:9px;font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink-4);}
    .ct-row.total-row{background:var(--green-bg);}
    .ct-name{font-family:var(--mono);font-size:12px;color:var(--ink-2);}
    .ct-name.primary{color:var(--ink);font-weight:500;}
    .ct-bar{height:4px;background:var(--line);border-radius:999px;overflow:hidden;}
    .ct-fill{height:100%;border-radius:inherit;transform-origin:left;animation:growX .8s cubic-bezier(.16,1,.3,1) both;}
    .ct-value{text-align:right;font-family:var(--mono);font-size:12px;color:var(--ink-3);}
    .ct-value.nd{color:var(--ink-4);}
    .ct-value.total{color:var(--green-2);font-weight:500;}

    /* TERPENES */
    .terp-layout{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);gap:18px;align-items:start;}
    .terp-card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-xl);padding:22px;box-shadow:var(--shadow);}
    .terp-total{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin-bottom:20px;}
    .terp-total-label{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink-4);white-space:nowrap;}
    .terp-total-track{height:7px;border-radius:999px;background:var(--line);overflow:hidden;}
    .terp-total-fill{width:86%;height:100%;background:linear-gradient(90deg,var(--green),var(--green-2));border-radius:inherit;transform-origin:left;animation:growX 1s cubic-bezier(.16,1,.3,1) both;}
    .terp-total-value{font-family:var(--mono);color:var(--ink);font-size:18px;white-space:nowrap;}
    .top-terp-list{display:grid;gap:16px;}
    .top-terp{border-bottom:1px solid var(--line);padding-bottom:14px;}
    .top-terp:last-child{border-bottom:none;padding-bottom:0;}
    .top-terp-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:4px;}
    .top-terp-name{font-size:14px;font-weight:700;color:var(--ink);}
    .top-terp-aroma{font-size:11px;color:var(--ink-4);font-style:italic;display:block;margin-top:1px;}
    .top-terp-pharm{font-size:12px;color:var(--ink-3);line-height:1.65;display:block;margin-top:5px;}
    .top-terp-value{font-family:var(--mono);color:var(--ink-3);font-size:12px;white-space:nowrap;}
    .top-terp-track{height:4px;border-radius:999px;background:var(--line);overflow:hidden;margin-bottom:6px;}
    .top-terp-fill{height:100%;border-radius:inherit;transform-origin:left;animation:growX .8s cubic-bezier(.16,1,.3,1) both;}
    .expand-box{margin-top:14px;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#fbfaf7;}
    .expand-toggle{width:100%;appearance:none;border:none;background:transparent;text-align:left;padding:13px 16px;font:inherit;color:var(--ink);cursor:pointer;display:flex;justify-content:space-between;gap:12px;align-items:center;font-weight:600;font-size:13px;}
    .expand-toggle span:last-child{color:var(--ink-4);font-family:var(--mono);font-size:11px;}
    .expand-content{display:none;padding:0 16px 16px;border-top:1px solid var(--line);}
    .expand-box.open .expand-content{display:block;}
    .mini-list{display:grid;gap:9px;padding-top:13px;}
    .mini-item{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:var(--ink-3);border-bottom:1px dashed #e8e3d9;padding-bottom:8px;}
    .mini-item:last-child{border-bottom:none;padding-bottom:0;}
    .mini-item strong{color:var(--ink);font-weight:500;}
    .fingerprint-card{background:linear-gradient(180deg,#1d1d1a,#121210);color:#fff;border-radius:var(--radius-xl);padding:20px;box-shadow:var(--shadow);overflow:hidden;position:relative;}
    .fingerprint-card::before{content:"";position:absolute;inset:0 0 auto 0;height:2px;background:linear-gradient(90deg,var(--green),#73d5ae);}
    .fingerprint-card .label{font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:rgba(255,255,255,.45);margin-bottom:8px;}
    .fingerprint-card .value{font-family:var(--mono);font-size:22px;letter-spacing:.12em;color:#77ddb6;margin-bottom:10px;}
    .fingerprint-card .copy{font-size:13px;color:rgba(255,255,255,.75);line-height:1.75;}
    .fingerprint-card .copy strong{color:#fff;}

    /* INTELLIGENCE */
    .intel-shell{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-xl);padding:22px;box-shadow:var(--shadow);}
    .intel-tabs{display:inline-flex;gap:4px;padding:4px;background:#f5f2ea;border:1px solid var(--line);border-radius:999px;margin-bottom:20px;}
    .intel-tab{appearance:none;border:none;background:transparent;padding:9px 16px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:.04em;color:var(--ink-4);cursor:pointer;font-family:var(--sans);transition:.18s;}
    .intel-tab.active{background:var(--green-2);color:#fff;}
    .intel-panel{display:none;}
    .intel-panel.active{display:grid;gap:12px;animation:fadeUp .3s ease both;}
    .intel-row{display:grid;grid-template-columns:24px minmax(0,1fr);gap:12px;align-items:start;padding-bottom:12px;border-bottom:1px solid var(--line);}
    .intel-row:last-child{border-bottom:none;padding-bottom:0;}
    .intel-row .num{font-family:var(--mono);font-size:11px;color:var(--green-2);padding-top:2px;}
    .intel-row .copy{font-size:14px;color:var(--ink-2);line-height:1.8;}
    .intel-row .copy strong{color:var(--ink);}
    .intel-row .copy em{color:var(--green-2);font-style:normal;font-weight:600;}

    /* HARVEST */
    .harvest-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;}
    .harvest-card{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--green);border-radius:var(--radius-lg);padding:18px;box-shadow:var(--shadow);}
    .harvest-label{font-size:9px;text-transform:uppercase;letter-spacing:.13em;color:var(--ink-4);margin-bottom:6px;}
    .harvest-value{font-size:18px;color:var(--green-2);font-weight:700;margin-bottom:6px;}
    .harvest-value.unknown{color:var(--ink-4);font-style:italic;font-size:14px;}
    .harvest-copy{font-size:12px;color:var(--ink-3);line-height:1.75;}

    /* ADVANCED */
    .advanced-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start;}
    .panel-card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-lg);padding:20px;box-shadow:var(--shadow);}
    .flav-row{display:grid;grid-template-columns:130px minmax(0,1fr) 86px;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line);}
    .flav-row:last-child{border-bottom:none;}
    .flav-name{font-size:12px;color:var(--ink-2);}
    .flav-name.cann{font-weight:600;color:var(--ink);}
    .flav-track{height:3px;background:var(--line);border-radius:999px;overflow:hidden;}
    .flav-fill{height:100%;background:linear-gradient(90deg,rgba(22,133,92,.3),rgba(22,133,92,.65));border-radius:inherit;transform-origin:left;animation:growX .9s cubic-bezier(.16,1,.3,1) both;}
    .flav-value{text-align:right;font-family:var(--mono);font-size:10px;color:var(--ink-4);}

    /* [data-role-panel] */
    [data-role-panel]{display:none;}
    [data-role-panel].active{display:block;animation:fadeUp .3s ease both;}

    /* LAB / FOOTER */
    .lab-strip{background:var(--surface);border-top:1px solid var(--line);border-bottom:1px solid var(--line);}
    .lab-grid{display:flex;gap:clamp(20px,3vw,44px);flex-wrap:wrap;align-items:center;padding-top:18px;padding-bottom:18px;}
    .lab-item strong{display:block;font-size:13px;color:var(--ink);}
    .lab-item strong.missing{color:var(--ink-4);font-style:italic;font-weight:400;}
    .lab-item span{display:block;margin-top:3px;font-size:9px;text-transform:uppercase;letter-spacing:.11em;color:var(--ink-4);}
    .lab-item.last{margin-left:auto;}
    .actions{display:flex;gap:10px;flex-wrap:wrap;padding-top:16px;padding-bottom:16px;}
    .btn{appearance:none;text-decoration:none;border-radius:999px;padding:11px 20px;font-size:12px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;transition:.18s ease;display:inline-flex;align-items:center;cursor:pointer;}
    .btn.primary{background:var(--green-2);color:#fff;border:1px solid var(--green-2);}
    .btn.primary:hover{background:var(--green);border-color:var(--green);}
    .btn.secondary{background:transparent;color:var(--ink-3);border:1px solid var(--line-2);}
    .btn.secondary:hover{color:var(--green-2);border-color:var(--green-2);}
    footer{background:var(--surface);padding:26px 0 34px;}
    .footer-inner{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;flex-wrap:wrap;}
    .footer-copy{font-size:11px;color:var(--ink-4);margin-top:8px;}
    .footer-line{font-family:var(--serif);font-size:15px;color:var(--ink-4);line-height:1.8;text-align:right;}

    /* RESPONSIVE */
    @media(max-width:1100px){
      .hero-grid,.summary-grid,.two-col,.terp-layout,.advanced-grid{grid-template-columns:1fr;}
      .score-panel{position:static;}
      .grid-4,.harvest-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
      .summary-main{grid-template-columns:1fr;}
      .lab-item.last{margin-left:0;}
    }
    @media(max-width:720px){
      .grid-4,.harvest-grid,.checks-row{grid-template-columns:1fr;}
      .stat-row{grid-template-columns:1fr;}
      .safety-banner{grid-template-columns:1fr;}
      .topbar-label,.divider{display:none;}
    }
  `;

  // ── POTENCY INTENSITY MARKER POSITION ──────────────────────────────────
  // Map THC% to position on scale: 0%→0px, 10%→35%, 18%→55%, 24%→78%, 30%+→95%
  const intensityLeft = !hasTHC ? 0
    : thc >= 30 ? 95
    : thc >= 28 ? 90
    : thc >= 24 ? 79
    : thc >= 20 ? 68
    : thc >= 15 ? 52
    : thc >= 10 ? 38
    : 18;

  const thcPct = !hasTHC ? 0 : Math.min(100, Math.round((thc / 30) * 82));

  // ── CANNABINOID ROWS ────────────────────────────────────────────────────
  const cannRows = cannabinoids.length > 0 ? cannabinoids.map(c => {
    const val   = toNum(c.value);
    const isNd  = !c.value || c.value === "ND" || val === 0;
    const isPrimary = ["THCA","D9-THC","Total THC","Total CBD"].includes(c.name);
    const maxCann   = Math.max(...cannabinoids.map(x => toNum(x.value)), 0.001);
    const barWidth  = isNd ? 0 : Math.max(2, (val / maxCann) * 100);
    const barColor  = c.name.includes("THC") || c.name.includes("THCA") ? "#16855c" : c.name.includes("CBD") ? "#285ca8" : "#b97b17";
    return `
    <div class="ct-row${isPrimary ? "" : ""}">
      <div class="ct-name${isPrimary ? " primary" : ""}">${h(c.name)}</div>
      <div class="ct-bar">${!isNd ? `<div class="ct-fill" style="width:${barWidth.toFixed(1)}%;background:${barColor}"></div>` : ""}</div>
      <div class="ct-value${isNd ? " nd" : ""}">${isNd ? "ND" : h(String(c.value)) + " " + h(c.unit || "wt%")}</div>
    </div>`;
  }).join("") : `<div class="ct-row"><div class="ct-name" style="color:var(--ink-4);font-style:italic;grid-column:1/-1">Cannabinoid table not available</div></div>`;

  // ── FLAVONOID BARS ──────────────────────────────────────────────────────
  const flavMaxVal = hasFlavonoids ? Math.max(...flavonoids.map(f => toNum(f.value)), 0.001) : 0.001;
  const flavBars = hasFlavonoids ? flavonoids.filter(f => toNum(f.value) > 0).map((f, i) => {
    const pct = Math.max(3, (toNum(f.value) / flavMaxVal) * 100);
    const isCann = /cannflavin/i.test(f.name);
    return `<div class="flav-row">
      <div class="flav-name${isCann ? " cann" : ""}">${h(f.name)}</div>
      <div class="flav-track"><div class="flav-fill" style="width:${pct.toFixed(1)}%;animation-delay:${(i*0.04).toFixed(2)}s"></div></div>
      <div class="flav-value">${h(String(f.value ?? ""))} ${h(f.unit || "wt%")}</div>
    </div>`;
  }).join("") : "";

  // ── BUILD AUDIENCE VERDICT SENTENCES ───────────────────────────────────
  const verdictPatient   = heroNarrative || (hasTHC ? `A <em>high-potency, fully tested</em> flower${hasTerpenes ? ` with a ${leadTerpene ? leadTerpene.split("-")[0].toLowerCase() + "-led" : "complex"} aromatic profile` : ""}. ${!hasAllSafety ? "Some safety panels are missing — check with your prescriber." : "<em>Safe across all tested categories.</em>"}` : "Chemistry data could not be fully extracted from this COA. Some sections below may be incomplete.");
  const verdictClinician = hasTHC ? `<em>THC ${h(chemistry.thc_total || "—")}%</em>, CBD ${cbd > 0.5 ? h(chemistry.cbd_total) + "%" : "not detected"}, ${fingerprintId} chemotype. ${hasAllSafety ? "Full safety panel compliance." : "Incomplete safety panel — review below."}` : `Chemistry extraction incomplete. THC data not captured. Review source COA directly.`;
  const verdictBuyer     = hasTHC ? `<em>${safeTotal}/100 ${safeTier}</em>${hasAllSafety ? " — full safety compliance" : " — safety data incomplete"}. ${hasTerpenes ? `${terpCount} aromatic compounds at ${h(String(chemistry.total_terpenes || "—"))} wt%.` : "Terpene data not available."}${hasFlavonoids ? " Cannflavins detected." : ""}` : `Incomplete COA — score is limited by missing data. ${safeTotal}/100 reflects only what was captured.`;

  // ── POST-HARVEST ─────────────────────────────────────────────────────────
  const phCards = [
    { label: "Freshness",        data: postHarvest.freshness   || { label: "Unknown", note: "Terpene data not available — freshness cannot be inferred." } },
    { label: "Curing signal",    data: postHarvest.curing      || { label: "Unknown", note: "Insufficient terpene breadth to infer curing quality." } },
    { label: "Degradation",      data: postHarvest.degradation || { label: "Unknown", note: "CBN/CBNA data not captured from this COA." } },
    { label: "Storage stability",data: postHarvest.stability   || { label: "Unknown", note: "Moisture and water activity not reported." } },
  ].map(ph => {
    const isUnknown = ph.data.label === "Unknown" || ph.data.label === "Limited";
    return `
    <div class="harvest-card">
      <div class="harvest-label">${h(ph.label)}</div>
      <div class="harvest-value${isUnknown ? " unknown" : ""}">${h(ph.data.label)}</div>
      <div class="harvest-copy">${h(ph.data.note)}</div>
    </div>`;
  }).join("");

  // ── COMPLETE HTML OUTPUT ─────────────────────────────────────────────────
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>${h(productName)} · Alem COA Intelligence</title>
  <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>${CSS}</style>
</head>
<body>

<!-- TOPBAR -->
<header class="topbar">
  <div class="wrap">
    <div class="topbar-inner">
      <div class="brand">
        <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png" alt="Alem"
          onerror="this.style.display='none';document.querySelector('.brand-fallback').style.display='block'"/>
        <span class="brand-fallback">Alem</span>
      </div>
      <div class="divider"></div>
      <div class="topbar-label">COA Intelligence Report</div>
      <div class="topbar-right">
        ${hasAnyCritical
          ? `<div class="pill incomplete">⚠ Incomplete data</div>`
          : `<div class="pill score">A+ · ${safeTotal} / 100</div>`
        }
      </div>
    </div>
  </div>
</header>

${/* ── CRITICAL MISSING BANNER ── */
hasAnyCritical ? `
<div class="critical-banner">
  <div class="wrap">
    <div class="critical-banner-inner">
      <div class="cb-icon">!</div>
      <div class="cb-body">
        <div class="cb-title">This COA has missing data that limits the report</div>
        <div class="cb-items">
          ${criticalMissing.map(m => `<div class="cb-item"><strong>${h(m.label)}:</strong> ${h(m.msg)}</div>`).join("")}
        </div>
      </div>
    </div>
  </div>
</div>` : minorMissing.length > 0 ? `
<div class="notice-banner">
  <div class="wrap">
    <div class="notice-inner">
      <div class="notice-label">Incomplete COA</div>
      <div class="notice-items">
        ${minorMissing.map(m => `<div class="notice-item">${h(m.label)}: ${h(m.msg)}</div>`).join("")}
      </div>
    </div>
  </div>
</div>` : ""}

<!-- HERO -->
<section class="hero">
  <div class="wrap">
    <div class="hero-grid">
      <div class="fade-up">
        <div class="eyebrow">Certificate of Analysis</div>
        <h1 class="product-name">${h(productName)}</h1>
        <div class="product-type">${h(chemistry.product_type || "Dried Flower")}</div>

        <div class="hero-verdict" id="heroVerdict">
          <span data-aud="patient" class="active">${verdictPatient}</span>
          <span data-aud="clinician">${verdictClinician}</span>
          <span data-aud="buyer">${verdictBuyer}</span>
        </div>

        <div class="meta-row">${[
          chemistry.laboratory_name || null,
          chemistry.batch_number ? "Batch " + chemistry.batch_number : null,
          chemistry.coa_report_date || null
        ].filter(Boolean).map(h).join(" &nbsp;·&nbsp;") || '<span style="color:var(--ink-4);font-style:italic">Batch, lab, and date details not captured</span>'}</div>

        <div class="tag-row">
          ${hasTerpenes && fingerprintId !== "UNK" ? `<span class="tag strong">${h(fingerprintId)}</span>` : ""}
          ${hasAllSafety && allSafetyPass ? `<span class="tag strong">Full safety compliance</span>` : !hasSafety ? `<span class="tag warn">Safety data missing</span>` : `<span class="tag warn">Partial safety data</span>`}
          ${isoPass ? `<span class="tag strong">ISO 17025:2017</span>` : ""}
          ${hasTerpenes ? `<span class="tag">${terpCount} terpenes · ${h(String(chemistry.total_terpenes || "—"))} wt%</span>` : ""}
          ${leadTerpene ? `<span class="tag">${h(leadTerpene)} dominant</span>` : ""}
          ${hasFlavonoids ? `<span class="tag">Cannflavins A, B, C present</span>` : ""}
          ${!hasTHC ? `<span class="tag warn">THC data missing</span>` : ""}
        </div>
      </div>

      <!-- SCORE PANEL -->
      <aside class="score-panel fade-up" style="animation-delay:.08s">
        <div class="score-ring">
          <svg viewBox="0 0 180 180" aria-hidden="true">
            <defs>
              <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#16855c"/>
                <stop offset="100%" stop-color="#0f6847"/>
              </linearGradient>
            </defs>
            <circle cx="90" cy="90" r="76" fill="none" stroke="#e7e3d8" stroke-width="9"/>
            <circle cx="90" cy="90" r="76" fill="none" stroke="url(#ringGrad)" stroke-width="9"
              stroke-linecap="round"
              stroke-dasharray="477.5"
              stroke-dashoffset="${(477.5 * (1 - safeTotal / 100)).toFixed(1)}"/>
          </svg>
          <div class="score-center">
            <div class="score-value">${safeTotal}</div>
            <div class="score-max">/100</div>
            <div class="score-grade">${h(safeGrade)}</div>
            <div class="score-label">${h(safeTier)}</div>
          </div>
        </div>

        <div class="mini-metrics">
          <div class="mini-metric">
            <div class="mini-metric-label">Potency</div>
            <div class="mini-track"><div class="mini-fill" style="width:${potencyPct}%;background:${potencyPct >= 75 ? "var(--green)" : "var(--amber)"}"></div></div>
            <div class="mini-value">${safePotency.score}/${safePotency.max}</div>
          </div>
          <div class="mini-metric">
            <div class="mini-metric-label">Terpenes</div>
            <div class="mini-track"><div class="mini-fill" style="width:${terpPct}%;background:${terpPct >= 75 ? "var(--green)" : "var(--amber)"}"></div></div>
            <div class="mini-value" style="${terpPct >= 100 ? "color:var(--green-2)" : ""}">${safeTerpScore.score}/${safeTerpScore.max}</div>
          </div>
          <div class="mini-metric">
            <div class="mini-metric-label">Minor Cann.</div>
            <div class="mini-track"><div class="mini-fill" style="width:${minorPct}%;background:${minorPct >= 75 ? "var(--green)" : "var(--amber)"}"></div></div>
            <div class="mini-value">${safeMinors.score}/${safeMinors.max}</div>
          </div>
          <div class="mini-metric">
            <div class="mini-metric-label">Safety</div>
            <div class="mini-track"><div class="mini-fill" style="width:${safetyPct}%;background:${safetyPct >= 75 ? "var(--green)" : "var(--amber)"}"></div></div>
            <div class="mini-value" style="${safetyPct >= 100 ? "color:var(--green-2)" : ""}">${safeSafetyS.score}/${safeSafetyS.max}</div>
          </div>
          <div class="mini-metric">
            <div class="mini-metric-label">Completeness</div>
            <div class="mini-track"><div class="mini-fill" style="width:${dataPct}%;background:${dataPct >= 75 ? "var(--green)" : "var(--amber)"}"></div></div>
            <div class="mini-value" style="${dataPct >= 100 ? "color:var(--green-2)" : ""}">${safeData.score}/${safeData.max}</div>
          </div>
        </div>

        <div class="score-note">Scores the <strong>COA document</strong>, not the product. More complete data raises the score ceiling.</div>

        ${hasAnyCritical || minorMissing.length > 0 ? `
        <div class="score-cap-note">
          <strong>Score is capped</strong> by missing data.
          ${criticalMissing.map(m => `${h(m.label)} not captured.`).join(" ")}
          ${minorMissing.map(m => `${h(m.label)} absent.`).join(" ")}
          A complete COA would score higher.
        </div>` : ""}
      </aside>
    </div>
  </div>
</section>

<!-- AUDIENCE BAR -->
<nav class="audience-bar">
  <div class="wrap">
    <div class="audience-inner">
      <div class="audience-label">View as</div>
      <button class="aud-btn active" data-aud-btn="patient" type="button">Patient</button>
      <button class="aud-btn" data-aud-btn="clinician" type="button">Clinician</button>
      <button class="aud-btn" data-aud-btn="buyer" type="button">Buyer / Brand</button>
    </div>
  </div>
</nav>

<!-- ══ SAFETY — always first ══ -->
<section class="section alt">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow">Safety first</div>
      <h2>${hasAllSafety ? "Fully tested across all four safety categories" : hasSafety ? "Partially tested — some safety panels missing" : "Safety panels not found in this COA"}</h2>
      <p>${hasAllSafety
        ? `This batch includes testing for <strong>pesticides</strong>, <strong>heavy metals</strong>, <strong>microbials</strong>, and <strong>mycotoxins</strong>. All panels present and clear — tested to <em>ISO 17025:2017 standards</em>.`
        : hasSafety
        ? `Some safety panels were captured. <strong>Missing: ${minorMissing.filter(m => m.key === "partial_safety").map(m => m.label).join(", ") || "one or more panels"}</strong>. Request the full compliance documentation from the producer before prescribing or listing.`
        : `<strong>No safety data was found in this COA.</strong> Pesticide, heavy metal, microbial, and mycotoxin results are all absent. This significantly limits what Alem can verify. Request the full compliance documentation from the producer before prescribing or listing this product.`
      }</p>
    </div>

    ${!hasSafety
      ? missingBlock("No safety panels found in this COA. The report continues below, but safety cannot be evaluated without this data. Contact the producer and request the full testing documentation.", "critical")
      : `
    <div class="${allSafetyPass ? "safety-banner" : "safety-banner warn"}">
      <div>
        <h3>${allSafetyPass ? "All clear — full panel safety compliance" : hasAllSafety ? "Review required — one or more panel results need attention" : "Partial safety data — not all panels present"}</h3>
        <p>${allSafetyPass
          ? `${contaminants.pesticides_compound_count ? contaminants.pesticides_compound_count + " pesticides, " : ""}4 heavy metals, 7 microbial endpoints, 5 mycotoxins — zero failed categories. ${isoPass ? "ISO 17025:2017 accredited. " : ""}${sccPass ? "SCC accredited." : ""}`
          : hasAllSafety
          ? "All four panels were tested. Review individual results below."
          : `Only ${[hasPesticides, hasMetals, hasMicrobials, hasMycotoxins].filter(Boolean).length} of 4 safety panels captured. Request remaining panels from producer.`
        }</p>
      </div>
      <div class="confidence">
        <strong class="${allSafetyPass ? "" : "warn"}">${allSafetyPass ? "20/20" : safeSafetyS.score + "/20"}</strong>
        <span>Safety score</span>
      </div>
    </div>

    <div class="grid-4">
      ${safetyCard("Pesticides",
        contaminants.pesticides_status || "All clear",
        pestPass, hasPesticides,
        `<div class="safe-row"><span>Compounds tested</span><strong class="ok">${h(contaminants.pesticides_compound_count || "—")}</strong></div>
         <div class="safe-row"><span>Above reporting limit</span><strong class="ok">0</strong></div>
         <div class="safe-row"><span>Failures</span><strong class="ok">0</strong></div>`,
        contaminants.pesticides_method || "")}
      ${safetyCard("Heavy Metals",
        contaminants.heavy_metals_status || "Pass",
        metalsPass, hasMetals,
        [
          ["Arsenic (As)", contaminants.arsenic_result],
          ["Cadmium (Cd)", contaminants.cadmium_result],
          ["Lead (Pb)",    contaminants.lead_result],
          ["Mercury (Hg)", contaminants.mercury_result],
        ].filter(([,v]) => v).map(([l,v]) => safeRow(l,v)).join("") ||
        `<div class="safe-nt-note">Individual metal results not captured. Status: ${h(contaminants.heavy_metals_status || "not reported")}</div>`,
        contaminants.heavy_metals_method || "")}
      ${safetyCard("Microbials",
        contaminants.microbials_status || "All absent",
        microPass, hasMicrobials,
        [
          ["Yeast & Mold",     contaminants.yeast_mold],
          ["Total Aerobic",    contaminants.total_aerobic],
          ["Salmonella",       contaminants.salmonella],
          ["E. coli",          contaminants.e_coli],
          ["S. aureus",        contaminants.s_aureus],
          ["P. aeruginosa",    contaminants.p_aeruginosa],
        ].filter(([,v]) => v).map(([l,v]) => safeRow(l,v)).join("") ||
        `<div class="safe-nt-note">Individual microbial results not captured. Status: ${h(contaminants.microbials_status || "not reported")}</div>`,
        contaminants.microbials_method || "")}
      ${safetyCard("Mycotoxins",
        contaminants.mycotoxins_status || "Not detected",
        mycoPass, hasMycotoxins,
        [
          ["Aflatoxin B1/B2/G1/G2", contaminants.aflatoxin_b1 ? `${contaminants.aflatoxin_b1} / ${contaminants.aflatoxin_b2 || "—"} / ${contaminants.aflatoxin_g1 || "—"} / ${contaminants.aflatoxin_g2 || "—"}` : null],
          ["Ochratoxin A",           contaminants.ochratoxin_a],
          ["Water activity",         waterActivity ? waterActivity + " aw" : null],
        ].filter(([,v]) => v).map(([l,v]) => safeRow(l,v)).join("") ||
        `<div class="safe-nt-note">Individual mycotoxin results not captured. Status: ${h(contaminants.mycotoxins_status || "not reported")}</div>`,
        contaminants.mycotoxins_method || "")}
    </div>`}

    <div class="checks-row">
      ${checkRow("ISO 17025:2017", isoPass ? "✓ Confirmed" : "Not confirmed",          isoPass, !isoPass)}
      ${checkRow("SCC Accreditation", sccPass ? "✓ Confirmed" : "Not confirmed",        sccPass, !sccPass)}
      ${checkRow("Foreign matter",  contaminants.foreign_matter_status || chemistry.foreign_matter || "not reported", /none detected|pass|nd/i.test(contaminants.foreign_matter_status || chemistry.foreign_matter || ""), !contaminants.foreign_matter_status)}
      ${checkRow("Moisture (LOD)",  hasMoisture ? moisture + "%" : "not reported",      hasMoisture, !hasMoisture)}
      ${checkRow("Water activity",  hasWaterActivity ? waterActivity + " aw" : "not reported", hasWaterActivity, !hasWaterActivity)}
      ${checkRow("Residual solvents", /pass|nd/i.test(contaminants.residual_solvents_status || "") ? "Pass" : (contaminants.residual_solvents_status || "not tested"), /pass|nd/i.test(contaminants.residual_solvents_status || ""), true)}
    </div>

    <div data-role-panel="patient" class="active">
      <div class="callout ${allSafetyPass ? "green" : hasSafety ? "amber" : "red"}" style="margin-top:20px">
        <strong>${allSafetyPass ? "This product has passed all safety checks" : hasSafety ? "Some safety panels are present — some are missing" : "Safety data was not found in this COA"}</strong>
        <p>${allSafetyPass
          ? "Testing was conducted by an accredited laboratory to pharmaceutical-grade standards. Every contaminant category came back clean. You can have confidence in the cleanliness of this product."
          : hasSafety
          ? "Some safety testing was completed. However, not all four categories are present in this COA. Ask your prescriber or the producer about the missing panels before use."
          : "This COA does not contain safety testing data. Always ask your prescriber or the producer for a complete safety panel before using any cannabis product."
        }</p>
      </div>
    </div>
    <div data-role-panel="clinician">
      <div class="callout ${allSafetyPass ? "green" : hasSafety ? "amber" : "red"}" style="margin-top:20px">
        <strong>Clinical safety interpretation</strong>
        <p>${allSafetyPass
          ? `Full panel compliance — all four categories present and clear. ${isoPass ? "ISO 17025:2017 + " : ""}${sccPass ? "SCC " : ""}${isoPass || sccPass ? "accreditation confirmed." : ""} Water activity ${hasWaterActivity ? waterActivity + " aw — microbiologically stable." : "not reported."} Supports clinical listing without caveat.`
          : hasSafety
          ? `Partial safety panel captured. Missing: ${[!hasPesticides && "Pesticides", !hasMetals && "Heavy Metals", !hasMicrobials && "Microbials", !hasMycotoxins && "Mycotoxins"].filter(Boolean).join(", ")}. Request complete panel before prescribing.`
          : "No safety data captured from this COA. Clinical prescription should not proceed without requesting and reviewing the full safety documentation from the producer."
        }</p>
      </div>
    </div>
    <div data-role-panel="buyer">
      <div class="callout ${allSafetyPass ? "green" : hasSafety ? "amber" : "red"}" style="margin-top:20px">
        <strong>Commercial safety story</strong>
        <p>${allSafetyPass
          ? `20/20 safety score with ${isoPass || sccPass ? "dual" : "no"} accreditation. Every panel present, every result clean. Fully defensible safety documentation for dispensary or clinical listing at premium pricing.`
          : hasSafety
          ? `Partial safety documentation. ${[!hasPesticides && "Pesticides", !hasMetals && "Metals", !hasMicrobials && "Microbials", !hasMycotoxins && "Mycotoxins"].filter(Boolean).join(", ")} missing. Listing with incomplete safety data exposes commercial and reputational risk — request full compliance panel from producer.`
          : "No safety data found. This product cannot be responsibly listed without complete safety documentation. Request and review the full compliance panel before proceeding."
        }</p>
      </div>
    </div>
  </div>
</section>

<!-- ══ CORE SUMMARY ══ -->
<section class="section">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow">Core summary</div>
      <h2>${hasTHC || hasTerpenes ? "One-page understanding before the deep dive" : "Limited data — partial summary only"}</h2>
      <p>${hasTHC && hasTerpenes
        ? `This batch shows <strong>${thc >= 24 ? "high" : thc >= 18 ? "moderate-high" : "moderate"} THC</strong>, <strong>CBD ${cbd > 0.5 ? h(chemistry.cbd_total) + "%" : "not detected"}</strong>, ${hasTerpenes ? `exceptional <strong>terpene richness</strong>,` : ""} and <em>${hasAllSafety && allSafetyPass ? "complete safety coverage" : hasSafety ? "partial safety data" : "no safety data captured"}</em>.`
        : `Some data could not be extracted from this COA. The sections below show everything that was captured.`
      }</p>
    </div>
    <div class="summary-grid">
      <div class="summary-score">
        <div class="small">Intelligence score</div>
        <div class="big${safeTotal < 30 ? " dim" : ""}">${safeTotal}</div>
        <div class="sub">${h(safeTier)} · ${h(safeGrade)}${hasAnyCritical ? "*" : ""}</div>
      </div>
      <div class="summary-main">
        <div class="card">
          <div class="card-title">Chemotype fingerprint</div>
          ${hasTerpenes && fingerprintId !== "UNK"
            ? `<div class="fingerprint">${h(fingerprintId)}</div>
               <div class="muted">${h(terpIntel.note)}</div>`
            : `<div class="fingerprint dim">Not available</div>
               <div class="muted">Terpene data not captured — chemotype cannot be determined from this COA.</div>`
          }
        </div>
        <div class="card">
          <div class="card-title">Three things that matter most</div>
          <div class="bullets">
            <div class="bullet"><div class="bullet-num">01</div><div>${hasTHC
              ? `<strong>${h(chemistry.thc_total)}% total THC</strong> with <strong>CBD ${cbd > 0.5 ? h(chemistry.cbd_total) + "%" : "not detected"}</strong> — a ${thc >= 24 ? "strong, unbuffered" : "moderate"} THC profile.`
              : `<strong>THC data not captured</strong> — potency cannot be evaluated from this COA. Request the full cannabinoid table from the producer.`
            }</div></div>
            <div class="bullet"><div class="bullet-num">02</div><div>${hasTerpenes
              ? `<strong>${h(String(chemistry.total_terpenes || "—"))} wt% total terpenes</strong> across <strong>${terpCount} compounds</strong>${terps >= 3 ? " — rare high-richness tier" : ""}.`
              : `<strong>Terpene data not captured</strong> — aromatic profile and effect direction cannot be determined.`
            }</div></div>
            <div class="bullet"><div class="bullet-num">03</div><div>${hasAllSafety && allSafetyPass
              ? `<strong>All four safety panels clear</strong> with ${isoPass ? "ISO 17025:2017 + " : ""}${sccPass ? "SCC " : ""}accredited lab backing.`
              : hasSafety
              ? `<strong>Partial safety data</strong> captured. ${[!hasPesticides&&"Pesticides",!hasMetals&&"Metals",!hasMicrobials&&"Microbials",!hasMycotoxins&&"Mycotoxins"].filter(Boolean).join(", ")} ${[!hasPesticides&&"Pesticides",!hasMetals&&"Metals",!hasMicrobials&&"Microbials",!hasMycotoxins&&"Mycotoxins"].filter(Boolean).length === 1 ? "panel is" : "panels are"} missing.`
              : `<strong>No safety data</strong> was found in this COA. Full compliance documentation must be requested from the producer.`
            }</div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ══ POTENCY ══ -->
${!hasTHC
  ? sectionUnavailable("Potency — data not available", "Potency", "THC values were not captured from this COA. The cannabinoid table may be absent, in an unreadable format, or expressed in an unrecognised unit. Review the source COA directly and request the cannabinoid analysis section from the producer if missing.")
  : `
<section class="section alt">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow">Potency</div>
      <h2>${thc >= 24 ? "High-THC, no-CBD — upper-tier intensity" : thc >= 18 ? "Moderate-high THC profile" : "Moderate THC profile"}</h2>
      <p>THC sits at <strong>${h(chemistry.thc_total)}%</strong>${cbd > 0.5 ? ` with CBD at <strong>${h(chemistry.cbd_total)}%</strong>` : " while <strong>CBD is not detected</strong>"}. ${cbd < 0.5 ? "The experience is overwhelmingly THC-led, with <em>no intrinsic CBD moderation</em>." : "<em>CBD is present</em> and may moderate the THC-dominant effect."}</p>
    </div>

    ${thc >= 24 && cbd < 0.5 ? `
    <div class="callout amber">
      <strong>Intensity note</strong>
      <p data-role-copy="patient" class="active-copy">This is a strong THC product. Start with a very small amount and wait 30–60 minutes before increasing. No CBD means no built-in moderating effect.</p>
      <p data-role-copy="clinician" style="display:none">Unbuffered THC profile — ${h(chemistry.thc_total)}% anhydrous. Conservative titration and tolerance-aware patient selection are appropriate.</p>
      <p data-role-copy="buyer" style="display:none">High-THC positioning supports premium placement. Strongest when paired with terpene depth — this batch delivers both.</p>
    </div>` : ""}

    <div class="two-col">
      <div>
        <div class="stat-row">
          <div class="stat"><div class="stat-value">${h(chemistry.thc_total || "ND")}</div><div class="stat-label">Total THC %</div></div>
          <div class="stat"><div class="stat-value${cbd < 0.5 ? " nd" : ""}">${h(chemistry.cbd_total || "ND")}</div><div class="stat-label">CBD Total %</div></div>
          <div class="stat"><div class="stat-value" style="font-size:clamp(16px,2vw,20px);color:${cbn < 0.3 && cbna < 0.1 ? "var(--green-2)" : "var(--amber)"};">${cbn < 0.3 && cbna < 0.1 ? "Fresh" : "Review"}</div><div class="stat-label">Degradation signal</div></div>
        </div>
        <div class="market-card">
          <div class="market-label">Market context · dried flower THC distribution</div>
          <div class="market-bar">
            <div class="zone-low"></div><div class="zone-mid"></div><div class="zone-high"></div>
            <div class="market-fill" style="width:${thcPct}%"></div>
            <div class="market-pin" style="left:${thcPct}%">${h(chemistry.thc_total)}%</div>
          </div>
          <div class="market-legend"><span>0%</span><span>avg 18%</span><span>premium 22%</span><span>top tier</span><span>30%+</span></div>
        </div>
      </div>
      <div>
        <div class="intensity">
          <div class="intensity-label">Experience intensity scale</div>
          <div class="intensity-scale"><div class="intensity-marker" style="left:${intensityLeft}%"></div></div>
          <div class="intensity-legend"><span>Low</span><span>Moderate</span><span>High</span><span>Very high</span></div>
        </div>
        <div class="callout green" style="margin-top:18px">
          <strong>THCA conversion</strong>
          <p>Most active potential is stored in <strong>THCA (${h(chemistry.thca || "—")}%)</strong>. It becomes psychoactively active when heated — total THC is calculated from THCA × 0.877 plus D9-THC.</p>
        </div>
      </div>
    </div>

    ${hasCannabinoids ? `
    <div class="cannabinoid-table">
      <div class="ct-row head"><div>Compound</div><div>Relative abundance</div><div style="text-align:right">wt%</div></div>
      ${cannRows}
      ${chemistry.total_cannabinoids ? `<div class="ct-row total-row"><div class="ct-name primary">Total cannabinoids</div><div></div><div class="ct-value total">${h(chemistry.total_cannabinoids)} wt%</div></div>` : ""}
    </div>` : missingBlock("Cannabinoid table not available — individual compound values not captured.", "minor")}
  </div>
</section>`}

<!-- ══ TERPENES ══ -->
${!hasTerpenes
  ? sectionUnavailable("Terpene architecture — data not available", "Terpene architecture", "No terpene data was captured from this COA. The terpene analysis section may be absent, in an unreadable format, or not included in this version of the report. The chemotype fingerprint and effect direction cannot be determined without terpene data.")
  : `
<section class="section">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow">Terpene architecture</div>
      <h2>Top-tier richness — ${h(fingerprintId)} chemotype</h2>
      <p><strong>${h(String(chemistry.total_terpenes || "—"))} wt% total terpenes</strong> across <strong>${terpCount} compounds</strong>${terps >= 3 ? " — top 5% for dried flower" : ""}. Led by <strong>${h(leadTerpene)}</strong>${leadTerpene === "Trans-Caryophyllene" ? ", the only terpene with confirmed cannabinoid receptor activity" : ""}.</p>
    </div>

    <div class="terp-layout">
      <div class="terp-card">
        <div class="terp-total">
          <div class="terp-total-label">Total terpenes · ${terpCount} compounds</div>
          <div class="terp-total-track"><div class="terp-total-fill" style="width:${Math.min(100, (terps / 4) * 100).toFixed(0)}%"></div></div>
          <div class="terp-total-value">${h(String(chemistry.total_terpenes || "—"))} wt%</div>
        </div>
        <div class="top-terp-list">
          ${activeTerps.slice(0, 3).map((t, i) => terpBar(t, i)).join("")}
        </div>
        ${activeTerps.length > 3 ? `
        <div class="expand-box" id="moreTerps">
          <button class="expand-toggle" type="button" onclick="toggleExpand('moreTerps')">
            <span>+${activeTerps.length - 3} more compounds</span>
            <span>Expand</span>
          </button>
          <div class="expand-content">
            <div class="mini-list">
              ${activeTerps.slice(3).map((t, i) => terpBar(t, i + 3)).join("")}
            </div>
          </div>
        </div>` : ""}
      </div>

      <div style="display:grid;gap:18px">
        <div class="fingerprint-card">
          <div class="label">Chemotype fingerprint</div>
          <div class="value">${h(fingerprintId)}</div>
          <div class="copy">${h(terpIntel.note)} <strong>${hasTerpenes ? "Complex, preserved, and commercially distinctive." : ""}</strong></div>
        </div>
        <div class="card">
          <div class="card-title">Entourage signal</div>
          <div style="font-size:22px;font-weight:700;color:var(--green-2);margin-bottom:8px">${terpCount >= 10 ? "Strong" : terpCount >= 6 ? "Moderate" : "Limited"}</div>
          <div class="muted">${cannabinoids.filter(c => toNum(c.value) > 0).length} cannabinoids + ${terpCount} terpenes create a ${terpCount >= 10 ? "richer" : "partial"} pharmacological picture than THC alone.</div>
        </div>
        ${leadTerpene === "Trans-Caryophyllene" ? `
        <div class="card">
          <div class="card-title">CB2 agonist activity</div>
          <div class="muted" style="font-size:13px;line-height:1.75">At ${h(String(terpenes[0]?.value || "—"))} wt%, Trans-Caryophyllene is the <strong>only terpene confirmed to directly engage the cannabinoid system</strong> via CB2 receptor agonism. Studied for anti-inflammatory, analgesic, and anxiolytic effects.</div>
        </div>` : ""}
      </div>
    </div>
  </div>
</section>`}

<!-- ══ INTELLIGENCE ══ -->
<section class="section alt">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow">What this means</div>
      <h2>The same chemistry, explained for different readers</h2>
      <p>Interpretation shifts depending on who is reading. The data stays the same${hasAnyCritical ? " — <strong>note that some data is missing</strong>, so some statements below reflect what could be determined from an incomplete COA" : ""}.</p>
    </div>
    <div class="intel-shell">
      <div class="intel-tabs">
        <button class="intel-tab active" type="button" data-intel-tab="patient">Patient</button>
        <button class="intel-tab" type="button" data-intel-tab="clinician">Clinician</button>
        <button class="intel-tab" type="button" data-intel-tab="buyer">Buyer / Brand</button>
      </div>
      <div class="intel-panel active" data-intel-panel="patient">${audiencePanel(audiences.patient)}</div>
      <div class="intel-panel" data-intel-panel="clinician">${audiencePanel(audiences.clinical)}</div>
      <div class="intel-panel" data-intel-panel="buyer">${audiencePanel(audiences.buyer)}</div>
    </div>
  </div>
</section>

<!-- ══ POST-HARVEST ══ -->
<section class="section">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow">Post-harvest signals</div>
      <h2>What the chemistry suggests about handling, curing, and preservation</h2>
      <p>Chemistry-based signals — not absolute claims. ${!hasTerpenes ? `<strong>Note: terpene data is absent from this COA, so freshness and curing signals cannot be reliably inferred.</strong>` : "They help infer how well the material was retained post-harvest."}</p>
    </div>
    <div class="harvest-grid">${phCards}</div>
  </div>
</section>

${/* ══ ADVANCED — only show if flavonoid data exists ══ */
hasFlavonoids ? `
<section class="section alt">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow">Advanced data</div>
      <h2>Cannflavins A, B, C detected — rare phytochemical depth</h2>
      <p>Flavonoid data is present in <strong>fewer than 10% of COAs</strong>. Cannabis-specific Cannflavins have attracted significant research attention for anti-inflammatory properties. Their presence is a marker of analytical completeness.</p>
    </div>
    <div class="advanced-grid">
      <div class="panel-card">
        <div class="card-title">Flavonoid profile — ${flavonoids.length} compounds</div>
        <h3 style="margin:0 0 10px;font-family:var(--serif);font-size:22px;font-weight:400">Cannflavins A, B &amp; C confirmed</h3>
        <div class="muted" style="margin-bottom:18px">Cannabis-specific flavonoids with preclinical anti-inflammatory data — PGE2 inhibition studied at up to 30× the potency of aspirin.</div>
        ${flavBars}
      </div>
      <div class="panel-card">
        <div class="card-title">Why this matters</div>
        <div class="bullets">
          <div class="bullet"><div class="bullet-num">01</div><div><strong>Cannflavins are unique to Cannabis sativa.</strong> Their presence is a genuine phytochemical signature — not found in other plants.</div></div>
          <div class="bullet"><div class="bullet-num">02</div><div>Preclinical research shows Cannflavin A and B may inhibit prostaglandin E2 at up to <strong>30× the potency of aspirin</strong> in cell models.</div></div>
          <div class="bullet"><div class="bullet-num">03</div><div><strong>Fewer than 10% of COAs</strong> include flavonoid panels. Having all three Cannflavins confirmed supports both clinical narrative and premium brand differentiation.</div></div>
        </div>
      </div>
    </div>
  </div>
</section>` : ""}

<!-- LAB STRIP -->
<div class="lab-strip">
  <div class="wrap">
    <div class="lab-grid">
      <div class="lab-item"><strong${!hasLabName ? ' class="missing"' : ""}>${hasLabName ? h(chemistry.laboratory_name) : "Not identified"}</strong><span>Laboratory</span></div>
      <div class="lab-item"><strong${!isoPass && !sccPass ? ' class="missing"' : ""}>${h(chemistry.laboratory_accreditation || (isoPass ? "ISO 17025:2017" : sccPass ? "SCC" : "Not confirmed"))}</strong><span>Accreditation</span></div>
      <div class="lab-item"><strong${!hasDate ? ' class="missing"' : ""}>${hasDate ? h(chemistry.coa_report_date) : "Not captured"}</strong><span>Report date</span></div>
      <div class="lab-item"><strong${!hasBatchNum ? ' class="missing"' : ""}>${hasBatchNum ? h(chemistry.batch_number) : "Not captured"}</strong><span>Batch</span></div>
      <div class="lab-item last"><strong style="font-family:var(--mono);color:var(--green-2)">Schema v7.0</strong><span>Alem Intelligence</span></div>
    </div>
    <div class="actions">
      <a href="/" class="btn primary">Analyse new COA</a>
      ${options.documentId ? `<a href="/pdf/${h(String(options.documentId))}" class="btn secondary" target="_blank">Export PDF</a>` : ""}
    </div>
  </div>
</div>

<footer>
  <div class="wrap">
    <div class="footer-inner">
      <div>
        <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png" alt="Alem"
          style="height:24px;display:block" onerror="this.style.display='none'"/>
        <div class="footer-copy">alem.solutions · COA Intelligence · Free. No account. No catch.</div>
      </div>
      <div class="footer-line">Upload your COA.<br>Understand your chemistry.</div>
    </div>
  </div>
</footer>

<script>
(function(){
  var current = "patient";
  var audBtns     = document.querySelectorAll("[data-aud-btn]");
  var verdictSpans= document.querySelectorAll("#heroVerdict [data-aud]");
  var intelTabs   = document.querySelectorAll("[data-intel-tab]");
  var intelPanels = document.querySelectorAll("[data-intel-panel]");
  var roleCopies  = document.querySelectorAll("[data-role-copy]");
  var rolePanels  = document.querySelectorAll("[data-role-panel]");

  function setAudience(aud) {
    current = aud;
    audBtns.forEach(function(b){ b.classList.toggle("active", b.dataset.audBtn === aud); });
    verdictSpans.forEach(function(s){ s.classList.toggle("active", s.dataset.aud === aud); });
    roleCopies.forEach(function(p){ p.style.display = p.dataset.roleCopy === aud ? "block" : "none"; });
    rolePanels.forEach(function(p){ p.classList.toggle("active", p.dataset.rolePanel === aud); });
    intelTabs.forEach(function(t){ t.classList.toggle("active", t.dataset.intelTab === aud); });
    intelPanels.forEach(function(p){ p.classList.toggle("active", p.dataset.intelPanel === aud); });
  }

  audBtns.forEach(function(b){ b.addEventListener("click", function(){ setAudience(this.dataset.audBtn); }); });
  intelTabs.forEach(function(t){ t.addEventListener("click", function(){ setAudience(this.dataset.intelTab); }); });

  window.toggleExpand = function(id){
    var box = document.getElementById(id);
    if(!box) return;
    box.classList.toggle("open");
    var lbl = box.querySelector(".expand-toggle span:last-child");
    if(lbl) lbl.textContent = box.classList.contains("open") ? "Collapse" : "Expand";
  };

  setAudience("patient");
})();
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

app.post("/upload-coa-multi", upload.array("files", 10), async (req, res) => {
  try {
    console.log(`📥 Multi-file COA upload: ${(req.files || []).length} file(s)`);
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ success: false, error: "No files uploaded" });
    }

    // Upload to Supabase for storage, but OCR from buffer directly
    const uploadResults = await Promise.all(
      req.files.map((file, idx) =>
        uploadBufferToSupabase({
          buffer: file.buffer,
          originalName: file.originalname || `upload-${Date.now()}-${idx}`,
          mimeType: file.mimetype,
          folder: "raw_documents",
        })
      )
    );
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
    return res.status(500).json({ success: false, error: error.message || "Upload pipeline failed" });
  }
});

app.post("/upload-coa", upload.single("file"), async (req, res) => {
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
    return res.send(renderReportHTML(row?.report_json || {}, { documentId: row.id }));
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
    const html = renderReportHTML(row?.report_json || {}, { documentId: row.id });

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
  // Only exit for truly fatal errors that corrupt state
  // For rendering/route errors, log and continue
});

app.listen(PORT, () => {
  console.log(`🌿 Alem Chemical Intelligence v6 running on port ${PORT}`);
});