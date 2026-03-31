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

// ─────────────────────────────────────────────
// LAYER 5 — REPORT HTML RENDERER (v7 — Full Chemistry + Full Safety)
// ─────────────────────────────────────────────

function renderReportHTML(data = {}) {
  const esc = (str) => String(str || "-").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const badges = (data.flags || []).map(flag => `
    <div class="badge ${flag.type || "neutral"}">${esc(flag.label)}</div>
  `).join("");

  const terps = (data.top_terpenes || []).map(t => `
    <div class="terp-row">
      <span>${esc(t.name)}</span>
      <span>${esc(t.value)}%</span>
    </div>
  `).join("");

  return `
  <!DOCTYPE html>
  <html>
  <head>
    <meta charset="UTF-8" />
    <title>Alem Intelligence Report</title>

    <style>
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0b0b0c;
        color: #f1f1f1;
      }

      .container {
        max-width: 1100px;
        margin: auto;
        padding: 40px 20px;
      }

      .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 40px;
      }

      .title {
        font-size: 28px;
        font-weight: 600;
      }

      .subtitle {
        font-size: 14px;
        color: #888;
      }

      .grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 30px;
      }

      .card {
        background: #111;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #1f1f1f;
      }

      .card h3 {
        font-size: 13px;
        color: #888;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
      }

      .big {
        font-size: 28px;
        font-weight: 600;
      }

      .badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
      }

      .badge {
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
      }

      .badge.positive {
        background: #0f2;
        color: #000;
      }

      .badge.warning {
        background: #ffb020;
        color: #000;
      }

      .badge.neutral {
        background: #333;
      }

      .section {
        margin-top: 40px;
      }

      .section-title {
        font-size: 18px;
        margin-bottom: 15px;
      }

      .terp-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #1f1f1f;
      }

      .summary {
        line-height: 1.6;
        color: #ccc;
      }

      .footer {
        margin-top: 60px;
        font-size: 12px;
        color: #666;
        text-align: center;
      }
    </style>
  </head>

  <body>
    <div class="container">

      <!-- HEADER -->
      <div class="header">
        <div>
          <div class="title">Alem Chemical Intelligence</div>
          <div class="subtitle">COA Analysis Report</div>
        </div>
      </div>

      <!-- KPI GRID -->
      <div class="grid">
        <div class="card">
          <h3>Total THC</h3>
          <div class="big">${esc(data.total_thc)}%</div>
        </div>

        <div class="card">
          <h3>Total CBD</h3>
          <div class="big">${esc(data.total_cbd)}%</div>
        </div>

        <div class="card">
          <h3>Total Terpenes</h3>
          <div class="big">${esc(data.total_terpenes)}%</div>
        </div>
      </div>

      <!-- FLAGS -->
      <div class="card">
        <h3>Quality Signals</h3>
        <div class="badges">
          ${badges || "<span>No data available</span>"}
        </div>
      </div>

      <!-- TERPENES -->
      <div class="section">
        <div class="section-title">Top Terpenes</div>
        <div class="card">
          ${terps || "<span>No terpene data</span>"}
        </div>
      </div>

      <!-- SUMMARY -->
      <div class="section">
        <div class="section-title">Clinical & Sensorial Interpretation</div>
        <div class="card summary">
          ${esc(data.summary || "No summary available")}
        </div>
      </div>

      <!-- FOOTER -->
      <div class="footer">
        Alem Solutions — Chemical Intelligence Platform  
        <br/>
        This report is for educational purposes only.
      </div>

    </div>
  </body>
  </html>
  `;
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
