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

For hero_narrative: write 2 precise, clinical sentences about this specific product's chemistry. State the chemotype identity (dominant terpene + potency tier), then describe what makes this profile distinctive. Use factual chemical language. Do NOT use phrases like "making it suitable for", "seeking strong effects", "unique aromatic character", or any consumer-facing promotional language. Focus on: what the chemistry IS, not what it does to users.
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
// ─────────────────────────────────────────────

function computeIntelligenceScore(extracted, contaminants) {
  let score = 0;
  const breakdown = {};

  const thc = toNum(extracted.thc_total);
  let potencyScore = 0;
  if (thc >= 28) potencyScore = 25;
  else if (thc >= 24) potencyScore = 22;
  else if (thc >= 20) potencyScore = 18;
  else if (thc >= 15) potencyScore = 13;
  else if (thc > 0) potencyScore = 8;
  const cbd = toNum(extracted.cbd_total);
  if (cbd >= 15 && thc < 5) potencyScore = Math.max(potencyScore, 20);
  breakdown.potency = { score: potencyScore, max: 25 };
  score += potencyScore;

  const terps = toNum(extracted.total_terpenes);
  const terpCount = (extracted.top_terpenes || []).filter(t => toNum(t.value) > 0).length;
  let terpScore = 0;
  if (terps >= 3.5) terpScore = 22;
  else if (terps >= 2.5) terpScore = 20;
  else if (terps >= 1.8) terpScore = 16;
  else if (terps >= 1.0) terpScore = 11;
  else if (terps > 0) terpScore = 6;
  if (terpCount >= 8) terpScore = Math.min(25, terpScore + 5);
  else if (terpCount >= 5) terpScore = Math.min(25, terpScore + 3);
  breakdown.terpenes = { score: terpScore, max: 25 };
  score += terpScore;

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
  if (cbn >= 1) minorScore = Math.max(0, minorScore - 4);
  breakdown.minors = { score: minorScore, max: 20 };
  score += minorScore;

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
  if (accred.includes("17025") || accred.includes("iso")) safetyScore = Math.min(20, safetyScore + 3);
  breakdown.safety = { score: safetyScore, max: 20 };
  score += safetyScore;

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

// Per-terpene educational data for info icons
const TERPENE_EDUCATION = {
  "Terpinolene": { aroma: "Fresh, piney, floral, slightly herbal — bright and clean.", therapeutic: "Antioxidant properties noted in preclinical studies. Mild anxiolytic signals. Found in Jack Herer, Durban Poison, Ghost Train Haze. Fewer than 15% of dried flower products are Terpinolene-dominant — a genuine market differentiator." },
  "β-Myrcene": { aroma: "Earthy, musky, fruity — ripe mango or hops. The most common cannabis terpene.", therapeutic: "May contribute sedating, relaxing effects at higher concentrations. Anti-inflammatory and analgesic properties in preclinical data. Some research suggests Myrcene may enhance cannabinoid absorption through cell membranes." },
  "Ocimene": { aroma: "Sweet, herbal, woody — fresh basil and tarragon. Marker of Haze-type genetics.", therapeutic: "Antifungal and antiviral properties noted in vitro. Contributes to the fresh, uplifting aromatic profile. Adds complexity to the entourage profile alongside Terpinolene." },
  "α-Pinene": { aroma: "Crisp pine, fresh forest air — also found in rosemary and eucalyptus.", therapeutic: "Bronchodilator in preclinical models. May counteract short-term memory effects sometimes associated with THC. Studied for anti-inflammatory and antimicrobial properties." },
  "Trans-Caryophyllene": { aroma: "Spicy, peppery, woody — the same compound that gives black pepper its heat.", therapeutic: "The only terpene confirmed to act as a CB2 cannabinoid receptor agonist. Studied for anti-inflammatory, analgesic, and anxiolytic effects. A unique pharmacological pathway distinct from classical terpene activity." },
  "Farnesene": { aroma: "Green apple, woody, floral. Found in apple skin, green tea. Marker of well-preserved plant material.", therapeutic: "Anti-inflammatory properties in preclinical research. Its presence suggests minimal terpene degradation post-harvest — a freshness and quality signal." },
  "β-Pinene": { aroma: "Piney, green, fresh — slightly more herbal than α-Pinene. Works synergistically with α-Pinene.", therapeutic: "Mild antiseptic properties. The Pinene family combination may contribute to bronchodilatory effects. Adds to the overall freshness of the aromatic profile." },
  "α-Phellandrene": { aroma: "Citrusy, minty, slightly peppery — found in eucalyptus, dill, cinnamon.", therapeutic: "Antifungal and potential anticancer properties in early research. Contributes to the bright, fresh quality of Terpinolene-dominant chemotypes." },
  "α-Bisabolol": { aroma: "Delicate floral, honey-like, slightly sweet. Widely used in cosmetics for skin-soothing.", therapeutic: "Well-documented anti-inflammatory and wound-healing properties. May enhance absorption of other compounds. Contributes to aromatic refinement and smooth experiential quality." },
  "Limonene": { aroma: "Bright citrus — lemon, orange. One of the most recognisable cannabis terpenes.", therapeutic: "Mood-elevating properties noted in early clinical studies. Potent antifungal and antibacterial. May reduce anxiety. Here it contributes citrus top notes as a background character." },
  "α-Humulene": { aroma: "Woody, earthy, spicy — like hops. An isomer of β-Caryophyllene.", therapeutic: "Anti-inflammatory, antibacterial, and appetite-suppressing properties in preclinical research. Often works synergistically with Caryophyllene. A trace contributor to the spicy background note." },
  "Guaiol": { aroma: "Piney, floral, rose-like with a woody undertone. A sesquiterpenoid alcohol — less volatile than most terpenes.", therapeutic: "Antimicrobial and anti-inflammatory properties in preclinical models. Its presence at detectable levels in flower suggests careful post-harvest preservation — a quality signal." },
};

// Per-cannabinoid educational data for info icons
const CANNABINOID_EDUCATION = {
  "THCA": { title: "THCA — Tetrahydrocannabinolic Acid", body: "The raw, non-psychoactive precursor to THC. Converts to psychoactive THC through decarboxylation (heat). Multiply by 0.877 to estimate active THC yield.", therapeutic: "Emerging research suggests THCA itself may have anti-inflammatory and neuroprotective properties without psychoactivity. Some patients use raw (unheated) cannabis specifically to access THCA." },
  "D9-THC": { title: "D9-THC — Delta-9-Tetrahydrocannabinol", body: "The primary psychoactive cannabinoid. This is already-converted active THC present before any heating. Combined with THCA conversion, delivers the total THC figure.", therapeutic: "Binds to CB1 receptors producing analgesic, antiemetic, and appetite-stimulating effects. Well-documented therapeutic and psychoactive properties." },
  "CBGA": { title: "CBGA — Cannabigerolic Acid (The Mother Cannabinoid)", body: "The biosynthetic precursor to ALL other cannabinoids including THCA, CBDA, and CBCA. Elevated CBGA suggests early harvest or genetics with slower enzymatic conversion.", therapeutic: "CBGA is under active research for anti-inflammatory, antibacterial, and metabolic effects. Its elevated presence signals genetic richness and biosynthetic complexity. A premium marker." },
  "CBCA": { title: "CBCA — Cannabichromenic Acid", body: "Precursor to CBC (Cannabichromene), a non-psychoactive minor cannabinoid. CBC is one of the 'big six' cannabinoids of research interest.", therapeutic: "Early research suggests CBC may contribute anti-inflammatory, antidepressant, and neurogenesis-promoting effects. Thought to synergise with THC to enhance analgesic outcomes." },
  "THCVA": { title: "THCVA — Tetrahydrocannabivarinic Acid", body: "A varin-type cannabinoid — a shorter-chain analogue of THCA. Associated with specific regional genetics (African landraces). Its decarboxylated form THCV is studied for appetite suppression.", therapeutic: "THCV may act as a CB1 antagonist at low doses. Studied for blood sugar regulation and anticonvulsant properties. Adds genetic complexity and entourage breadth." },
  "CBG": { title: "CBG — Cannabigerol", body: "The active (non-acid) form of CBG. Sometimes called the 'Rolls-Royce of cannabinoids' due to rarity. Non-psychoactive and binds to both CB1 and CB2 receptors.", therapeutic: "Studied for antibacterial, neuroprotective, and anti-inflammatory properties. Emerging research links CBG to potential benefits in inflammatory bowel conditions and glaucoma." },
  "CBN": { title: "CBN — Cannabinol", body: "CBN is a degradation product of THC — it forms when THC oxidises over time or with exposure to heat/light. High CBN signals age or poor storage.", therapeutic: "Mildly psychoactive. Studied for sedative, antibacterial, and appetite-stimulating properties. Its presence (or absence) is a key freshness and storage quality indicator." },
  "CBD": { title: "CBD — Cannabidiol", body: "The second most researched cannabinoid. Non-psychoactive. Often associated with modulating or balancing the intensity of THC effects when both are present.", therapeutic: "Extensive research into anxiolytic, anticonvulsant, anti-inflammatory, and neuroprotective properties. The FDA-approved drug Epidiolex is CBD-based. CBD absence means no inherent THC modulation." },
};

function getTerpeneIntel(terpName = "") {
  const key = String(terpName).toLowerCase().trim();
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

  const freshness = terps >= 3 ? { label: "Strong", note: "Terpene retention is high — consistent with excellent post-harvest preservation." }
    : terps >= 2 ? { label: "Good", note: "Terpene retention suggests well-preserved aromatic content through handling and storage." }
    : terps >= 1.2 ? { label: "Moderate", note: "Some aromatic expression remains, though terpene density is not exceptional." }
    : terps > 0 ? { label: "Light", note: "Lower terpene density may suggest some aromatic loss through processing or storage." }
    : { label: "Unknown", note: "Freshness cannot be inferred — terpene density not captured." };

  const curing = terpCount >= 6 && terps >= 1.5 ? { label: "Positive signal", note: "Broad terpene spectrum without flat profile suggests preserved aromatic complexity through curing." }
    : terpCount >= 3 ? { label: "Moderate signal", note: "Some terpene layering visible. Confidence in curing quality is limited without additional data." }
    : { label: "Limited signal", note: "Insufficient terpene breadth to draw a confident curing quality inference." };

  const degradation = cbn >= 1.5 ? { label: "Notable", note: `CBN at ${cbn.toFixed(2)} wt% suggests meaningful THC oxidative degradation. Review storage conditions.` }
    : cbn >= 0.5 ? { label: "Low signal", note: `CBN detected at ${cbn.toFixed(2)} wt% — minor degradation signal. Not clinically significant at this level.` }
    : { label: "Minimal", note: "CBN not detected — minimal evidence of THC oxidative degradation." };

  const moisture = extracted.moisture_content || contaminants.moisture_content;
  const stability = moisture ? { label: "Data present", note: `Moisture content: ${moisture}. ${parseFloat(moisture) <= 12 ? "Within acceptable range for storage stability." : "Review against target range for product category."}` }
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

function prepareOCRText(cleanText = "") {
  const text = String(cleanText || "").trim();
  if (!text) return "";
  if (text.length <= MAX_OCR_CHARS) return text;
  const headLen = Math.floor(MAX_OCR_CHARS * 0.7);
  const tailLen = MAX_OCR_CHARS - headLen;
  return [text.slice(0, headLen), "\n\n[TRUNCATED]\n\n", text.slice(-tailLen)].join("");
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
  const topTerpenes = a(data.top_terpenes).slice(0, 12).map(i => ({ name: s(i?.name), value: s(i?.value), unit: s(i?.unit) || "wt%" }));
  let totalTerpenes = s(data.total_terpenes);
  if (!totalTerpenes || totalTerpenes === "ND" || totalTerpenes === "") {
    const computed = topTerpenes.reduce((sum, t) => { const n = parseFloat(t.value); return sum + (isNaN(n) ? 0 : n); }, 0);
    if (computed > 0) { totalTerpenes = computed.toFixed(4); console.log(`⚠️  total_terpenes missing — computed from terpene sum: ${totalTerpenes}`); }
  }
  const fixUnit = (u) => { const clean = s(u).trim(); if (!clean || clean === "%") return "wt%"; return clean; };
  const normCannName = (name) => { const n = s(name).trim(); if (/^thca-a$/i.test(n)) return "THCA"; if (/^cbda-a$/i.test(n)) return "CBDA"; if (/^d9-?thc$/i.test(n)) return "D9-THC"; if (/^d8-?thc$/i.test(n)) return "D8-THC"; return n; };
  return {
    product_name: s(data.product_name), batch_number: s(data.batch_number), coa_report_date: s(data.coa_report_date),
    product_type: s(data.product_type), laboratory_name: s(data.laboratory_name), laboratory_accreditation: s(data.laboratory_accreditation),
    laboratory_method: s(data.laboratory_method), thc_total: s(data.thc_total), thc_total_unit: fixUnit(data.thc_total_unit),
    cbd_total: s(data.cbd_total), cbd_total_unit: fixUnit(data.cbd_total_unit), thca: s(data.thca), thca_unit: fixUnit(data.thca_unit),
    cbda: s(data.cbda), cbn: s(data.cbn), cbg: s(data.cbg), cbga: s(data.cbga), cbca: s(data.cbca), thcva: s(data.thcva),
    total_terpenes: totalTerpenes, total_terpenes_unit: fixUnit(data.total_terpenes_unit) || "wt%",
    total_cannabinoids: s(data.total_cannabinoids), hero_narrative: s(data.hero_narrative),
    top_cannabinoids: a(data.top_cannabinoids).slice(0, 10).map(i => ({ name: normCannName(i?.name), value: s(i?.value), unit: fixUnit(i?.unit), notes: s(i?.notes) })),
    top_terpenes: topTerpenes,
  };
}

function normalizeContaminants(data = {}) {
  const s = (v) => (v === null || v === undefined ? "" : String(v));
  const a = (v) => (Array.isArray(v) ? v : []);
  return {
    pesticides_status: s(data.pesticides_status), pesticides_detail: s(data.pesticides_detail),
    heavy_metals_status: s(data.heavy_metals_status), heavy_metals_detail: s(data.heavy_metals_detail),
    microbials_status: s(data.microbials_status), microbials_detail: s(data.microbials_detail),
    mycotoxins_status: s(data.mycotoxins_status), residual_solvents_status: s(data.residual_solvents_status),
    moisture_content: s(data.moisture_content), water_activity: s(data.water_activity),
    foreign_matter_status: s(data.foreign_matter_status), contaminant_narrative: s(data.contaminant_narrative),
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
  const [chemistry, contaminants] = await Promise.all([extractChemistry(ocrText), extractContaminants(ocrText)]);
  return { chemistry, contaminants };
}

// ─────────────────────────────────────────────
// AZURE OCR
// ─────────────────────────────────────────────

async function extractDocumentFromUrl(fileUrl) {
  if (!fileUrl || typeof fileUrl !== "string") throw new Error("fileUrl must be a non-empty string");
  const fileResponse = await axios.get(fileUrl, { responseType: "arraybuffer", timeout: 60000, maxRedirects: 5, validateStatus: (status) => status >= 200 && status < 300 });
  const mimeType = detectMimeType(fileUrl, fileResponse.headers["content-type"]);
  const poller = await azureClient.beginAnalyzeDocument("prebuilt-layout", fileResponse.data, { contentType: mimeType });
  const result = await poller.pollUntilDone();
  const pages = (result.pages || []).map(page => ({ page_number: page.pageNumber, text: (page.lines || []).map(line => line.content).join("\n") }));
  return { mimeType, plain_text: pages.map(p => p.text).join("\n\n").trim(), page_count: pages.length, table_count: (result.tables || []).length };
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
    report_json: { chemistry, contaminants, scoring, intelligence, _meta: { source_url: sourceUrl || "", storage_path: storagePath || "", original_filename: originalFilename || "", mime_type: mimeType || "", saved_at: new Date().toISOString(), schema_version: "5.0" } },
    overall_score: scoring.total, report_confidence_score: scoring.breakdown.dataCompleteness.score,
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
// LAYER 5 — REPORT HTML RENDERER (v6 — Full Alem Design)
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
  const cbga = toNum(chemistry.cbga);
  const cbn = toNum(chemistry.cbn);
  const leadTerpene = terpenes[0]?.name || "";
  const secondTerpene = terpenes[1]?.name || "";
  const thirdTerpene = terpenes[2]?.name || "";
  const terpIntel = getTerpeneIntel(leadTerpene);
  const fingerprintId = intelligence.fingerprintId || generateFingerprintId(terpenes);
  const audiences = intelligence.audiences || buildAudienceNarratives(chemistry, contaminants, scoring);
  const postHarvest = intelligence.postHarvest || buildPostHarvestIntel(chemistry, contaminants);

  const maxTerpVal = Math.max(...terpenes.map(t => toNum(t.value)), 0.001);
  const productName = chemistry.product_name || "COA Report";
  const heroNarrative = chemistry.hero_narrative || `${leadTerpene ? leadTerpene + "-dominant" : "Cannabis"} profile with ${terps > 2 ? "strong" : terps > 1 ? "moderate" : "light"} terpene expression.`;

  // Score ring: circumference = 2π × 56 = 351.86, offset = (1 - score/100) × 351.86
  const ringCirc = 351.86;
  const ringOffset = ((100 - scoring.total) / 100 * ringCirc).toFixed(1);

  // Helper: info icon with tooltip
  function infoIcon(content, variant = "") {
    return `<span class="info-icon${variant ? " " + variant : ""}" tabindex="0">ⓘ${content}</span>`;
  }

  function tooltip(title, body, extra = "") {
    return `<span class="tooltip"><strong>${esc(title)}</strong><span class="tt-body">${esc(body)}</span>${extra}</span>`;
  }

  function ttExtra(label, text) {
    return `<span class="tt-coa"><strong>${esc(label)}</strong> ${esc(text)}</span>`;
  }

  // Score breakdown rows with info icons
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

  // Terpene bar with info icon
  function terpBar(t, i) {
    const val = toNum(t.value);
    const width = Math.max(4, (val / maxTerpVal) * 100);
    const isLead = i === 0;
    const edu = TERPENE_EDUCATION[t.name] || null;
    const icon = edu
      ? infoIcon(`<span class="tooltip"><strong>${esc(t.name)}</strong><span class="tt-body">${esc("Aroma: " + edu.aroma)}</span><span class="tt-coa">${esc(edu.therapeutic)}</span></span>`, "tt-right row-info")
      : "";
    return `
    <div class="tb">
      <div class="tb-row">
        <span class="tb-name${isLead ? " lead" : ""}">${esc(t.name)}${icon}</span>
        <span class="tb-pct">${esc(t.value)} ${esc(t.unit || "wt%")}</span>
      </div>
      <div class="tb-track"><div class="tb-fill${isLead ? " lead" : ""}" style="width:${width.toFixed(1)}%;animation-delay:${(i * 0.04 + 0.04).toFixed(2)}s"></div></div>
    </div>`;
  }

  // Cannabinoid cell with info icon
  function cannCell(name, value, unit, note, isHighlight = false) {
    const edu = CANNABINOID_EDUCATION[name] || null;
    const icon = edu
      ? infoIcon(`<span class="tooltip"><strong>${esc(edu.title)}</strong><span class="tt-body">${esc(edu.body)}</span><span class="tt-coa">${esc(edu.therapeutic)}</span></span>`, "cann-info")
      : "";
    return `
    <div class="cann-cell${isHighlight ? " hl" : ""}">
      ${icon}
      <div class="cann-lbl">${esc(name)}</div>
      <div class="cann-val">${esc(value)}<span class="cann-unit"> ${esc(unit || "wt%")}</span></div>
      ${note ? `<div class="cann-note">${esc(note)}</div>` : ""}
    </div>`;
  }

  // Post-harvest row with info icon
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

  // Safety dot row
  function sfRow(label, statusClass, statusText, pass) {
    return `
    <div class="sf-row">
      <div class="sf-dot ${pass ? "pass" : "nt"}"></div>
      <div class="sf-name">${esc(label)}</div>
      <div class="sf-val${pass ? " pass" : ""}">${esc(statusText || (pass ? "✓ Certified" : "not tested"))}</div>
    </div>`;
  }

  // Audience panel
  function audiencePanel(items = []) {
    if (!items.length) return `<div class="ins-row"><div class="ins-arr"></div><span class="ins-text">No intelligence available.</span></div>`;
    return items.map(line => `<div class="ins-row"><div class="ins-arr"></div><span class="ins-text">${esc(line)}</span></div>`).join("");
  }

  const pestPass = /pass|nd|not detected/i.test(contaminants.pesticides_status || "");
  const metalsPass = /pass|nd|not detected/i.test(contaminants.heavy_metals_status || "");
  const microPass = /pass|nd|not detected/i.test(contaminants.microbials_status || "");
  const isoPass = /17025|iso/i.test(contaminants.lab_quality_summary || chemistry.laboratory_accreditation || "");
  const sccPass = /scc/i.test(contaminants.positive_flags?.join(" ") || "");

  // Score breakdown thresholds
  const potencyPct = Math.round(scoring.breakdown.potency.score / scoring.breakdown.potency.max * 100);
  const terpPct = Math.round(scoring.breakdown.terpenes.score / scoring.breakdown.terpenes.max * 100);
  const minorPct = Math.round(scoring.breakdown.minors.score / scoring.breakdown.minors.max * 100);
  const safetyPct = Math.round(scoring.breakdown.safety.score / scoring.breakdown.safety.max * 100);
  const dataPct = Math.round(scoring.breakdown.dataCompleteness.score / scoring.breakdown.dataCompleteness.max * 100);

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
  --sh:          rgba(13,45,62,.09);
}
body { background:var(--alem-wash); font-family:'Nunito',sans-serif; font-size:13.5px; line-height:1.6; color:var(--t-body); min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:48px 16px 72px; }
@keyframes rise { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes grow { from{transform:scaleX(0)} to{transform:scaleX(1)} }

/* ── CARD ── */
.rcard { width:100%; max-width:580px; background:var(--white); border:1px solid var(--border); box-shadow:0 2px 20px var(--sh),0 8px 40px var(--sh); animation:rise .55s cubic-bezier(.16,1,.3,1) both; }

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
.hero-narrative { font-family:'Nunito',sans-serif; font-style:italic; font-size:14.5px; font-weight:300; line-height:1.65; color:var(--t-mid); margin-top:14px; max-width:310px; }

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
.tooltip { display:none; position:absolute; left:22px; top:50%; transform:translateY(-50%); width:280px; background:var(--white); border:1px solid var(--border); border-radius:6px; padding:14px 16px; box-shadow:0 8px 32px rgba(13,45,62,.14),0 2px 8px rgba(13,45,62,.08); z-index:999; pointer-events:none; font-size:11px; font-weight:400; color:var(--t-body); line-height:1.6; font-family:'Nunito',sans-serif; text-transform:none; letter-spacing:0; }
.tooltip strong { display:block; font-size:11.5px; font-weight:800; color:var(--alem-dark); margin-bottom:7px; letter-spacing:0; }
.tooltip .tt-body { display:block; margin-bottom:9px; color:var(--t-body); }
.tooltip .tt-how,.tooltip .tt-coa { display:block; font-size:10.5px; color:var(--t-mid); margin-top:7px; padding-top:7px; border-top:1px solid var(--border-l); line-height:1.55; }
.tooltip .tt-how strong,.tooltip .tt-coa strong { display:inline; font-size:10.5px; font-weight:700; color:var(--alem-dark); margin-bottom:0; }
.tooltip::before { content:''; position:absolute; left:-6px; top:50%; transform:translateY(-50%); width:0; height:0; border-top:6px solid transparent; border-bottom:6px solid transparent; border-right:6px solid var(--border); }
.tooltip::after { content:''; position:absolute; left:-5px; top:50%; transform:translateY(-50%); width:0; height:0; border-top:5px solid transparent; border-bottom:5px solid transparent; border-right:5px solid var(--white); }
.info-icon:hover .tooltip,.info-icon:focus .tooltip { display:block; }
/* Tooltip position variants */
.tt-right .tooltip { left:22px; right:auto; top:50%; transform:translateY(-50%); }
.tt-right .tooltip::before { left:-6px; right:auto; border-left:none; border-right:6px solid var(--border); border-top:6px solid transparent; border-bottom:6px solid transparent; top:50%; transform:translateY(-50%); }
.tt-right .tooltip::after { left:-5px; right:auto; border-left:none; border-right:5px solid var(--white); border-top:5px solid transparent; border-bottom:5px solid transparent; top:50%; transform:translateY(-50%); }
.sec-info { vertical-align:middle; margin-left:6px; }
.row-info { margin-left:4px; flex-shrink:0; }
.cann-cell { position:relative; }
.cann-info { position:absolute; top:8px; right:8px; }
.cann-info .tooltip { width:260px; right:22px; left:auto; top:0; transform:none; }
.cann-info .tooltip::before { right:-6px; left:auto; border-right:none; border-left:6px solid var(--border); border-top:6px solid transparent; border-bottom:6px solid transparent; top:12px; transform:none; }
.cann-info .tooltip::after { right:-5px; left:auto; border-right:none; border-left:5px solid var(--white); border-top:5px solid transparent; border-bottom:5px solid transparent; top:13px; transform:none; }
.ph-info .tooltip { width:260px; left:22px; top:50%; transform:translateY(-50%); right:auto; }
.ph-info .tooltip::before { left:-6px; border-right:6px solid var(--border); border-left:none; border-top:6px solid transparent; border-bottom:6px solid transparent; top:50%; transform:translateY(-50%); right:auto; }
.ph-info .tooltip::after { left:-5px; border-right:5px solid var(--white); border-left:none; border-top:5px solid transparent; border-bottom:5px solid transparent; top:50%; transform:translateY(-50%); right:auto; }
/* m-info (metrics) — tooltip opens downward */
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
.summary-body strong { font-weight:600; color:var(--alem-dark); }

/* ── SECTIONS ── */
.sec { padding:24px 28px; border-bottom:1px solid var(--border-l); }
.sec-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:18px; }
.sec-title { font-size:8px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--t-light); }
.sec-badge { font-family:'Space Mono',monospace; font-size:8px; color:var(--alem-mid); background:var(--alem-tint); padding:2px 10px; border-radius:20px; }

/* ── TERPENES ── */
.terp-intro { font-size:12px; font-weight:400; color:var(--t-mid); background:var(--off); border:1px solid var(--border-l); padding:10px 14px; margin-bottom:16px; line-height:1.55; }
.terp-intro strong { font-weight:700; color:var(--alem-dark); }
.terp-wrap { display:flex; gap:20px; align-items:flex-start; }
.t-bars { flex:1; display:flex; flex-direction:column; gap:9px; }
.tb-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.tb-name { font-size:9px; font-weight:400; color:var(--t-body); display:flex; align-items:center; gap:4px; }
.tb-name.lead { font-weight:700; color:var(--alem-dark); }
.tb-pct { font-family:'Space Mono',monospace; font-size:9px; color:var(--alem-mid); }
.tb-track { height:3px; background:var(--border-l); border-radius:2px; overflow:hidden; }
.tb-fill { height:100%; background:var(--alem-mid); border-radius:2px; transform-origin:left; animation:grow .8s cubic-bezier(.16,1,.3,1) both; opacity:.75; }
.tb-fill.lead { background:var(--alem-dark); opacity:1; }

/* ── CANNABINOIDS ── */
.cann-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.cann-cell { padding:12px 13px; background:var(--off); border:1px solid var(--border-l); position:relative; }
.cann-cell.hl { background:var(--gold-tint); border-color:var(--gold-bord); }
.cann-lbl { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:1px; color:var(--t-mid); margin-bottom:5px; }
.cann-cell.hl .cann-lbl { color:var(--gold); }
.cann-val { font-family:'Nunito',sans-serif; font-size:20px; font-weight:800; color:var(--alem-dark); line-height:1; }
.cann-cell.hl .cann-val { color:var(--gold); }
.cann-unit { font-family:'Nunito',sans-serif; font-size:9px; color:var(--t-light); }
.cann-note { font-size:8px; font-weight:300; color:var(--t-light); margin-top:5px; line-height:1.4; }
.cann-cell.hl .cann-note { color:rgba(154,120,34,.7); }
.cann-total { margin-top:10px; padding:11px 16px; background:var(--alem-tint); border:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
.ct-lbl2 { font-size:8px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--t-mid); }
.ct-val2 { font-family:'Nunito',sans-serif; font-size:20px; font-weight:800; color:var(--alem-dark); }

/* ── ENTOURAGE EFFECT BANNER ── */
.entourage { margin-top:14px; padding:11px 14px; background:linear-gradient(135deg,var(--alem-tint),#f0f8ff); border:1px solid var(--border); border-left:3px solid var(--alem-accent); display:flex; align-items:flex-start; gap:10px; }
.entourage-icon { font-size:16px; flex-shrink:0; margin-top:1px; }
.entourage-body { font-size:10.5px; color:var(--t-body); line-height:1.55; }
.entourage-body strong { font-weight:700; color:var(--alem-dark); }

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
.safety-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.sf-row { display:flex; align-items:center; gap:9px; padding:8px 11px; background:var(--off); border:1px solid var(--border-l); }
.sf-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.sf-dot.pass { background:var(--alem-accent); }
.sf-dot.nt { background:#ddc8b4; border:1px solid #c8a888; }
.sf-name { font-size:9px; font-weight:500; color:var(--t-body); }
.sf-val { margin-left:auto; font-family:'Space Mono',monospace; font-size:8px; color:var(--t-faint); font-style:italic; }
.sf-val.pass { color:var(--alem-accent); font-style:normal; font-weight:700; }
.safety-note { margin-top:12px; padding:12px 14px; background:var(--warn-bg); border-left:3px solid var(--warn-bord); font-size:11px; font-weight:400; color:var(--warn-text); line-height:1.6; }
.safety-note strong { font-weight:700; }

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
.ractions { width:100%; max-width:580px; margin-top:14px; display:flex; gap:10px; animation:rise .55s cubic-bezier(.16,1,.3,1) .2s both; }
.rbtn { flex:1; padding:14px 20px; font-family:'Nunito',sans-serif; font-size:8.5px; font-weight:700; letter-spacing:2px; text-transform:uppercase; cursor:pointer; transition:all .15s; border-radius:30px; text-decoration:none; display:inline-block; text-align:center; }
.rbtn-p { background:var(--alem-dark); border:1px solid var(--alem-dark); color:var(--white); }
.rbtn-p:hover { background:var(--alem-mid); }
.rbtn-s { background:var(--white); border:1px solid var(--border); color:var(--t-mid); }
.rbtn-s:hover { border-color:var(--alem-accent); color:var(--alem-dark); }
.page-foot { margin-top:20px; font-size:8px; font-weight:500; letter-spacing:2px; text-transform:uppercase; color:var(--t-faint); text-align:center; animation:rise .55s ease .35s both; }
.page-foot a { color:var(--alem-mid); text-decoration:none; }
</style>
</head>
<body>

<div class="rcard">

  <!-- NAV with logo -->
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
      <div class="nav-schema">Schema v5.0 · ${esc(chemistry.laboratory_accreditation || "COA Report")}</div>
    </div>
  </div>

  <!-- HERO -->
  <div class="hero">
    <div class="hero-top">
      <div class="hero-left">
        <div class="hero-eye">Certificate of Analysis · Intelligence Report</div>
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
            <div class="score-n">${scoring.total}</div>
            <div class="score-d">/100</div>
          </div>
        </div>
        <div class="score-grade">${esc(scoring.grade)} &nbsp; Grade</div>
        <div class="score-tier">${esc(scoring.tier)}</div>
      </div>
    </div>
  </div>

  <!-- SCORE BREAKDOWN -->
  <div class="score-bd">
    ${sbRow("Potency", potencyPct >= 75 ? "f-g" : potencyPct >= 45 ? "f-o" : "f-r", potencyPct, 0.06, `${scoring.breakdown.potency.score} / ${scoring.breakdown.potency.max}`,
      `Potency Score: ${scoring.breakdown.potency.score}/${scoring.breakdown.potency.max}`,
      `THC Total of ${chemistry.thc_total || "N/A"} wt% places this product in the ${thc >= 24 ? "high" : thc >= 18 ? "moderate-to-high" : "moderate"} range. THCA at ${chemistry.thca || "N/A"} wt% is the primary precursor — multiply by 0.877 to estimate post-decarboxylation THC.`,
      `Max potency score (25pts) requires THC total above 28 wt%. Score tiers: ≥28=25, ≥24=22, ≥20=18, ≥15=13, >0=8.`,
      `Look for "THC Total" or calculate it as (THCA × 0.877) + D9-THC.`)}

    ${sbRow("Terpenes", terpPct >= 75 ? "f-g" : terpPct >= 45 ? "f-o" : "f-r", terpPct, 0.12, `${scoring.breakdown.terpenes.score} / ${scoring.breakdown.terpenes.max}`,
      `Terpene Score: ${scoring.breakdown.terpenes.score}/${scoring.breakdown.terpenes.max}`,
      `Total terpene content of ${chemistry.total_terpenes || "N/A"} wt% with ${terpenes.filter(t => toNum(t.value) > 0).length} compounds detected. Scores improve with both density (wt%) and breadth (number of compounds).`,
      `Tiers: ≥3.5 wt%=22, ≥2.5=20, ≥1.8=16, ≥1.0=11. Breadth bonus: +5 for ≥8 compounds, +3 for ≥5 compounds.`,
      `Find the "Terpene Profile" section. Sum all individual values if no total is listed.`)}

    ${sbRow("Minor Cannabinoids", minorPct >= 75 ? "f-g" : minorPct >= 45 ? "f-o" : "f-r", minorPct, 0.18, `${scoring.breakdown.minors.score} / ${scoring.breakdown.minors.max}`,
      `Minor Cannabinoid Score: ${scoring.breakdown.minors.score}/${scoring.breakdown.minors.max}`,
      `${[{k:"CBGA",v:cbga},{k:"CBN",v:cbn},{k:"CBG",v:toNum(chemistry.cbg)},{k:"CBCA",v:toNum(chemistry.cbca)},{k:"THCVA",v:toNum(chemistry.thcva)}].filter(x=>x.v>0).map(x=>`${x.k}: ${x.v.toFixed(2)} wt%`).join(", ") || "No minor cannabinoids detected"}. Minor cannabinoids contribute to the entourage effect — the synergistic interaction between all cannabis compounds.`,
      `Score based on sum of minors (total - THC - CBD) and count of detected minors. CBN ≥1% applies a 4-point degradation penalty.`,
      `Look for "Cannabinoid Profile" table. Count all non-ND values beyond THC and CBD.`)}

    ${sbRow("Safety Data", safetyPct >= 75 ? "f-g" : safetyPct >= 45 ? "f-o" : "f-r", safetyPct, 0.24, `${scoring.breakdown.safety.score} / ${scoring.breakdown.safety.max}`,
      `Safety Score: ${scoring.breakdown.safety.score}/${scoring.breakdown.safety.max}`,
      `Pesticides: ${contaminants.pesticides_status || "Not tested"} · Heavy Metals: ${contaminants.heavy_metals_status || "Not tested"} · Microbials: ${contaminants.microbials_status || "Not tested"}. Missing panels reduce the maximum achievable safety score — this is not a failed test.`,
      `7pts for pesticide pass/ND, 7pts for heavy metals pass/ND, 6pts for microbials pass/ND. +3 bonus for ISO 17025 accreditation.`,
      `Look for "Pesticide Analysis", "Heavy Metals", and "Microbiological" sections. If absent, request the full compliance package from the producer.`)}

    ${sbRow("Data Completeness", dataPct >= 75 ? "f-g" : dataPct >= 45 ? "f-o" : "f-r", dataPct, 0.30, `${scoring.breakdown.dataCompleteness.score} / ${scoring.breakdown.dataCompleteness.max}`,
      `Data Completeness: ${scoring.breakdown.dataCompleteness.score}/${scoring.breakdown.dataCompleteness.max}`,
      `We check for 10 mandatory fields: THC total (2pts), total terpenes (2pts), ≥3 terpenes reported (2pts), ≥3 cannabinoids (1pt), report date (1pt), laboratory name (1pt), batch number (1pt).`,
      `A complete submission scores all 10 points. Missing fields reduce completeness and signal a less thorough lab submission.`,
      `A complete COA should include: lab name, accreditation, batch ID, report date, product type, cannabinoid table, terpene table, and method references.`)}
  </div>

  <!-- CHEMOTYPE BAND -->
  <div class="ct-band">
    <div class="ct-code">
      ${esc(fingerprintId)}
      ${infoIcon(`<span class="tooltip"><strong>Chemotype Fingerprint: ${esc(fingerprintId)}</strong><span class="tt-body">The top-3 dominant terpenes in order of concentration. This fingerprint clusters products by chemical identity across our database, enabling market comparisons.</span><span class="tt-coa"><strong>Derived from:</strong> We rank all reported terpenes by wt% and take the top 3. Each terpene is abbreviated (TPN = Terpinolene, MYR = β-Myrcene, OCI = Ocimene, etc.).</span></span>`, "tt-right")}
    </div>
    <div class="ct-sep"></div>
    <div>
      <div class="ct-lbl">Chemotype fingerprint</div>
      <div class="ct-lin">${esc(terpIntel.lineage)}</div>
    </div>
    <div class="ct-right">
      <div class="ct-effect">
        ☀ &nbsp;${esc(terpIntel.direction)}
        ${infoIcon(`<span class="tooltip"><strong>Effect Direction: ${esc(terpIntel.direction)}</strong><span class="tt-body">Effect direction is inferred from the terpene chemotype using observational data — not from clinical trials on this specific batch. Individual responses vary significantly based on tolerance, dose, method of consumption, and personal physiology.</span><span class="tt-coa"><strong>Important:</strong> This is a population-level directional signal. Always consult a healthcare professional for personalised guidance.</span></span>`, "tt-right")}
      </div>
      <div class="ct-conf">
        ${esc(terpIntel.lineageConfidence)} confidence
        ${infoIcon(`<span class="tooltip"><strong>Lineage Confidence: ${esc(terpIntel.lineageConfidence)}</strong><span class="tt-body">Confidence reflects how closely this chemotype cluster matches known genetic lineages in our reference database. It improves with multiple batches from the same cultivar and broader database coverage.</span><span class="tt-coa"><strong>How to improve it:</strong> Submit multiple COA batches from the same cultivar. Consistent chemotypes across batches increase lineage confidence.</span></span>`, "tt-right")}
      </div>
    </div>
  </div>

  <!-- KEY METRICS -->
  <div class="metrics">
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.thc_total || "ND")}<span class="m-unit">${chemistry.thc_total && chemistry.thc_total !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">THC Total</div>
      <div class="m-ctx">${thc >= 24 ? "↑ above avg." : thc >= 18 ? "moderate-high" : thc > 0 ? "moderate" : "not detected"}</div>
      ${infoIcon(`<span class="tooltip"><strong>THC Total: ${esc(chemistry.thc_total || "ND")} wt%</strong><span class="tt-body">Market average for premium dried flower is 18–22 wt%. At ${esc(chemistry.thc_total || "ND")}%, this product ${thc >= 24 ? "sits above average — suitable for experienced consumers." : "is in a typical therapeutic range."}</span><span class="tt-coa"><strong>Check on your COA:</strong> THC Total = (THCA × 0.877) + D9-THC. Some labs report it directly; others require manual calculation.</span></span>`, "m-info")}
    </div>
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.thca || "ND")}<span class="m-unit">${chemistry.thca && chemistry.thca !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">THCA</div>
      <div class="m-ctx">×0.877 on heat</div>
      ${infoIcon(`<span class="tooltip"><strong>THCA: ${esc(chemistry.thca || "ND")} wt%</strong><span class="tt-body">THCA is the raw, non-psychoactive acid form of THC found in fresh cannabis. When heated (smoked, vaporised, or cooked), it decarboxylates into active THC. Multiply THCA × 0.877 to estimate active THC produced.</span><span class="tt-coa"><strong>Check on your COA:</strong> Listed as "THCA" or "THCA-A". Typically the largest value in the cannabinoid profile for potent flower.</span></span>`, "m-info")}
    </div>
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.total_terpenes || "ND")}<span class="m-unit">${chemistry.total_terpenes && chemistry.total_terpenes !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">Terpenes</div>
      <div class="m-ctx">${terps >= 3 ? "strong" : terps >= 2 ? "top 20%" : terps > 0 ? "moderate" : "not reported"}</div>
      ${infoIcon(`<span class="tooltip"><strong>Total Terpenes: ${esc(chemistry.total_terpenes || "ND")} wt%</strong><span class="tt-body">Terpenes drive aroma, flavour, and may influence therapeutic outcomes via the entourage effect. A total above 2.0 wt% is considered aromatic and complex. Above 3.0 wt% is exceptional.</span><span class="tt-coa"><strong>Check on your COA:</strong> Look for "Total Terpenes" or sum all individual terpene values. Not all labs report a total — you may need to calculate manually.</span></span>`, "m-info")}
    </div>
    <div class="m-cell">
      <div class="m-val">${esc(chemistry.cbd_total || "ND")}<span class="m-unit">${chemistry.cbd_total && chemistry.cbd_total !== "ND" ? "%" : ""}</span></div>
      <div class="m-lbl">CBD Total</div>
      <div class="m-ctx ${cbd < 0.5 ? "nd" : ""}">${cbd >= 0.5 ? "modulating" : "not detected"}</div>
      ${infoIcon(`<span class="tooltip"><strong>CBD Total: ${esc(chemistry.cbd_total || "ND")} wt%</strong><span class="tt-body">CBD may modulate the intensity of THC effects when both are present. In this product, CBD is ${cbd >= 0.5 ? `present at ${chemistry.cbd_total} wt% — providing some inherent THC modulation.` : "absent — meaning no built-in CBD buffer. The full THC effect is unmodified."}</span><span class="tt-coa"><strong>Check on your COA:</strong> CBD Total = (CBDA × 0.877) + CBD. If both read ND, CBD is absent from this product.</span></span>`, "m-info")}
    </div>
  </div>

  <!-- SUMMARY -->
  <div class="summary">
    <div class="summary-lbl">what this means</div>
    <div class="summary-body">
      ${esc(heroNarrative)}
    </div>
  </div>

  <!-- TERPENE FINGERPRINT -->
  <div class="sec">
    <div class="sec-head">
      <div class="sec-title">terpene fingerprint ${infoIcon(`<span class="tooltip"><strong>What are Terpenes?</strong><span class="tt-body">Terpenes are aromatic compounds produced by the cannabis plant. They shape scent, flavour, and may modulate the therapeutic effects of cannabinoids via the Entourage Effect. Over 200 terpenes have been identified in cannabis.</span><span class="tt-coa"><strong>Entourage Effect:</strong> Research suggests terpenes and cannabinoids may work synergistically — the combination produces different outcomes than either compound alone. A rich terpene profile is a marker of full-spectrum complexity.</span></span>`, "sec-info")}</div>
      <div class="sec-badge">${esc(chemistry.total_terpenes || "—")} wt% · ${terpenes.filter(t => toNum(t.value) > 0).length} compounds</div>
    </div>
    ${leadTerpene ? `<div class="terp-intro"><strong>${esc(leadTerpene)}-dominant.</strong> ${esc(TERPENE_EDUCATION[leadTerpene]?.aroma || terpIntel.note)} ${leadTerpene === "Terpinolene" ? "Fewer than 15% of dried flower products lead with this compound — a genuine shelf differentiator in a Myrcene-heavy market." : ""}</div>` : ""}
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
        ${terpenes.slice(0, 6).length >= 3 ? (() => {
          const pts = terpenes.slice(0, 6).map((t, i) => {
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
        <text x="75"  y="7"   text-anchor="middle" fill="var(--t-light)" font-size="8" font-family="Space Mono,monospace">${esc(terpenes[0]?.name?.slice(0,3).toUpperCase() || "T1")}</text>
        <text x="138" y="47"  text-anchor="start"  fill="var(--t-light)" font-size="8" font-family="Space Mono,monospace">${esc(terpenes[1]?.name?.slice(0,3).toUpperCase() || "T2")}</text>
        <text x="138" y="110" text-anchor="start"  fill="var(--t-light)" font-size="8" font-family="Space Mono,monospace">${esc(terpenes[2]?.name?.slice(0,3).toUpperCase() || "T3")}</text>
        <text x="75"  y="148" text-anchor="middle" fill="var(--t-light)" font-size="8" font-family="Space Mono,monospace">${esc(terpenes[3]?.name?.slice(0,3).toUpperCase() || "T4")}</text>
        <text x="12"  y="110" text-anchor="end"    fill="var(--t-light)" font-size="8" font-family="Space Mono,monospace">${esc(terpenes[4]?.name?.slice(0,3).toUpperCase() || "T5")}</text>
        <text x="12"  y="47"  text-anchor="end"    fill="var(--t-light)" font-size="8" font-family="Space Mono,monospace">${esc(terpenes[5]?.name?.slice(0,3).toUpperCase() || "T6")}</text>
      </svg>
      <div class="t-bars">
        ${terpenes.length ? terpenes.map((t, i) => terpBar(t, i)).join("") : "<div style='color:var(--t-faint);font-size:11px;'>No terpene data reported.</div>"}
      </div>
    </div>
  </div>

  <!-- CANNABINOID PROFILE -->
  <div class="sec">
    <div class="sec-head">
      <div class="sec-title">cannabinoid profile ${infoIcon(`<span class="tooltip"><strong>The Entourage Effect</strong><span class="tt-body">The "entourage effect" is the hypothesis that cannabinoids and terpenes work synergistically — producing different therapeutic outcomes together than in isolation. THC alone differs from THC alongside CBC, CBGA, CBG, and a rich terpene profile.</span><span class="tt-coa"><strong>Clinical implication:</strong> A product with multiple detected minor cannabinoids is thought to produce more nuanced effects. The breadth of minors here (${cannabinoids.filter(c => !["THCA","D9-THC","CBD","CBDA"].includes(c.name)).length} detected) is a positive signal for entourage complexity.</span></span>`, "sec-info")}</div>
      <div class="sec-badge">${esc(chemistry.total_cannabinoids || "—")} wt% total</div>
    </div>
    <div class="cann-grid">
      ${cannabinoids.length ? cannabinoids.map(c => {
        const isHL = c.name === "CBGA" || (toNum(c.value) > 0 && ["CBGA"].includes(c.name));
        const noteMap = { "THCA": "Precursor · ×0.877 on heat", "D9-THC": "Active form", "CBGA": "Notable · biosynthetic depth", "CBN": "Degradation marker" };
        return cannCell(c.name, c.value, c.unit, c.notes || noteMap[c.name] || "", isHL);
      }).join("") : "<div style='color:var(--t-faint);'>No cannabinoid data reported.</div>"}
    </div>
    ${chemistry.total_cannabinoids ? `<div class="cann-total"><span class="ct-lbl2">Total Cannabinoid Sum</span><span class="ct-val2">${esc(chemistry.total_cannabinoids)} wt%</span></div>` : ""}
    <!-- Entourage Effect Banner -->
    <div class="entourage">
      <span class="entourage-icon">⬡</span>
      <div class="entourage-body">
        <strong>Entourage Effect signal: ${cannabinoids.filter(c => toNum(c.value) > 0).length >= 5 ? "Strong" : cannabinoids.filter(c => toNum(c.value) > 0).length >= 3 ? "Moderate" : "Limited"}.</strong>
        With ${cannabinoids.filter(c => toNum(c.value) > 0).length} quantified cannabinoids and ${terpenes.filter(t => toNum(t.value) > 0).length} terpenes, this profile has the biochemical complexity associated with full-spectrum, synergistic activity. THC does not act alone here.
        ${infoIcon(`<span class="tooltip"><strong>Why Entourage Matters</strong><span class="tt-body">The entourage effect (Russo, 2011) describes how whole-plant cannabis extracts produce effects greater than the sum of their individual parts. THC embedded in a complex cannabinoid-terpene matrix behaves differently to THC in isolation.</span><span class="tt-coa"><strong>Clinical implication:</strong> Patients and clinicians should consider the full profile, not just THC%. A product with rich minor cannabinoids and terpenes will likely behave differently to a THC-dominant, terpene-poor product of the same potency.</span></span>`, "tt-right")}
      </div>
    </div>
  </div>

  <!-- EFFECT + POST HARVEST -->
  <div class="two-col">
    <div class="sec" style="border-bottom:1px solid var(--border-l)">
      <div class="sec-title" style="margin-bottom:14px">effect direction ${infoIcon(`<span class="tooltip"><strong>How Effect Direction is Inferred</strong><span class="tt-body">Direction is derived from the chemotype fingerprint using observational data correlating terpene dominance to user-reported experiences across thousands of products. Not based on clinical trials.</span><span class="tt-coa"><strong>Limitations:</strong> Individual responses vary widely. Tolerance, consumption method, dose, physiology, and set & setting all influence the experience. This is directional, not predictive.</span></span>`, "sec-info")}</div>
      <div class="eff-pill"><span class="eff-icon">☀</span><span class="eff-dir">${esc(terpIntel.direction)}</span></div>
      <div class="eff-body">
        <strong>Lead:</strong> ${esc(leadTerpene || "Not identified")}<br>
        ${secondTerpene ? `<strong>Secondary:</strong> ${esc(secondTerpene)}${thirdTerpene ? " · " + esc(thirdTerpene) : ""}<br>` : ""}
        <br>${esc(terpIntel.note)}
      </div>
    </div>
    <div class="sec" style="border-bottom:1px solid var(--border-l)">
      <div class="sec-title" style="margin-bottom:14px">post-harvest</div>
      <div class="ph-list">
        ${phRow(terps >= 2 ? "good" : "mild", `Freshness — ${postHarvest.freshness.label}`, postHarvest.freshness.note,
          "Freshness Signal",
          "Inferred from total terpene content. Terpenes degrade with heat, light, and time. High terpene retention relative to batch type suggests good post-harvest handling.",
          "Compare terpene content against producer's historical batches. A significant drop batch-to-batch may indicate storage or handling issues.")}
        ${phRow(terpenes.length >= 6 ? "good" : "mild", `Curing — ${postHarvest.curing.label}`, postHarvest.curing.note,
          "Curing Quality Signal",
          "Curing is controlled post-harvest drying. A broad, intact terpene spectrum with volatile terpenes (Terpinolene, Ocimene) present suggests gentle, appropriate curing conditions.",
          "If only low-volatility terpenes survive (Caryophyllene, Bisabolol), it suggests aggressive drying. Volatile terpenes present = positive curing signal.")}
        ${phRow(cbn < 0.3 ? "good" : "mild", `Degradation — ${postHarvest.degradation.label}`, postHarvest.degradation.note,
          "THC Oxidative Degradation (CBN)",
          "CBN (Cannabinol) forms when THC oxidises over time or with exposure to heat/light. It is the primary chemical marker of cannabis age and storage quality.",
          "Find 'CBN' in the cannabinoid table. Values above 0.3 wt% may indicate significant age or improper storage. ND (not detected) is the ideal result.")}
        ${phRow(cbga >= 1 ? "good" : "mild", `CBGA Signal — ${cbga >= 1 ? "Notable" : cbga > 0 ? "Present" : "Not detected"}`, `${cbga >= 1 ? cbga.toFixed(2) + " wt% — elevated CBGA may indicate specific genetics or early harvest timing." : "CBGA not elevated in this profile."}`,
          "CBGA Signal (Biosynthetic Marker)",
          "CBGA is the precursor to all other cannabinoids. Residual CBGA after full maturation suggests either specific genetics that leave more unconverted CBGA, or harvest timing optimised for CBGA retention.",
          "Find 'CBGA' in the cannabinoid table. Values above 1.0 wt% are considered notable and signal genetic depth.")}
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

  <!-- QUALITY & SAFETY -->
  <div class="sec">
    <div class="sec-head">
      <div class="sec-title">quality &amp; safety signals ${infoIcon(`<span class="tooltip"><strong>What Safety Panels Mean</strong><span class="tt-body">A complete cannabis COA includes chemical profiling AND safety screening. The chemical profile tells you what IS in the product. Safety panels tell you what should NOT be there.</span><span class="tt-coa"><strong>Key panels to request:</strong> Pesticides (TGO93/TGO100), Heavy Metals (ICH Q3D), Microbials (USP 2021/2023), Mycotoxins (EC 1881/2006), Residual Solvents (USP 467). If absent from a COA, request them before prescribing or listing.</span></span>`, "sec-info")}</div>
    </div>
    <div class="safety-grid">
      ${sfRow("ISO 17025:2017", "pass", isoPass ? "✓ Certified" : "Not confirmed", isoPass)}
      ${sfRow("SCC Accredited", "pass", sccPass ? "✓ Pass" : "Not confirmed", sccPass)}
      ${sfRow("Pesticides", pestPass ? "pass" : "nt", pestPass ? "✓ " + (contaminants.pesticides_status || "Pass") : "not tested", pestPass)}
      ${sfRow("Heavy Metals", metalsPass ? "pass" : "nt", metalsPass ? "✓ " + (contaminants.heavy_metals_status || "Pass") : "not tested", metalsPass)}
      ${sfRow("Microbials", microPass ? "pass" : "nt", microPass ? "✓ " + (contaminants.microbials_status || "Pass") : "not tested", microPass)}
      ${sfRow("Mycotoxins", /pass|nd/i.test(contaminants.mycotoxins_status || "") ? "pass" : "nt", /pass|nd/i.test(contaminants.mycotoxins_status || "") ? "✓ Pass" : "not tested", /pass|nd/i.test(contaminants.mycotoxins_status || ""))}
      ${sfRow("Residual Solvents", /pass|nd/i.test(contaminants.residual_solvents_status || "") ? "pass" : "nt", /pass|nd/i.test(contaminants.residual_solvents_status || "") ? "✓ Pass" : "not tested", /pass|nd/i.test(contaminants.residual_solvents_status || ""))}
      ${sfRow("Moisture / Water Activity", !!(contaminants.moisture_content || contaminants.water_activity) ? "pass" : "nt", contaminants.moisture_content || contaminants.water_activity || "not tested", !!(contaminants.moisture_content || contaminants.water_activity))}
    </div>
    ${contaminants.contaminant_narrative || (!pestPass && !metalsPass) ? `<div class="safety-note"><strong>${pestPass && metalsPass ? "Safety Profile:" : "Incomplete safety profile."}</strong> ${esc(contaminants.contaminant_narrative || "Contaminant panels were not included in this COA submission. This is not a failed test — the panels were simply not submitted. Request the full compliance panel before clinical prescription or commercial listing.")}</div>` : ""}
  </div>

  <!-- LAB STRIP -->
  <div class="lab-strip">
    <div class="lab-cell">
      <div class="lab-val">${esc(chemistry.laboratory_name || "—")}</div>
      <div class="lab-lbl">Laboratory ${infoIcon(`<span class="tooltip"><strong>${esc(chemistry.laboratory_name || "Laboratory")}</strong><span class="tt-body">${esc(contaminants.lab_quality_summary || (chemistry.laboratory_accreditation ? `Accredited under ${chemistry.laboratory_accreditation}.` : "Laboratory accreditation details not captured."))} ISO 17025 is the international gold standard for testing laboratories — it verifies competence, impartiality, and consistent operation.</span><span class="tt-coa"><strong>Why it matters:</strong> Not all cannabis labs hold ISO 17025. Accreditation means results are traceable to international measurement standards. Always verify lab accreditation before relying on a COA.</span></span>`, "tt-right")}</div>
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
  <a class="rbtn rbtn-p" href="/" onclick="return false;" style="cursor:pointer;" onclick="window.location='/'">↗ &nbsp; Analyse New COA</a>
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
  return res.json({ success: true, message: "Alem v5 running", schema: "5.0" });
});

app.post("/upload-coa", upload.single("file"), async (req, res) => {
  try {
    console.log("📥 COA upload received");
    if (!req.file) return res.status(400).json({ success: false, error: "No file uploaded" });

    const originalFilename = req.file.originalname || `upload-${Date.now()}.pdf`;
    const mimeType = req.file.mimetype || "application/octet-stream";

    const { publicUrl, storagePath } = await uploadBufferToSupabase({ buffer: req.file.buffer, originalName: originalFilename, mimeType, folder: "raw_documents" });
    console.log("☁️  Stored:", storagePath);

    console.log("🔍 Azure OCR...");
    const extracted = await extractDocumentFromUrl(publicUrl);
    console.log(`✅ OCR: ${extracted.plain_text.length} chars, ${extracted.page_count} pages`);

    const { chemistry, contaminants } = await runDualPassExtraction(extracted.plain_text);
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

app.listen(PORT, () => {
  console.log(`🌿 Alem Chemical Intelligence v5 running on port ${PORT}`);
});