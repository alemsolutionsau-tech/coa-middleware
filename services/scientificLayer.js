"use strict";

/**
 * scientificLayer.js
 * Builds a structured scientificEvidence object from a processed COA.
 * Queries PubMed for peer-reviewed literature relevant to the COA's
 * terpene profile, cannabinoid ratios, effect archetype, and strain family.
 *
 * Total execution budget: 8 seconds (SCIENTIFIC_LAYER_TIMEOUT_MS).
 * Returns partial results if the budget is exceeded.
 * Never throws — all errors are caught and logged.
 */

const { searchAndFetch } = require("./pubmedClient");

const TIMEOUT_MS = Number(process.env.SCIENTIFIC_LAYER_TIMEOUT_MS || 8000);

// ── Terpene column → PubMed search term ──────────────────────────────────
const TERPENE_QUERIES = {
  beta_myrcene:        "beta-myrcene cannabis pharmacology",
  d_limonene:          "d-limonene cannabis therapeutic",
  beta_caryophyllene:  "beta-caryophyllene cannabinoid receptor CB2",
  alpha_pinene:        "alpha-pinene cannabis memory",
  linalool:            "linalool anxiolytic GABA receptor",
  terpinolene:         "terpinolene cannabis sativa effects",
  alpha_humulene:      "humulene anti-inflammatory terpene",
  alpha_bisabolol:     "bisabolol anti-inflammatory therapeutic",
  ocimene:             "ocimene terpene therapeutic",
  guaiol:              "guaiol cannabis terpene",
  nerolidol:           "nerolidol therapeutic antiparasitic",
  valencene:           "valencene terpene biological activity",
  geraniol:            "geraniol neuroprotective terpene",
  borneol:             "borneol CNS cannabis terpene",
  camphene:            "camphene cardiovascular terpene",
  fenchol:             "fenchol Alzheimer terpene neuroprotective",
};

// ── Terpene column → display label for relevance pill ────────────────────
const TERPENE_LABEL = {
  beta_myrcene:       "β-Myrcene",
  d_limonene:         "Limonene",
  beta_caryophyllene: "β-Caryophyllene",
  alpha_pinene:       "α-Pinene",
  linalool:           "Linalool",
  terpinolene:        "Terpinolene",
  alpha_humulene:     "α-Humulene",
  alpha_bisabolol:    "Bisabolol",
};

// ── Effect archetype → PubMed search term ────────────────────────────────
const ARCHETYPE_QUERIES = {
  "relaxing":          "cannabis sleep insomnia clinical study",
  "sedating":          "cannabis sleep insomnia clinical study",
  "myrcene":           "cannabis sleep insomnia clinical study",
  "uplifting":         "cannabis anxiety depression clinical trial",
  "limonene":          "cannabis anxiety depression clinical trial",
  "Uplifting / Energising": "cannabis anxiety depression clinical trial",
  "Relaxing / Sedating":    "cannabis sleep insomnia clinical study",
  "grounding":         "beta-caryophyllene CB2 receptor inflammation",
  "Grounding / Anti-inflam.": "beta-caryophyllene CB2 receptor inflammation",
  "anti-inflammatory": "beta-caryophyllene CB2 receptor inflammation",
  "cerebral":          "cannabis sativa cognitive effects clinical",
  "anxiolytic":        "linalool anxiety GABA clinical",
  "alerting":          "alpha-pinene acetylcholinesterase memory cognition",
};

// ── Helpers ───────────────────────────────────────────────────────────────

function toNum(v) { const n = parseFloat(v); return isNaN(n) ? 0 : n; }

/**
 * Extract the dominant terpene columns from chemistry.top_terpenes.
 * Returns array of { col, fraction } for terpenes with fraction > 0.10.
 */
function getDominantTerpenes(chemistry) {
  const topTerpenes = chemistry.top_terpenes || [];
  const totalTerps  = toNum(chemistry.total_terpenes) || 0.001;

  const NAME_TO_COL = {
    "beta-myrcene": "beta_myrcene", "β-myrcene": "beta_myrcene", "myrcene": "beta_myrcene",
    "(r)-(+)-limonene": "d_limonene", "limonene": "d_limonene", "d-limonene": "d_limonene",
    "trans-caryophyllene": "beta_caryophyllene", "beta-caryophyllene": "beta_caryophyllene",
    "caryophyllene": "beta_caryophyllene",
    "alpha-pinene": "alpha_pinene", "α-pinene": "alpha_pinene",
    "linalool": "linalool",
    "terpinolene": "terpinolene",
    "alpha-humulene": "alpha_humulene", "humulene": "alpha_humulene",
    "alpha-bisabolol": "alpha_bisabolol", "bisabolol": "alpha_bisabolol",
    "ocimene": "ocimene", "β-ocimene": "ocimene",
    "guaiol": "guaiol", "nerolidol": "nerolidol", "trans-nerolidol": "nerolidol",
    "valencene": "valencene", "geraniol": "geraniol", "borneol": "borneol",
    "camphene": "camphene", "fenchol": "fenchol",
  };

  const result = [];
  for (const t of topTerpenes) {
    const col = NAME_TO_COL[String(t.name || "").toLowerCase().trim()];
    const val = toNum(t.value);
    if (col && val > 0 && TERPENE_QUERIES[col]) {
      result.push({ col, val, fraction: val / totalTerps, label: t.name });
    }
  }
  return result.filter(t => t.fraction > 0.10).slice(0, 3);
}

/**
 * Deduplicate articles across sections by PMID.
 * Priority order (highest wins): safety > terpene > indication > cannabinoid > strain
 */
function deduplicateAcrossSections(evidence) {
  const seen    = new Map(); // pmid → sectionName
  const priority = ["safetyStudies", "terpeneStudies", "indicationStudies", "cannabinoidStudies", "strainFamilyStudies"];

  for (const section of priority) {
    const items = evidence[section];
    if (!items) continue;

    if (section === "terpeneStudies") {
      for (const group of items) {
        group.articles = group.articles.filter(a => {
          if (seen.has(a.pmid)) return false;
          seen.set(a.pmid, section);
          return true;
        });
      }
    } else {
      evidence[section] = items.filter(a => {
        if (seen.has(a.pmid)) return false;
        seen.set(a.pmid, section);
        return true;
      });
    }
  }
  return evidence;
}

/** Keep max 3 per section, sorted by year descending. */
function trimAndSort(articles) {
  return articles
    .sort((a, b) => (Number(b.year) || 0) - (Number(a.year) || 0))
    .slice(0, 3);
}

// ── Main export ───────────────────────────────────────────────────────────

async function buildScientificEvidence(coaData) {
  const start      = Date.now();
  const chemistry  = coaData.chemistry  || coaData;
  const intel      = coaData.intelligence || {};
  const strainIntel = intel.strainIntel  || null;

  const totalTerps = toNum(chemistry.total_terpenes);
  const thc        = toNum(chemistry.thc_total);
  const cbd        = toNum(chemistry.cbd_total);

  // Insufficient data guard
  if (totalTerps < 0.1) {
    return {
      terpeneStudies: [], cannabinoidStudies: [], strainFamilyStudies: [],
      indicationStudies: [], safetyStudies: [],
      queriesRun: 0, totalArticles: 0, executionMs: Date.now() - start,
      insufficient: true,
    };
  }

  // ── Build query plan ──────────────────────────────────────────────────
  const dominantTerps  = getDominantTerpenes(chemistry);
  const terpeneJobs    = [];
  const cannJobs       = [];
  const archetypeJobs  = [];
  const strainJobs     = [];
  const safetyJobs     = [];

  // Priority 1: Dominant terpenes (up to 3)
  for (const { col, label } of dominantTerps) {
    if (TERPENE_QUERIES[col]) {
      terpeneJobs.push({ col, label, query: TERPENE_QUERIES[col] });
    }
  }

  // Priority 2: Cannabinoid ratios
  if (thc > 0 && cbd > 0) {
    const ratio = cbd / thc;
    const q = ratio > 1    ? "CBD THC ratio anxiety clinical trial"
            : ratio > 0.1  ? "cannabidiol THC interaction entourage effect"
                           : "THC medicinal dose response";
    cannJobs.push({ query: q, hint: `CBD:THC ${(ratio).toFixed(2)}` });
  }
  cannJobs.push({ query: "cannabinoid entourage effect terpenes clinical", hint: "Entourage Effect" });

  // Priority 3: Effect archetype
  const archetype    = intel.effectDirection || intel.lineageCluster || "";
  const archetypeQ   = ARCHETYPE_QUERIES[archetype] || ARCHETYPE_QUERIES[archetype.toLowerCase()] || null;
  if (archetypeQ) archetypeJobs.push({ query: archetypeQ, hint: archetype });

  // Priority 4: Strain family (if match confidence warrants it)
  const strainMatch  = strainIntel?.match;
  if (strainMatch?.strain_name && strainMatch.similarity >= 70) {
    strainJobs.push({
      query: `${strainMatch.strain_name} cannabis terpene pharmacology`,
      hint:  strainMatch.strain_name,
    });
  }

  // Priority 5: Safety (always 2 queries)
  safetyJobs.push(
    { query: "cannabis drug interaction cytochrome P450", hint: "Drug interactions" },
    { query: "medicinal cannabis adverse effects systematic review", hint: "Adverse effects" }
  );

  // ── Execute within timeout budget ─────────────────────────────────────
  const allJobs = [
    ...terpeneJobs.map(j => ({ ...j, section: "terpene" })),
    ...cannJobs.map(j => ({ ...j, section: "cannabinoid" })),
    ...archetypeJobs.map(j => ({ ...j, section: "archetype" })),
    ...strainJobs.map(j => ({ ...j, section: "strain" })),
    ...safetyJobs.map(j => ({ ...j, section: "safety" })),
  ];

  const runJobs = Promise.all(
    allJobs.map(j =>
      searchAndFetch(j.query, 4)
        .then(articles => ({ ...j, articles }))
        .catch(() => ({ ...j, articles: [] }))
    )
  );

  const timeout = new Promise(resolve =>
    setTimeout(() => resolve(null), TIMEOUT_MS - 500)
  );

  const results = await Promise.race([runJobs, timeout]) || [];

  // ── Assemble sections ─────────────────────────────────────────────────
  const terpeneMap   = new Map(); // col → {terpene, articles[]}
  const cannArticles = [];
  const archArticles = [];
  const strainArticles = [];
  const safetyArticles = [];

  for (const r of results) {
    const arts = trimAndSort(r.articles || []);
    switch (r.section) {
      case "terpene": {
        const col = r.col;
        if (!terpeneMap.has(col)) terpeneMap.set(col, { terpene: TERPENE_LABEL[col] || r.label || col, articles: [] });
        terpeneMap.get(col).articles.push(...arts);
        break;
      }
      case "cannabinoid": cannArticles.push(...arts); break;
      case "archetype":   archArticles.push(...arts); break;
      case "strain":      strainArticles.push(...arts); break;
      case "safety":      safetyArticles.push(...arts); break;
    }
  }

  const evidence = {
    terpeneStudies:      [...terpeneMap.values()].map(g => ({ ...g, articles: trimAndSort(g.articles) })),
    cannabinoidStudies:  trimAndSort(cannArticles),
    indicationStudies:   trimAndSort(archArticles),
    strainFamilyStudies: trimAndSort(strainArticles),
    safetyStudies:       trimAndSort(safetyArticles),
    queriesRun:    allJobs.length,
    totalArticles: 0,
    executionMs:   Date.now() - start,
  };

  deduplicateAcrossSections(evidence);

  evidence.totalArticles =
    evidence.terpeneStudies.reduce((s, g) => s + g.articles.length, 0) +
    evidence.cannabinoidStudies.length +
    evidence.indicationStudies.length +
    evidence.strainFamilyStudies.length +
    evidence.safetyStudies.length;

  return evidence;
}

module.exports = { buildScientificEvidence };
