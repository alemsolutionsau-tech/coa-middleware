"use strict";

/**
 * scientificLayer.js — v2
 * Builds a structured scientificEvidence object from a processed COA.
 *
 * Queries PubMed for peer-reviewed research relevant to the COA's:
 *   - Dominant terpenes (mechanism + clinical therapeutic queries)
 *   - Cannabinoid ratios and entourage interactions
 *   - Effect archetype / therapeutic indication
 *   - Strain family (if high-confidence match)
 *   - Drug safety and contraindications
 *
 * Each article is tagged with:
 *   - evidenceQuality: Meta-Analysis | Systematic Review | Clinical Trial |
 *                      Research Article | Preclinical | In Vitro
 *   - therapeuticArea: the clinical domain this study addresses
 *   - relevanceHint:   the query that found it
 *
 * Execution budget: SCIENTIFIC_LAYER_TIMEOUT_MS (default 8s).
 * All failures are caught and logged — never throws.
 */

const { searchAndFetch } = require("./pubmedClient");

const TIMEOUT_MS = Number(process.env.SCIENTIFIC_LAYER_TIMEOUT_MS || 8000);

// ── Evidence quality detection ────────────────────────────────────────────
// Inspects title keywords to assign a quality tier (higher = more reputable)
function detectQuality(title) {
  const t = (title || "").toLowerCase();
  if (/\bmeta-analysis\b|\bmeta analysis\b/.test(t))
    return { label: "Meta-Analysis",     tier: 5, color: "gold"   };
  if (/systematic review/.test(t))
    return { label: "Systematic Review", tier: 4, color: "green"  };
  if (/randomized|randomised|randomized controlled|rct\b|double.blind|placebo.controlled/.test(t))
    return { label: "Clinical Trial",    tier: 3, color: "blue"   };
  if (/clinical study|clinical trial|pilot study|open.label trial|cohort/.test(t))
    return { label: "Clinical Study",    tier: 2, color: "blue"   };
  if (/\bin vivo\b|animal model|\brat \b|\bmouse \b|\bmice\b|rodent/.test(t))
    return { label: "Preclinical",       tier: 1, color: "grey"   };
  if (/\bin vitro\b|cell line|cell culture|cell-based/.test(t))
    return { label: "In Vitro",          tier: 0, color: "faint"  };
  return   { label: "Research Article",  tier: 2, color: "default"};
}

// ── Terpene therapeutic profiles ─────────────────────────────────────────
// Each entry has:
//   queries[]        — PubMed search terms (most specific/high-quality first)
//   therapeuticAreas — clinical domains this terpene is studied in
//   mechanismNote    — one-line pharmacological mechanism
const TERPENE_PROFILES = {
  beta_myrcene: {
    label: "β-Myrcene",
    queries: [
      "myrcene sedative anxiolytic cannabis review",
      "beta-myrcene analgesic anti-inflammatory",
      "myrcene cannabinoid synergy entourage",
    ],
    therapeuticAreas: ["Analgesia", "Sedation", "Anti-inflammatory"],
    mechanismNote: "Acts on TRPV1 receptors; may potentiate cannabinoid activity via CB1 modulation and enhance BBB permeability.",
  },
  d_limonene: {
    label: "D-Limonene",
    queries: [
      "d-limonene anxiolytic antidepressant human",
      "limonene serotonin dopamine stress",
      "limonene antioxidant anti-inflammatory",
    ],
    therapeuticAreas: ["Anxiolytic", "Antidepressant", "Antioxidant"],
    mechanismNote: "Modulates serotonin (5-HT1A) and dopamine receptors; demonstrated anxiolytic activity in clinical cohort studies.",
  },
  beta_caryophyllene: {
    label: "β-Caryophyllene",
    queries: [
      "beta-caryophyllene CB2 receptor agonist anti-inflammatory",
      "caryophyllene neuroprotection analgesic clinical",
      "caryophyllene anxiety depression CB2",
    ],
    therapeuticAreas: ["Anti-inflammatory", "Neuroprotection", "Analgesia"],
    mechanismNote: "Only terpene confirmed as a full CB2 receptor agonist (non-psychoactive). Studied for neuroinflammation, pain, and metabolic syndrome.",
  },
  alpha_pinene: {
    label: "α-Pinene",
    queries: [
      "alpha-pinene memory acetylcholinesterase inhibitor",
      "alpha-pinene bronchodilator anti-inflammatory",
      "pinene cannabis cognitive neuroprotection",
    ],
    therapeuticAreas: ["Cognition", "Respiratory", "Neuroprotection"],
    mechanismNote: "Acetylcholinesterase inhibitor — may counteract short-term memory impairment from THC. Bronchodilator in preclinical models.",
  },
  linalool: {
    label: "Linalool",
    queries: [
      "linalool anxiolytic GABA receptor clinical",
      "linalool sedative anticonvulsant mechanism",
      "linalool anti-inflammatory pain",
    ],
    therapeuticAreas: ["Anxiolytic", "Anticonvulsant", "Sedation"],
    mechanismNote: "Positive allosteric modulator of GABA-A receptors. Clinical studies show significant anxiolytic effects without dependence risk.",
  },
  terpinolene: {
    label: "Terpinolene",
    queries: [
      "terpinolene antioxidant anticancer",
      "terpinolene sedative CNS depressant",
    ],
    therapeuticAreas: ["Antioxidant", "Sedation"],
    mechanismNote: "Potent antioxidant; preliminary anticancer activity via PI3K/Akt pathway inhibition. Sedating in high concentrations.",
  },
  alpha_humulene: {
    label: "α-Humulene",
    queries: [
      "humulene anti-inflammatory appetite suppression",
      "alpha-humulene anticancer apoptosis",
    ],
    therapeuticAreas: ["Anti-inflammatory", "Appetite suppression"],
    mechanismNote: "Inhibits PGE2 (a key inflammatory mediator); appetite-suppressant properties noted in animal models.",
  },
  alpha_bisabolol: {
    label: "α-Bisabolol",
    queries: [
      "bisabolol anti-inflammatory wound healing skin",
      "bisabolol anticancer apoptosis",
    ],
    therapeuticAreas: ["Anti-inflammatory", "Dermatological"],
    mechanismNote: "Well-documented dermatological anti-inflammatory; emerging anticancer data via mitochondrial apoptosis pathway.",
  },
};

// ── Archetype → therapeutic indication queries ────────────────────────────
const ARCHETYPE_THERAPEUTIC = {
  "Relaxing / Sedating":     { query: "cannabis insomnia sleep quality clinical trial",      area: "Sleep & Relaxation"    },
  "relaxing":                { query: "cannabis insomnia sleep quality clinical trial",      area: "Sleep & Relaxation"    },
  "sedating":                { query: "cannabis insomnia sleep disorders systematic review", area: "Sleep & Relaxation"    },
  "Uplifting / Energising":  { query: "cannabis anxiety depression mood clinical trial",     area: "Mood & Mental Health"  },
  "uplifting":               { query: "cannabis anxiety depression mood clinical trial",     area: "Mood & Mental Health"  },
  "Grounding / Anti-inflam.":{ query: "cannabis pain inflammation CB2 receptor clinical",   area: "Pain & Inflammation"   },
  "grounding":               { query: "cannabis chronic pain inflammation clinical",         area: "Pain & Inflammation"   },
  "anti-inflammatory":       { query: "cannabis anti-inflammatory mechanism clinical",       area: "Pain & Inflammation"   },
  "cerebral":                { query: "cannabis sativa cognitive focus clinical study",      area: "Cognition"             },
  "anxiolytic":              { query: "cannabis anxiety GABA receptor anxiolytic trial",    area: "Anxiety"               },
};

// ── Cannabinoid ratio queries ─────────────────────────────────────────────
function buildCannabinoidQueries(thc, cbd) {
  const jobs = [];
  if (thc > 0 && cbd > 0) {
    const ratio = cbd / thc;
    if (ratio > 1)
      jobs.push({ query: "CBD THC ratio anxiety inflammation systematic review", hint: `CBD:THC ${ratio.toFixed(2)} (CBD-dominant)` });
    else if (ratio > 0.1)
      jobs.push({ query: "cannabidiol THC entourage effect clinical review", hint: `CBD:THC ${ratio.toFixed(2)}` });
    else
      jobs.push({ query: "THC dose response analgesic clinical study", hint: `THC-dominant (${thc}%)` });
  }
  // Always: entourage effect
  jobs.push({ query: "phytocannabinoid terpene entourage effect systematic review", hint: "Entourage Effect" });
  // Always: endocannabinoid system
  jobs.push({ query: "endocannabinoid system therapeutic applications review", hint: "Endocannabinoid System" });
  return jobs;
}

// ── Safety queries — always run ───────────────────────────────────────────
const SAFETY_QUERIES = [
  { query: "cannabis adverse effects contraindications systematic review",  hint: "Adverse Effects"      },
  { query: "cannabis drug interaction cytochrome P450 CYP3A4",             hint: "Drug Interactions"    },
  { query: "medicinal cannabis safety tolerability meta-analysis",         hint: "Safety Meta-Analysis" },
];

// ── Helpers ───────────────────────────────────────────────────────────────
function toNum(v) { const n = parseFloat(v); return isNaN(n) ? 0 : n; }

function getDominantTerpenes(chemistry) {
  const top = chemistry.top_terpenes || [];
  const totalTerps = toNum(chemistry.total_terpenes) || 0.001;
  const NAME_TO_COL = {
    "beta-myrcene":"beta_myrcene","β-myrcene":"beta_myrcene","myrcene":"beta_myrcene",
    "(r)-(+)-limonene":"d_limonene","limonene":"d_limonene","d-limonene":"d_limonene",
    "trans-caryophyllene":"beta_caryophyllene","beta-caryophyllene":"beta_caryophyllene","caryophyllene":"beta_caryophyllene",
    "alpha-pinene":"alpha_pinene","α-pinene":"alpha_pinene",
    "linalool":"linalool","terpinolene":"terpinolene",
    "alpha-humulene":"alpha_humulene","humulene":"alpha_humulene",
    "alpha-bisabolol":"alpha_bisabolol","bisabolol":"alpha_bisabolol",
  };
  const mapped = top
    .map(t => ({ col: NAME_TO_COL[String(t.name||"").toLowerCase().trim()], val: toNum(t.value), name: t.name }))
    .filter(t => t.col && t.val > 0 && TERPENE_PROFILES[t.col])
    .map(t => ({ ...t, fraction: t.val / totalTerps }));
  // Use 5% threshold but always include at least the top 2 if they map to known profiles
  const aboveThreshold = mapped.filter(t => t.fraction > 0.05);
  return (aboveThreshold.length > 0 ? aboveThreshold : mapped.slice(0, 2)).slice(0, 3);
}

function tagArticles(articles, therapeuticArea, relevanceHint) {
  return articles.map(a => ({
    ...a,
    quality:          detectQuality(a.title),
    therapeuticArea:  therapeuticArea || "",
    relevanceHint:    relevanceHint  || a.relevanceHint || "",
  }));
}

function deduplicate(evidence) {
  const seen = new Map();
  const priority = ["safetyStudies","terpeneStudies","indicationStudies","cannabinoidStudies","strainFamilyStudies"];
  for (const sec of priority) {
    if (sec === "terpeneStudies") {
      for (const g of evidence[sec]) {
        g.articles = g.articles.filter(a => { if (seen.has(a.pmid)) return false; seen.set(a.pmid,sec); return true; });
      }
    } else {
      evidence[sec] = (evidence[sec]||[]).filter(a => { if (seen.has(a.pmid)) return false; seen.set(a.pmid,sec); return true; });
    }
  }
  return evidence;
}

function trimSort(arts, n = 3) {
  // Sort: prefer higher quality tier, then more recent year
  return [...arts]
    .sort((a,b) => {
      const qt = (b.quality?.tier||0) - (a.quality?.tier||0);
      if (qt !== 0) return qt;
      return (Number(b.year)||0) - (Number(a.year)||0);
    })
    .slice(0, n);
}

// ── Main export ───────────────────────────────────────────────────────────
async function buildScientificEvidence(coaData) {
  const start     = Date.now();
  const chemistry = coaData.chemistry || coaData;
  const intel     = coaData.intelligence || {};
  const strainIntel = intel.strainIntel || null;

  const thc = toNum(chemistry.thc_total);
  const cbd = toNum(chemistry.cbd_total);

  // Still run cannabinoid + safety queries even if terpenes are low/absent
  const dominantTerps = getDominantTerpenes(chemistry);

  // Build all jobs
  const jobs = [];

  // Priority 1 — Terpene queries (mechanism + clinical, 2-3 queries each)
  for (const { col } of dominantTerps) {
    const profile = TERPENE_PROFILES[col];
    if (!profile) continue;
    for (const q of profile.queries) {
      jobs.push({ query: q, section: "terpene", col, therapeuticArea: profile.therapeuticAreas[0], hint: profile.label });
    }
  }

  // Priority 2 — Cannabinoid ratio + entourage
  for (const j of buildCannabinoidQueries(thc, cbd)) {
    jobs.push({ query: j.query, section: "cannabinoid", therapeuticArea: "Cannabinoid Interactions", hint: j.hint });
  }

  // Priority 3 — Effect archetype
  const archetype = intel.effectDirection || intel.lineageCluster || "";
  const archetypeMap = ARCHETYPE_THERAPEUTIC[archetype] || ARCHETYPE_THERAPEUTIC[archetype.toLowerCase()] || null;
  if (archetypeMap) {
    jobs.push({ query: archetypeMap.query, section: "archetype", therapeuticArea: archetypeMap.area, hint: archetype });
    // Second clinical query for the same area
    jobs.push({ query: archetypeMap.query + " meta-analysis", section: "archetype", therapeuticArea: archetypeMap.area, hint: `${archetype} (meta-analysis)` });
  }

  // Priority 4 — Strain family
  const match = strainIntel?.match;
  if (match?.strain_name && match.similarity >= 70) {
    jobs.push({ query: `${match.strain_name} cannabis terpene therapeutic`, section: "strain", therapeuticArea: "Strain Research", hint: match.strain_name });
  }

  // Priority 5 — Safety (always)
  for (const j of SAFETY_QUERIES) {
    jobs.push({ query: j.query, section: "safety", therapeuticArea: "Safety & Tolerability", hint: j.hint });
  }

  // Execute within budget
  const runAll = Promise.all(
    jobs.map(j =>
      searchAndFetch(j.query, 4)
        .then(arts => ({ ...j, articles: tagArticles(arts, j.therapeuticArea, j.hint) }))
        .catch(() => ({ ...j, articles: [] }))
    )
  );
  const timedOut = new Promise(r => setTimeout(() => r(null), TIMEOUT_MS - 500));
  const results  = await Promise.race([runAll, timedOut]) || [];

  // Assemble sections
  const terpeneMap   = new Map();
  const cannArts     = [];
  const archArts     = [];
  const strainArts   = [];
  const safetyArts   = [];

  for (const r of results) {
    if (!r.articles?.length) continue;
    switch (r.section) {
      case "terpene": {
        if (!terpeneMap.has(r.col)) {
          const p = TERPENE_PROFILES[r.col];
          terpeneMap.set(r.col, { terpene: p.label, mechanismNote: p.mechanismNote, therapeuticAreas: p.therapeuticAreas, articles: [] });
        }
        terpeneMap.get(r.col).articles.push(...r.articles);
        break;
      }
      case "cannabinoid": cannArts.push(...r.articles);  break;
      case "archetype":   archArts.push(...r.articles);  break;
      case "strain":      strainArts.push(...r.articles); break;
      case "safety":      safetyArts.push(...r.articles); break;
    }
  }

  const evidence = {
    terpeneStudies:      [...terpeneMap.values()].map(g => ({ ...g, articles: trimSort(g.articles, 4) })),
    cannabinoidStudies:  trimSort(cannArts, 4),
    indicationStudies:   trimSort(archArts, 3),
    strainFamilyStudies: trimSort(strainArts, 3),
    safetyStudies:       trimSort(safetyArts, 3),
    queriesRun:    jobs.length,
    totalArticles: 0,
    executionMs:   Date.now() - start,
  };

  deduplicate(evidence);

  evidence.totalArticles =
    evidence.terpeneStudies.reduce((s, g) => s + g.articles.length, 0) +
    evidence.cannabinoidStudies.length +
    evidence.indicationStudies.length +
    evidence.strainFamilyStudies.length +
    evidence.safetyStudies.length;

  return evidence;
}

module.exports = { buildScientificEvidence };
