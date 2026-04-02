"use strict";

/**
 * strainIntelligence.js
 * Compares a COA's terpene fingerprint against mart_strain_profiles in BigQuery.
 * Returns three sections: STRAIN MATCH, SUBSTITUTE STRAINS, COMPLEMENTARY STRAINS.
 *
 * Similarity metric: cosine similarity on the terpene vector (20 compounds).
 * Complementary match: strains whose dominant terpene belongs to a different
 * effect cluster (relaxing ↔ uplifting ↔ grounding).
 *
 * Results are 24h-cached per form_factor so BQ is queried at most once/day.
 */

const { BigQuery } = require("@google-cloud/bigquery");

const PROJECT = "alem-coa-ai";
const CLEAN   = `${PROJECT}.cannabis_coa_clean`;

function makeBqClient() {
  const raw = process.env.BQ_CREDENTIALS_JSON;
  if (raw) {
    try {
      const start = raw.indexOf("{");
      const end   = raw.lastIndexOf("}");
      return new BigQuery({ projectId: PROJECT, credentials: JSON.parse(raw.slice(start, end + 1)) });
    } catch (e) {
      console.error("strainIntelligence: BQ credential parse failed:", e.message);
    }
  }
  return new BigQuery({ projectId: PROJECT });
}
const bq = makeBqClient();

// ── 24h cache (keyed per form_factor) ────────────────────────────────────
const _cache = new Map();
function _get(k) { const e = _cache.get(k); if (!e || Date.now() - e.ts > 86_400_000) return undefined; return e.v; }
function _set(k, v) { _cache.set(k, { ts: Date.now(), v }); }

// ── 20 terpene columns used in all computations ───────────────────────────
const TERPENE_COLS = [
  "beta_myrcene", "d_limonene", "beta_caryophyllene", "alpha_pinene",
  "beta_pinene",  "linalool",   "ocimene",            "terpinolene",
  "humulene",     "alpha_bisabolol", "camphene",       "guaiol",
  "valencene",    "geraniol",   "nerolidol",          "fenchol",
  "borneol",      "sabinene",   "delta3_carene",      "cymene",
];

// ── Terpene display names ─────────────────────────────────────────────────
const COL_TO_LABEL = {
  beta_myrcene:        "Beta-Myrcene",
  d_limonene:          "Limonene",
  beta_caryophyllene:  "Caryophyllene",
  alpha_pinene:        "Alpha-Pinene",
  beta_pinene:         "Beta-Pinene",
  linalool:            "Linalool",
  ocimene:             "Ocimene",
  terpinolene:         "Terpinolene",
  humulene:            "Alpha-Humulene",
  alpha_bisabolol:     "Bisabolol",
  camphene:            "Camphene",
  guaiol:              "Guaiol",
  valencene:           "Valencene",
  geraniol:            "Geraniol",
  nerolidol:           "Nerolidol",
  fenchol:             "Fenchol",
  borneol:             "Borneol",
  sabinene:            "Sabinene",
  delta3_carene:       "Delta-3-Carene",
  cymene:              "p-Cymene",
};

// ── Name → column (for extracting vector from GPT-4o top_terpenes) ────────
const NAME_TO_COL = {
  "beta-myrcene":        "beta_myrcene",
  "β-myrcene":           "beta_myrcene",
  "myrcene":             "beta_myrcene",
  "(r)-(+)-limonene":    "d_limonene",
  "limonene":            "d_limonene",
  "d-limonene":          "d_limonene",
  "trans-caryophyllene": "beta_caryophyllene",
  "beta-caryophyllene":  "beta_caryophyllene",
  "caryophyllene":       "beta_caryophyllene",
  "α-caryophyllene":     "beta_caryophyllene",
  "alpha-pinene":        "alpha_pinene",
  "α-pinene":            "alpha_pinene",
  "beta-pinene":         "beta_pinene",
  "β-pinene":            "beta_pinene",
  "linalool":            "linalool",
  "ocimene":             "ocimene",
  "β-ocimene":           "ocimene",
  "terpinolene":         "terpinolene",
  "alpha-humulene":      "humulene",
  "α-humulene":          "humulene",
  "humulene":            "humulene",
  "alpha-bisabolol":     "alpha_bisabolol",
  "α-bisabolol":         "alpha_bisabolol",
  "bisabolol":           "alpha_bisabolol",
  "camphene":            "camphene",
  "guaiol":              "guaiol",
  "valencene":           "valencene",
  "geraniol":            "geraniol",
  "nerolidol":           "nerolidol",
  "trans-nerolidol":     "nerolidol",
  "fenchol":             "fenchol",
  "borneol":             "borneol",
  "sabinene":            "sabinene",
  "delta-3-carene":      "delta3_carene",
  "3-carene":            "delta3_carene",
  "p-cymene":            "cymene",
  "cymene":              "cymene",
};

// ── Effect clusters and complementary mappings ───────────────────────────
const EFFECT_CLUSTERS = {
  relaxing:  { cols: ["beta_myrcene", "linalool", "alpha_bisabolol"],          label: "Relaxing / Sedating"      },
  uplifting: { cols: ["d_limonene", "terpinolene", "alpha_pinene", "ocimene"], label: "Uplifting / Energising"   },
  grounding: { cols: ["beta_caryophyllene", "humulene"],                        label: "Grounding / Anti-inflam." },
  complex:   { cols: ["valencene", "geraniol", "guaiol", "nerolidol"],          label: "Complex / Exotic"         },
};
const COMPLEMENTARY_MAP = {
  relaxing:  ["uplifting", "grounding"],
  uplifting: ["relaxing",  "grounding"],
  grounding: ["uplifting", "relaxing"],
  complex:   ["uplifting", "relaxing"],
};

function getCluster(dominantCol) {
  for (const [name, { cols }] of Object.entries(EFFECT_CLUSTERS)) {
    if (cols.includes(dominantCol)) return name;
  }
  return null;
}

// ── Build a numeric terpene vector from chemistry.top_terpenes ─────────────
function buildInputVector(topTerpenes = []) {
  const vec = {};
  for (const t of topTerpenes) {
    const col = NAME_TO_COL[String(t.name || "").toLowerCase().trim()];
    const val = parseFloat(t.value) || 0;
    if (col && val > 0) vec[col] = (vec[col] || 0) + val;
  }
  const mag = Math.sqrt(Object.values(vec).reduce((s, v) => s + v * v, 0));
  return { vec, mag };
}

// ── Cosine similarity between input vector and a BQ row ───────────────────
function cosine(inputVec, inputMag, row) {
  if (inputMag === 0) return 0;
  let dot = 0, strainMag2 = 0;
  for (const col of TERPENE_COLS) {
    const a = inputVec[col] || 0;
    const b = Number(row[`avg_${col}_pct`] || 0);
    dot       += a * b;
    strainMag2 += b * b;
  }
  const strainMag = Math.sqrt(strainMag2);
  return strainMag === 0 ? 0 : dot / (inputMag * strainMag);
}

// ── Dominant terpene column for a BQ row ──────────────────────────────────
function dominantCol(row) {
  let best = null, bestVal = -1;
  for (const col of TERPENE_COLS) {
    const v = Number(row[`avg_${col}_pct`] || 0);
    if (v > bestVal) { bestVal = v; best = col; }
  }
  return best;
}

// ── Fetch all strain profiles for a form_factor (cached 24h) ─────────────
async function fetchAllStrains(formFactor) {
  const cacheKey = `strains|${formFactor}`;
  const cached = _get(cacheKey);
  if (cached !== undefined) return cached;

  const [rows] = await bq.query({
    query: `
      SELECT
        strain_name, form_factor, coa_count,
        avg_total_thc_pct, avg_total_terpenes_pct,
        p10_total_thc_pct, p90_total_thc_pct,
        avg_beta_myrcene_pct,        avg_d_limonene_pct,
        avg_beta_caryophyllene_pct,  avg_alpha_pinene_pct,
        avg_beta_pinene_pct,         avg_linalool_pct,
        avg_ocimene_pct,             avg_terpinolene_pct,
        avg_humulene_pct,            avg_alpha_bisabolol_pct,
        avg_camphene_pct,            avg_guaiol_pct,
        avg_valencene_pct,           avg_geraniol_pct,
        avg_nerolidol_pct,           avg_fenchol_pct,
        avg_borneol_pct,             avg_sabinene_pct,
        avg_delta3_carene_pct,       avg_cymene_pct
      FROM \`${CLEAN}.mart_strain_profiles\`
      WHERE form_factor = @ff
        AND coa_count >= 3
    `,
    params: { ff: formFactor },
  });

  _set(cacheKey, rows);
  return rows;
}

// ── Main entry point ──────────────────────────────────────────────────────
async function fetchStrainIntelligence(chemistry) {
  try {
    const { normalizeFormFactor } = require("./bqBenchmark");
    const formFactor = normalizeFormFactor(chemistry.product_type);
    if (!formFactor || formFactor === "other") return null;

    const { vec: inputVec, mag: inputMag } = buildInputVector(chemistry.top_terpenes || []);
    if (inputMag === 0) return null; // no terpene data → can't match

    const rows = await fetchAllStrains(formFactor);
    if (!rows || rows.length === 0) return null;

    // Score every strain by cosine similarity
    const scored = rows
      .map(row => {
        const dc = dominantCol(row);
        return {
          strain_name:  String(row.strain_name || ""),
          form_factor:  String(row.form_factor  || ""),
          coa_count:    Number(row.coa_count     || 0),
          avg_thc:      Number(row.avg_total_thc_pct      || 0).toFixed(1),
          avg_terpenes: Number(row.avg_total_terpenes_pct || 0).toFixed(2),
          p10_thc:      row.p10_total_thc_pct != null ? Number(row.p10_total_thc_pct).toFixed(1) : null,
          p90_thc:      row.p90_total_thc_pct != null ? Number(row.p90_total_thc_pct).toFixed(1) : null,
          similarity:   Math.round(cosine(inputVec, inputMag, row) * 100),
          dominantCol:  dc,
          dominantLabel: dc ? (COL_TO_LABEL[dc] || dc) : null,
          cluster:      dc ? getCluster(dc) : null,
          clusterLabel: dc ? (EFFECT_CLUSTERS[getCluster(dc)]?.label || null) : null,
        };
      })
      .sort((a, b) => b.similarity - a.similarity);

    // STRAIN MATCH — top hit
    const match = scored[0] || null;

    // SUBSTITUTES — next 4 at >= 55% similarity
    const substitutes = scored.slice(1).filter(s => s.similarity >= 55).slice(0, 4);

    // COMPLEMENTARY — different effect cluster, lower similarity range (avoids dupes)
    const inputDominantCol = Object.entries(inputVec).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
    const inputCluster = inputDominantCol ? getCluster(inputDominantCol) : null;
    const compClusters = new Set((COMPLEMENTARY_MAP[inputCluster] || []).flatMap(c => EFFECT_CLUSTERS[c]?.cols || []));

    const complementary = scored
      .filter(s => s.dominantCol && compClusters.has(s.dominantCol) && s.similarity < 75)
      .slice(0, 3);

    return {
      match,
      substitutes,
      complementary,
      inputCluster,
      inputClusterLabel: inputCluster ? EFFECT_CLUSTERS[inputCluster]?.label : null,
      totalStrains: rows.length,
      formFactor,
    };

  } catch (err) {
    console.error("Strain intelligence failed (non-fatal):", err.message);
    return null;
  }
}

module.exports = { fetchStrainIntelligence, COL_TO_LABEL, EFFECT_CLUSTERS };
