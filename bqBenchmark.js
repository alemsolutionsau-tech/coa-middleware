"use strict";

/**
 * bqBenchmark.js
 * Fetches live market benchmark data from the cannabis_coa_clean BigQuery
 * dataset and returns percentile ranks + market medians for a given COA.
 *
 * Results are cached in-memory for 24 hours to avoid redundant BQ queries.
 * All failures are swallowed — returns null so reports still render fine
 * without BQ credentials in dev environments.
 */

const { BigQuery } = require("@google-cloud/bigquery");

const PROJECT = "alem-coa-ai";
const CLEAN   = `${PROJECT}.cannabis_coa_clean`;

// On Render (and most cloud hosts) you can't point to a key file.
// Set BQ_CREDENTIALS_JSON to the full contents of your service account JSON.
// Falls back to GOOGLE_APPLICATION_CREDENTIALS / ADC if not set.
function makeBqClient() {
  const raw = process.env.BQ_CREDENTIALS_JSON;
  if (raw) {
    try {
      const credentials = JSON.parse(raw);
      return new BigQuery({ projectId: PROJECT, credentials });
    } catch (e) {
      console.error("BQ_CREDENTIALS_JSON is set but failed to parse:", e.message);
    }
  }
  return new BigQuery({ projectId: PROJECT });
}
const bq = makeBqClient();

// ── 24-hour in-memory cache ───────────────────────────────────────────────
const _cache = new Map();
function _get(key) {
  const e = _cache.get(key);
  if (!e) return undefined;
  if (Date.now() - e.ts > 86_400_000) { _cache.delete(key); return undefined; }
  return e.v;
}
function _set(key, v) { _cache.set(key, { ts: Date.now(), v }); }

// ── Form factor normalisation (mirrors bq_phase3_pipeline.py logic) ───────
function normalizeFormFactor(productType) {
  const pt = String(productType || "").toLowerCase();
  if (/flower|bud|shake|trim|hemp|plant|nug/.test(pt))                          return "flower";
  if (/preroll|pre-roll|pre roll|joint|cone|blunt/.test(pt))                    return "preroll";
  if (/\boil\b|tincture|vape|cartridge|vaporizer|\bpen\b/.test(pt))             return "oil";
  if (/capsule|tablet|softgel|pill/.test(pt))                                   return "capsule";
  if (/extract|concentrate|wax|shatter|rosin|resin|distillate|crumble|budder|sauce|live|hash|kief|isolate|co2|bho|rso|dab|sugar|badder|diamonds|crystalline/.test(pt)) return "extract";
  return "other";
}

const FORM_FACTOR_LABELS = {
  flower:  "Dried Flower",
  preroll: "Pre-Roll",
  oil:     "Oil / Vape",
  capsule: "Capsule",
  extract: "Extract / Concentrate",
  other:   "Cannabis Product",
};

// ── Core BQ query ─────────────────────────────────────────────────────────
async function fetchMarketBenchmark(formFactor, thcPct, terpPct) {
  const thcBucket  = Math.round(thcPct  * 10) / 10;
  const terpBucket = Math.round(terpPct * 10) / 10;
  const key = `market|${formFactor}|${thcBucket}|${terpBucket}`;
  const cached = _get(key);
  if (cached !== undefined) return cached;

  const [rows] = await bq.query({
    query: `
      SELECT
        COUNT(*)                                                                      AS total,
        COUNTIF(total_thc_pct  < @thc)                                               AS below_thc,
        COUNTIF(total_terpenes_pct < @terp)                                          AS below_terp,
        ROUND(APPROX_QUANTILES(total_thc_pct,        100)[OFFSET(50)], 2)            AS median_thc,
        ROUND(APPROX_QUANTILES(total_terpenes_pct,   100)[OFFSET(50)], 2)            AS median_terp,
        ROUND(APPROX_QUANTILES(total_thc_pct,         10)[OFFSET(9)],  2)            AS p90_thc,
        ROUND(APPROX_QUANTILES(total_terpenes_pct,    10)[OFFSET(9)],  2)            AS p90_terp,
        COUNT(DISTINCT supplier_id)                                                   AS supplier_count
      FROM \`${CLEAN}.fact_coa\`
      WHERE form_factor = @ff
        AND total_thc_pct IS NOT NULL
        AND test_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
    `,
    params: { ff: formFactor, thc: thcPct, terp: terpPct },
  });

  const r = rows[0];
  const n = Number(r.total) || 1;

  const result = {
    n,
    formFactor,
    formFactorLabel: FORM_FACTOR_LABELS[formFactor] || formFactor,
    thcPercentile:   Math.round(Number(r.below_thc)  / n * 100),
    terpPercentile:  terpPct > 0 ? Math.round(Number(r.below_terp) / n * 100) : null,
    medianThc:       r.median_thc  != null ? Number(r.median_thc)  : null,
    medianTerp:      r.median_terp != null ? Number(r.median_terp) : null,
    p90Thc:          r.p90_thc     != null ? Number(r.p90_thc)     : null,
    p90Terp:         r.p90_terp    != null ? Number(r.p90_terp)    : null,
    supplierCount:   r.supplier_count != null ? Number(r.supplier_count) : null,
  };

  _set(key, result);
  return result;
}

/**
 * Main entry point.
 * @param {object} chemistry - reportJson.chemistry from Supabase
 * @returns {object|null} benchmark data, or null if unavailable
 */
async function fetchBenchmark(chemistry) {
  try {
    const formFactor = normalizeFormFactor(chemistry.product_type);
    const thcPct     = parseFloat(chemistry.thc_total)       || 0;
    const terpPct    = parseFloat(chemistry.total_terpenes)  || 0;
    if (!formFactor || thcPct === 0) return null;
    return await fetchMarketBenchmark(formFactor, thcPct, terpPct);
  } catch (err) {
    console.error("BQ benchmark fetch failed (non-fatal):", err.message);
    return null;
  }
}

module.exports = { fetchBenchmark, normalizeFormFactor };
