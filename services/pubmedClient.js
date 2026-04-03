"use strict";

/**
 * pubmedClient.js
 * Rate-limited, cached wrapper around NCBI PubMed E-utilities.
 *
 * Rate limits:
 *   Without API key: 3 req/sec (MIN_GAP = 334ms)
 *   With NCBI_API_KEY: 10 req/sec (MIN_GAP = 100ms)
 *
 * Cache: in-memory 24h TTL + disk persistence to /tmp/pubmed_cache.json
 */

const axios  = require("axios");
const fs     = require("fs");
const path   = require("path");

const NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";
const API_KEY      = process.env.NCBI_API_KEY || null;
const CACHE_TTL_MS = Number(process.env.PUBMED_CACHE_TTL_HOURS || 168) * 3_600_000;
const CACHE_FILE   = path.join("/tmp", "pubmed_cache.json");
const USER_AGENT   = "AlemSolutions-COA-Intelligence/1.0 (contact@alem.solutions)";
const MIN_GAP_MS   = API_KEY ? 100 : 334;

// ── In-memory cache ───────────────────────────────────────────────────────
const _cache = new Map();

// Load from disk on module init
try {
  if (fs.existsSync(CACHE_FILE)) {
    const raw  = JSON.parse(fs.readFileSync(CACHE_FILE, "utf8"));
    const now  = Date.now();
    let loaded = 0;
    for (const [k, v] of Object.entries(raw)) {
      if (now - v.cachedAt < CACHE_TTL_MS) { _cache.set(k, v); loaded++; }
    }
    if (loaded) console.log(`📚 PubMed disk cache loaded: ${loaded} entries`);
  }
} catch (_) {}

function _persistCache() {
  try {
    const obj = {};
    for (const [k, v] of _cache.entries()) obj[k] = v;
    fs.writeFileSync(CACHE_FILE, JSON.stringify(obj));
  } catch (_) {}
}

// ── Rate limiter — single serial promise chain ────────────────────────────
let _chain   = Promise.resolve();
let _lastAt  = 0;
let _hourCount = 0;
let _hourStart = Date.now();

function _enqueue(fn) {
  const run = () => {
    const gap = Math.max(0, _lastAt + MIN_GAP_MS - Date.now());
    return new Promise(r => setTimeout(r, gap)).then(() => {
      _lastAt = Date.now();
      // Rate warning counter
      if (Date.now() - _hourStart > 3_600_000) { _hourCount = 0; _hourStart = Date.now(); }
      _hourCount++;
      if (_hourCount > 100) console.warn(`⚠️  PubMed: ${_hourCount} requests this hour`);
      return fn();
    });
  };
  _chain = _chain.then(run, run); // continue queue even on error
  return _chain;
}

// ── Core HTTP fetch with retry ────────────────────────────────────────────
async function _ncbiGet(endpoint, params) {
  const p = { ...params, retmode: "json", tool: "AlemCOA", email: "contact@alem.solutions" };
  if (API_KEY) p.api_key = API_KEY;
  const url = `${NCBI_BASE}/${endpoint}?${new URLSearchParams(p)}`;

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await _enqueue(() =>
        axios.get(url, { timeout: 8000, headers: { "User-Agent": USER_AGENT } })
      );
      return res.data;
    } catch (err) {
      if (err.response?.status === 429) {
        await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
      } else {
        console.error(`PubMed ${endpoint} failed (attempt ${attempt + 1}):`, err.message);
        break;
      }
    }
  }
  return null;
}

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Search PubMed and return an array of PMIDs.
 */
async function searchPubMed(query, maxResults = 5) {
  const key    = `s:${query}:${maxResults}`;
  const cached = _cache.get(key);
  if (cached && Date.now() - cached.cachedAt < CACHE_TTL_MS) return cached.result;

  const data = await _ncbiGet("esearch.fcgi", {
    db: "pubmed", term: query, retmax: maxResults, sort: "relevance",
  });

  const pmids = data?.esearchresult?.idlist || [];
  _cache.set(key, { result: pmids, cachedAt: Date.now() });
  return pmids;
}

/**
 * Fetch article metadata for an array of PMIDs.
 * @param {string[]} pmids
 * @param {string}   relevanceHint  — which query/terpene triggered this
 */
async function fetchSummaries(pmids, relevanceHint = "") {
  if (!pmids.length) return [];
  const key    = `sum:${[...pmids].sort().join(",")}`;
  const cached = _cache.get(key);
  if (cached && Date.now() - cached.cachedAt < CACHE_TTL_MS) {
    // Patch relevanceHint onto cached results without mutating cache
    return cached.result.map(a => ({ ...a, relevanceHint }));
  }

  const data = await _ncbiGet("esummary.fcgi", { db: "pubmed", id: pmids.join(",") });
  if (!data?.result) return [];

  const articles = pmids.map(pmid => {
    const r = data.result[pmid];
    if (!r || r.error) return null;
    const rawAuthors = r.authors || [];
    const authors = rawAuthors.slice(0, 3).map(a => a.name).join(", ")
                  + (rawAuthors.length > 3 ? " et al." : "");
    const doi = (r.articleids || []).find(a => a.idtype === "doi")?.value || null;
    return {
      pmid,
      title:   r.title  || "(No title)",
      authors,
      journal: r.source || "",
      year:    r.pubdate ? String(r.pubdate).split(" ")[0] : "",
      doi,
      url:     `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`,
      relevanceHint,
    };
  }).filter(Boolean);

  _cache.set(key, { result: articles, cachedAt: Date.now() });
  _persistCache();
  return articles;
}

/**
 * Convenience: search then immediately fetch summaries.
 */
async function searchAndFetch(query, maxResults = 5) {
  try {
    const pmids = await searchPubMed(query, maxResults);
    if (!pmids.length) return [];
    return fetchSummaries(pmids, query);
  } catch (err) {
    console.error("searchAndFetch error:", err.message);
    return [];
  }
}

// Startup banner
console.log(`📖 PubMed integration active. Cache TTL: ${CACHE_TTL_MS / 3_600_000}h. API key: ${API_KEY ? "YES" : "NO"}`);

module.exports = { searchPubMed, fetchSummaries, searchAndFetch };
