const express = require("express");
const cors = require("cors");
const puppeteer = require("puppeteer");
const { createClient } = require("@supabase/supabase-js");
const supabase = require("./supabase");

const app = express();

app.use(cors());
app.use(express.text({ type: "*/*", limit: "50mb" }));

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

function esc(v = "") {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sanitizeFileName(name = "") {
  return String(name || "")
    .replace(/[^a-zA-Z0-9-_.]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function numberFromPercentString(v) {
  if (v === null || v === undefined) return 0;
  const match = String(v).match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : 0;
}

function getSafeArray(arr) {
  return Array.isArray(arr) ? arr : [];
}

function renderBadges(items = [], type = "neutral") {
  if (!Array.isArray(items) || !items.length) {
    return `<span class="muted-inline">None reported</span>`;
  }

  return items
    .map((x) => `<span class="badge ${type}">${esc(x)}</span>`)
    .join("");
}

function renderMetricCards(data) {
  return `
    <div class="metrics">
      <div class="metric hero-metric">
        <div class="label">Total THC</div>
        <div class="value">${esc(data?.thc_total || "Not reported")}</div>
      </div>
      <div class="metric hero-metric">
        <div class="label">Total CBD</div>
        <div class="value">${esc(data?.cbd_total || "Not reported")}</div>
      </div>
      <div class="metric hero-metric">
        <div class="label">Total Terpenes</div>
        <div class="value">${esc(data?.total_terpenes || "Not reported")}</div>
      </div>
      <div class="metric hero-metric">
        <div class="label">Confidence</div>
        <div class="value">${esc(data?.report_confidence_score || "—")}</div>
      </div>
    </div>
  `;
}

function renderMetaGrid(data) {
  return `
    <div class="meta">
      <div class="meta-item"><div class="meta-label">Batch</div><div class="meta-value">${esc(data?.batch_number || "Not reported")}</div></div>
      <div class="meta-item"><div class="meta-label">COA date</div><div class="meta-value">${esc(data?.coa_report_date || "Not reported")}</div></div>
      <div class="meta-item"><div class="meta-label">Product type</div><div class="meta-value">${esc(data?.product_type || "Not reported")}</div></div>
      <div class="meta-item"><div class="meta-label">Laboratory</div><div class="meta-value">${esc(data?.laboratory_name || "Not reported")}</div></div>
      <div class="meta-item"><div class="meta-label">Sample date</div><div class="meta-value">${esc(data?.sample_date || "Not reported")}</div></div>
      <div class="meta-item"><div class="meta-label">Received date</div><div class="meta-value">${esc(data?.received_date || "Not reported")}</div></div>
      <div class="meta-item"><div class="meta-label">Certificate ID</div><div class="meta-value">${esc(data?.certificate_id || "Not reported")}</div></div>
      <div class="meta-item"><div class="meta-label">Total cannabinoids</div><div class="meta-value">${esc(data?.total_cannabinoids || "Not reported")}</div></div>
    </div>
  `;
}

function renderTableRows(items, type) {
  if (!Array.isArray(items) || !items.length) {
    if (type === "cannabinoids") {
      return `<tr><td colspan="4">No explicit cannabinoid rows available.</td></tr>`;
    }
    if (type === "terpenes") {
      return `<tr><td colspan="3">No explicit terpene rows available.</td></tr>`;
    }
    return `<tr><td colspan="3">No explicit rows available.</td></tr>`;
  }

  if (type === "cannabinoids") {
    return items.map((c) => `
      <tr>
        <td>${esc(c.name)}</td>
        <td>${esc(c.value)}</td>
        <td>${esc(c.unit)}</td>
        <td>${esc(c.notes)}</td>
      </tr>
    `).join("");
  }

  if (type === "terpenes") {
    return items.map((t) => `
      <tr>
        <td>${esc(t.name)}</td>
        <td>${esc(t.value)}</td>
        <td>${esc(t.unit)}</td>
      </tr>
    `).join("");
  }

  return items.map((x) => `
    <tr>
      <td>${esc(x.label)}</td>
      <td>${esc(x.status)}</td>
      <td>${esc(x.notes)}</td>
    </tr>
  `).join("");
}

function renderComparisonCards(items = []) {
  if (!Array.isArray(items) || !items.length) {
    return `<div class="copy">No comparative classification available.</div>`;
  }

  return `
    <div class="comparison-grid">
      ${items.map((item) => `
        <div class="comparison-card">
          <div class="comparison-metric">${esc(item.metric)}</div>
          <div class="comparison-level">${esc(item.level)}</div>
          <div class="comparison-notes">${esc(item.notes)}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function buildFingerprintId(data) {
  const chemotype = String(data?.chemotype_identity || "UNCLASSIFIED")
    .toUpperCase()
    .replace(/[^A-Z0-9 ]/g, " ")
    .trim()
    .split(/\s+/)
    .slice(0, 3)
    .map((x) => x.slice(0, 3))
    .join("-");

  const thc = Math.round(Number(data?.fingerprint_radar?.thc_intensity ?? 0));
  const terp = Math.round(Number(data?.fingerprint_radar?.terpene_intensity ?? 0));
  const aroma = Math.round(Number(data?.fingerprint_radar?.aromatic_complexity ?? 0));

  return `${chemotype || "UNK"}-${thc}-${terp}-${aroma}`;
}

function renderRadarSVG(radar = {}) {
  const labels = [
    { key: "thc_intensity", label: "THC" },
    { key: "cbd_intensity", label: "CBD" },
    { key: "minor_cannabinoid_richness", label: "Minors" },
    { key: "terpene_intensity", label: "Terpenes" },
    { key: "terpene_diversity", label: "Diversity" },
    { key: "aromatic_complexity", label: "Aroma" }
  ];

  const size = 360;
  const cx = 180;
  const cy = 180;
  const levels = [20, 40, 60, 80, 100];

  function pointFor(index, value, radiusScale = 1) {
    const angle = (-Math.PI / 2) + (index * 2 * Math.PI / labels.length);
    const radius = (value / 100) * 115 * radiusScale;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    return { x, y };
  }

  const gridPolygons = levels.map((level) => {
    const points = labels.map((_, i) => {
      const p = pointFor(i, level);
      return `${p.x},${p.y}`;
    }).join(" ");
    return `<polygon points="${points}" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="1"/>`;
  }).join("");

  const axisLines = labels.map((_, i) => {
    const p = pointFor(i, 100, 1);
    return `<line x1="${cx}" y1="${cy}" x2="${p.x}" y2="${p.y}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>`;
  }).join("");

  const dataPoints = labels.map((item, i) => {
    const value = Number(radar?.[item.key] ?? 0);
    const p = pointFor(i, value);
    return `${p.x},${p.y}`;
  }).join(" ");

  const dots = labels.map((item, i) => {
    const value = Number(radar?.[item.key] ?? 0);
    const p = pointFor(i, value);
    return `<circle cx="${p.x}" cy="${p.y}" r="4" fill="#9bf19a" stroke="#07100f" stroke-width="2"/>`;
  }).join("");

  const labelNodes = labels.map((item, i) => {
    const p = pointFor(i, 100, 1.24);
    return `
      <text x="${p.x}" y="${p.y}" fill="rgba(255,255,255,0.82)" font-size="12" text-anchor="middle" dominant-baseline="middle">
        ${esc(item.label)}
      </text>
    `;
  }).join("");

  return `
    <div class="radar-wrap">
      <svg viewBox="0 0 ${size} ${size}" class="radar-svg" aria-label="Chemical signature radar chart">
        ${gridPolygons}
        ${axisLines}
        <polygon points="${dataPoints}" fill="rgba(155,241,154,0.18)" stroke="#9bf19a" stroke-width="2.2"/>
        ${dots}
        ${labelNodes}
        <circle cx="${cx}" cy="${cy}" r="2.5" fill="rgba(255,255,255,0.45)"/>
      </svg>
    </div>
  `;
}

function renderFingerprintStatRows(radar = {}) {
  const rows = [
    ["THC intensity", radar?.thc_intensity ?? 0],
    ["CBD intensity", radar?.cbd_intensity ?? 0],
    ["Minor richness", radar?.minor_cannabinoid_richness ?? 0],
    ["Terpene intensity", radar?.terpene_intensity ?? 0],
    ["Terpene diversity", radar?.terpene_diversity ?? 0],
    ["Aromatic complexity", radar?.aromatic_complexity ?? 0]
  ];

  return `
    <div class="fingerprint-list">
      ${rows.map(([label, value]) => `
        <div class="fingerprint-row">
          <div class="fingerprint-label">${esc(label)}</div>
          <div class="fingerprint-bar-wrap">
            <div class="fingerprint-bar">
              <span style="width:${Math.max(0, Math.min(100, Number(value) || 0))}%"></span>
            </div>
          </div>
          <div class="fingerprint-value">${esc(value)}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderTerpeneSpectrum(topTerpenes = []) {
  const items = getSafeArray(topTerpenes).slice(0, 8);
  if (!items.length) {
    return `<div class="copy">No terpene spectrum available.</div>`;
  }

  const total = items.reduce((sum, item) => sum + numberFromPercentString(item.value), 0) || 1;

  const palette = [
    "#9bf19a",
    "#83e7b9",
    "#67d8d4",
    "#79bff3",
    "#b4a2ff",
    "#f0ba6d",
    "#f59aa8",
    "#c9d48b"
  ];

  const bar = items.map((item, idx) => {
    const width = (numberFromPercentString(item.value) / total) * 100;
    return `<span style="width:${width}%; background:${palette[idx % palette.length]};"></span>`;
  }).join("");

  const legend = items.map((item, idx) => `
    <div class="spectrum-legend-item">
      <span class="spectrum-dot" style="background:${palette[idx % palette.length]};"></span>
      <span class="spectrum-name">${esc(item.name)}</span>
      <span class="spectrum-val">${esc(item.value)} ${esc(item.unit || "")}</span>
    </div>
  `).join("");

  return `
    <div class="spectrum-bar">${bar}</div>
    <div class="spectrum-legend">${legend}</div>
  `;
}

function renderDivider() {
  return `<div class="divider"></div>`;
}

function renderReportHTML(data) {
  const topCannabinoids = getSafeArray(data?.top_cannabinoids);
  const topTerpenes = getSafeArray(data?.top_terpenes);
  const compliance = getSafeArray(data?.compliance_indicators);
  const positiveFlags = getSafeArray(data?.positive_flags);
  const warningFlags = getSafeArray(data?.warning_flags);
  const aromaProfile = getSafeArray(data?.aroma_profile);
  const wowHighlights = getSafeArray(data?.wow_highlights);
  const comparativeClassification = getSafeArray(data?.comparative_classification);

  const cannabinoidRows = renderTableRows(topCannabinoids, "cannabinoids");
  const terpeneRows = renderTableRows(topTerpenes, "terpenes");
  const complianceRows = renderTableRows(compliance, "compliance");
  const fingerprintId = buildFingerprintId(data);

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>${esc(data?.product_name || "COA Intelligence Report")}</title>
<style>
  :root{
    --bg:#09100f;
    --bg-soft:#0f1716;
    --panel:rgba(255,255,255,.025);
    --line:rgba(255,255,255,.07);
    --line-strong:rgba(255,255,255,.14);
    --text:#edf3ef;
    --muted:#97aca2;
    --soft:#d8e2dd;
    --green:#9bf19a;
    --green-deep:#6bcf72;
    --amber:#f0ba6d;
    --white:rgba(255,255,255,.98);
  }

  *{box-sizing:border-box}

  html, body{
    margin:0;
    padding:0;
    background:var(--bg);
    color:var(--text);
    font-family:Arial, Helvetica, sans-serif;
    -webkit-print-color-adjust:exact;
    print-color-adjust:exact;
  }

  body{
    background:#09100f;
  }

  .report{
    width:100%;
    padding:42px 58px 54px;
  }

  .hero{
    position:relative;
    overflow:hidden;
    padding:52px 48px 42px;
    margin-bottom:24px;
    border-radius:34px;
    background:
      radial-gradient(circle at 18% 20%, rgba(155,241,154,.13), transparent 26%),
      radial-gradient(circle at 82% 14%, rgba(240,186,109,.10), transparent 18%),
      linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
    border:1px solid var(--line-strong);
    box-shadow:0 10px 44px rgba(0,0,0,.18);
  }

  .hero-top{
    display:grid;
    grid-template-columns:1.35fr .85fr;
    gap:28px;
    align-items:start;
    margin-bottom:24px;
  }

  .brand{
    font-size:11px;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:var(--muted);
    margin-bottom:12px;
  }

  .eyebrow{
    display:inline-block;
    padding:7px 12px;
    border-radius:999px;
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.08);
    color:var(--soft);
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:.14em;
    margin-bottom:18px;
  }

  h1{
    margin:0 0 10px;
    font-size:56px;
    line-height:1;
    letter-spacing:-0.05em;
    color:var(--white);
  }

  h2{
    margin:0 0 12px;
    font-size:22px;
    line-height:1.15;
    letter-spacing:-.02em;
    color:var(--white);
  }

  .subhead{
    font-size:17px;
    color:var(--muted);
    line-height:1.72;
    max-width:760px;
  }

  .hero-product-sheet{
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.08);
    border-radius:22px;
    padding:18px;
  }

  .hero-sheet-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px;
  }

  .hero-sheet-item{
    padding:10px 0;
    border-bottom:1px solid rgba(255,255,255,.06);
  }

  .hero-sheet-item:last-child,
  .hero-sheet-item:nth-last-child(2){
    border-bottom:none;
  }

  .sheet-k{
    font-size:11px;
    letter-spacing:.1em;
    text-transform:uppercase;
    color:var(--muted);
    margin-bottom:6px;
  }

  .sheet-v{
    font-size:14px;
    color:var(--white);
    font-weight:700;
    line-height:1.45;
  }

  .chemotype-strip{
    display:grid;
    grid-template-columns:1.15fr .85fr;
    gap:20px;
    margin-bottom:22px;
  }

  .chemotype-panel,
  .descriptor-panel{
    background:rgba(255,255,255,.025);
    border:1px solid rgba(255,255,255,.07);
    border-radius:22px;
    padding:18px 18px 16px;
  }

  .section-kicker{
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:.14em;
    color:var(--muted);
    margin-bottom:10px;
  }

  .chemotype-title{
    font-size:30px;
    font-weight:800;
    line-height:1.08;
    margin-bottom:10px;
    color:var(--white);
  }

  .chemotype-copy{
    font-size:15px;
    line-height:1.82;
    color:#dbe6e0;
  }

  .metrics{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
  }

  .metric{
    padding:16px;
    border:1px solid rgba(255,255,255,.07);
    border-radius:18px;
    background:rgba(255,255,255,.03);
  }

  .label{
    font-size:11px;
    text-transform:uppercase;
    color:var(--muted);
    letter-spacing:.1em;
    margin-bottom:8px;
  }

  .value{
    font-size:24px;
    font-weight:800;
    color:var(--white);
  }

  .hero-metric .value{
    font-size:31px;
    letter-spacing:-.03em;
    line-height:1.1;
  }

  .section-band,
  .story-block,
  .flow-section{
    margin-bottom:8px;
  }

  .badge-row{
    display:flex;
    flex-wrap:wrap;
    gap:10px;
  }

  .badge{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    padding:9px 13px;
    border-radius:999px;
    font-size:12px;
    font-weight:700;
    border:1px solid rgba(255,255,255,.06);
    background:rgba(255,255,255,.03);
    color:var(--soft);
  }

  .badge.good{
    background:rgba(155,241,154,.11);
    color:var(--green);
  }

  .badge.warn{
    background:rgba(240,186,109,.12);
    color:var(--amber);
  }

  .badge.neutral{
    background:rgba(255,255,255,.04);
    color:#dce7e1;
  }

  .muted-inline{
    color:var(--muted);
    font-size:13px;
  }

  .story-block{
    padding:4px 0;
  }

  .story-quote{
    font-size:21px;
    line-height:1.84;
    color:var(--white);
    max-width:1020px;
    margin-bottom:16px;
  }

  .story-support{
    font-size:15px;
    line-height:1.88;
    color:#d8e3dd;
  }

  .divider{
    height:1px;
    background:rgba(255,255,255,.08);
    margin:18px 0;
  }

  .two-col{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:26px;
  }

  .three-col{
    display:grid;
    grid-template-columns:1fr 1fr 1fr;
    gap:18px;
  }

  .card{
    padding:4px 0;
  }

  .copy{
    font-size:15px;
    line-height:1.88;
    color:#dbe6e0;
  }

  .copy strong{
    color:var(--white);
  }

  .meta{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:10px;
  }

  .meta-item{
    padding:12px 0 14px;
    border:none;
    border-bottom:1px solid rgba(255,255,255,.08);
    background:transparent;
  }

  .meta-label{
    font-size:11px;
    text-transform:uppercase;
    color:var(--muted);
    letter-spacing:.08em;
    margin-bottom:6px;
  }

  .meta-value{
    font-size:14px;
    font-weight:700;
    line-height:1.45;
    color:var(--white);
  }

  table{
    width:100%;
    border-collapse:collapse;
    font-size:13px;
    margin-top:12px;
  }

  th, td{
    text-align:left;
    padding:11px 8px;
    vertical-align:top;
  }

  th{
    font-size:11px;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:.08em;
    border-bottom:1px solid rgba(255,255,255,.2);
  }

  td{
    border-bottom:1px solid rgba(255,255,255,.06);
  }

  .spectrum-bar{
    display:flex;
    width:100%;
    height:16px;
    border-radius:999px;
    overflow:hidden;
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.06);
    margin:10px 0 16px;
  }

  .spectrum-bar span{
    display:block;
    height:100%;
  }

  .spectrum-legend{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px 18px;
  }

  .spectrum-legend-item{
    display:flex;
    align-items:center;
    gap:8px;
    font-size:13px;
    line-height:1.5;
  }

  .spectrum-dot{
    width:10px;
    height:10px;
    border-radius:50%;
    flex:0 0 auto;
  }

  .spectrum-name{
    color:var(--white);
    font-weight:700;
  }

  .spectrum-val{
    color:var(--muted);
    margin-left:auto;
  }

  .comparison-grid{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:12px;
    margin-top:8px;
  }

  .comparison-card{
    border:1px solid rgba(255,255,255,.06);
    background:rgba(255,255,255,.02);
    border-radius:18px;
    padding:16px;
  }

  .comparison-metric{
    font-size:11px;
    text-transform:uppercase;
    color:var(--muted);
    letter-spacing:.08em;
    margin-bottom:8px;
  }

  .comparison-level{
    font-size:24px;
    font-weight:800;
    margin-bottom:8px;
    color:var(--white);
  }

  .comparison-notes{
    font-size:13px;
    line-height:1.65;
    color:#d8e3dd;
  }

  .signature-wrap{
    display:grid;
    grid-template-columns:420px 1fr;
    gap:28px;
    align-items:start;
    margin-top:8px;
  }

  .fingerprint-id{
    display:inline-block;
    margin-top:12px;
    padding:9px 12px;
    border-radius:999px;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(255,255,255,.08);
    color:var(--soft);
    font-size:12px;
    font-weight:700;
    letter-spacing:.12em;
    text-transform:uppercase;
  }

  .radar-wrap{
    width:100%;
    max-width:420px;
  }

  .radar-svg{
    width:100%;
    height:auto;
    display:block;
  }

  .fingerprint-list{
    display:flex;
    flex-direction:column;
    gap:14px;
    margin-top:8px;
  }

  .fingerprint-row{
    display:grid;
    grid-template-columns:150px 1fr 42px;
    gap:12px;
    align-items:center;
  }

  .fingerprint-label{
    font-size:13px;
    color:var(--soft);
  }

  .fingerprint-bar{
    width:100%;
    height:14px;
    border-radius:999px;
    background:rgba(255,255,255,.06);
    overflow:hidden;
    border:1px solid rgba(255,255,255,.04);
  }

  .fingerprint-bar span{
    display:block;
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg, var(--green-deep), var(--green));
  }

  .fingerprint-value{
    font-size:13px;
    font-weight:800;
    text-align:right;
    color:var(--white);
  }

  .footer{
    margin-top:18px;
    font-size:11px;
    color:#8ea19a;
    line-height:1.75;
    text-align:center;
    padding:12px 0 0;
  }

  @page{
    margin:0;
  }
</style>
</head>
<body>
  <div class="report">

    <section class="hero">
      <div class="hero-top">
        <div>
          <div class="brand">Alem Solutions · COA Intelligence Report</div>
          <div class="eyebrow">Chemical Intelligence Dossier</div>
          <h1>${esc(data?.product_name || "Cannabis Product")}</h1>
          <div class="subhead">
            ${esc(data?.executive_summary || "A premium interpretive layer derived from certificate of analysis data.")}
          </div>
        </div>

        <div class="hero-product-sheet">
          <div class="section-kicker">Luxury product sheet</div>
          <div class="hero-sheet-grid">
            <div class="hero-sheet-item">
              <div class="sheet-k">Chemotype</div>
              <div class="sheet-v">${esc(data?.chemotype_identity || "Not reported")}</div>
            </div>
            <div class="hero-sheet-item">
              <div class="sheet-k">Descriptor</div>
              <div class="sheet-v">${esc(data?.chemotype_descriptor || "Not reported")}</div>
            </div>
            <div class="hero-sheet-item">
              <div class="sheet-k">Batch</div>
              <div class="sheet-v">${esc(data?.batch_number || "Not reported")}</div>
            </div>
            <div class="hero-sheet-item">
              <div class="sheet-k">Lab</div>
              <div class="sheet-v">${esc(data?.laboratory_name || "Not reported")}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="chemotype-strip">
        <div class="chemotype-panel">
          <div class="section-kicker">Chemotype identity</div>
          <div class="chemotype-title">${esc(data?.chemotype_identity || data?.overall_score || "Profile not classified")}</div>
          <div class="chemotype-copy">${esc(data?.opening_statement || "No opening statement available.")}</div>
        </div>

        <div class="descriptor-panel">
          <div class="section-kicker">Profile descriptor</div>
          <div class="chemotype-title" style="font-size:21px;">${esc(data?.chemotype_descriptor || "No descriptor available.")}</div>
          <div class="chemotype-copy">${esc(data?.aromatic_profile_summary || "No aromatic profile summary available.")}</div>
        </div>
      </div>

      ${renderMetricCards(data)}
    </section>

    <section class="story-block">
      <div class="section-kicker">Chemical story</div>
      <div class="story-quote">${esc(data?.chemical_story || data?.opening_statement || "No chemical story available.")}</div>
      <div class="story-support"><strong>Overall profile:</strong> ${esc(data?.overall_score || "Not reported")}</div>
    </section>

    ${renderDivider()}

    <section class="section-band">
      <div class="section-kicker">WOW highlights</div>
      <div class="badge-row">${renderBadges(wowHighlights, "good")}</div>
    </section>

    ${renderDivider()}

    <section class="section-band">
      <div class="section-kicker">Aroma profile</div>
      <div class="badge-row">${renderBadges(aromaProfile, "neutral")}</div>
    </section>

    ${renderDivider()}

    <section class="flow-section">
      <div class="two-col">
        <div class="card">
          <h2>Interpretive summary</h2>
          <div class="copy"><strong>Cannabinoid architecture:</strong> ${esc(data?.cannabinoid_architecture || "Not reported")}</div>
          <div class="copy" style="margin-top:14px;"><strong>Terpene architecture:</strong> ${esc(data?.terpene_architecture || "Not reported")}</div>
          <div class="copy" style="margin-top:14px;"><strong>Minor cannabinoids:</strong> ${esc(data?.minor_cannabinoids || "Not reported")}</div>
        </div>

        <div class="card">
          <h2>Batch & document details</h2>
          ${renderMetaGrid(data)}
        </div>
      </div>
    </section>

    ${renderDivider()}

    <section class="flow-section">
      <div class="two-col">
        <div class="card">
          <h2>Cannabinoid architecture</h2>
          <div class="copy">${esc(data?.cannabinoid_architecture || "Not reported")}</div>
          <table>
            <thead>
              <tr><th>Name</th><th>Value</th><th>Unit</th><th>Notes</th></tr>
            </thead>
            <tbody>
              ${cannabinoidRows}
            </tbody>
          </table>
        </div>

        <div class="card">
          <h2>Aromatic map</h2>
          <div class="copy">${esc(data?.terpene_architecture || "Not reported")}</div>
          ${renderTerpeneSpectrum(topTerpenes)}
          <table>
            <thead>
              <tr><th>Name</th><th>Value</th><th>Unit</th></tr>
            </thead>
            <tbody>
              ${terpeneRows}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    ${renderDivider()}

    <section class="flow-section">
      <div class="two-col">
        <div class="card">
          <h2>Cultivation & post-harvest signals</h2>
          <div class="copy"><strong>Cultivation insights:</strong> ${esc(data?.cultivation_insights || "Not reported")}</div>
          <div class="copy" style="margin-top:14px;"><strong>Entourage effect:</strong> ${esc(data?.entourage_effect || "Not reported")}</div>
          <div class="copy" style="margin-top:14px;"><strong>Therapeutic potential:</strong> ${esc(data?.therapeutic_potential || "Not reported")}</div>
        </div>

        <div class="card">
          <h2>Experience profile</h2>
          <div class="copy">${esc(data?.experience_profile || "Not reported")}</div>
          <h2 style="margin-top:18px;">Best use cases</h2>
          <div class="copy">${esc(data?.best_use_cases || "Not reported")}</div>
        </div>
      </div>
    </section>

    ${renderDivider()}

    <section class="flow-section">
      <div class="section-kicker">Positive and cautionary signals</div>
      <div class="two-col">
        <div class="card">
          <h2>Positive flags</h2>
          <div class="badge-row">${renderBadges(positiveFlags, "good")}</div>
        </div>

        <div class="card">
          <h2>Watchouts</h2>
          <div class="badge-row">${renderBadges(warningFlags, "warn")}</div>
        </div>
      </div>
    </section>

    ${renderDivider()}

    <section class="flow-section">
      <h2>Chemical signature</h2>
      <div class="signature-wrap">
        <div>
          ${renderRadarSVG(data?.fingerprint_radar || {})}
          <div class="fingerprint-id">Fingerprint ID · ${esc(fingerprintId)}</div>
        </div>
        <div>
          ${renderFingerprintStatRows(data?.fingerprint_radar || {})}
        </div>
      </div>
    </section>

    ${renderDivider()}

    <section class="flow-section">
      <h2>Comparative classification</h2>
      ${renderComparisonCards(comparativeClassification)}
    </section>

    ${renderDivider()}

    <section class="flow-section">
      <div class="two-col">
        <div class="card">
          <h2>Lab quality & compliance</h2>
          <div class="copy"><strong>Lab quality summary:</strong> ${esc(data?.lab_quality_summary || "Not reported")}</div>
          <div class="copy" style="margin-top:14px;"><strong>Contaminant overview:</strong> ${esc(data?.contaminant_overview || "Not reported")}</div>
          <table style="margin-top:14px;">
            <thead>
              <tr><th>Category</th><th>Status</th><th>Notes</th></tr>
            </thead>
            <tbody>
              ${complianceRows}
            </tbody>
          </table>
        </div>

        <div class="card">
          <h2>Scientific literature notes</h2>
          <div class="copy">${esc(data?.scientific_references || "Not reported")}</div>

          <h2 style="margin-top:18px;">Reader snapshot</h2>
          <div class="copy"><strong>Chemotype:</strong> ${esc(data?.chemotype_identity || "Not reported")}</div>
          <div class="copy" style="margin-top:12px;"><strong>Descriptor:</strong> ${esc(data?.chemotype_descriptor || "Not reported")}</div>
          <div class="copy" style="margin-top:12px;"><strong>Aroma summary:</strong> ${esc(data?.aromatic_profile_summary || "Not reported")}</div>
        </div>
      </div>
    </section>

    <div class="footer">
      This report is an educational interpretive layer based on certificate of analysis data and does not replace physician advice, pharmacist counselling, regulatory review, or direct laboratory confirmation.
    </div>
  </div>
</body>
</html>
`;
}

function parseIncomingBody(rawBody) {
  if (rawBody === undefined || rawBody === null) {
    throw new Error("Request body is empty");
  }

  const trimmed = String(rawBody).trim();

  if (!trimmed || trimmed === "null" || trimmed === "undefined") {
    throw new Error("Request body is empty or null");
  }

  let payload;
  try {
    payload = JSON.parse(trimmed);
  } catch (err) {
    throw new Error(`Request body is not valid JSON: ${err.message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Parsed JSON is null or not a valid object");
  }

  if (payload.report_json && typeof payload.report_json === "object") {
    return {
      fileName: sanitizeFileName(payload.file_name || payload.report_json?.product_name || `coa-${Date.now()}`) + ".pdf",
      data: payload.report_json
    };
  }

  if (payload.report_json_string && typeof payload.report_json_string === "string") {
    let parsedInner;
    try {
      parsedInner = JSON.parse(payload.report_json_string);
    } catch (err) {
      throw new Error(`report_json_string is not valid JSON: ${err.message}`);
    }

    if (!parsedInner || typeof parsedInner !== "object" || Array.isArray(parsedInner)) {
      throw new Error("report_json_string parsed to null or invalid object");
    }

    return {
      fileName: sanitizeFileName(payload.file_name || parsedInner?.product_name || `coa-${Date.now()}`) + ".pdf",
      data: parsedInner
    };
  }

  if (payload.product_name || payload.top_cannabinoids || payload.fingerprint_radar) {
    return {
      fileName: sanitizeFileName(payload.file_name || payload.product_name || `coa-${Date.now()}`) + ".pdf",
      data: payload
    };
  }

  throw new Error("Missing valid report_json or report_json_string");
}

app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Middleware is running"
  });
});

app.get("/health", async (req, res) => {
  try {
    const { error } = await supabase.from("documents").select("id").limit(1);

    res.json({
      success: true,
      status: "ok",
      hasSupabaseUrl: !!process.env.SUPABASE_URL,
      hasSupabaseKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
      bucket: process.env.SUPABASE_BUCKET || null,
      db_connected: !error,
      db_error: error ? error.message : null
    });
  } catch (err) {
    res.json({
      success: false,
      status: "error",
      message: err.message
    });
  }
});

app.post("/generate-report", async (req, res) => {
  let browser;

  try {
    console.log("STEP 1: request received");
    console.log("RAW BODY PREVIEW:");
    console.log(String(req.body).slice(0, 3000));

    const { data, fileName } = parseIncomingBody(req.body);

    console.log("STEP 2: rendering HTML");
    const html = renderReportHTML(data);

    console.log("STEP 3: launching puppeteer");
    browser = await puppeteer.launch({
      headless: "new",
      args: ["--no-sandbox", "--disable-setuid-sandbox"]
    });

    console.log("STEP 4: opening new page");
    const page = await browser.newPage();

    await page.setViewport({
      width: 1400,
      height: 2000,
      deviceScaleFactor: 1
    });

    console.log("STEP 5: setting HTML content");
    await page.setContent(html, { waitUntil: "networkidle0" });
    await page.emulateMediaType("screen");

    await page.addStyleTag({
      content: `
        html, body {
          height: auto !important;
          overflow: visible !important;
        }
      `
    });

    const pageHeight = await page.evaluate(() => {
      const body = document.body;
      const html = document.documentElement;
      return Math.max(
        body.scrollHeight,
        body.offsetHeight,
        html.clientHeight,
        html.scrollHeight,
        html.offsetHeight
      );
    });

    const pdfHeight = Math.max(pageHeight + 30, 2200);

    console.log("STEP 6: generating PDF");
    const pdfBuffer = await page.pdf({
      printBackground: true,
      width: "1280px",
      height: `${pdfHeight}px`,
      margin: {
        top: "0px",
        right: "0px",
        bottom: "0px",
        left: "0px"
      },
      pageRanges: "1"
    });

    console.log("STEP 7: uploading PDF to Supabase");
    const { error: uploadError } = await supabase.storage
      .from(process.env.SUPABASE_BUCKET)
      .upload(fileName, pdfBuffer, {
        contentType: "application/pdf",
        upsert: true
      });

    if (uploadError) {
      throw uploadError;
    }

    const { data: publicUrlData } = supabase.storage
      .from(process.env.SUPABASE_BUCKET)
      .getPublicUrl(fileName);

    const pdfUrl = publicUrlData?.publicUrl;

    if (!pdfUrl) {
      throw new Error("Could not generate public PDF URL");
    }

    console.log("STEP 8: closing browser");
    await browser.close();
    browser = null;

    console.log("STEP 9: sending response with pdf_url");
    return res.json({
      success: true,
      file_name: fileName,
      pdf_url: pdfUrl
    });

  } catch (error) {
    console.error("ERROR IN /generate-report:");
    console.error(error);

    if (browser) {
      try {
        await browser.close();
      } catch (closeErr) {
        console.error("ERROR CLOSING BROWSER:", closeErr);
      }
    }

    return res.status(500).json({
      success: false,
      error: error.message || "Unknown server error"
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});