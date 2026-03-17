const express = require("express");
const cors = require("cors");
const puppeteer = require("puppeteer");
const { createClient } = require("@supabase/supabase-js");

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
        <div class="label">Report Confidence</div>
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

function renderRadarStats(radar = {}) {
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

function renderReportHTML(data) {
  const topCannabinoids = Array.isArray(data?.top_cannabinoids) ? data.top_cannabinoids : [];
  const topTerpenes = Array.isArray(data?.top_terpenes) ? data.top_terpenes : [];
  const compliance = Array.isArray(data?.compliance_indicators) ? data.compliance_indicators : [];
  const positiveFlags = Array.isArray(data?.positive_flags) ? data.positive_flags : [];
  const warningFlags = Array.isArray(data?.warning_flags) ? data.warning_flags : [];
  const aromaProfile = Array.isArray(data?.aroma_profile) ? data.aroma_profile : [];
  const wowHighlights = Array.isArray(data?.wow_highlights) ? data.wow_highlights : [];
  const comparativeClassification = Array.isArray(data?.comparative_classification)
    ? data.comparative_classification
    : [];

  const cannabinoidRows = renderTableRows(topCannabinoids, "cannabinoids");
  const terpeneRows = renderTableRows(topTerpenes, "terpenes");
  const complianceRows = renderTableRows(compliance, "compliance");

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
    --panel:rgba(255,255,255,.035);
    --panel-2:rgba(255,255,255,.025);
    --line:rgba(255,255,255,.08);
    --line-strong:rgba(255,255,255,.14);
    --text:#edf3ef;
    --muted:#99ada4;
    --soft:#d7e2dc;
    --green:#9bf19a;
    --green-deep:#6bcf72;
    --amber:#f0ba6d;
    --white:rgba(255,255,255,.98);
  }

  *{box-sizing:border-box}

  body{
    margin:0;
    padding:0;
    font-family:Arial, Helvetica, sans-serif;
    background:
      radial-gradient(circle at top left, rgba(107,207,114,.08), transparent 28%),
      radial-gradient(circle at top right, rgba(240,186,109,.06), transparent 22%),
      var(--bg);
    color:var(--text);
  }

  .report{
    max-width:1100px;
    margin:0 auto;
  }

  .hero,
  .card,
  .statement,
  .section-band{
    background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02));
    border:1px solid var(--line);
    border-radius:26px;
    box-shadow:0 10px 40px rgba(0,0,0,.18);
  }

  .hero{
    padding:34px;
    margin-bottom:20px;
    position:relative;
    overflow:hidden;
  }

  .hero::after{
    content:"";
    position:absolute;
    inset:auto -80px -90px auto;
    width:260px;
    height:260px;
    border-radius:50%;
    background:radial-gradient(circle, rgba(155,241,154,.09), transparent 65%);
    pointer-events:none;
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
    border:1px solid var(--line);
    color:var(--soft);
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:.14em;
    margin-bottom:16px;
  }

  h1{
    margin:0 0 8px;
    font-size:42px;
    line-height:1.02;
    letter-spacing:-.03em;
    color:var(--white);
  }

  .subhead{
    font-size:15px;
    color:var(--muted);
    line-height:1.6;
    margin-bottom:20px;
    max-width:760px;
  }

  .chemotype-block{
    display:grid;
    grid-template-columns:1.3fr .9fr;
    gap:18px;
    margin-bottom:18px;
  }

  .chemotype-card,
  .descriptor-card{
    background:rgba(255,255,255,.035);
    border:1px solid var(--line);
    border-radius:20px;
    padding:18px;
  }

  .section-kicker{
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:.14em;
    color:var(--muted);
    margin-bottom:10px;
  }

  .chemotype-title{
    font-size:26px;
    font-weight:800;
    line-height:1.12;
    margin-bottom:8px;
  }

  .chemotype-copy{
    font-size:14px;
    line-height:1.7;
    color:#dbe6e0;
  }

  .metrics{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin-top:8px;
  }

  .metric{
    padding:16px;
    border:1px solid var(--line);
    border-radius:18px;
    background:rgba(255,255,255,.035);
  }

  .hero-metric .value{
    font-size:30px;
    font-weight:800;
    line-height:1.1;
    letter-spacing:-.03em;
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
  }

  .band{
    margin-bottom:20px;
  }

  .section-band{
    padding:18px 20px;
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
    border:1px solid var(--line);
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

  .statement{
    padding:24px;
    margin-bottom:20px;
  }

  .statement .quote{
    font-size:18px;
    line-height:1.7;
    color:var(--white);
    margin-bottom:16px;
  }

  .quote-secondary{
    font-size:14px;
    line-height:1.8;
    color:#d8e3dd;
  }

  .grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:18px;
    margin-bottom:20px;
  }

  .card{
    padding:24px;
  }

  h2{
    margin:0 0 14px;
    font-size:19px;
    letter-spacing:-.02em;
  }

  .copy{
    font-size:14px;
    line-height:1.75;
    color:#dbe6e0;
  }

  .copy strong{
    color:var(--white);
  }

  .meta{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px;
  }

  .meta-item{
    padding:12px;
    border:1px solid var(--line);
    border-radius:16px;
    background:rgba(255,255,255,.03);
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
    font-weight:600;
    line-height:1.45;
  }

  table{
    width:100%;
    border-collapse:collapse;
    font-size:13px;
  }

  th, td{
    text-align:left;
    padding:10px 8px;
    border-bottom:1px solid rgba(255,255,255,.06);
    vertical-align:top;
  }

  th{
    font-size:11px;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:.08em;
  }

  .story-columns{
    display:grid;
    grid-template-columns:1.15fr .85fr;
    gap:18px;
    margin-bottom:20px;
  }

  .comparison-grid{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:12px;
  }

  .comparison-card{
    border:1px solid var(--line);
    background:rgba(255,255,255,.03);
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
    line-height:1.6;
    color:#d8e3dd;
  }

  .fingerprint-list{
    display:flex;
    flex-direction:column;
    gap:12px;
  }

  .fingerprint-row{
    display:grid;
    grid-template-columns:140px 1fr 38px;
    gap:10px;
    align-items:center;
  }

  .fingerprint-label{
    font-size:12px;
    color:var(--soft);
  }

  .fingerprint-bar{
    position:relative;
    width:100%;
    height:10px;
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
  }

  .footer{
    margin-top:20px;
    font-size:11px;
    color:#8ea19a;
    line-height:1.75;
    text-align:center;
    padding:10px 20px;
  }

  @page{
  margin:0;
   }
</style>
</head>
<body>
  <div class="report">

    <section class="hero">
      <div class="brand">Alem Solutions · COA Intelligence Report</div>
      <div class="eyebrow">ULTRA PREMIUM TEST VERSION</div>
      <h1>${esc(data?.product_name || "Cannabis Product")}</h1>
      <div class="subhead">
        ${esc(data?.executive_summary || "A premium interpretive layer derived from certificate of analysis data.")}
      </div>

      <div class="chemotype-block">
        <div class="chemotype-card">
          <div class="section-kicker">Chemotype identity</div>
          <div class="chemotype-title">${esc(data?.chemotype_identity || data?.overall_score || "Profile not classified")}</div>
          <div class="chemotype-copy">${esc(data?.opening_statement || "No opening statement available.")}</div>
        </div>

        <div class="descriptor-card">
          <div class="section-kicker">Profile descriptor</div>
          <div class="chemotype-title" style="font-size:20px;">${esc(data?.chemotype_descriptor || "No descriptor available.")}</div>
          <div class="chemotype-copy">${esc(data?.aromatic_profile_summary || "No aromatic profile summary available.")}</div>
        </div>
      </div>

      ${renderMetricCards(data)}
    </section>

    <section class="band section-band">
      <div class="section-kicker">Aroma profile</div>
      <div class="badge-row">${renderBadges(aromaProfile, "neutral")}</div>
    </section>

    <section class="band section-band">
      <div class="section-kicker">WOW highlights</div>
      <div class="badge-row">${renderBadges(wowHighlights, "good")}</div>
    </section>

    <section class="statement">
      <div class="section-kicker">Chemical story</div>
      <div class="quote">${esc(data?.chemical_story || data?.opening_statement || "No chemical story available.")}</div>
      <div class="quote-secondary"><strong>Overall profile:</strong> ${esc(data?.overall_score || "Not reported")}</div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Batch & document details</h2>
        ${renderMetaGrid(data)}
      </div>

      <div class="card">
        <h2>Interpretive summary</h2>
        <div class="copy"><strong>Cannabinoid architecture:</strong> ${esc(data?.cannabinoid_architecture || "Not reported")}</div>
        <div class="copy" style="margin-top:14px;"><strong>Terpene architecture:</strong> ${esc(data?.terpene_architecture || "Not reported")}</div>
        <div class="copy" style="margin-top:14px;"><strong>Minor cannabinoids:</strong> ${esc(data?.minor_cannabinoids || "Not reported")}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Positive flags</h2>
        <div class="badge-row">${renderBadges(positiveFlags, "good")}</div>
      </div>

      <div class="card">
        <h2>Watchouts</h2>
        <div class="badge-row">${renderBadges(warningFlags, "warn")}</div>
      </div>
    </section>

    <section class="story-columns">
      <div class="card">
        <h2>Cannabinoid architecture</h2>
        <div class="copy" style="margin-bottom:14px;">${esc(data?.cannabinoid_architecture || "Not reported")}</div>
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
        <div class="copy" style="margin-bottom:14px;">${esc(data?.terpene_architecture || "Not reported")}</div>
        <table>
          <thead>
            <tr><th>Name</th><th>Value</th><th>Unit</th></tr>
          </thead>
          <tbody>
            ${terpeneRows}
          </tbody>
        </table>
      </div>
    </section>

    <section class="grid">
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
        <h2>Cultivation & post-harvest signals</h2>
        <div class="copy"><strong>Cultivation insights:</strong> ${esc(data?.cultivation_insights || "Not reported")}</div>
        <div class="copy" style="margin-top:14px;"><strong>Entourage effect:</strong> ${esc(data?.entourage_effect || "Not reported")}</div>
        <div class="copy" style="margin-top:14px;"><strong>Therapeutic potential:</strong> ${esc(data?.therapeutic_potential || "Not reported")}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Experience profile</h2>
        <div class="copy">${esc(data?.experience_profile || "Not reported")}</div>
        <h2 style="margin-top:18px;">Best use cases</h2>
        <div class="copy">${esc(data?.best_use_cases || "Not reported")}</div>
      </div>

      <div class="card">
        <h2>Chemical fingerprint</h2>
        ${renderRadarStats(data?.fingerprint_radar || {})}
      </div>
    </section>

    <section class="card" style="margin-bottom:20px;">
      <h2>Comparative classification</h2>
      ${renderComparisonCards(comparativeClassification)}
    </section>

    <section class="grid">
      <div class="card">
        <h2>Scientific literature notes</h2>
        <div class="copy">${esc(data?.scientific_references || "Not reported")}</div>
      </div>

      <div class="card">
        <h2>Reader snapshot</h2>
        <div class="copy"><strong>Chemotype:</strong> ${esc(data?.chemotype_identity || "Not reported")}</div>
        <div class="copy" style="margin-top:12px;"><strong>Descriptor:</strong> ${esc(data?.chemotype_descriptor || "Not reported")}</div>
        <div class="copy" style="margin-top:12px;"><strong>Aroma summary:</strong> ${esc(data?.aromatic_profile_summary || "Not reported")}</div>
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

app.get("/health", (req, res) => {
  res.json({
    success: true,
    status: "ok",
    hasSupabaseUrl: !!process.env.SUPABASE_URL,
    hasSupabaseKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
    bucket: process.env.SUPABASE_BUCKET || null
  });
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

    console.log("STEP 5: setting HTML content");
    await page.setContent(html, { waitUntil: "networkidle0" });

    console.log("STEP 6: generating PDF");
    const pdfBuffer = await page.pdf({
      width: "1100px",
      height: "auto",
      printBackground: true,
      margin: {
        top: "0px",
        right: "0px",
        bottom: "0px",
        left: "0px"
      }
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