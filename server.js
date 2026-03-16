const express = require("express");
const cors = require("cors");
const puppeteer = require("puppeteer");

const app = express();

app.use(cors());
app.use(express.text({ type: "*/*", limit: "50mb" }));

function esc(v = "") {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderReportHTML(data) {
  const topCannabinoids = Array.isArray(data?.top_cannabinoids) ? data.top_cannabinoids : [];
  const topTerpenes = Array.isArray(data?.top_terpenes) ? data.top_terpenes : [];
  const compliance = Array.isArray(data?.compliance_indicators) ? data.compliance_indicators : [];
  const positiveFlags = Array.isArray(data?.positive_flags) ? data.positive_flags : [];
  const warningFlags = Array.isArray(data?.warning_flags) ? data.warning_flags : [];

  const cannabinoidRows = topCannabinoids.map(c => `
    <tr>
      <td>${esc(c.name)}</td>
      <td>${esc(c.value)}</td>
      <td>${esc(c.unit)}</td>
      <td>${esc(c.notes)}</td>
    </tr>
  `).join("");

  const terpeneRows = topTerpenes.map(t => `
    <tr>
      <td>${esc(t.name)}</td>
      <td>${esc(t.value)}</td>
      <td>${esc(t.unit)}</td>
    </tr>
  `).join("");

  const complianceRows = compliance.map(c => `
    <tr>
      <td>${esc(c.label)}</td>
      <td>${esc(c.status)}</td>
      <td>${esc(c.notes)}</td>
    </tr>
  `).join("");

  const goodBadges = positiveFlags.map(x => `<span class="badge good">${esc(x)}</span>`).join("");
  const warnBadges = warningFlags.map(x => `<span class="badge warn">${esc(x)}</span>`).join("");

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>${esc(data?.product_name || "COA Report")}</title>
<style>
  :root{
    --bg:#0b1110;
    --line:rgba(255,255,255,.08);
    --text:#edf3ef;
    --muted:#9fb0a7;
    --green:#95f08e;
    --amber:#f0b562;
  }
  *{box-sizing:border-box}
  body{
    margin:0;
    padding:24px;
    font-family:Arial, Helvetica, sans-serif;
    background:var(--bg);
    color:var(--text);
  }
  .report{max-width:1100px;margin:0 auto}
  .hero,.card{
    background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
    border:1px solid var(--line);
    border-radius:22px;
  }
  .hero{padding:30px;margin-bottom:18px}
  .brand{
    font-size:12px;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:var(--muted);
    margin-bottom:14px;
  }
  h1{margin:0 0 10px;font-size:34px;line-height:1.05}
  h2{margin:0 0 14px;font-size:18px}
  .lead,.copy{
    font-size:14px;
    line-height:1.75;
    color:#dde7e1;
  }
  .metrics{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin-top:18px;
  }
  .metric{
    padding:16px;
    border:1px solid var(--line);
    border-radius:16px;
    background:rgba(255,255,255,.03);
  }
  .label{
    font-size:11px;
    text-transform:uppercase;
    color:var(--muted);
    letter-spacing:.08em;
    margin-bottom:8px;
  }
  .value{
    font-size:24px;
    font-weight:800;
  }
  .grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:18px;
    margin-bottom:18px;
  }
  .card{padding:22px}
  .meta{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px;
  }
  .meta-item{
    padding:12px;
    border:1px solid var(--line);
    border-radius:14px;
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
  th,td{
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
  .badge-row{
    display:flex;
    flex-wrap:wrap;
    gap:10px;
  }
  .badge{
    display:inline-block;
    padding:9px 12px;
    border-radius:999px;
    font-size:12px;
    font-weight:700;
    border:1px solid var(--line);
  }
  .badge.good{
    background:rgba(149,240,142,.12);
    color:var(--green);
  }
  .badge.warn{
    background:rgba(240,181,98,.12);
    color:var(--amber);
  }
  .radar{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:12px;
    margin-top:12px;
  }
  .radar-item{
    padding:14px;
    border-radius:14px;
    background:rgba(255,255,255,.03);
    border:1px solid var(--line);
  }
  .radar-label{
    font-size:11px;
    color:var(--muted);
    text-transform:uppercase;
    margin-bottom:8px;
  }
  .radar-value{
    font-size:22px;
    font-weight:800;
  }
  .footer{
    margin-top:18px;
    font-size:11px;
    color:#8fa099;
    line-height:1.7;
    text-align:center;
    padding:10px;
  }
  @page{
    size:A4;
    margin:14mm;
  }
</style>
</head>
<body>
  <div class="report">
    <section class="hero">
      <div class="brand">Alem Solutions · COA Intelligence Report</div>
      <h1>${esc(data?.product_name || "Cannabis Product")}</h1>
      <div class="lead">${esc(data?.opening_statement || "No opening statement available.")}</div>

      <div class="metrics">
        <div class="metric">
          <div class="label">Total THC</div>
          <div class="value">${esc(data?.thc_total || "Not reported")}</div>
        </div>
        <div class="metric">
          <div class="label">Total CBD</div>
          <div class="value">${esc(data?.cbd_total || "Not reported")}</div>
        </div>
        <div class="metric">
          <div class="label">Total Terpenes</div>
          <div class="value">${esc(data?.total_terpenes || "Not reported")}</div>
        </div>
        <div class="metric">
          <div class="label">Report Confidence</div>
          <div class="value">${esc(data?.report_confidence_score || "—")}</div>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Batch details</h2>
        <div class="meta">
          <div class="meta-item"><div class="meta-label">Batch</div><div class="meta-value">${esc(data?.batch_number || "Not reported")}</div></div>
          <div class="meta-item"><div class="meta-label">COA date</div><div class="meta-value">${esc(data?.coa_report_date || "Not reported")}</div></div>
          <div class="meta-item"><div class="meta-label">Product type</div><div class="meta-value">${esc(data?.product_type || "Not reported")}</div></div>
          <div class="meta-item"><div class="meta-label">Laboratory</div><div class="meta-value">${esc(data?.laboratory_name || "Not reported")}</div></div>
        </div>
        <div class="copy" style="margin-top:16px;">${esc(data?.executive_summary || "No executive summary available.")}</div>
      </div>

      <div class="card">
        <h2>Interpretive summary</h2>
        <div class="copy"><strong>Overall profile:</strong> ${esc(data?.overall_score || "Not reported")}</div>
        <div class="copy" style="margin-top:14px;"><strong>Minor cannabinoids:</strong> ${esc(data?.minor_cannabinoids || "Not reported")}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Positive flags</h2>
        <div class="badge-row">${goodBadges || '<span class="copy">None reported</span>'}</div>
      </div>

      <div class="card">
        <h2>Watchouts</h2>
        <div class="badge-row">${warnBadges || '<span class="copy">None reported</span>'}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Top cannabinoids</h2>
        <table>
          <thead>
            <tr><th>Name</th><th>Value</th><th>Unit</th><th>Notes</th></tr>
          </thead>
          <tbody>
            ${cannabinoidRows || '<tr><td colspan="4">No explicit cannabinoid rows available.</td></tr>'}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2>Top terpenes</h2>
        <table>
          <thead>
            <tr><th>Name</th><th>Value</th><th>Unit</th></tr>
          </thead>
          <tbody>
            ${terpeneRows || '<tr><td colspan="3">No explicit terpene rows available.</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Contaminant & compliance overview</h2>
        <div class="copy">${esc(data?.contaminant_overview || "Not reported")}</div>
        <table style="margin-top:14px;">
          <thead>
            <tr><th>Category</th><th>Status</th><th>Notes</th></tr>
          </thead>
          <tbody>
            ${complianceRows || '<tr><td colspan="3">No explicit compliance indicators available.</td></tr>'}
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2>Therapeutic potential</h2>
        <div class="copy">${esc(data?.therapeutic_potential || "Not reported")}</div>
        <h2 style="margin-top:18px;">Entourage effect</h2>
        <div class="copy">${esc(data?.entourage_effect || "Not reported")}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Cultivation insights</h2>
        <div class="copy">${esc(data?.cultivation_insights || "Not reported")}</div>
      </div>

      <div class="card">
        <h2>Best use cases</h2>
        <div class="copy">${esc(data?.best_use_cases || "Not reported")}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Chemical fingerprint</h2>
        <div class="radar">
          <div class="radar-item"><div class="radar-label">THC intensity</div><div class="radar-value">${esc(data?.fingerprint_radar?.thc_intensity ?? "0")}</div></div>
          <div class="radar-item"><div class="radar-label">CBD intensity</div><div class="radar-value">${esc(data?.fingerprint_radar?.cbd_intensity ?? "0")}</div></div>
          <div class="radar-item"><div class="radar-label">Minor richness</div><div class="radar-value">${esc(data?.fingerprint_radar?.minor_cannabinoid_richness ?? "0")}</div></div>
          <div class="radar-item"><div class="radar-label">Terpene intensity</div><div class="radar-value">${esc(data?.fingerprint_radar?.terpene_intensity ?? "0")}</div></div>
          <div class="radar-item"><div class="radar-label">Terpene diversity</div><div class="radar-value">${esc(data?.fingerprint_radar?.terpene_diversity ?? "0")}</div></div>
          <div class="radar-item"><div class="radar-label">Aromatic complexity</div><div class="radar-value">${esc(data?.fingerprint_radar?.aromatic_complexity ?? "0")}</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Scientific literature notes</h2>
        <div class="copy">${esc(data?.scientific_references || "Not reported")}</div>
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
  if (!rawBody || typeof rawBody !== "string") {
    throw new Error("Request body is empty or not text");
  }

  const trimmed = rawBody.trim();

  // First: try parsing the whole body as JSON
  let payload;
  try {
    payload = JSON.parse(trimmed);
  } catch (err) {
    throw new Error(`Request body is not valid JSON: ${err.message}`);
  }

  // Case 1: Bubble sends report_json as real object
  if (payload.report_json && typeof payload.report_json === "object") {
    return {
      fileName: payload.file_name || `coa-${Date.now()}.pdf`,
      data: payload.report_json
    };
  }

  // Case 2: Bubble sends report_json_string as stringified JSON
  if (payload.report_json_string && typeof payload.report_json_string === "string") {
    try {
      return {
        fileName: payload.file_name || `coa-${Date.now()}.pdf`,
        data: JSON.parse(payload.report_json_string)
      };
    } catch (err) {
      throw new Error(`report_json_string is not valid JSON: ${err.message}`);
    }
  }

  // Case 3: Bubble sends raw report JSON directly as top-level body
  if (payload.product_name || payload.top_cannabinoids || payload.fingerprint_radar) {
    return {
      fileName: payload.file_name || `coa-${Date.now()}.pdf`,
      data: payload
    };
  }

  throw new Error("Missing valid report_json or report_json_string");
}

app.get("/", (req, res) => {
  res.send("Middleware is running");
});

app.post("/generate-pdf", async (req, res) => {
  try {
    console.log("RAW BODY PREVIEW:");
    console.log(String(req.body).slice(0, 1000));

    const { fileName, data } = parseIncomingBody(req.body);

    const html = renderReportHTML(data);

    const browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"]
    });

    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "networkidle0" });
    await new Promise(resolve => setTimeout(resolve, 500));

    const pdf = await page.pdf({
      format: "A4",
      printBackground: true,
      margin: {
        top: "10mm",
        right: "10mm",
        bottom: "12mm",
        left: "10mm"
      }
    });

    await browser.close();

    return res.json({
      success: true,
      file_name: fileName,
      pdf_base64: pdf.toString("base64")
    });
  } catch (error) {
    console.error("Middleware error:", error.message);

    return res.status(400).json({
      success: false,
      error: error.message
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});