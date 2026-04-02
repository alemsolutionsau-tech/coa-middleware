"use strict";

// ─────────────────────────────────────────────
// V7 PDF/PRINT RENDERER — Pharma-grade A4 document
// Alem COA Intelligence Platform
// Fed to Puppeteer via page.setContent() for PDF generation.
// ─────────────────────────────────────────────

// ── Inline utilities (self-contained) ─────────────────────────────────────

function toNum(value) {
  if (value === null || value === undefined) return 0;
  const raw = String(value).trim().toLowerCase();
  if (!raw || raw === "nd" || raw === "n/d" || raw.includes("not detected") ||
      raw.includes("<loq") || raw.includes("< lod") || raw === "blq" || raw === "absent") return 0;
  const match = raw.match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : 0;
}

function esc(v) {
  return String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Terpene education ──────────────────────────────────────────────────────

const TERPENE_EDUCATION = {
  "Terpinolene":        { aroma: "Fresh, piney, floral, slightly herbal — bright and clean.", therapeutic: "Antioxidant properties noted in preclinical studies. Mild anxiolytic signals." },
  "β-Myrcene":          { aroma: "Earthy, musky, fruity — ripe mango or hops.", therapeutic: "Anti-inflammatory and analgesic properties in preclinical data." },
  "Beta-Myrcene":       { aroma: "Earthy, musky, fruity — ripe mango or hops.", therapeutic: "Anti-inflammatory and analgesic properties in preclinical data." },
  "Ocimene":            { aroma: "Sweet, herbal, woody — fresh basil and tarragon.", therapeutic: "Antifungal and antiviral properties noted in vitro." },
  "Alpha-Pinene":       { aroma: "Crisp pine, fresh forest air.", therapeutic: "Bronchodilator in preclinical models. May counteract short-term memory effects associated with THC." },
  "Beta-Pinene":        { aroma: "Piney, green, fresh — slightly more herbal than Alpha-Pinene.", therapeutic: "Mild antiseptic properties." },
  "Trans-Caryophyllene":{ aroma: "Spicy, peppery, woody — the same compound that gives black pepper its heat.", therapeutic: "The only terpene confirmed to act as a CB2 cannabinoid receptor agonist." },
  "Farnesene":          { aroma: "Green apple, woody, floral.", therapeutic: "Anti-inflammatory properties in preclinical research." },
  "Alpha-Humulene":     { aroma: "Woody, earthy, spicy — like hops.", therapeutic: "Anti-inflammatory, antibacterial, and appetite-suppressing properties." },
  "Linalool":           { aroma: "Floral, lavender-like with a subtle citrus note.", therapeutic: "Associated with calming, relaxing experiential profiles." },
  "Alpha-Bisabolol":    { aroma: "Delicate floral, honey-like, slightly sweet.", therapeutic: "Anti-inflammatory and wound-healing properties." },
  "Guaiol":             { aroma: "Piney, floral, rose-like with a woody undertone.", therapeutic: "Antimicrobial and anti-inflammatory properties." },
  "Alpha-Terpineol":    { aroma: "Lilac-like, floral, slightly citrusy.", therapeutic: "Sedative and antimicrobial properties." },
  "Caryophyllene oxide":{ aroma: "Woody, spicy — the oxidised form of Caryophyllene.", therapeutic: "Antifungal properties. Marker of terpene oxidation." },
  "Camphene":           { aroma: "Damp woodlands, fir needles, herbal.", therapeutic: "Antioxidant properties. Minor aromatic contributor." },
  "Limonene":           { aroma: "Bright citrus — lemon, orange.", therapeutic: "Mood-elevating properties noted in early clinical studies." },
};

// ── Main export ─────────────────────────────────────────────────────────────

module.exports = function renderReportPDF(reportJson = {}, options = {}) {
  const chemistry    = reportJson.chemistry    || {};
  const contaminants = reportJson.contaminants || {};
  const scoring      = reportJson.scoring      || { total: 0, grade: "—", tier: "—", breakdown: {} };
  const intelligence = reportJson.intelligence || {};

  const terpenes    = chemistry.top_terpenes    || [];
  const cannabinoids = chemistry.top_cannabinoids || [];
  const flavonoids  = chemistry.flavonoids      || [];

  const thc   = toNum(chemistry.thc_total);
  const cbd   = toNum(chemistry.cbd_total);
  const terps = toNum(chemistry.total_terpenes);
  const cbn   = toNum(chemistry.cbn);
  const cbna  = toNum(chemistry.cbna);

  const moisture     = chemistry.moisture_content  || contaminants.moisture_content  || "";
  const waterActivity = chemistry.water_activity   || contaminants.water_activity    || "";

  const productName  = chemistry.product_name   || "COA Report";
  const batchNumber  = chemistry.batch_number   || "—";
  const reportDate   = chemistry.coa_report_date || "";

  const leadTerpene   = terpenes[0]?.name || "";
  const secondTerpene = terpenes[1]?.name || "";
  const thirdTerpene  = terpenes[2]?.name || "";

  const fingerprintId = intelligence.fingerprintId   || "UNK";
  const effectDir     = intelligence.effectDirection || "Chemistry-led";
  const lineageConf   = intelligence.lineageConfidence || "";

  const heroNarrative = chemistry.hero_narrative ||
    `${leadTerpene ? leadTerpene + "-dominant" : "Cannabis"} profile with ${terps > 2 ? "strong" : terps > 1 ? "moderate" : "light"} terpene expression.`;

  // Audiences
  let audiences = intelligence.audiences || {};
  if (!Array.isArray(audiences.brand))    audiences.brand    = [];
  if (!Array.isArray(audiences.clinical)) audiences.clinical = [];
  if (!Array.isArray(audiences.patient))  audiences.patient  = [];
  if (!Array.isArray(audiences.buyer))    audiences.buyer    = [];

  // Post-harvest
  let postHarvest = intelligence.postHarvest || {};
  const safePH = {
    freshness:   postHarvest.freshness   || { label: "Unknown", note: "Data not available." },
    curing:      postHarvest.curing      || { label: "Unknown", note: "Data not available." },
    degradation: postHarvest.degradation || { label: "Unknown", note: "Data not available." },
    stability:   postHarvest.stability   || { label: "Unknown", note: "Data not available." },
  };

  const activeTerps = terpenes.filter(t => toNum(t.value) > 0);
  const blqTerps    = terpenes.filter(t => String(t.value || "").toLowerCase() === "blq");
  const maxTerpVal  = activeTerps.length > 0 ? Math.max(...activeTerps.map(t => toNum(t.value)), 0.001) : 0.001;
  const terpCount   = activeTerps.length;
  const blqCount    = blqTerps.length;

  const safeTotal = scoring.total ?? 0;
  const safeGrade = scoring.grade || "—";
  const safeTier  = scoring.tier  || "—";

  const safeBreakdown = scoring.breakdown || {};
  const safePotency   = safeBreakdown.potency          || { score: 0, max: 25 };
  const safeTerpenes  = safeBreakdown.terpenes         || { score: 0, max: 25 };
  const safeMinors    = safeBreakdown.minors           || { score: 0, max: 20 };
  const safeSafety    = safeBreakdown.safety           || { score: 0, max: 20 };
  const safeData      = safeBreakdown.dataCompleteness || { score: 0, max: 10 };

  const pestPass  = /pass|nd|not detected/i.test(contaminants.pesticides_status   || "");
  const metalsPass = /pass|nd|not detected|blq/i.test(contaminants.heavy_metals_status || "");
  const microPass = /pass|nd|not detected|absent/i.test(contaminants.microbials_status || "");
  const mycoPass  = /pass|nd|not detected/i.test(contaminants.mycotoxins_status   || "");
  const isoPass   = contaminants.iso_17025 === true || /17025/i.test(chemistry.laboratory_accreditation || "");
  const sccPass   = contaminants.scc_accredited === true;
  const hasFlavonoids = flavonoids.length > 0;

  // Score ring SVG
  const RING_R = 44;
  const RING_CIRC = (2 * Math.PI * RING_R).toFixed(1);
  const RING_FILL  = (safeTotal / 100 * 2 * Math.PI * RING_R).toFixed(1);
  const scoreRingSVG = `
  <svg width="110" height="110" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="${RING_R}" fill="none" stroke="#e8e8e0" stroke-width="8"/>
    <circle cx="50" cy="50" r="${RING_R}" fill="none" stroke="#16855c" stroke-width="8"
      stroke-dasharray="${RING_FILL} ${RING_CIRC}"
      stroke-dashoffset="0"
      stroke-linecap="round"
      transform="rotate(-90 50 50)"/>
    <text x="50" y="44" text-anchor="middle" font-family="'Libre Baskerville',Georgia,serif"
      font-size="22" fill="#16855c" font-weight="700">${safeTotal}</text>
    <text x="50" y="58" text-anchor="middle" font-family="'DM Sans',sans-serif"
      font-size="9" fill="#999992">/100</text>
    <text x="50" y="71" text-anchor="middle" font-family="'Libre Baskerville',Georgia,serif"
      font-size="10" fill="#1a1a1a">${esc(safeGrade)}</text>
  </svg>`;

  // Sub-score bar for PDF (inline style, no animation needed)
  function pdfScoreBar(label, score, max) {
    const pct = max > 0 ? Math.min(100, Math.round(score / max * 100)) : 0;
    const w = pct.toFixed(0);
    return `
    <div style="display:grid;grid-template-columns:100pt 1fr 36pt;align-items:center;gap:6pt;margin-bottom:5pt;">
      <span style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#555550;">${esc(label)}</span>
      <div style="height:4pt;background:#e8e8e0;border-radius:2pt;overflow:hidden;">
        <div style="height:100%;width:${w}%;background:#16855c;border-radius:2pt;"></div>
      </div>
      <span style="font-family:'DM Mono',monospace;font-size:8pt;color:#999992;text-align:right;">${esc(String(score))}/${esc(String(max))}</span>
    </div>`;
  }

  // Terpene bar row for PDF
  function pdfTerpBar(t, i) {
    if (!t || !t.name) return "";
    const val   = toNum(t.value);
    const isBlq = String(t.value || "").toLowerCase() === "blq";
    const width = isBlq ? 2 : Math.max(2, maxTerpVal > 0 ? (val / maxTerpVal) * 100 : 0);
    const edu   = TERPENE_EDUCATION[t.name] || TERPENE_EDUCATION[t.name.replace(/^beta-/i,"Beta-").replace(/^alpha-/i,"Alpha-")] || null;
    const aroma = edu ? edu.aroma.split("—")[0].trim() : "";
    const rowBg = i % 2 === 0 ? "#ffffff" : "#f7f5ef";
    return `
    <tr style="background:${rowBg}${isBlq ? ";opacity:0.65" : ""}">
      <td style="font-family:'DM Mono',monospace;font-size:7pt;color:#999992;padding:3pt 4pt;width:18pt;">${i + 1}</td>
      <td style="font-family:'DM Sans',sans-serif;font-size:8pt;color:${isBlq ? "#999992" : "#1a1a1a"};padding:3pt 4pt;${isBlq ? "font-style:italic;" : i === 0 ? "font-weight:600;" : ""}">${esc(t.name)}</td>
      <td style="padding:3pt 4pt;width:80pt;">
        <div class="ptbar"><div class="ptbar-fill" style="width:${width.toFixed(1)}%;"></div></div>
      </td>
      <td style="font-family:'DM Mono',monospace;font-size:8pt;color:${isBlq ? "#999992" : "#16855c"};padding:3pt 4pt;white-space:nowrap;">${isBlq ? "BLQ" : esc(String(t.value || "")) + " " + esc(t.unit || "wt%")}</td>
      <td style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#999992;padding:3pt 4pt;max-width:120pt;">${esc(aroma)}</td>
    </tr>`;
  }

  // Cannabinoid cell for PDF (compact table row)
  function pdfCannRow(c) {
    if (!c || !c.name) return "";
    const safeVal = String(c.value ?? "");
    const isNd = !safeVal || safeVal === "ND" || safeVal === "" || toNum(safeVal) === 0;
    const noteMap = {
      "THCA":  "Precursor · ×0.877 on heat",
      "D9-THC":"Active form",
      "CBNA":  "Degradation marker",
      "CBN":   "Degradation marker",
      "CBGA":  "Mother cannabinoid",
    };
    const note = c.notes || noteMap[c.name] || "";
    return `
    <tr>
      <td style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#1a1a1a;padding:3pt 6pt;font-weight:${["THCA","D9-THC","CBD"].includes(c.name) ? "600" : "400"};">${esc(c.name)}</td>
      <td style="font-family:'DM Mono',monospace;font-size:9pt;color:${isNd ? "#999992" : "#16855c"};padding:3pt 6pt;${isNd ? "font-style:italic;" : ""}">${isNd ? "ND" : esc(safeVal)}</td>
      <td style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#999992;padding:3pt 6pt;">${isNd ? "" : esc(c.unit || "wt%")}</td>
      <td style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#999992;padding:3pt 6pt;font-style:italic;">${esc(note)}</td>
    </tr>`;
  }

  // Safety status badge
  function pdfSafetyBadge(pass, status) {
    const bg   = pass   ? "#16855c" : (status ? "#b93030" : "#999992");
    const text = pass   ? (status || "PASS") : (status ? status.toUpperCase() : "NOT TESTED");
    return `<span style="display:inline-block;background:${bg};color:#fff;font-family:'DM Mono',monospace;font-size:8pt;padding:2pt 7pt;border-radius:3pt;">${esc(text)}</span>`;
  }

  // Metal/microbial/myco detail row
  function pdfDetailRow(name, result) {
    const r = String(result || "");
    const ok = !r || /nd|blq|absent|not detected|< 10/i.test(r);
    return `<div style="display:flex;align-items:center;gap:4pt;padding:2pt 0;">
      <span style="display:inline-block;width:5pt;height:5pt;border-radius:50%;background:${ok ? "#16855c" : "#b93030"};flex-shrink:0;"></span>
      <span style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#555550;flex:1;">${esc(name)}</span>
      <span style="font-family:'DM Mono',monospace;font-size:7pt;color:${ok ? "#16855c" : "#b93030"};">${esc(r || "—")}</span>
    </div>`;
  }

  // Post-harvest row
  function pdfPHRow(icon, signalName, signal) {
    const lbl = signal.label || "Unknown";
    const isGood  = /strong|good|positive|minimal|data present/i.test(lbl);
    const isAmber = /moderate|low signal|light/i.test(lbl);
    const pillBg  = isGood ? "#eaf4ef" : isAmber ? "#fef5e7" : "#f0f0ea";
    const pillCol = isGood ? "#16855c" : isAmber ? "#c8860a" : "#999992";
    return `
    <tr>
      <td style="padding:4pt 6pt;font-size:14pt;">${icon}</td>
      <td style="font-family:'DM Sans',sans-serif;font-size:8pt;font-weight:600;color:#1a1a1a;padding:4pt 6pt;">${esc(signalName)}</td>
      <td style="padding:4pt 6pt;">
        <span style="background:${pillBg};color:${pillCol};font-family:'DM Sans',sans-serif;font-size:7pt;padding:2pt 6pt;border-radius:8pt;">${esc(lbl)}</span>
      </td>
      <td style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#555550;padding:4pt 6pt;line-height:1.4;">${esc(signal.note || "")}</td>
    </tr>`;
  }

  // Audience bullet list
  function pdfAudienceList(title, items) {
    if (!items || !items.length) return "";
    return `
    <div class="pdf-sec" style="margin-bottom:10pt;">
      <div style="font-family:'DM Sans',sans-serif;font-size:8pt;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#555550;margin-bottom:6pt;">${esc(title)}</div>
      <ul style="margin:0;padding-left:14pt;list-style:disc;">
        ${items.map(line => `<li style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#333333;margin-bottom:4pt;line-height:1.5;">${esc(line)}</li>`).join("")}
      </ul>
    </div>`;
  }

  // Detected cannabinoid count for entourage signal
  const detectedCannCount = cannabinoids.filter(c => toNum(c.value) > 0).length;
  const entourageStrength = detectedCannCount >= 5 ? "Strong" : detectedCannCount >= 3 ? "Moderate" : "Limited";

  const panelCount = [
    contaminants.pesticides_status,
    contaminants.heavy_metals_status,
    contaminants.microbials_status,
    contaminants.mycotoxins_status,
  ].filter(s => s && !/not tested/i.test(s)).length;

  const docId = options.documentId ? String(options.documentId) : "";
  const docIdShort = docId ? docId.slice(0, 8) : "";

  // ── HTML ─────────────────────────────────────────────────────────────────
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>${esc(productName)} — Alem Chemical Intelligence Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'DM Sans', sans-serif;
  font-size: 9pt;
  color: #333333;
  background: #ffffff;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

@page {
  size: A4 portrait;
  margin: 20mm 15mm 22mm 15mm;
}

@media print {
  .no-print { display: none !important; }
}

/* ── Fixed header (every page) ────────────────────────────────────── */
.pdf-hdr {
  position: fixed;
  top: -20mm;
  left: 0;
  right: 0;
  height: 16mm;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 0.5pt solid rgba(22,133,92,0.20);
  padding-bottom: 3pt;
  font-size: 8pt;
}

/* ── Fixed footer (every page) ───────────────────────────────────── */
.pdf-ftr {
  position: fixed;
  bottom: -22mm;
  left: 0;
  right: 0;
  height: 16mm;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  border-top: 0.5pt solid rgba(22,133,92,0.20);
  padding-top: 3pt;
  font-size: 8pt;
  color: #999992;
  font-family: 'DM Sans', sans-serif;
}
.pdf-ftr-centre {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  font-family: 'DM Mono', monospace;
  font-size: 7.5pt;
}
.pdf-ftr-centre::after {
  content: "Page " counter(page) " of " counter(pages);
}

/* ── Sections ─────────────────────────────────────────────────────── */
.pdf-sec {
  page-break-inside: avoid;
  break-inside: avoid;
  margin-bottom: 14pt;
}
h2 {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 11pt;
  color: #1a1a1a;
  border-bottom: 0.5pt solid #e0e0d8;
  padding-bottom: 3pt;
  margin: 12pt 0 6pt;
  page-break-after: avoid;
  break-after: avoid;
}

/* ── Cover block ──────────────────────────────────────────────────── */
.cover-grid {
  display: grid;
  grid-template-columns: 110pt 1fr;
  gap: 20pt;
  align-items: start;
  margin-bottom: 16pt;
  padding-bottom: 16pt;
  border-bottom: 1pt solid #e0e0d8;
}

/* ── Terpene bars ──────────────────────────────────────────────────  */
.ptbar {
  height: 5pt;
  background: #e8e8e0;
  border-radius: 3pt;
  overflow: hidden;
}
.ptbar-fill {
  height: 100%;
  background: linear-gradient(to right, #16855c, #4aab84);
  border-radius: 3pt;
}

/* ── Table base ───────────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; }
td, th { vertical-align: middle; }
th {
  font-family: 'DM Sans', sans-serif;
  font-size: 7pt;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #999992;
  padding: 4pt 4pt;
  text-align: left;
  border-bottom: 0.5pt solid #e0e0d8;
}

/* ── Safety grid ───────────────────────────────────────────────────── */
.safety-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8pt;
}
.safety-card {
  border: 0.5pt solid #e0e0d8;
  border-radius: 4pt;
  padding: 8pt;
}
.safety-card-title {
  font-family: 'DM Sans', sans-serif;
  font-size: 7.5pt;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 5pt;
}

/* ── Post-harvest table ────────────────────────────────────────────── */
.ph-table td { border-bottom: 0.5pt solid #f0f0ea; }
.ph-table tr:last-child td { border-bottom: none; }
</style>
</head>
<body>

<!-- Fixed header -->
<div class="pdf-hdr">
  <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png"
       height="20"
       onerror="this.style.display='none'">
  <span style="font-family:'DM Sans',sans-serif;font-size:9pt;color:#16855c;">${esc(productName)} · Batch ${esc(batchNumber)}</span>
  <span style="font-family:'DM Mono',monospace;font-size:8pt;color:#999992;letter-spacing:0.04em;">CONFIDENTIAL · Chemical Intelligence Report</span>
</div>

<!-- Fixed footer -->
<div class="pdf-ftr">
  <span style="font-family:'DM Mono',monospace;font-size:7.5pt;">alem.solutions · COA Intelligence</span>
  <div class="pdf-ftr-centre"></div>
  <span style="font-family:'DM Mono',monospace;font-size:7.5pt;">${docIdShort ? esc(docIdShort) + " · " : ""}${esc(reportDate)}</span>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     PAGE 1 — COVER: Score + Key Metrics + Sub-scores
════════════════════════════════════════════════════════════════ -->
<div class="pdf-sec">

  <!-- Masthead -->
  <div style="margin-bottom:12pt;">
    <div style="font-family:'DM Mono',monospace;font-size:9pt;color:#16855c;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6pt;">Chemical Intelligence Report</div>
    <div style="font-family:'Libre Baskerville',Georgia,serif;font-size:28pt;color:#1a1a1a;line-height:1.1;margin-bottom:4pt;">${esc(productName)}</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:10pt;color:#555550;">
      ${[chemistry.product_type, chemistry.batch_number, chemistry.laboratory_name].filter(Boolean).map(esc).join(" · ")}
    </div>
    <div style="border-top:1pt solid rgba(22,133,92,0.3);margin:10pt 0;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:8pt;color:#999992;">
      ${esc(reportDate)}${chemistry.laboratory_accreditation ? " · " + esc(chemistry.laboratory_accreditation) : ""} · Schema v7.0
    </div>
  </div>

  <!-- Cover grid: score ring + sub-scores -->
  <div class="cover-grid">

    <!-- Left: Score ring -->
    <div style="text-align:center;">
      ${scoreRingSVG}
      <div style="font-family:'Libre Baskerville',Georgia,serif;font-size:13pt;color:#1a1a1a;margin-top:4pt;">${esc(safeGrade)}</div>
      <div style="font-family:'DM Sans',sans-serif;font-size:8pt;font-style:italic;color:#555550;">${esc(safeTier)}</div>
    </div>

    <!-- Right: sub-score bars + hero narrative -->
    <div>
      <div style="font-family:'DM Sans',sans-serif;font-size:9pt;color:#333333;margin-bottom:12pt;line-height:1.5;border-left:2pt solid #16855c;padding-left:8pt;">${esc(heroNarrative)}</div>
      ${pdfScoreBar("Potency",          safePotency.score,  safePotency.max)}
      ${pdfScoreBar("Terpenes",         safeTerpenes.score, safeTerpenes.max)}
      ${pdfScoreBar("Minor Cannabinoids", safeMinors.score, safeMinors.max)}
      ${pdfScoreBar("Safety",           safeSafety.score,   safeSafety.max)}
      ${pdfScoreBar("Data Completeness",safeData.score,     safeData.max)}
    </div>
  </div>

  <!-- Key metrics strip: 4 cols -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8pt;margin-bottom:14pt;">
    ${[
      { label: "THC TOTAL",       val: chemistry.thc_total  ? chemistry.thc_total  + " " + (chemistry.thc_total_unit || "wt%") : "ND", ctx: thc >= 24 ? "Above avg." : thc >= 18 ? "Moderate-high" : "Moderate" },
      { label: "THCA",            val: chemistry.thca       ? chemistry.thca + " wt%"              : "ND", ctx: "× 0.877 on heat" },
      { label: "TOTAL TERPENES",  val: chemistry.total_terpenes ? chemistry.total_terpenes + " wt%" : "ND", ctx: terps >= 3 ? "Exceptional" : terps >= 2 ? "Strong" : "Moderate" },
      { label: "CBD TOTAL",       val: chemistry.cbd_total && chemistry.cbd_total !== "ND" ? chemistry.cbd_total + " wt%" : "ND", ctx: cbd >= 0.5 ? "Modulating" : "Not detected" },
    ].map(m => `
    <div style="background:#f7f5ef;border-radius:4pt;padding:10pt 10pt;">
      <div style="font-family:'DM Mono',monospace;font-size:16pt;color:#16855c;line-height:1;">${esc(m.val)}</div>
      <div style="font-family:'DM Sans',sans-serif;font-size:7pt;text-transform:uppercase;letter-spacing:0.08em;color:#999992;margin-top:3pt;">${esc(m.label)}</div>
      <div style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#555550;margin-top:3pt;">${esc(m.ctx)}</div>
    </div>`).join("")}
  </div>

  <!-- Chemotype fingerprint band -->
  <div style="background:#eaf4ef;border:0.5pt solid #cce8da;border-radius:4pt;padding:10pt 14pt;display:flex;align-items:center;gap:16pt;flex-wrap:wrap;">
    <div style="font-family:'DM Mono',monospace;font-size:18pt;font-weight:500;color:#16855c;">${esc(fingerprintId)}</div>
    <div style="flex:1;">
      <div style="font-family:'DM Sans',sans-serif;font-size:9pt;color:#1a1a1a;">${esc(effectDir)}${lineageConf ? `<span style="background:#16855c;color:#fff;font-size:7pt;padding:1pt 6pt;border-radius:8pt;margin-left:6pt;">${esc(lineageConf)} confidence</span>` : ""}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2pt;">
      ${[leadTerpene, secondTerpene, thirdTerpene].filter(Boolean).map(n =>
        `<span style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#555550;">${esc(n)}</span>`
      ).join("")}
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     TERPENE FINGERPRINT
════════════════════════════════════════════════════════════════ -->
<div class="pdf-sec">
  <h2>Terpene Fingerprint</h2>

  <!-- Summary pills -->
  <div style="display:flex;gap:6pt;flex-wrap:wrap;margin-bottom:10pt;">
    ${[
      chemistry.total_terpenes ? chemistry.total_terpenes + " wt% total terpenes" : null,
      terpCount > 0 ? terpCount + " compound" + (terpCount !== 1 ? "s" : "") + " detected" : null,
      blqCount > 0 ? blqCount + " at BLQ" : null,
    ].filter(Boolean).map(p => `<span style="background:#eaf4ef;color:#16855c;border:0.5pt solid #cce8da;font-family:'DM Mono',monospace;font-size:7.5pt;padding:2pt 8pt;border-radius:10pt;">${esc(p)}</span>`).join("")}
  </div>

  <!-- Dominant terpene callout -->
  ${leadTerpene ? (() => {
    const edu = TERPENE_EDUCATION[leadTerpene] || null;
    const val = terpenes[0] ? `${esc(String(terpenes[0].value || ""))} ${esc(terpenes[0].unit || "wt%")}` : "—";
    return `
  <div style="border-left:2pt solid #16855c;background:#eaf4ef;padding:8pt 12pt;margin-bottom:10pt;">
    <div style="display:flex;align-items:baseline;gap:10pt;margin-bottom:4pt;">
      <span style="font-family:'Libre Baskerville',Georgia,serif;font-size:16pt;color:#1a1a1a;">${esc(leadTerpene)}</span>
      <span style="font-family:'DM Mono',monospace;font-size:13pt;color:#16855c;">${val}</span>
    </div>
    ${edu ? `<div style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#555550;font-style:italic;margin-bottom:3pt;">${esc(edu.aroma)}</div>` : ""}
    ${edu ? `<div style="font-family:'DM Sans',sans-serif;font-size:7.5pt;color:#999992;line-height:1.4;">${esc(edu.therapeutic)}</div>` : ""}
  </div>`;
  })() : ""}

  <!-- All terpenes table -->
  ${terpenes.length > 0 ? `
  <table style="width:100%;">
    <thead><tr>
      <th style="width:18pt;">#</th>
      <th>Compound</th>
      <th style="width:80pt;">Relative</th>
      <th style="width:70pt;">Value</th>
      <th>Aroma note</th>
    </tr></thead>
    <tbody>
      ${terpenes.map((t, i) => pdfTerpBar(t, i)).join("")}
    </tbody>
  </table>` : `<p style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#999992;font-style:italic;">No terpene data reported.</p>`}
</div>

<!-- ═══════════════════════════════════════════════════════════════
     CANNABINOID PROFILE
════════════════════════════════════════════════════════════════ -->
<div class="pdf-sec">
  <h2>Cannabinoid Profile</h2>

  ${chemistry.total_cannabinoids ? `
  <div style="margin-bottom:8pt;">
    <span style="font-family:'DM Mono',monospace;font-size:20pt;color:#16855c;">${esc(chemistry.total_cannabinoids)} wt%</span>
    <span style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#999992;margin-left:6pt;text-transform:uppercase;letter-spacing:0.08em;">Total Cannabinoids</span>
  </div>` : ""}

  ${cannabinoids.length > 0 ? `
  <table>
    <thead><tr>
      <th style="width:100pt;">Cannabinoid</th>
      <th style="width:60pt;">Value</th>
      <th style="width:40pt;">Unit</th>
      <th>Note</th>
    </tr></thead>
    <tbody>
      ${cannabinoids.map(c => pdfCannRow(c)).join("")}
    </tbody>
  </table>` : `<p style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#999992;font-style:italic;">No cannabinoid data reported.</p>`}

  ${chemistry.thc_total_anhydrous && chemistry.thc_total_anhydrous !== chemistry.thc_total ? `
  <div style="margin-top:8pt;padding:6pt 10pt;background:#eaf4ef;border-radius:3pt;font-family:'DM Sans',sans-serif;font-size:7.5pt;color:#555550;">
    Anhydrous (moisture-corrected) THC: <strong style="color:#16855c;">${esc(chemistry.thc_total_anhydrous)} wt%</strong> ·
    As-received: ${esc(chemistry.thc_total)} wt% ·
    Moisture: ${esc(chemistry.moisture_content || "—")}%
  </div>` : ""}

  ${detectedCannCount >= 3 ? `
  <div style="margin-top:8pt;border-left:2pt solid #c8860a;padding:8pt 10pt;background:#fffbf5;font-family:'DM Sans',sans-serif;font-size:8pt;color:#555550;">
    <strong style="color:#c8860a;">Entourage Effect Signal: ${esc(entourageStrength)}</strong> —
    ${detectedCannCount} cannabinoids + ${terpCount} terpenes detected.
    This profile has biochemical complexity associated with full-spectrum, synergistic activity.
  </div>` : ""}
</div>

<!-- ═══════════════════════════════════════════════════════════════
     QUALITY & SAFETY
════════════════════════════════════════════════════════════════ -->
<div class="pdf-sec">
  <h2>Quality &amp; Safety</h2>

  <div style="display:flex;align-items:center;gap:8pt;margin-bottom:10pt;flex-wrap:wrap;">
    ${panelCount === 4
      ? `<span style="background:#eaf4ef;color:#16855c;font-family:'DM Sans',sans-serif;font-size:8pt;padding:2pt 8pt;border-radius:8pt;">✓ Complete Panel (4/4)</span>`
      : `<span style="background:#fef5e7;color:#c8860a;font-family:'DM Sans',sans-serif;font-size:8pt;padding:2pt 8pt;border-radius:8pt;">⚠ Partial Panel (${panelCount}/4)</span>`}
    ${isoPass ? `<span style="background:#eaf4ef;color:#16855c;font-family:'DM Mono',monospace;font-size:7.5pt;padding:2pt 8pt;border-radius:8pt;">ISO 17025</span>` : ""}
    ${sccPass ? `<span style="background:#eaf4ef;color:#16855c;font-family:'DM Mono',monospace;font-size:7.5pt;padding:2pt 8pt;border-radius:8pt;">SCC Accredited</span>` : ""}
  </div>

  <div class="safety-grid">

    <!-- Pesticides -->
    <div class="safety-card">
      <div class="safety-card-title">Pesticides</div>
      ${pdfSafetyBadge(pestPass, contaminants.pesticides_status)}
      ${contaminants.pesticides_method ? `<div style="font-family:'DM Mono',monospace;font-size:6.5pt;color:#999992;margin-top:4pt;">${esc(contaminants.pesticides_method)}</div>` : ""}
      ${contaminants.pesticides_compound_count ? `<div style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#555550;margin-top:3pt;">${esc(contaminants.pesticides_compound_count)} compounds tested</div>` : ""}
      ${!contaminants.pesticides_status ? `<div style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#999992;font-style:italic;margin-top:4pt;">Panel not captured</div>` : ""}
    </div>

    <!-- Heavy Metals -->
    <div class="safety-card">
      <div class="safety-card-title">Heavy Metals</div>
      ${pdfSafetyBadge(metalsPass, contaminants.heavy_metals_status)}
      <div style="margin-top:5pt;">
        ${contaminants.arsenic_result  ? pdfDetailRow("Arsenic (As)", contaminants.arsenic_result)  : ""}
        ${contaminants.cadmium_result  ? pdfDetailRow("Cadmium (Cd)", contaminants.cadmium_result)  : ""}
        ${contaminants.lead_result     ? pdfDetailRow("Lead (Pb)",    contaminants.lead_result)     : ""}
        ${contaminants.mercury_result  ? pdfDetailRow("Mercury (Hg)", contaminants.mercury_result)  : ""}
        ${!contaminants.arsenic_result && !contaminants.cadmium_result ? `<div style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#999992;font-style:italic;">No individual results captured</div>` : ""}
      </div>
    </div>

    <!-- Microbials -->
    <div class="safety-card">
      <div class="safety-card-title">Microbials</div>
      ${pdfSafetyBadge(microPass, contaminants.microbials_status)}
      <div style="margin-top:5pt;">
        ${contaminants.salmonella   ? pdfDetailRow("Salmonella",   contaminants.salmonella)   : ""}
        ${contaminants.e_coli       ? pdfDetailRow("E. coli",      contaminants.e_coli)       : ""}
        ${contaminants.yeast_mold   ? pdfDetailRow("Yeast & Mold", contaminants.yeast_mold)   : ""}
        ${contaminants.s_aureus     ? pdfDetailRow("S. aureus",    contaminants.s_aureus)     : ""}
        ${!contaminants.yeast_mold && !contaminants.salmonella ? `<div style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#999992;font-style:italic;">No individual results captured</div>` : ""}
      </div>
    </div>

    <!-- Mycotoxins -->
    <div class="safety-card">
      <div class="safety-card-title">Mycotoxins</div>
      ${pdfSafetyBadge(mycoPass, contaminants.mycotoxins_status)}
      <div style="margin-top:5pt;">
        ${contaminants.aflatoxin_b1 ? pdfDetailRow("Aflatoxin B1", contaminants.aflatoxin_b1) : ""}
        ${contaminants.aflatoxin_b2 ? pdfDetailRow("Aflatoxin B2", contaminants.aflatoxin_b2) : ""}
        ${contaminants.aflatoxin_g1 ? pdfDetailRow("Aflatoxin G1", contaminants.aflatoxin_g1) : ""}
        ${contaminants.aflatoxin_g2 ? pdfDetailRow("Aflatoxin G2", contaminants.aflatoxin_g2) : ""}
        ${contaminants.ochratoxin_a ? pdfDetailRow("Ochratoxin A", contaminants.ochratoxin_a) : ""}
        ${!contaminants.aflatoxin_b1 && !contaminants.ochratoxin_a ? `<div style="font-family:'DM Sans',sans-serif;font-size:7pt;color:#999992;font-style:italic;">No individual results captured</div>` : ""}
      </div>
    </div>
  </div>

  ${(moisture || waterActivity) ? `
  <div style="margin-top:8pt;display:flex;gap:8pt;flex-wrap:wrap;">
    ${moisture ? `<span style="background:#eaf4ef;color:#16855c;font-family:'DM Mono',monospace;font-size:8pt;padding:2pt 8pt;border-radius:8pt;">Moisture: ${esc(moisture)}%</span>` : ""}
    ${waterActivity ? `<span style="background:#eaf4ef;color:#16855c;font-family:'DM Mono',monospace;font-size:8pt;padding:2pt 8pt;border-radius:8pt;">Water activity: ${esc(waterActivity)} aw</span>` : ""}
  </div>` : ""}
</div>

<!-- ═══════════════════════════════════════════════════════════════
     POST-HARVEST SIGNALS
════════════════════════════════════════════════════════════════ -->
<div class="pdf-sec">
  <h2>Post-Harvest Signals</h2>
  <table class="ph-table" style="width:100%;">
    <colgroup>
      <col style="width:18pt;">
      <col style="width:70pt;">
      <col style="width:80pt;">
      <col>
    </colgroup>
    <tbody>
      ${pdfPHRow("🌿", "Freshness",   safePH.freshness)}
      ${pdfPHRow("🔬", "Curing",      safePH.curing)}
      ${pdfPHRow("⚗️", "Degradation", safePH.degradation)}
      ${pdfPHRow("💧", "Stability",   safePH.stability)}
    </tbody>
  </table>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     AUDIENCE INTELLIGENCE (Brand + Clinical only for print)
════════════════════════════════════════════════════════════════ -->
<div class="pdf-sec">
  <h2>Audience Intelligence</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16pt;">
    ${audiences.brand.length ? `
    <div>
      <div style="font-family:'DM Mono',monospace;font-size:7.5pt;text-transform:uppercase;letter-spacing:0.1em;color:#16855c;margin-bottom:6pt;">◈ Brand</div>
      <ul style="margin:0;padding-left:14pt;list-style:disc;">
        ${audiences.brand.map(line => `<li style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#333333;margin-bottom:4pt;line-height:1.5;">${esc(line)}</li>`).join("")}
      </ul>
    </div>` : ""}
    ${audiences.clinical.length ? `
    <div>
      <div style="font-family:'DM Mono',monospace;font-size:7.5pt;text-transform:uppercase;letter-spacing:0.1em;color:#16855c;margin-bottom:6pt;">✚ Clinical</div>
      <ul style="margin:0;padding-left:14pt;list-style:disc;">
        ${audiences.clinical.map(line => `<li style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#333333;margin-bottom:4pt;line-height:1.5;">${esc(line)}</li>`).join("")}
      </ul>
    </div>` : ""}
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     LAB CARD
════════════════════════════════════════════════════════════════ -->
<div class="pdf-sec">
  <h2>Laboratory</h2>
  <div style="display:flex;align-items:center;gap:16pt;flex-wrap:wrap;padding:10pt;background:#f7f5ef;border-radius:4pt;">
    <div>
      <div style="font-family:'Libre Baskerville',Georgia,serif;font-size:12pt;color:#1a1a1a;">${esc(chemistry.laboratory_name || "Laboratory not reported")}</div>
      ${chemistry.laboratory_accreditation ? `<div style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#555550;margin-top:2pt;">${esc(chemistry.laboratory_accreditation)}</div>` : ""}
    </div>
    <div style="margin-left:auto;display:flex;gap:6pt;align-items:center;flex-wrap:wrap;">
      ${isoPass ? `<span style="background:#eaf4ef;color:#16855c;font-family:'DM Mono',monospace;font-size:7.5pt;padding:2pt 8pt;border-radius:8pt;">ISO 17025</span>` : ""}
      ${sccPass ? `<span style="background:#eaf4ef;color:#16855c;font-family:'DM Mono',monospace;font-size:7.5pt;padding:2pt 8pt;border-radius:8pt;">SCC</span>` : ""}
      ${chemistry.laboratory_method ? `<span style="font-family:'DM Mono',monospace;font-size:7.5pt;color:#999992;">${esc(chemistry.laboratory_method)}</span>` : ""}
      ${reportDate ? `<span style="font-family:'DM Sans',sans-serif;font-size:8pt;color:#555550;">${esc(reportDate)}</span>` : ""}
    </div>
  </div>
  ${docId ? `
  <div style="margin-top:6pt;font-family:'DM Mono',monospace;font-size:7pt;color:#cccccc;text-align:right;">
    Report ID: ${esc(docId)}
  </div>` : ""}
</div>

</body>
</html>`;
};
