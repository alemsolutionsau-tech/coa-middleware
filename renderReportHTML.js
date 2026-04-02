"use strict";

// ─────────────────────────────────────────────
// V7 BROWSER REPORT RENDERER
// Alem COA Intelligence Platform
// ─────────────────────────────────────────────

// ── Inline utilities (keep this file self-contained) ──────────────────────

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

// ── Terpene education lookup (copy of server.js TERPENE_EDUCATION) ─────────

const TERPENE_EDUCATION = {
  "Terpinolene": { aroma: "Fresh, piney, floral, slightly herbal — bright and clean.", therapeutic: "Antioxidant properties noted in preclinical studies. Mild anxiolytic signals. Found in Jack Herer, Durban Poison, Ghost Train Haze. Fewer than 15% of dried flower products are Terpinolene-dominant — a genuine market differentiator." },
  "β-Myrcene":   { aroma: "Earthy, musky, fruity — ripe mango or hops.", therapeutic: "May contribute sedating, relaxing effects at higher concentrations. Anti-inflammatory and analgesic properties in preclinical data." },
  "Beta-Myrcene": { aroma: "Earthy, musky, fruity — ripe mango or hops.", therapeutic: "Anti-inflammatory and analgesic properties in preclinical data." },
  "Ocimene": { aroma: "Sweet, herbal, woody — fresh basil and tarragon. Marker of Haze-type genetics.", therapeutic: "Antifungal and antiviral properties noted in vitro." },
  "Alpha-Pinene": { aroma: "Crisp pine, fresh forest air — also found in rosemary and eucalyptus.", therapeutic: "Bronchodilator in preclinical models. May counteract short-term memory effects sometimes associated with THC." },
  "Beta-Pinene": { aroma: "Piney, green, fresh — slightly more herbal than Alpha-Pinene.", therapeutic: "Mild antiseptic properties. Works synergistically with Alpha-Pinene." },
  "Trans-Caryophyllene": { aroma: "Spicy, peppery, woody — the same compound that gives black pepper its heat.", therapeutic: "The only terpene confirmed to act as a CB2 cannabinoid receptor agonist. Studied for anti-inflammatory, analgesic, and anxiolytic effects." },
  "Farnesene": { aroma: "Green apple, woody, floral. Found in apple skin, green tea.", therapeutic: "Anti-inflammatory properties in preclinical research. Its presence suggests minimal terpene degradation post-harvest." },
  "Alpha-Humulene": { aroma: "Woody, earthy, spicy — like hops.", therapeutic: "Anti-inflammatory, antibacterial, and appetite-suppressing properties in preclinical research." },
  "Linalool": { aroma: "Floral, lavender-like with a subtle citrus note.", therapeutic: "Associated with calming, relaxing experiential profiles. Anti-anxiety and analgesic properties studied in preclinical models." },
  "Alpha-Bisabolol": { aroma: "Delicate floral, honey-like, slightly sweet.", therapeutic: "Well-documented anti-inflammatory and wound-healing properties." },
  "Guaiol": { aroma: "Piney, floral, rose-like with a woody undertone.", therapeutic: "Antimicrobial and anti-inflammatory properties in preclinical models." },
  "Alpha-Terpineol": { aroma: "Lilac-like, floral, slightly citrusy.", therapeutic: "Sedative and antimicrobial properties noted in preclinical studies." },
  "Caryophyllene oxide": { aroma: "Woody, spicy — the oxidised form of Caryophyllene.", therapeutic: "Antifungal properties. Marker of terpene oxidation." },
  "Camphene": { aroma: "Damp woodlands, fir needles, herbal.", therapeutic: "Antioxidant properties. Minor aromatic contributor." },
  "Limonene": { aroma: "Bright citrus — lemon, orange.", therapeutic: "Mood-elevating properties noted in early clinical studies. Potent antifungal and antibacterial." },
};

function normaliseTerpeneName(name) {
  return String(name || "")
    .replace(/β-|β /gi, "Beta-").replace(/α-|α /gi, "Alpha-")
    .replace(/\(R\)-\(\+\)-/gi, "").replace(/\(R\)-/gi, "").replace(/\(S\)-/gi, "")
    .replace(/d-Limonene/gi, "Limonene").replace(/D-Limonene/gi, "Limonene")
    .replace(/trans-/gi, "Trans-").replace(/cis-/gi, "Cis-").trim();
}

function getTerpEdu(name) {
  return TERPENE_EDUCATION[name] || TERPENE_EDUCATION[normaliseTerpeneName(name)] || null;
}

// ── CSS-only tab switcher helper ────────────────────────────────────────────
// Uses hidden radio inputs — zero JS required.

function tabSwitch(panels) {
  // panels: array of { id, label, icon, content }
  const name = "aud-tab";
  const radios = panels.map((p, i) =>
    `<input type="radio" name="${name}" id="tab-${p.id}" class="tab-radio" ${i === 0 ? "checked" : ""} hidden>`
  ).join("\n");
  const labels = panels.map(p =>
    `<label for="tab-${p.id}" class="tab-lbl">${p.icon ? `<span class="tab-icon">${p.icon}</span>` : ""}${esc(p.label)}</label>`
  ).join("\n");
  const contents = panels.map(p =>
    `<div class="tab-content" id="tc-${p.id}">${p.content}</div>`
  ).join("\n");
  // CSS rules are included in the main <style> block
  return `${radios}<div class="tab-bar">${labels}</div><div class="tab-panels">${contents}</div>`;
}

// ── Main export ─────────────────────────────────────────────────────────────

module.exports = function renderReportHTML(reportJson = {}, options = {}) {
  const chemistry    = reportJson.chemistry    || {};
  const contaminants = reportJson.contaminants || {};
  const scoring      = reportJson.scoring      || { total: 0, grade: "—", tier: "—", breakdown: {} };
  const intelligence = reportJson.intelligence || {};
  const benchmark    = options.benchmark       || null;

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

  const leadTerpene   = terpenes[0]?.name || "";
  const secondTerpene = terpenes[1]?.name || "";
  const thirdTerpene  = terpenes[2]?.name || "";

  const fingerprintId = intelligence.fingerprintId || "UNK";
  const effectDir     = intelligence.effectDirection || "Chemistry-led";

  const productName = chemistry.product_name || "COA Report";
  const heroNarrative = chemistry.hero_narrative ||
    `${leadTerpene ? leadTerpene + "-dominant" : "Cannabis"} profile with ${terps > 2 ? "strong" : terps > 1 ? "moderate" : "light"} terpene expression.`;

  // Safely build audiences
  let audiences = intelligence.audiences || {};
  if (!Array.isArray(audiences.brand)) audiences.brand    = [];
  if (!Array.isArray(audiences.clinical)) audiences.clinical = [];
  if (!Array.isArray(audiences.patient))  audiences.patient  = [];
  if (!Array.isArray(audiences.buyer))    audiences.buyer    = [];

  // Safely build postHarvest
  let postHarvest = intelligence.postHarvest || {};
  const safePH = {
    freshness:   postHarvest.freshness   || { label: "Unknown", note: "Data not available." },
    curing:      postHarvest.curing      || { label: "Unknown", note: "Data not available." },
    degradation: postHarvest.degradation || { label: "Unknown", note: "Data not available." },
    stability:   postHarvest.stability   || { label: "Unknown", note: "Data not available." },
  };

  const activeTerps  = terpenes.filter(t => toNum(t.value) > 0);
  const blqTerps     = terpenes.filter(t => String(t.value || "").toLowerCase() === "blq");
  const maxTerpVal   = activeTerps.length > 0 ? Math.max(...activeTerps.map(t => toNum(t.value)), 0.001) : 0.001;
  const terpCount    = activeTerps.length;
  const blqCount     = blqTerps.length;

  const safeTotal = scoring.total ?? 0;
  const safeGrade = scoring.grade || "—";
  const safeTier  = scoring.tier  || "—";

  const safeBreakdown = scoring.breakdown || {};
  const safePotency   = safeBreakdown.potency          || { score: 0, max: 25 };
  const safeTerpenes  = safeBreakdown.terpenes         || { score: 0, max: 25 };
  const safeMinors    = safeBreakdown.minors           || { score: 0, max: 20 };
  const safeSafety    = safeBreakdown.safety           || { score: 0, max: 20 };
  const safeData      = safeBreakdown.dataCompleteness || { score: 0, max: 10 };

  // Status helpers
  const pestPass  = /pass|nd|not detected/i.test(contaminants.pesticides_status   || "");
  const metalsPass = /pass|nd|not detected|blq/i.test(contaminants.heavy_metals_status || "");
  const microPass = /pass|nd|not detected|absent/i.test(contaminants.microbials_status || "");
  const mycoPass  = /pass|nd|not detected/i.test(contaminants.mycotoxins_status   || "");
  const isoPass   = contaminants.iso_17025 === true || /17025/i.test(chemistry.laboratory_accreditation || "");
  const sccPass   = contaminants.scc_accredited === true;
  const hasFlavonoids = flavonoids.length > 0;
  const hasMoisture     = !!(moisture && moisture !== "");
  const hasWaterActivity = !!(waterActivity && waterActivity !== "");

  // ── Sub-score bar (for hero section) ───────────────────────────────
  function scoreBar(label, score, max) {
    const pct = max > 0 ? Math.min(100, Math.round(score / max * 100)) : 0;
    return `
    <div class="sb-row">
      <span class="sb-lbl">${esc(label)}</span>
      <div class="sb-track"><div class="sb-fill" style="width:${pct}%" data-pct="${pct}"></div></div>
      <span class="sb-num">${esc(String(score))}/${esc(String(max))}</span>
    </div>`;
  }

  // ── Metric card ─────────────────────────────────────────────────────
  function metricCard(value, label, context) {
    const safeVal = value || "—";
    return `
    <div class="metric-card">
      <div class="metric-val">${esc(safeVal)}</div>
      <div class="metric-lbl">${esc(label)}</div>
      ${context ? `<div class="metric-ctx">${esc(context)}</div>` : ""}
    </div>`;
  }

  // ── Terpene table row ────────────────────────────────────────────────
  function terpRow(t, i) {
    if (!t || !t.name) return "";
    const val    = toNum(t.value);
    const isBlq  = String(t.value || "").toLowerCase() === "blq";
    const width  = isBlq ? 2 : Math.max(2, maxTerpVal > 0 ? (val / maxTerpVal) * 100 : 0);
    const edu    = getTerpEdu(t.name);
    const aroma  = edu ? edu.aroma : "";
    const isLead = i === 0;
    const rowBg  = i % 2 === 0 ? "#ffffff" : "#fafaf7";
    return `
    <tr style="background:${rowBg}${isBlq ? ";opacity:0.7" : ""}">
      <td class="trow-rank">${i + 1}</td>
      <td class="trow-name${isLead ? " lead" : ""}${isBlq ? " blq" : ""}">${esc(t.name)}</td>
      <td class="trow-bar">
        <div class="tbar-track"><div class="tbar-fill" style="width:${width.toFixed(1)}%" data-w="${width.toFixed(1)}"></div></div>
      </td>
      <td class="trow-val${isBlq ? " blq" : ""}">${isBlq ? "BLQ" : esc(String(t.value || "")) + " " + esc(t.unit || "wt%")}</td>
      <td class="trow-aroma">${esc(aroma)}</td>
    </tr>`;
  }

  // ── Post-harvest signal card ─────────────────────────────────────────
  function phCard(icon, signalName, signal) {
    const lbl = signal.label || "Unknown";
    const isGood = /strong|good|positive|minimal|data present/i.test(lbl);
    const isAmber = /moderate|low signal|light/i.test(lbl);
    const pillClass = isGood ? "pill-green" : isAmber ? "pill-amber" : "pill-grey";
    return `
    <div class="ph-card">
      <div class="ph-card-head">
        <span class="ph-icon">${icon}</span>
        <span class="ph-name">${esc(signalName)}</span>
        <span class="status-pill ${pillClass}">${esc(lbl)}</span>
      </div>
      <div class="ph-note">${esc(signal.note || "")}</div>
    </div>`;
  }

  // ── Safety card ──────────────────────────────────────────────────────
  function safetyCard(title, status, pass, method, details) {
    const badgeClass = pass ? "badge-pass" : (status ? "badge-fail" : "badge-nt");
    const badgeText  = pass ? "PASS" : (status ? esc(status).toUpperCase() : "NOT TESTED");
    return `
    <div class="safety-card">
      <div class="safety-card-head">
        <span class="safety-title">${esc(title)}</span>
        <span class="safety-badge ${badgeClass}">${badgeText}</span>
      </div>
      ${method ? `<div class="safety-method">${esc(method)}</div>` : ""}
      <div class="safety-details">${details || ""}</div>
    </div>`;
  }

  // ── Metal/microbial row inside safety card ───────────────────────────
  function metalRow(name, result) {
    const r = String(result || "");
    const ok = !r || /nd|blq|absent|not detected/i.test(r);
    return `<div class="detail-row"><span class="detail-dot ${ok ? "dot-green" : "dot-red"}"></span><span class="detail-name">${esc(name)}</span><span class="detail-val ${ok ? "val-green" : "val-red"}">${esc(r || "—")}</span></div>`;
  }

  // ── Audience bullet list ─────────────────────────────────────────────
  function audienceList(items) {
    if (!items || !items.length) return `<p class="aud-empty">No intelligence available.</p>`;
    return `<ul class="aud-list">${items.map(line => `<li>${esc(line)}</li>`).join("")}</ul>`;
  }

  // ── Dominant terpene callout ─────────────────────────────────────────
  function dominantCallout() {
    if (!leadTerpene) return "";
    const edu = getTerpEdu(leadTerpene);
    const val = terpenes[0] ? `${esc(String(terpenes[0].value || ""))} ${esc(terpenes[0].unit || "wt%")}` : "—";
    return `
    <div class="dominant-callout">
      <div class="dominant-head">
        <span class="dominant-name">${esc(leadTerpene)}</span>
        <span class="dominant-val">${val}</span>
      </div>
      ${edu ? `<div class="dominant-aroma">${esc(edu.aroma)}</div>` : ""}
      ${edu ? `<div class="dominant-bio">${esc(edu.therapeutic)}</div>` : ""}
    </div>`;
  }

  // ── Flavonoid pills ──────────────────────────────────────────────────
  function flavPill(f) {
    const isCann = /cannflavin/i.test(f.name || "");
    const cls = isCann ? "flav-pill-green" : "flav-pill-default";
    return `<span class="flav-pill ${cls}">${esc(f.name)} <span class="flav-pill-val">${esc(String(f.value || ""))} ${esc(f.unit || "wt%")}</span></span>`;
  }

  // Build safety card detail bodies
  const pesticideDetails = [
    contaminants.pesticides_method     ? `<div class="detail-row"><span class="detail-name">Method:</span><span class="detail-val">${esc(contaminants.pesticides_method)}</span></div>` : "",
    contaminants.pesticides_compound_count ? `<div class="detail-row"><span class="detail-name">Compounds tested:</span><span class="detail-val">${esc(contaminants.pesticides_compound_count)}</span></div>` : "",
    contaminants.pesticides_detail ? `<div class="detail-note">${esc(contaminants.pesticides_detail)}</div>` : "",
    !contaminants.pesticides_status ? `<div class="detail-note nt-note">Pesticide panel not captured. Request from producer.</div>` : "",
  ].join("");

  const metalsDetails = [
    contaminants.arsenic_result  ? metalRow("Arsenic (As)", contaminants.arsenic_result)  : "",
    contaminants.cadmium_result  ? metalRow("Cadmium (Cd)", contaminants.cadmium_result)  : "",
    contaminants.lead_result     ? metalRow("Lead (Pb)",    contaminants.lead_result)     : "",
    contaminants.mercury_result  ? metalRow("Mercury (Hg)", contaminants.mercury_result)  : "",
    (!contaminants.arsenic_result && !contaminants.cadmium_result && !contaminants.lead_result && !contaminants.mercury_result)
      ? `<div class="detail-note nt-note">No individual metal results captured.</div>` : "",
    contaminants.heavy_metals_method ? `<div class="detail-note">${esc(contaminants.heavy_metals_method)}</div>` : "",
  ].join("");

  const microDetails = [
    contaminants.yeast_mold                   ? metalRow("Yeast & Mold", contaminants.yeast_mold) : "",
    contaminants.total_aerobic                ? metalRow("Total Aerobic", contaminants.total_aerobic) : "",
    contaminants.bile_tolerant_gram_negative  ? metalRow("Bile-Tolerant Gram-Neg", contaminants.bile_tolerant_gram_negative) : "",
    contaminants.salmonella                   ? metalRow("Salmonella", contaminants.salmonella) : "",
    contaminants.s_aureus                     ? metalRow("S. aureus", contaminants.s_aureus) : "",
    contaminants.p_aeruginosa                 ? metalRow("P. aeruginosa", contaminants.p_aeruginosa) : "",
    contaminants.e_coli                       ? metalRow("E. coli", contaminants.e_coli) : "",
    (!contaminants.yeast_mold && !contaminants.salmonella)
      ? `<div class="detail-note nt-note">No individual microbial results captured.</div>` : "",
    contaminants.microbials_method ? `<div class="detail-note">${esc(contaminants.microbials_method)}</div>` : "",
  ].join("");

  const mycoDetails = [
    contaminants.aflatoxin_b1  ? metalRow("Aflatoxin B1", contaminants.aflatoxin_b1) : "",
    contaminants.aflatoxin_b2  ? metalRow("Aflatoxin B2", contaminants.aflatoxin_b2) : "",
    contaminants.aflatoxin_g1  ? metalRow("Aflatoxin G1", contaminants.aflatoxin_g1) : "",
    contaminants.aflatoxin_g2  ? metalRow("Aflatoxin G2", contaminants.aflatoxin_g2) : "",
    (contaminants.sum_aflatoxins !== undefined && contaminants.sum_aflatoxins !== "")
      ? metalRow("Sum Aflatoxins", contaminants.sum_aflatoxins) : "",
    contaminants.ochratoxin_a  ? metalRow("Ochratoxin A", contaminants.ochratoxin_a) : "",
    (!contaminants.aflatoxin_b1 && !contaminants.ochratoxin_a)
      ? `<div class="detail-note nt-note">No individual mycotoxin results captured.</div>` : "",
    contaminants.mycotoxins_method ? `<div class="detail-note">${esc(contaminants.mycotoxins_method)}</div>` : "",
  ].join("");

  // Accreditation badges for safety section
  const accredBadges = [
    isoPass ? `<span class="accred-badge accred-green">ISO 17025</span>` : "",
    sccPass ? `<span class="accred-badge accred-green">SCC</span>` : "",
    (!isoPass && !sccPass) ? `<span class="accred-badge accred-grey">Accreditation not confirmed</span>` : "",
  ].join("");

  // Panel completeness badge
  const panelCount = [
    contaminants.pesticides_status,
    contaminants.heavy_metals_status,
    contaminants.microbials_status,
    contaminants.mycotoxins_status,
  ].filter(s => s && !/not tested/i.test(s)).length;
  const panelBadge = panelCount === 4
    ? `<span class="panel-badge panel-complete">✓ Complete Panel</span>`
    : `<span class="panel-badge panel-partial">⚠ Partial Panel (${panelCount}/4)</span>`;

  // Audience tabs (CSS-only via radio inputs)
  const tabPanels = [
    { id: "brand",    label: "Brand",    icon: "◈", content: audienceList(audiences.brand)    },
    { id: "clinical", label: "Clinical", icon: "✚", content: audienceList(audiences.clinical) },
    { id: "patient",  label: "Patient",  icon: "◯", content: audienceList(audiences.patient)  },
    { id: "buyer",    label: "Buyer",    icon: "◇", content: audienceList(audiences.buyer)    },
  ];

  // Entourage signal
  const detectedCannCount = cannabinoids.filter(c => toNum(c.value) > 0).length;
  const entourageStrength = detectedCannCount >= 5 ? "Strong" : detectedCannCount >= 3 ? "Moderate" : "Limited";

  // ── HTML ─────────────────────────────────────────────────────────────────
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${esc(productName)} — Alem Chemical Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── Reset ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Design tokens ──────────────────────────────────────────────── */
:root {
  --cream:        #f7f5ef;
  --ink:          #1a1a1a;
  --ink-muted:    #555550;
  --ink-faint:    #999992;
  --green:        #16855c;
  --green-light:  #eaf4ef;
  --green-mid:    #cce8da;
  --amber:        #c8860a;
  --red:          #b93030;
  --card-bg:      #ffffff;
  --card-shadow:  0 1px 4px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.04);
  --radius:       6px;
}

/* ── Base ────────────────────────────────────────────────────────── */
body {
  background: var(--cream);
  font-family: 'DM Sans', system-ui, sans-serif;
  color: var(--ink);
  line-height: 1.6;
  min-height: 100vh;
}

.report-wrap {
  max-width: 860px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

/* ── Section spacing ─────────────────────────────────────────────── */
.report-section { margin-top: 48px; }
.section-heading {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 20px;
  color: var(--ink);
  margin-bottom: 4px;
}
.section-sub {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
  color: var(--ink-faint);
  margin-bottom: 20px;
}

/* ── Card ─────────────────────────────────────────────────────────── */
.card {
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  border-radius: var(--radius);
  padding: 20px 24px;
}

/* ═══════════════════════════════════════════════════════════════════
   1. MASTHEAD
══════════════════════════════════════════════════════════════════ */
.masthead { margin-bottom: 8px; }
.masthead-eyebrow {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 10px;
  color: var(--green);
  letter-spacing: 0.15em;
  text-transform: uppercase;
}
.masthead-name {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 36px;
  font-weight: 700;
  color: var(--ink);
  line-height: 1.1;
  margin-top: 8px;
}
.masthead-sub {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 14px;
  color: var(--ink-muted);
  margin-top: 6px;
}
.masthead-rule {
  border: none;
  border-top: 1px solid rgba(22,133,92,0.30);
  margin: 16px 0;
}
.masthead-meta {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 11px;
  color: var(--ink-faint);
}

/* ═══════════════════════════════════════════════════════════════════
   2. INTELLIGENCE SCORE HERO
══════════════════════════════════════════════════════════════════ */
.hero-card {
  background: var(--cream);
  box-shadow: var(--card-shadow);
  border-radius: var(--radius);
  border-top: 3px solid var(--green);
  padding: 28px 28px 24px;
  display: flex;
  gap: 32px;
  align-items: flex-start;
}
.hero-left { flex: 0 0 58%; }
.hero-right { flex: 1; }

.hero-score-row {
  display: flex;
  align-items: baseline;
  gap: 6px;
  margin-bottom: 8px;
}
.hero-score-num {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 72px;
  color: var(--green);
  line-height: 1;
}
.hero-score-denom {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 18px;
  color: var(--ink-faint);
}

.grade-badge {
  display: inline-block;
  background: var(--green);
  color: #fff;
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 16px;
  border-radius: 20px;
  padding: 3px 14px;
  margin-right: 8px;
}
.tier-label {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-style: italic;
  font-size: 13px;
  color: var(--ink-muted);
}

.hero-narrative {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 14px;
  color: #333333;
  margin-top: 16px;
  line-height: 1.55;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Sub-score bars */
.sb-row {
  display: grid;
  grid-template-columns: 120px 1fr 52px;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.sb-lbl {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
  color: var(--ink-muted);
}
.sb-track {
  background: #e8e8e0;
  height: 4px;
  border-radius: 2px;
  overflow: hidden;
}
.sb-fill {
  height: 100%;
  background: var(--green);
  border-radius: 2px;
  width: 0;
  transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}
.sb-num {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 11px;
  color: var(--ink-faint);
  text-align: right;
}

/* ═══════════════════════════════════════════════════════════════════
   3. KEY METRICS STRIP
══════════════════════════════════════════════════════════════════ */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
.metric-card {
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  border-radius: var(--radius);
  padding: 16px;
}
.metric-val {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 24px;
  color: var(--green);
  line-height: 1.1;
}
.metric-lbl {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 11px;
  text-transform: uppercase;
  color: var(--ink-faint);
  letter-spacing: 0.08em;
  margin-top: 4px;
}
.metric-ctx {
  display: inline-block;
  background: var(--green-light);
  color: var(--green);
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 12px;
  margin-top: 6px;
}

/* ═══════════════════════════════════════════════════════════════════
   4. CHEMOTYPE FINGERPRINT BAND
══════════════════════════════════════════════════════════════════ */
.fingerprint-band {
  background: var(--green-light);
  border: 1px solid var(--green-mid);
  border-radius: var(--radius);
  padding: 14px 24px;
  display: flex;
  align-items: center;
  gap: 24px;
  flex-wrap: wrap;
}
.fp-code {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 22px;
  font-weight: 500;
  color: var(--green);
  flex-shrink: 0;
}
.fp-center {
  flex: 1;
  min-width: 160px;
}
.fp-dir {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink);
}
.fp-conf {
  display: inline-block;
  background: var(--green);
  color: #fff;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 12px;
  margin-left: 6px;
}
.fp-terps {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}
.fp-terpname {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 11px;
  color: var(--ink-muted);
}

/* ═══════════════════════════════════════════════════════════════════
   5. TERPENE SECTION
══════════════════════════════════════════════════════════════════ */
.terp-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 20px;
}
.terp-pill {
  background: var(--green-light);
  color: var(--green);
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 12px;
  padding: 4px 12px;
  border-radius: 20px;
}

.dominant-callout {
  background: var(--green-light);
  border-left: 3px solid var(--green);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 16px 20px;
  margin-bottom: 20px;
}
.dominant-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 6px;
}
.dominant-name {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 24px;
  color: var(--ink);
}
.dominant-val {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 20px;
  color: var(--green);
}
.dominant-aroma {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink-muted);
  margin-bottom: 4px;
}
.dominant-bio {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
  color: var(--ink-faint);
  line-height: 1.55;
}

/* Terpene table */
.terp-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
}
.terp-table th {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-faint);
  padding: 8px 6px;
  text-align: left;
  border-bottom: 1px solid #e8e8e0;
}
.terp-table td { padding: 7px 6px; vertical-align: middle; }
.trow-rank { color: var(--ink-faint); font-size: 11px; width: 28px; }
.trow-name { color: var(--ink); }
.trow-name.lead { font-weight: 600; }
.trow-name.blq  { color: var(--ink-faint); font-style: italic; }
.trow-bar { width: 100px; }
.tbar-track { height: 6px; background: #e8e8e0; border-radius: 3px; overflow: hidden; }
.tbar-fill  { height: 100%; background: var(--green); border-radius: 3px; width: 0; transition: width 0.7s cubic-bezier(0.16,1,0.3,1); }
.trow-val   { font-family: 'DM Mono','Courier New',monospace; font-size: 11px; color: var(--green); white-space: nowrap; }
.trow-val.blq { color: var(--ink-faint); font-style: italic; }
.trow-aroma { color: var(--ink-faint); font-size: 11px; max-width: 200px; }

/* ═══════════════════════════════════════════════════════════════════
   6. CANNABINOID PROFILE
══════════════════════════════════════════════════════════════════ */
.cann-total-stat {
  margin-bottom: 16px;
}
.cann-total-val {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 28px;
  color: var(--green);
}
.cann-total-lbl {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
  color: var(--ink-faint);
}
.cann-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
.cann-cell {
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  border-radius: var(--radius);
  padding: 14px 16px;
  position: relative;
}
.cann-cell-name {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 4px;
}
.cann-cell-val {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 18px;
  color: var(--green);
}
.cann-cell-val.nd { color: var(--ink-faint); font-style: italic; }
.cann-cell-unit {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 10px;
  color: var(--ink-faint);
}
.cann-cell-note {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 10px;
  color: var(--ink-faint);
  margin-top: 2px;
}

.entourage-card {
  border-left: 3px solid var(--amber);
  background: #fffbf5;
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 14px 18px;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink-muted);
}
.entourage-card strong { color: var(--amber); }

/* ═══════════════════════════════════════════════════════════════════
   7. FLAVONOIDS
══════════════════════════════════════════════════════════════════ */
.flav-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.flav-pill {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
  padding: 5px 12px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.flav-pill-green   { background: var(--green-light); color: var(--green); }
.flav-pill-default { background: #f0f0ea; color: var(--ink-muted); }
.flav-pill-val     { font-family: 'DM Mono','Courier New',monospace; font-size: 11px; }

/* ═══════════════════════════════════════════════════════════════════
   8. POST-HARVEST SIGNALS
══════════════════════════════════════════════════════════════════ */
.ph-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}
.ph-card {
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  border-radius: var(--radius);
  padding: 16px;
}
.ph-card-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.ph-icon  { font-size: 16px; }
.ph-name  { font-family: 'DM Sans', system-ui, sans-serif; font-size: 13px; font-weight: 600; flex: 1; }
.ph-note  { font-family: 'DM Sans', system-ui, sans-serif; font-size: 12px; color: var(--ink-muted); line-height: 1.5; }

/* Status pills */
.status-pill {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 12px;
  white-space: nowrap;
}
.pill-green { background: var(--green-light); color: var(--green); }
.pill-amber { background: #fef5e7; color: var(--amber); }
.pill-grey  { background: #f0f0ea; color: var(--ink-faint); }

/* ═══════════════════════════════════════════════════════════════════
   9. QUALITY & SAFETY
══════════════════════════════════════════════════════════════════ */
.safety-head-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.panel-badge {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 12px;
}
.panel-complete { background: var(--green-light); color: var(--green); }
.panel-partial  { background: #fef5e7; color: var(--amber); }

.safety-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
.safety-card {
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  border-radius: var(--radius);
  padding: 14px 16px;
}
.safety-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.safety-title { font-family: 'DM Sans', system-ui, sans-serif; font-size: 13px; font-weight: 600; }
.safety-badge {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 4px;
}
.badge-pass { background: var(--green-light); color: var(--green); }
.badge-fail { background: #fdecea; color: var(--red); }
.badge-nt   { background: #f0f0ea; color: var(--ink-faint); }

.safety-method { font-family: 'DM Mono','Courier New',monospace; font-size: 10px; color: var(--ink-faint); margin-bottom: 8px; }
.safety-details { font-family: 'DM Sans', system-ui, sans-serif; font-size: 12px; }

.detail-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
}
.detail-dot   { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.dot-green    { background: var(--green); }
.dot-red      { background: var(--red); }
.detail-name  { color: var(--ink-muted); flex: 1; }
.detail-val   { font-family: 'DM Mono','Courier New',monospace; font-size: 11px; }
.val-green    { color: var(--green); }
.val-red      { color: var(--red); }
.detail-note  { font-size: 11px; color: var(--ink-faint); margin-top: 4px; line-height: 1.4; }
.nt-note      { font-style: italic; }

.accred-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.accred-badge {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 12px;
}
.accred-green { background: var(--green-light); color: var(--green); }
.accred-grey  { background: #f0f0ea; color: var(--ink-faint); }

.stability-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.stability-pill {
  background: var(--green-light);
  color: var(--green);
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 12px;
}

/* ═══════════════════════════════════════════════════════════════════
   10. AUDIENCE INTELLIGENCE — CSS-only tabs
══════════════════════════════════════════════════════════════════ */
.tab-radio { display: none; }

.tab-bar {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid #e8e8e0;
}
.tab-lbl {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink-faint);
  padding: 8px 14px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color 0.15s, border-color 0.15s;
}
.tab-lbl:hover { color: var(--green); }
.tab-icon { font-size: 11px; }

.tab-content { display: none; }

/* show each tab when its radio is checked */
#tab-brand:checked    ~ .tab-bar label[for="tab-brand"],
#tab-clinical:checked ~ .tab-bar label[for="tab-clinical"],
#tab-patient:checked  ~ .tab-bar label[for="tab-patient"],
#tab-buyer:checked    ~ .tab-bar label[for="tab-buyer"] {
  color: var(--green);
  border-bottom-color: var(--green);
}
#tab-brand:checked    ~ .tab-panels #tc-brand,
#tab-clinical:checked ~ .tab-panels #tc-clinical,
#tab-patient:checked  ~ .tab-panels #tc-patient,
#tab-buyer:checked    ~ .tab-panels #tc-buyer {
  display: block;
}

.aud-list {
  list-style: none;
  border-left: 3px solid var(--green);
  padding-left: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.aud-list li {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 14px;
  color: #333333;
  line-height: 1.6;
}
.aud-empty {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink-faint);
  font-style: italic;
}

/* ═══════════════════════════════════════════════════════════════════
   11. LAB CARD
══════════════════════════════════════════════════════════════════ */
.lab-card {
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}
.lab-name {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: 16px;
  color: var(--ink);
  flex: 1;
}
.lab-meta {
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 11px;
  color: var(--ink-faint);
}
.lab-date {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
  color: var(--ink-muted);
}

/* ═══════════════════════════════════════════════════════════════════
   12. FOOTER CTA
══════════════════════════════════════════════════════════════════ */
.footer-cta {
  background: var(--cream);
  text-align: center;
  padding: 48px 24px;
  margin-top: 48px;
}
.footer-logo { height: 24px; display: inline-block; }
.footer-logo img { height: 24px; }
.footer-tagline {
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  color: var(--ink-muted);
  margin-top: 10px;
  margin-bottom: 20px;
}
.footer-btns { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.btn-primary {
  background: var(--green);
  color: #fff;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  font-weight: 600;
  padding: 10px 22px;
  border-radius: var(--radius);
  text-decoration: none;
  border: 2px solid var(--green);
  display: inline-block;
}
.btn-outline {
  background: transparent;
  color: var(--green);
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  font-weight: 600;
  padding: 10px 22px;
  border-radius: var(--radius);
  text-decoration: none;
  border: 2px solid var(--green);
  display: inline-block;
}
.btn-primary:hover { background: #12724e; border-color: #12724e; }
.btn-outline:hover { background: var(--green-light); }
.footer-report-id {
  margin-top: 16px;
  font-family: 'DM Mono', 'Courier New', monospace;
  font-size: 10px;
  color: #bbbbbb;
}

/* ── Animation trigger (JS fallback: bars animate on load) ── */
@keyframes barGrow { from { width: 0 } to { width: var(--w, 100%) } }

/* ═══════════════════════════════════════════════════════════════════
   MARKET BENCHMARK SECTION
══════════════════════════════════════════════════════════════════ */
.bm-card { padding: 20px 24px; }
.bm-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px 32px;
}
@media (max-width: 560px) { .bm-grid { grid-template-columns: 1fr; } }
.bm-metric { display: flex; flex-direction: column; gap: 8px; }
.bm-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--ink-faint);
}
.bm-track-row { display: flex; align-items: center; gap: 10px; }
.bm-track {
  flex: 1;
  height: 8px;
  background: #ececec;
  border-radius: 4px;
  overflow: hidden;
}
.bm-fill {
  height: 100%;
  border-radius: 4px;
  background: var(--green);
  transition: width 0.6s ease;
}
.bm-fill-amber { background: var(--amber); }
.bm-fill-grey  { background: #bbb; }
.bm-badge {
  flex-shrink: 0;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 9px;
  border-radius: 20px;
  white-space: nowrap;
}
.bm-badge-top    { background: var(--green-light);  color: var(--green); }
.bm-badge-mid    { background: #fff8e7;              color: var(--amber); }
.bm-badge-low    { background: #f3f3f3;              color: #888; }
.bm-compare {
  display: flex;
  align-items: baseline;
  gap: 6px;
  font-size: 12px;
  color: var(--ink-muted);
  flex-wrap: wrap;
}
.bm-this-val  { font-size: 18px; font-weight: 700; color: var(--ink); font-family: 'DM Mono', monospace; }
.bm-vs        { font-size: 10px; color: var(--ink-faint); }
.bm-median    { font-size: 12px; color: var(--ink-muted); }
.bm-delta-up  { font-size: 11px; font-weight: 600; color: var(--green); }
.bm-delta-dn  { font-size: 11px; font-weight: 600; color: var(--amber); }
.bm-divider   { border: none; border-top: 1px solid #eee; margin: 16px 0; }
.bm-context-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  font-size: 11px;
  color: var(--ink-faint);
}
.bm-context-item strong { color: var(--ink-muted); font-weight: 600; }
</style>
</head>
<body>
<div class="report-wrap">

<!-- ═══════════════════════════════════════════════════════════════
     1. MASTHEAD
════════════════════════════════════════════════════════════════ -->
<div class="masthead">
  <div class="masthead-eyebrow">Chemical Intelligence Report</div>
  <div class="masthead-name">${esc(productName)}</div>
  <div class="masthead-sub">${
    [chemistry.product_type, chemistry.batch_number, chemistry.laboratory_name]
      .filter(Boolean).map(esc).join(" · ")
  }</div>
  <hr class="masthead-rule">
  <div class="masthead-meta">${esc(chemistry.coa_report_date || "")} · Schema v7.0${chemistry.laboratory_name ? " · " + esc(chemistry.laboratory_name) : ""}</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     2. INTELLIGENCE SCORE HERO
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="hero-card">
  <div class="hero-left">
    <div class="hero-score-row">
      <span class="hero-score-num">${safeTotal}</span>
      <span class="hero-score-denom">/100</span>
    </div>
    <div>
      <span class="grade-badge">${esc(safeGrade)}</span>
      <span class="tier-label">${esc(safeTier)}</span>
    </div>
    <div class="hero-narrative">${esc(heroNarrative)}</div>
  </div>
  <div class="hero-right">
    ${scoreBar("Potency",          safePotency.score,  safePotency.max)}
    ${scoreBar("Terpenes",         safeTerpenes.score, safeTerpenes.max)}
    ${scoreBar("Minors",           safeMinors.score,   safeMinors.max)}
    ${scoreBar("Safety",           safeSafety.score,   safeSafety.max)}
    ${scoreBar("Data Completeness",safeData.score,     safeData.max)}
  </div>
</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     3. KEY METRICS STRIP
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="metrics-grid">
  ${metricCard(
    chemistry.thc_total ? chemistry.thc_total + " " + (chemistry.thc_total_unit || "wt%") : "ND",
    "THC TOTAL",
    thc >= 24 ? "Above avg." : thc >= 18 ? "Moderate-high" : thc > 0 ? "Moderate" : "Not detected"
  )}
  ${metricCard(
    chemistry.thca ? chemistry.thca + " wt%" : "ND",
    "THCA",
    "× 0.877 on heat"
  )}
  ${metricCard(
    chemistry.total_terpenes ? chemistry.total_terpenes + " " + (chemistry.total_terpenes_unit || "wt%") : "ND",
    "TOTAL TERPENES",
    terps >= 3 ? "Exceptional" : terps >= 2 ? "Strong" : terps > 0 ? "Moderate" : "Not reported"
  )}
  ${metricCard(
    chemistry.cbd_total && chemistry.cbd_total !== "ND" ? chemistry.cbd_total : "ND",
    "CBD TOTAL",
    cbd >= 0.5 ? "Modulating" : "Not detected"
  )}
</div>
</div>

${(() => {
  if (!benchmark) return "";
  const {
    n, formFactorLabel,
    thcPercentile, terpPercentile,
    medianThc, medianTerp,
    p90Thc, p90Terp,
    supplierCount,
  } = benchmark;

  const thcVal  = parseFloat(chemistry.thc_total)      || 0;
  const terpVal = parseFloat(chemistry.total_terpenes) || 0;

  function pctBadge(pct) {
    const top = 100 - pct;
    const cls = top <= 20 ? "bm-badge-top" : top <= 50 ? "bm-badge-mid" : "bm-badge-low";
    return `<span class="bm-badge ${cls}">Top ${top}%</span>`;
  }
  function fillCls(pct) {
    const top = 100 - pct;
    return top <= 20 ? "" : top <= 50 ? "bm-fill-amber" : "bm-fill-grey";
  }
  function delta(actual, median) {
    if (!median || !actual) return "";
    const d = ((actual - median) / median * 100).toFixed(1);
    if (d > 0)  return `<span class="bm-delta-up">+${d}% above median</span>`;
    if (d < 0)  return `<span class="bm-delta-dn">${d}% below median</span>`;
    return "";
  }

  const thcMetric = `
    <div class="bm-metric">
      <div class="bm-label">THC Potency Rank</div>
      <div class="bm-track-row">
        <div class="bm-track"><div class="bm-fill ${fillCls(thcPercentile)}" style="width:${thcPercentile}%"></div></div>
        ${pctBadge(thcPercentile)}
      </div>
      <div class="bm-compare">
        <span class="bm-this-val">${esc(String(thcVal))} wt%</span>
        <span class="bm-vs">vs</span>
        <span class="bm-median">${medianThc != null ? medianThc + " wt% median" : "—"}</span>
        ${delta(thcVal, medianThc)}
      </div>
    </div>`;

  const terpMetric = terpPercentile != null ? `
    <div class="bm-metric">
      <div class="bm-label">Terpene Richness Rank</div>
      <div class="bm-track-row">
        <div class="bm-track"><div class="bm-fill ${fillCls(terpPercentile)}" style="width:${terpPercentile}%"></div></div>
        ${pctBadge(terpPercentile)}
      </div>
      <div class="bm-compare">
        <span class="bm-this-val">${esc(String(terpVal))} wt%</span>
        <span class="bm-vs">vs</span>
        <span class="bm-median">${medianTerp != null ? medianTerp + " wt% median" : "—"}</span>
        ${delta(terpVal, medianTerp)}
      </div>
    </div>` : "";

  return `
<!-- ═══════════════════════════════════════════════════════════════
     3b. MARKET BENCHMARK
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="section-heading">Market Benchmark</div>
<div class="section-sub">${esc(formFactorLabel)} · ${n.toLocaleString()} COAs · last 2 years${supplierCount ? " · " + supplierCount.toLocaleString() + " suppliers" : ""}</div>
<div class="card bm-card">
  <div class="bm-grid">
    ${thcMetric}
    ${terpMetric}
  </div>
  ${(p90Thc != null || p90Terp != null) ? `
  <hr class="bm-divider">
  <div class="bm-context-row">
    ${p90Thc  != null ? `<span class="bm-context-item">Top 10% THC threshold: <strong>${p90Thc} wt%</strong></span>`  : ""}
    ${p90Terp != null ? `<span class="bm-context-item">Top 10% terpene threshold: <strong>${p90Terp} wt%</strong></span>` : ""}
  </div>` : ""}
</div>
</div>`;
})()}

<!-- ═══════════════════════════════════════════════════════════════
     4. CHEMOTYPE FINGERPRINT BAND
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="fingerprint-band">
  <div class="fp-code">${esc(fingerprintId)}</div>
  <div class="fp-center">
    <div class="fp-dir">
      ${esc(effectDir)}
      ${intelligence.lineageConfidence ? `<span class="fp-conf">${esc(intelligence.lineageConfidence)} confidence</span>` : ""}
    </div>
  </div>
  <div class="fp-terps">
    ${[leadTerpene, secondTerpene, thirdTerpene].filter(Boolean).map(n =>
      `<span class="fp-terpname">${esc(n)}</span>`
    ).join("")}
  </div>
</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     5. TERPENE FINGERPRINT
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="section-heading">Terpene Fingerprint</div>
<div class="section-sub">Aroma compounds detected in this profile</div>

<!-- Summary pills -->
<div class="terp-pills">
  <span class="terp-pill">${esc(chemistry.total_terpenes || "—")} wt% total</span>
  <span class="terp-pill">${terpCount} compound${terpCount !== 1 ? "s" : ""} detected</span>
  ${blqCount > 0 ? `<span class="terp-pill">${blqCount} at BLQ</span>` : ""}
</div>

${dominantCallout()}

${terpenes.length > 1 ? `
<div class="card" style="padding:0;overflow:hidden;">
  <table class="terp-table">
    <thead><tr>
      <th>#</th><th>Compound</th><th>Relative</th><th>Value</th><th>Aroma note</th>
    </tr></thead>
    <tbody>
      ${terpenes.map((t, i) => i > 0 ? terpRow(t, i) : "").join("")}
    </tbody>
  </table>
</div>
` : ""}
</div>

<!-- ═══════════════════════════════════════════════════════════════
     6. CANNABINOID PROFILE
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="section-heading">Cannabinoid Profile</div>
<div class="section-sub">Full cannabinoid panel from COA</div>

${chemistry.total_cannabinoids ? `
<div class="cann-total-stat">
  <span class="cann-total-val">${esc(chemistry.total_cannabinoids)} wt%</span>
  <span class="cann-total-lbl"> total cannabinoids</span>
</div>` : ""}

<div class="cann-grid">
${cannabinoids.length ? cannabinoids.map(c => {
  const noteMap = {
    "THCA":  "Precursor · ×0.877 on heat",
    "D9-THC":"Active form",
    "CBNA":  "Degradation marker (acid)",
    "CBN":   "Degradation marker",
    "CBGA":  "Mother cannabinoid",
  };
  const safeVal = String(c.value ?? "");
  const isNd = !safeVal || safeVal === "ND" || safeVal === "" || toNum(safeVal) === 0;
  const note = c.notes || noteMap[c.name] || "";
  return `
  <div class="cann-cell">
    <div class="cann-cell-name">${esc(c.name || "")}</div>
    <div class="cann-cell-val${isNd ? " nd" : ""}">
      ${isNd ? "ND" : `${esc(safeVal)}<span class="cann-cell-unit"> ${esc(c.unit || "wt%")}</span>`}
    </div>
    ${note ? `<div class="cann-cell-note">${esc(note)}</div>` : ""}
  </div>`;
}).join("") : `<div style="color:var(--ink-faint);font-family:'DM Sans',sans-serif;padding:16px;">No cannabinoid data reported.</div>`}
</div>

<!-- Entourage signal -->
${detectedCannCount >= 3 ? `
<div class="entourage-card">
  <strong>Entourage Effect Signal: ${esc(entourageStrength)}</strong> — ${detectedCannCount} cannabinoids + ${terpCount} terpenes detected.
  This profile has biochemical complexity associated with full-spectrum, synergistic activity.
</div>` : ""}

${chemistry.thc_total_anhydrous && chemistry.thc_total_anhydrous !== chemistry.thc_total ? `
<div style="margin-top:12px;padding:10px 14px;background:var(--green-light);border-radius:var(--radius);font-family:'DM Sans',sans-serif;font-size:12px;color:var(--ink-muted);">
  <strong>Anhydrous (moisture-corrected) THC:</strong> ${esc(chemistry.thc_total_anhydrous)} wt% &nbsp;·&nbsp;
  As-received: ${esc(chemistry.thc_total)} wt% &nbsp;·&nbsp;
  Moisture: ${esc(chemistry.moisture_content || "—")}%
</div>` : ""}
</div>

<!-- ═══════════════════════════════════════════════════════════════
     7. FLAVONOIDS (conditional)
════════════════════════════════════════════════════════════════ -->
${hasFlavonoids ? `
<div class="report-section">
<div class="section-heading">Flavonoid Profile</div>
<div class="section-sub">${flavonoids.length} flavonoid compound${flavonoids.length !== 1 ? "s" : ""} quantified${chemistry.total_flavonoids ? " · " + esc(chemistry.total_flavonoids) + " wt% total" : ""}</div>
<div class="card">
  <div class="flav-pills">
    ${flavonoids.filter(f => toNum(f.value) > 0 || String(f.value).toLowerCase() === "blq").map(f => flavPill(f)).join("")}
  </div>
</div>
</div>` : ""}

<!-- ═══════════════════════════════════════════════════════════════
     8. POST-HARVEST SIGNALS
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="section-heading">Post-Harvest Signals</div>
<div class="section-sub">Inferred from terpene chemistry and stability data</div>
<div class="ph-grid">
  ${phCard("🌿", "Freshness", safePH.freshness)}
  ${phCard("🔬", "Curing",    safePH.curing)}
  ${phCard("⚗️", "Degradation", safePH.degradation)}
  ${phCard("💧", "Stability", safePH.stability)}
</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     9. QUALITY & SAFETY
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="safety-head-row">
  <div class="section-heading" style="margin-bottom:0">Quality &amp; Safety</div>
  ${panelBadge}
</div>
<div class="safety-grid">
  ${safetyCard("Pesticides",    contaminants.pesticides_status,    pestPass,  contaminants.pesticides_method,    pesticideDetails)}
  ${safetyCard("Heavy Metals",  contaminants.heavy_metals_status,  metalsPass, contaminants.heavy_metals_method, metalsDetails)}
  ${safetyCard("Microbials",    contaminants.microbials_status,    microPass, contaminants.microbials_method,    microDetails)}
  ${safetyCard("Mycotoxins",    contaminants.mycotoxins_status,    mycoPass,  contaminants.mycotoxins_method,    mycoDetails)}
</div>

<div class="accred-row">${accredBadges}</div>

${(hasMoisture || hasWaterActivity) ? `
<div class="stability-pills">
  ${hasMoisture ? `<span class="stability-pill">Moisture: ${esc(moisture)}%</span>` : ""}
  ${hasWaterActivity ? `<span class="stability-pill">Water activity: ${esc(waterActivity)} aw</span>` : ""}
</div>` : ""}

${(pestPass && metalsPass && microPass && mycoPass) ? `
<div style="margin-top:14px;padding:12px 16px;background:var(--green-light);border-left:3px solid var(--green);border-radius:0 var(--radius) var(--radius) 0;font-family:'DM Sans',sans-serif;font-size:13px;color:var(--green);">
  <strong>✓ Full Compliance Panel — All Categories Clear.</strong>${isoPass ? " ISO 17025:2017 accreditation confirmed." : ""} ${sccPass ? "SCC accredited." : ""}
</div>` : (!contaminants.pesticides_status && !contaminants.heavy_metals_status) ? `
<div style="margin-top:14px;padding:12px 16px;background:#fef5e7;border-left:3px solid var(--amber);border-radius:0 var(--radius) var(--radius) 0;font-family:'DM Sans',sans-serif;font-size:13px;color:var(--ink-muted);">
  <strong>Incomplete safety panel:</strong> ${esc(contaminants.contaminant_narrative || "Contaminant panels were not captured from this COA. Request the full compliance documentation from the producer.")}
</div>` : ""}
</div>

<!-- ═══════════════════════════════════════════════════════════════
     10. AUDIENCE INTELLIGENCE (CSS-only tabs)
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="section-heading">Audience Intelligence</div>
<div class="section-sub">Tailored insights by stakeholder type</div>
<div class="card">
  ${tabSwitch(tabPanels)}
</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     11. LAB CARD
════════════════════════════════════════════════════════════════ -->
<div class="report-section">
<div class="card">
  <div class="lab-card">
    <div class="lab-name">${esc(chemistry.laboratory_name || "Laboratory not reported")}</div>
    ${isoPass ? `<span class="accred-badge accred-green">ISO 17025</span>` : ""}
    ${sccPass  ? `<span class="accred-badge accred-green">SCC</span>` : ""}
    ${chemistry.laboratory_method ? `<div class="lab-meta">${esc(chemistry.laboratory_method)}</div>` : ""}
    ${chemistry.coa_report_date ? `<div class="lab-date">${esc(chemistry.coa_report_date)}</div>` : ""}
  </div>
</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     12. FOOTER CTA
════════════════════════════════════════════════════════════════ -->
<div class="footer-cta">
  <div class="footer-logo">
    <img src="https://www.alem.solutions/wp-content/uploads/2024/04/Alem-Brand-Green.png"
         alt="ALEM" height="24"
         onerror="this.style.display='none';this.nextElementSibling.style.display='inline'">
    <span style="display:none;font-family:'Libre Baskerville',serif;font-size:18px;color:var(--green);font-weight:700;">ALEM</span>
  </div>
  <div class="footer-tagline">Upload your COA. Understand your chemistry.</div>
  <div class="footer-btns">
    <a class="btn-primary" href="/">Analyse New COA</a>
    ${options.documentId ? `<a class="btn-outline" href="/pdf/${esc(String(options.documentId))}" target="_blank">Export PDF</a>` : ""}
  </div>
  ${options.documentId ? `<div class="footer-report-id">Report ID: ${esc(String(options.documentId))}</div>` : ""}
</div>

</div><!-- /.report-wrap -->

<script>
// Animate bars after paint
(function() {
  function animateBars() {
    document.querySelectorAll('.sb-fill[data-pct]').forEach(function(el) {
      el.style.width = el.getAttribute('data-pct') + '%';
    });
    document.querySelectorAll('.tbar-fill[data-w]').forEach(function(el) {
      el.style.width = el.getAttribute('data-w') + '%';
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', animateBars);
  } else {
    requestAnimationFrame(animateBars);
  }
})();
</script>
</body>
</html>`;
};
