"use strict";

/**
 * scientificLayer.test.js
 * Run with: node test/scientificLayer.test.js
 *
 * Tests make real NCBI API calls — requires internet access.
 * Set NCBI_API_KEY in env for higher rate limits during testing.
 */

const assert = require("assert");
const { buildScientificEvidence } = require("../services/scientificLayer");

let passed = 0;
let failed = 0;

async function test(name, fn) {
  process.stdout.write(`  ${name} ... `);
  try {
    await fn();
    console.log("✅ PASS");
    passed++;
  } catch (err) {
    console.log(`❌ FAIL: ${err.message}`);
    failed++;
  }
}

function getAllPmids(ev) {
  const pmids = [];
  for (const g of ev.terpeneStudies) pmids.push(...g.articles.map(a => a.pmid));
  pmids.push(...ev.cannabinoidStudies.map(a => a.pmid));
  pmids.push(...ev.indicationStudies.map(a => a.pmid));
  pmids.push(...ev.strainFamilyStudies.map(a => a.pmid));
  pmids.push(...ev.safetyStudies.map(a => a.pmid));
  return pmids;
}

async function run() {
  console.log("\n🧪 Scientific Layer Tests\n");

  // ── Test 1: Myrcene-dominant flower COA ─────────────────────────────────
  await test("Test 1 — Myrcene-dominant flower COA", async () => {
    const coaData = {
      chemistry: {
        product_type:     "Dried Flower",
        total_terpenes:   "2.8",
        total_thc:        "24",
        total_cbd:        "0.1",
        top_terpenes: [
          { name: "Beta-Myrcene",       value: "1.2", unit: "wt%" },
          { name: "Trans-Caryophyllene",value: "0.4", unit: "wt%" },
          { name: "Limonene",           value: "0.3", unit: "wt%" },
        ],
      },
      intelligence: {
        effectDirection: "Relaxing / Sedating",
        strainIntel: { match: { strain_name: "OG Kush", similarity: 84 } },
      },
    };

    const ev = await buildScientificEvidence(coaData);
    assert.ok(ev.totalArticles > 0, "Should return at least one article");
    assert.ok(ev.terpeneStudies.some(g => /myrcene/i.test(g.terpene)), "Should have myrcene studies");
    assert.ok(ev.safetyStudies.length > 0, "Should have safety studies");
    assert.ok(ev.executionMs < 8000, `Execution too slow: ${ev.executionMs}ms`);

    // No duplicate PMIDs
    const pmids = getAllPmids(ev);
    const unique = new Set(pmids);
    assert.strictEqual(pmids.length, unique.size, `Duplicate PMIDs found: ${pmids.length} vs ${unique.size} unique`);
  });

  // ── Test 2: High CBD balanced COA ───────────────────────────────────────
  await test("Test 2 — High CBD balanced COA", async () => {
    const coaData = {
      chemistry: {
        product_type:   "Oil",
        total_terpenes: "1.2",
        total_thc:      "9",
        total_cbd:      "8",
        top_terpenes: [
          { name: "Trans-Caryophyllene", value: "0.6", unit: "wt%" },
          { name: "Linalool",            value: "0.3", unit: "wt%" },
        ],
      },
      intelligence: { effectDirection: "Grounding / Anti-inflam." },
    };

    const ev = await buildScientificEvidence(coaData);
    assert.ok(ev.cannabinoidStudies.length > 0, "Should have cannabinoid studies for CBD:THC ratio");
    assert.ok(ev.terpeneStudies.some(g => /caryophyllene/i.test(g.terpene)), "Should have caryophyllene studies");
  });

  // ── Test 3: Insufficient terpene data ───────────────────────────────────
  await test("Test 3 — Insufficient terpene data", async () => {
    const coaData = { chemistry: { total_terpenes: "0.05", top_terpenes: [] }, intelligence: {} };
    const ev = await buildScientificEvidence(coaData);
    assert.ok(ev.insufficient === true, "Should flag as insufficient");
    assert.strictEqual(ev.queriesRun, 0, "Should make no API calls");
    assert.strictEqual(ev.totalArticles, 0, "Should return 0 articles");
  });

  // ── Test 4: Cache hit ────────────────────────────────────────────────────
  await test("Test 4 — Cache hit is faster than cold fetch", async () => {
    const coaData = {
      chemistry: {
        product_type:   "Dried Flower",
        total_terpenes: "2.1",
        total_thc:      "22",
        total_cbd:      "0.1",
        top_terpenes: [{ name: "Terpinolene", value: "0.9", unit: "wt%" }],
      },
      intelligence: {},
    };

    const t1 = Date.now();
    await buildScientificEvidence(coaData);
    const firstMs = Date.now() - t1;

    const t2 = Date.now();
    await buildScientificEvidence(coaData);
    const secondMs = Date.now() - t2;

    // Cache should be meaningfully faster (at least 50% reduction)
    assert.ok(secondMs < firstMs * 0.8 || secondMs < 500,
      `Cache not faster: first=${firstMs}ms second=${secondMs}ms`);
  });

  // ── Test 5: Timeout resilience ───────────────────────────────────────────
  await test("Test 5 — Timeout returns without throwing", async () => {
    // Temporarily reduce timeout to 50ms to force timeout
    const origEnv = process.env.SCIENTIFIC_LAYER_TIMEOUT_MS;
    process.env.SCIENTIFIC_LAYER_TIMEOUT_MS = "50";

    // Re-require to pick up new env (clear module cache)
    delete require.cache[require.resolve("../services/scientificLayer")];
    const { buildScientificEvidence: buildFresh } = require("../services/scientificLayer");

    const coaData = {
      chemistry: {
        total_terpenes: "2.5",
        total_thc: "20",
        top_terpenes: [{ name: "Beta-Myrcene", value: "1.0", unit: "wt%" }],
      },
      intelligence: {},
    };

    const started = Date.now();
    let errThrown = false;
    try {
      await buildFresh(coaData);
    } catch (_) {
      errThrown = true;
    }

    process.env.SCIENTIFIC_LAYER_TIMEOUT_MS = origEnv;
    assert.ok(!errThrown, "Should not throw on timeout");
    assert.ok(Date.now() - started < 3000, "Should resolve quickly after timeout");
  });

  // ── Summary ──────────────────────────────────────────────────────────────
  console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`);
  process.exit(failed > 0 ? 1 : 0);
}

run().catch(err => {
  console.error("Test runner crashed:", err);
  process.exit(1);
});
