const express = require("express");

const app = express();

// Accept raw text instead of JSON
app.use(express.text({ type: "*/*", limit: "10mb" }));

const EXPECTED_FIELDS = [
  "product_name",
  "batch_number",
  "coa_report_date",
  "product_type",
  "laboratory_name",
  "overall_score",
  "opening_statement",
  "thc_total",
  "cbd_total",
  "total_terpenes",
  "minor_cannabinoids",
  "terpene_list_left",
  "terpene_list_right",
  "contaminants_status",
  "tgo93_status",
  "regulatory_summary",
  "complexity_score",
  "coa_score",
  "quality_check_terpene_test",
  "quality_check_cannabinoid_test",
  "quality_check_contaminants_test",
  "quality_check_lab_information",
  "flavor_profile",
  "experience_profile",
  "use_cases",
  "scientific_notes"
];

function normalizeReportData(input) {
  const safe = {};

  for (const key of EXPECTED_FIELDS) {
    let value = input[key];

    if (
      value === undefined ||
      value === null ||
      value === "null" ||
      value === "undefined"
    ) {
      value = "";
    }

    if (typeof value !== "string") {
      value = String(value);
    }

    safe[key] = value.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  }

  return safe;
}

app.get("/", (req, res) => {
  res.send("Middleware is running");
});

app.post("/generate-pdf", (req, res) => {
  try {
    const rawBody = req.body;

    console.log("RAW BODY:");
    console.log(rawBody);

    const parsed = JSON.parse(rawBody);
    const data = normalizeReportData(parsed);

    console.log("NORMALIZED REPORT:");
    console.log(JSON.stringify(data, null, 2));

    return res.json({
      success: true,
      message: "Payload received successfully",
      placeholders: data,
      pdf_url: `https://example.com/reports/${data.batch_number || "test"}.pdf`
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