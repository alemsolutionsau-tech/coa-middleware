"""
ALEM COA DATABASE — BigQuery → Supabase Upload Script
------------------------------------------------------
Reads from BigQuery in batches, cleans data, uploads to Supabase.

Requirements:
  pip install google-cloud-bigquery supabase pandas python-dotenv

Setup:
  1. Create a .env file with your credentials (see below)
  2. Run: python upload_to_supabase.py
"""

import os
import json
import math
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import bigquery
from supabase import create_client, Client

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────
SUPABASE_URL      = os.getenv("SUPABASE_URL")        # https://xxxx.supabase.co
SUPABASE_KEY      = os.getenv("SUPABASE_SERVICE_KEY") # service_role key (not anon)
BQ_PROJECT        = os.getenv("BQ_PROJECT", "alem-coa-ai")
BQ_DATASET        = os.getenv("BQ_DATASET", "cannabis_coa")
BQ_TABLE          = os.getenv("BQ_TABLE",   "coa_results")
BATCH_SIZE        = 500     # rows per Supabase upsert
SLEEP_BETWEEN     = 0.3     # seconds between batches
BQ_BATCH_SIZE     = 10_000  # rows fetched from BQ at a time

# Numeric cannabinoid/terpene/safety columns — coerce to float
NUMERIC_COLS = [
    "total_thc","total_cbd","total_cannabinoids","sum_of_cannabinoids",
    "total_potential_thc","thca","delta_9_thc","delta_8_thc","cbd","cbda",
    "cbg","cbga","cbc","cbca","cbn","thcv","thcva","cbdv","cbdva","cbl",
    "total_terpenes","beta_myrcene","beta_caryophyllene","d_limonene",
    "alpha_pinene","beta_pinene","linalool","terpinolene","alpha_humulene",
    "alpha_bisabolol","caryophyllene_oxide","ocimene","camphene","borneol",
    "fenchol","geraniol","guaiol","nerolidol","trans_nerolidol","valencene",
    "terpineol","eucalyptol","arsenic","cadmium","lead","mercury","chromium",
    "total_aflatoxins","aflatoxin_b1","aflatoxin_b2","aflatoxin_g1","aflatoxin_g2",
    "ochratoxin_a","water_activity","total_yeast_and_mold",
    "total_aerobic_microbial_count","total_viable_aerobic_bacteria",
    "thc_cbd_ratio","thc_cbn_ratio","lab_latitude","lab_longitude",
]

DATE_COLS = [
    "date_tested","date_collected","date_received","date_sampled",
    "date_harvested","date_released","coa_parsed_at",
]

# ── CLEAN ─────────────────────────────────────────────────────
def clean_batch(df: pd.DataFrame) -> list[dict]:
    """Clean a DataFrame batch and return list of dicts for Supabase."""

    # Rename BQ V2 column name mangling back
    df.columns = [c.lower().strip() for c in df.columns]
    if "date_tested_1" in df.columns:
        df.drop(columns=["date_tested_1"], inplace=True, errors="ignore")

    # Coerce numerics — ND, <LOQ, N/A → None
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce dates
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )

    # Parse results JSON column
    if "results" in df.columns:
        def safe_json(val):
            if pd.isna(val) or val == "":
                return None
            if isinstance(val, (dict, list)):
                return val
            try:
                return json.loads(val.replace("'", '"'))
            except Exception:
                return None
        df["results"] = df["results"].apply(safe_json)

    # Replace NaN with None
    df = df.where(pd.notna(df), None)

    # Drop unnamed/index columns
    drop_cols = [c for c in df.columns if c.startswith("unnamed") or c == "index"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    return df.to_dict(orient="records")


# ── UPLOAD ────────────────────────────────────────────────────
def upload_batch(supabase: Client, records: list[dict]):
    """Upsert a batch of records into Supabase."""
    try:
        supabase.table("coa_results").upsert(
            records,
            on_conflict="results_hash"
        ).execute()
    except Exception as e:
        print(f"  ⚠ Batch error: {e}")


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("🔌 Connecting to BigQuery and Supabase...")
    bq     = bigquery.Client(project=BQ_PROJECT)
    supa   = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Count total rows
    count_query = f"SELECT COUNT(*) as n FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
    total = list(bq.query(count_query))[0]["n"]
    print(f"📊 Total rows in BigQuery: {total:,}")

    pages     = math.ceil(total / BQ_BATCH_SIZE)
    uploaded  = 0
    skipped   = 0

    for page in range(pages):
        offset = page * BQ_BATCH_SIZE
        print(f"\n📦 Fetching BQ page {page+1}/{pages} (offset {offset:,})...")

        query = f"""
            SELECT *
            FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
            LIMIT {BQ_BATCH_SIZE}
            OFFSET {offset}
        """
        df = bq.query(query).to_dataframe()
        records = clean_batch(df)

        # Split into Supabase batches
        for i in range(0, len(records), BATCH_SIZE):
            chunk = records[i : i + BATCH_SIZE]
            # Remove records missing the unique key
            chunk = [r for r in chunk if r.get("results_hash")]
            if not chunk:
                skipped += len(chunk)
                continue
            upload_batch(supa, chunk)
            uploaded += len(chunk)
            print(f"  ✓ Uploaded {uploaded:,} / {total:,} rows", end="\r")
            time.sleep(SLEEP_BETWEEN)

    print(f"\n\n✅ Done! {uploaded:,} rows uploaded, {skipped:,} skipped (no hash).")


if __name__ == "__main__":
    main()
