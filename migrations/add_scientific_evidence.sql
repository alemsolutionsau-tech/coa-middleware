-- Migration: Add scientific evidence support
-- Run in Supabase SQL editor

-- 1. PubMed query cache (Tier 2)
CREATE TABLE IF NOT EXISTS pubmed_cache (
  query_hash   TEXT        PRIMARY KEY,
  query_term   TEXT        NOT NULL,
  results      JSONB       NOT NULL DEFAULT '[]',
  fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  hit_count    INTEGER     NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pubmed_cache_fetched ON pubmed_cache (fetched_at);

-- 2. Scientific evidence per COA report (Tier 3)
ALTER TABLE coa_ai_reports
  ADD COLUMN IF NOT EXISTS scientific_evidence JSONB;

-- 3. Auto-expire pubmed_cache rows older than 7 days (optional scheduled job)
-- DELETE FROM pubmed_cache WHERE fetched_at < NOW() - INTERVAL '7 days';
