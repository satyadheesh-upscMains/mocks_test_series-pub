-- Additive, idempotent UPSC DB migration. Apply explicitly; no changes to existing tables.
CREATE TABLE IF NOT EXISTS mains_submissions (
  id TEXT PRIMARY KEY,
  owner_hash TEXT NOT NULL,
  network_hash TEXT NOT NULL,
  article_id INTEGER NOT NULL,
  lang TEXT NOT NULL CHECK (lang IN ('en','hi')),
  question TEXT NOT NULL,
  references_json TEXT NOT NULL,
  answer TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','reviewing','ready','failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  retry_at INTEGER NOT NULL DEFAULT 0,
  lease TEXT,
  lease_until INTEGER,
  result_json TEXT,
  rubric_version TEXT,
  evaluated_at INTEGER,
  contact_kind TEXT CHECK (contact_kind IN ('push','email')),
  contact_hash TEXT,
  contact_json TEXT,
  delivery TEXT CHECK (delivery IN ('pending','sending','sent','failed')),
  delivery_attempts INTEGER NOT NULL DEFAULT 0,
  delivery_at INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS mains_queue_idx ON mains_submissions(status,retry_at,lease_until,created_at);
CREATE INDEX IF NOT EXISTS mains_owner_idx ON mains_submissions(owner_hash,created_at);
CREATE INDEX IF NOT EXISTS mains_network_idx ON mains_submissions(network_hash,created_at);
CREATE INDEX IF NOT EXISTS mains_expiry_idx ON mains_submissions(expires_at);
CREATE INDEX IF NOT EXISTS mains_delivery_idx ON mains_submissions(delivery,delivery_at);
CREATE INDEX IF NOT EXISTS mains_contact_idx ON mains_submissions(contact_hash,created_at);
