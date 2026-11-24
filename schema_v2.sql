-- Mains feedback v2: photo answers and the full student report. Additive and idempotent: new tables only, no
-- change to mains_submissions or any other table. Apply to the ACTIVE UPSC database only, after review.
--
-- Lifetime follows the submission: the site deletes these rows with their mains_submissions row (30-day expiry,
-- student deletion), in the same statement batch, the way it already clears mains_review_preview.

-- Answer-page photos, as the browser uploaded them (re-encoded JPEG, no EXIF). At most 2 pages per submission.
CREATE TABLE IF NOT EXISTS mains_submission_pages (
  id TEXT NOT NULL,
  page INTEGER NOT NULL CHECK (page BETWEEN 1 AND 2),
  jpeg_b64 TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (id, page)
);

-- What the worker produced for one submission.
--   input            typed | photos
--   transcript_json  photo answers: lines with page boxes, read status, every read (the evidence), retake reasons
--   blocked          1 when a photo could not be read with certainty and the student was asked to retake it
--   report_json      the student report (report_version mains-report-v10), with page marks for photo answers
CREATE TABLE IF NOT EXISTS mains_reviews (
  id TEXT PRIMARY KEY,
  input TEXT NOT NULL CHECK (input IN ('typed','photos')),
  transcript_json TEXT,
  blocked INTEGER NOT NULL DEFAULT 0 CHECK (blocked IN (0,1)),
  report_json TEXT,
  report_version TEXT,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS mains_reviews_updated_idx ON mains_reviews(updated_at);
