-- Mains feedback v3: the question's format, chosen by the student. Additive and idempotent: one new table, no change
-- to any existing table. Apply to the ACTIVE UPSC database only, after review.
--
-- Lifetime follows the submission: the site deletes the row with its mains_submissions row, in the same batch as
-- mains_submission_pages and mains_reviews.
--
--   format      gs-10 | gs-15 | ethics-10 | ethics-case-20 | essay-125 | optional-10 | optional-15 | optional-20 | custom
--   marks       the marks the question carries (from the format, or entered for custom)
--   word_limit  the word limit the student wrote to (the printed "(Answer in N words)" still wins in the grader)
CREATE TABLE IF NOT EXISTS mains_submission_format (
  id TEXT PRIMARY KEY,
  format TEXT NOT NULL CHECK (format IN ('gs-10','gs-15','ethics-10','ethics-case-20','essay-125',
                                         'optional-10','optional-15','optional-20','custom')),
  marks INTEGER NOT NULL CHECK (marks BETWEEN 1 AND 250),
  word_limit INTEGER CHECK (word_limit IS NULL OR word_limit BETWEEN 30 AND 3000),
  created_at INTEGER NOT NULL
);
