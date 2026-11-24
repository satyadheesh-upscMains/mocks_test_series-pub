"""Database migrations: additive, idempotent, and never touching existing rows. No model, no network."""
import sqlite3
import unittest
from pathlib import Path

HERE = Path(__file__).parent


class Migrations(unittest.TestCase):
    def test_v1_and_v2_are_additive_and_idempotent(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE upsc_articles (id INTEGER PRIMARY KEY)")
            db.execute("INSERT INTO upsc_articles VALUES (17)")
            for name in ("schema.sql", "schema_v2.sql", "schema_v3.sql", "schema.sql", "schema_v2.sql", "schema_v3.sql"):
                db.executescript((HERE / name).read_text())
            self.assertEqual(db.execute("SELECT id FROM upsc_articles").fetchone(), (17,))
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"mains_submissions", "mains_submission_pages", "mains_reviews",
                             "mains_submission_format"} <= tables)

    def test_v3_rejects_bad_formats(self):
        with sqlite3.connect(":memory:") as db:
            db.executescript((HERE / "schema_v3.sql").read_text())
            db.execute("INSERT INTO mains_submission_format VALUES ('a','gs-15',15,250,1)")
            for bad in (("b", "gs-20", 20, 250), ("c", "gs-10", 0, 150), ("d", "custom", 10, 5)):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute("INSERT INTO mains_submission_format VALUES (?,?,?,?,1)", bad)

    def test_v2_rejects_bad_values(self):
        with sqlite3.connect(":memory:") as db:
            db.executescript((HERE / "schema.sql").read_text() + (HERE / "schema_v2.sql").read_text())
            db.execute("INSERT INTO mains_reviews(id,input,updated_at) VALUES ('a','photos',1)")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("INSERT INTO mains_reviews(id,input,updated_at) VALUES ('b','video',1)")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("INSERT INTO mains_submission_pages(id,page,jpeg_b64,created_at) VALUES ('a',3,'x',1)")

    def test_v2_does_not_alter_existing_tables(self):
        text = (HERE / "schema_v2.sql").read_text().upper()
        self.assertNotIn("ALTER TABLE", text)
        self.assertNotIn("DROP ", text)
        self.assertNotIn("DELETE ", text)


if __name__ == "__main__":
    unittest.main()
