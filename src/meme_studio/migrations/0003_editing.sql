ALTER TABLE versions ADD COLUMN region_id TEXT REFERENCES regions(id);
