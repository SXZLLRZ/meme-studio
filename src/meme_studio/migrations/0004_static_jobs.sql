ALTER TABLE projects ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN source_revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN paused INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN next_action_at REAL NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN wait_until REAL NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN poll_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN lease_token TEXT;
ALTER TABLE jobs ADD COLUMN lease_until REAL NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN result_url TEXT;
ALTER TABLE jobs ADD COLUMN link_refreshes INTEGER NOT NULL DEFAULT 0;

CREATE INDEX jobs_worker_ready ON jobs(provider, kind, paused, next_action_at, lease_until);
CREATE UNIQUE INDEX jobs_active_request ON jobs(project_id, source_version_id, request_hash)
    WHERE local_status NOT IN ('FAILED', 'RESULT_EXPIRED');
