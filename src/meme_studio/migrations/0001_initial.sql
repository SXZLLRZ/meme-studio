CREATE TABLE galleries (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    source_dir TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE assets (
    id TEXT PRIMARY KEY NOT NULL,
    sha256 TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    mime TEXT NOT NULL,
    width INTEGER CHECK (width > 0),
    height INTEGER CHECK (height > 0),
    duration REAL CHECK (duration >= 0),
    byte_size INTEGER NOT NULL CHECK (byte_size > 0),
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE gallery_images (
    id TEXT PRIMARY KEY NOT NULL,
    gallery_id TEXT NOT NULL REFERENCES galleries(id),
    source_path TEXT,
    source_hash TEXT,
    asset_id TEXT REFERENCES assets(id),
    tags_json TEXT NOT NULL,
    import_status TEXT NOT NULL,
    error_code TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (gallery_id, source_hash)
);

CREATE TABLE feature_sets (
    id TEXT PRIMARY KEY NOT NULL,
    gallery_id TEXT NOT NULL REFERENCES galleries(id),
    model_name TEXT NOT NULL,
    checkpoint_hash TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    index_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    generation TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL
);

CREATE TABLE image_features (
    image_id TEXT NOT NULL REFERENCES gallery_images(id),
    feature_set_id TEXT NOT NULL REFERENCES feature_sets(id),
    row_index INTEGER NOT NULL CHECK (row_index >= 0),
    asset_hash TEXT NOT NULL,
    PRIMARY KEY (image_id, feature_set_id),
    UNIQUE (feature_set_id, row_index)
);

CREATE TABLE projects (
    id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    source_image_id TEXT NOT NULL REFERENCES gallery_images(id),
    root_version_id TEXT REFERENCES versions(id),
    current_version_id TEXT REFERENCES versions(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE versions (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL REFERENCES projects(id),
    parent_id TEXT REFERENCES versions(id),
    kind TEXT NOT NULL,
    base_image_asset_id TEXT NOT NULL REFERENCES assets(id),
    base_video_asset_id TEXT REFERENCES assets(id),
    anchor_image_version_id TEXT REFERENCES versions(id),
    text_layers_json TEXT NOT NULL,
    constraints_json TEXT NOT NULL,
    operation_json TEXT NOT NULL,
    job_id TEXT UNIQUE REFERENCES jobs(id),
    created_at TEXT NOT NULL
);

CREATE TABLE regions (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL REFERENCES projects(id),
    source_version_id TEXT NOT NULL REFERENCES versions(id),
    base_asset_id TEXT NOT NULL REFERENCES assets(id),
    base_asset_hash TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    mask_asset_id TEXT NOT NULL REFERENCES assets(id),
    prompts_json TEXT NOT NULL,
    transform_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL REFERENCES projects(id),
    source_version_id TEXT NOT NULL REFERENCES versions(id),
    kind TEXT NOT NULL,
    local_status TEXT NOT NULL,
    remote_status TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    remote_task_id TEXT,
    request_hash TEXT NOT NULL,
    request_path TEXT NOT NULL,
    result_path TEXT,
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (provider, remote_task_id)
);

CREATE TABLE usage_events (
    id TEXT PRIMARY KEY NOT NULL,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    image_count INTEGER,
    video_seconds REAL,
    resolution TEXT,
    audio INTEGER CHECK (audio IN (0, 1)),
    estimated_cost TEXT,
    price_revision TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE exports (
    id TEXT PRIMARY KEY NOT NULL,
    version_id TEXT NOT NULL REFERENCES versions(id),
    format TEXT NOT NULL,
    options_json TEXT NOT NULL,
    asset_id TEXT NOT NULL REFERENCES assets(id),
    created_at TEXT NOT NULL,
    UNIQUE (version_id, format, options_json)
);

CREATE INDEX versions_project_created ON versions(project_id, created_at);
CREATE INDEX jobs_project_status ON jobs(project_id, local_status);
