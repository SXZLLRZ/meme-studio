ALTER TABLE galleries ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE gallery_images ADD COLUMN display_name TEXT NOT NULL DEFAULT '';
ALTER TABLE gallery_images ADD COLUMN thumbnail_path TEXT;
ALTER TABLE gallery_images ADD COLUMN removed_at TEXT;

CREATE UNIQUE INDEX gallery_asset_once
    ON gallery_images(gallery_id, asset_id)
    WHERE asset_id IS NOT NULL;

ALTER TABLE feature_sets ADD COLUMN gallery_revision INTEGER NOT NULL DEFAULT 0;

CREATE UNIQUE INDEX one_active_gallery_index
    ON feature_sets(gallery_id)
    WHERE status = 'active';

CREATE TABLE image_vectors (
    asset_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    checkpoint_hash TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    vector BLOB NOT NULL CHECK (length(vector) = 4 * dimension),
    PRIMARY KEY (asset_hash, model_name, checkpoint_hash, preprocess_version)
);
