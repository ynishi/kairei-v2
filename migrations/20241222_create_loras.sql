-- Create loras table
CREATE TABLE IF NOT EXISTS loras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    base_model_id UUID REFERENCES base_models(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'available',
    archived BOOLEAN NOT NULL DEFAULT FALSE,
    file_path TEXT,
    size_bytes BIGINT,
    metadata JSONB NOT NULL
);

-- Create indexes for faster lookups
CREATE INDEX idx_loras_name ON loras(name);
CREATE INDEX idx_loras_base_model_id ON loras(base_model_id);
CREATE INDEX idx_loras_status ON loras(status);
CREATE INDEX idx_loras_archived ON loras(archived);
CREATE INDEX idx_loras_created_at ON loras(created_at);

-- Create updated_at trigger (reuse the function from base_models migration)
CREATE TRIGGER update_loras_updated_at BEFORE UPDATE
    ON loras FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();