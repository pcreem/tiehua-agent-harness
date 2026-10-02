-- PGVector initialization script
-- Run on PostgreSQL startup

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables for tourism knowledge
CREATE TABLE IF NOT EXISTS tourism_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS tourism_documents_embedding_idx
ON tourism_documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create table for evaluation reports
CREATE TABLE IF NOT EXISTS evaluation_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id VARCHAR(64) NOT NULL,
    query TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    score FLOAT,
    tool_calls JSONB,
    tool_results JSONB,
    evaluation JSONB,
    feedback JSONB,
    retry_count INTEGER DEFAULT 0,
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS evaluation_reports_run_id_idx ON evaluation_reports(run_id);
CREATE INDEX IF NOT EXISTS evaluation_reports_status_idx ON evaluation_reports(status);
CREATE INDEX IF NOT EXISTS evaluation_reports_created_at_idx ON evaluation_reports(created_at);

-- Create table for agent runs (for observability)
CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id VARCHAR(64) NOT NULL,
    query TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    final_answer TEXT,
    tool_calls JSONB,
    tool_results JSONB,
    evaluation JSONB,
    feedback JSONB,
    retry_count INTEGER DEFAULT 0,
    attempt INTEGER DEFAULT 0,
    duration_ms INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS agent_runs_run_id_idx ON agent_runs(run_id);
CREATE INDEX IF NOT EXISTS agent_runs_status_idx ON agent_runs(status);
CREATE INDEX IF NOT EXISTS agent_runs_created_at_idx ON agent_runs(created_at);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO tiehua;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO tiehua;