-- PostgreSQL Schema for Student Lifestyle Analytics System

-- Create students table
CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    roll_no VARCHAR(20) UNIQUE NOT NULL,
    age INT,
    department VARCHAR(50),
    year_of_study INT,
    
    -- Lifestyle Features
    sleep_hours FLOAT,
    sleep_consistency FLOAT,
    exercise FLOAT,
    tired_during_class FLOAT,
    study_hours FLOAT,
    attendance FLOAT,
    assignment_submission FLOAT,
    concentration FLOAT,
    screen_time FLOAT,
    late_phone FLOAT,
    social_media_usage FLOAT,
    stress_level FLOAT,
    overwhelmed FLOAT,
    time_management FLOAT,
    productivity_score FLOAT,
    gpa FLOAT,
    academic_performance VARCHAR(50),
    
    -- Engineered Features
    lifestyle_score FLOAT,
    burnout_risk FLOAT,
    digital_addiction_score FLOAT,
    productivity_index FLOAT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create AI logs table for MCP interactions
CREATE TABLE IF NOT EXISTS ai_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT,
    context JSONB,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    student_id INT REFERENCES students(id),
    model_name VARCHAR(100),
    input_features JSONB,
    predicted_lifestyle_score FLOAT,
    predicted_burnout_risk FLOAT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_students_roll_no ON students(roll_no);
CREATE INDEX IF NOT EXISTS idx_students_created_at ON students(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_ai_logs_created_at ON ai_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_student_id ON predictions(student_id);

-- Create views for analytics
CREATE OR REPLACE VIEW v_student_summary AS
SELECT 
    COUNT(*) as total_students,
    AVG(lifestyle_score) as avg_lifestyle_score,
    AVG(burnout_risk) as avg_burnout_risk,
    AVG(gpa) as avg_gpa,
    AVG(sleep_hours) as avg_sleep_hours,
    AVG(study_hours) as avg_study_hours,
    MAX(updated_at) as last_update
FROM students;

CREATE OR REPLACE VIEW v_high_risk_students AS
SELECT 
    roll_no,
    lifestyle_score,
    burnout_risk,
    stress_level,
    sleep_hours,
    study_hours
FROM students
WHERE burnout_risk > (SELECT AVG(burnout_risk) FROM students)
  AND lifestyle_score < (SELECT AVG(lifestyle_score) FROM students)
ORDER BY burnout_risk DESC;
