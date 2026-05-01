"""
PostgreSQL Database Connection and Management Layer
Handles all database operations for the Student Lifestyle Analytics System
"""

import psycopg2
from psycopg2.extras import DictCursor, Json
import json
import logging
import difflib
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize database manager with configuration
        
        Args:
            db_config: Dictionary with keys: host, port, database, user, password
        """
        self.db_config = db_config
        self.connection = None
        self.is_connected = False
        self.last_error = None
        self.connect()

    def connect(self):
        """Establish connection to PostgreSQL database"""
        database_name = self.db_config.get('database') or self.db_config.get('dbname')
        conn_kwargs = {
            'host': self.db_config.get('host'),
            'port': self.db_config.get('port'),
            'database': database_name,
            'user': self.db_config.get('user'),
            'password': self.db_config.get('password')
        }
        sslmode = self.db_config.get('sslmode')
        if sslmode:
            conn_kwargs['sslmode'] = sslmode

        try:
            self.connection = psycopg2.connect(**conn_kwargs)
            self.is_connected = True
            self.last_error = None
            self.ensure_schema()
            logger.info("✅ Connected to PostgreSQL database")
        except KeyError as e:
            self.is_connected = False
            self.connection = None
            self.last_error = f"Missing config key: {e}"
            logger.warning(f"⚠️  Database config missing required key: {e}")
        except Exception as e:
            if database_name and "does not exist" in str(e):
                resolved_name = self._resolve_database_name(database_name)
                if resolved_name and resolved_name != database_name:
                    retry_kwargs = dict(conn_kwargs)
                    retry_kwargs["database"] = resolved_name
                    try:
                        self.connection = psycopg2.connect(**retry_kwargs)
                        self.is_connected = True
                        self.last_error = None
                        self.db_config["database"] = resolved_name
                        self.ensure_schema()
                        logger.info(
                            f"✅ Connected using resolved database name: {resolved_name!r}"
                        )
                        return
                    except Exception:
                        pass
            self.is_connected = False
            self.connection = None
            self.last_error = f"{type(e).__name__}: {e}"
            safe_config = {
                'host': self.db_config.get('host'),
                'port': self.db_config.get('port'),
                'database': database_name,
                'user': self.db_config.get('user')
            }
            logger.warning(
                f"⚠️  Database connection unavailable: {type(e).__name__}: {e} | "
                f"config={safe_config}"
            )

    def _resolve_database_name(self, requested_name: str) -> Optional[str]:
        """Best-effort resolution for case/whitespace mismatches in DB names."""
        lookup = requested_name.strip().lower()
        if not lookup:
            return None
        try:
            maintenance_kwargs = {
                'host': self.db_config.get('host'),
                'port': self.db_config.get('port'),
                'database': 'postgres',
                'user': self.db_config.get('user'),
                'password': self.db_config.get('password')
            }
            sslmode = self.db_config.get('sslmode')
            if sslmode:
                maintenance_kwargs['sslmode'] = sslmode

            conn = psycopg2.connect(**maintenance_kwargs)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname"
            )
            names = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()

            matches = [name for name in names if name.strip().lower() == lookup]
            if matches:
                return matches[0]

            normalized_map = {name.strip().lower(): name for name in names}
            closest = difflib.get_close_matches(lookup, list(normalized_map.keys()), n=1, cutoff=0.65)
            return normalized_map[closest[0]] if closest else None
        except Exception:
            return None

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """
        Execute SELECT query and return results
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            List of dictionaries with query results
        """
        if not self.is_connected:
            logger.warning("⚠️  Database not connected. Cannot execute query.")
            return []
        
        try:
            cursor = self.connection.cursor(cursor_factory=DictCursor)
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in rows]
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Query execution failed: {e}")
            return []

    def ensure_schema(self):
        """Create required tables/views if they do not exist."""
        if not self.connection:
            return
        schema_path = Path(__file__).with_name("schema.sql")
        if not schema_path.exists():
            logger.warning("⚠️  Schema file not found. Skipping schema bootstrap.")
            return
        try:
            sql = schema_path.read_text(encoding="utf-8")
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            cursor.close()
        except Exception as e:
            self.connection.rollback()
            logger.warning(f"⚠️  Schema bootstrap failed: {e}")

    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            Number of rows affected
        """
        if not self.is_connected:
            logger.warning("⚠️  Database not connected. Cannot execute update.")
            return 0
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            self.connection.commit()
            rows_affected = cursor.rowcount
            cursor.close()
            return rows_affected
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"❌ Update execution failed: {e}")
            return 0

    def insert_student_data(self, student_data: pd.DataFrame) -> int:
        """
        Insert processed student data into database
        
        Args:
            student_data: Polars DataFrame with processed student data
            
        Returns:
            Number of rows inserted
        """
        rows_inserted = 0
        try:
            cursor = self.connection.cursor()

            def _to_int(value, default=0):
                if value is None:
                    return default
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                text = str(value).strip()
                if not text:
                    return default
                match = re.search(r"-?\d+", text)
                return int(match.group()) if match else default

            def _to_float(value, default=0.0):
                if value is None:
                    return default
                try:
                    return float(value)
                except Exception:
                    text = str(value).strip()
                    match = re.search(r"-?\d+(\.\d+)?", text)
                    return float(match.group()) if match else default

            def _to_timestamp(value):
                if isinstance(value, datetime):
                    return value
                text = str(value).strip() if value is not None else ""
                if not text:
                    return datetime.now()
                for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
                    try:
                        return datetime.strptime(text, fmt)
                    except ValueError:
                        continue
                return datetime.now()
            
            for row in student_data.iter_rows(named=True):
                insert_query = """
                    INSERT INTO students (
                        timestamp, roll_no, age, department, year_of_study,
                        sleep_hours, sleep_consistency, exercise, tired_during_class,
                        study_hours, attendance, assignment_submission, concentration,
                        screen_time, late_phone, social_media_usage, stress_level,
                        overwhelmed, time_management, productivity_score, gpa,
                        academic_performance, lifestyle_score, burnout_risk,
                        digital_addiction_score, productivity_index, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (roll_no) DO UPDATE SET
                        timestamp = EXCLUDED.timestamp,
                        sleep_hours = EXCLUDED.sleep_hours,
                        study_hours = EXCLUDED.study_hours,
                        screen_time = EXCLUDED.screen_time,
                        lifestyle_score = EXCLUDED.lifestyle_score,
                        burnout_risk = EXCLUDED.burnout_risk,
                        updated_at = NOW()
                """
                
                values = (
                    _to_timestamp(row.get('Timestamp', datetime.now())),
                    row.get('Roll No'),
                    _to_int(row.get('Age')),
                    row.get('Department'),
                    _to_int(row.get('Year of Study')),
                    _to_float(row.get('sleep_hours', 0)),
                    _to_float(row.get('sleep_consistency', 0)),
                    _to_float(row.get('exercise', 0)),
                    _to_float(row.get('tired_during_class', 0)),
                    _to_float(row.get('study_hours', 0)),
                    _to_float(row.get('attendance', 0)),
                    _to_float(row.get('assignment_submission', 0)),
                    _to_float(row.get('concentration', 0)),
                    _to_float(row.get('screen_time', 0)),
                    _to_float(row.get('late_phone', 0)),
                    _to_float(row.get('social_media_usage', 0)),
                    _to_float(row.get('stress_level', 0)),
                    _to_float(row.get('overwhelmed', 0)),
                    _to_float(row.get('time_management', 0)),
                    _to_float(row.get('productivity_score', 0)),
                    _to_float(row.get('gpa', 0)),
                    row.get('How do you rate your academic performance?', 'Average'),
                    _to_float(row.get('lifestyle_score', 0)),
                    _to_float(row.get('burnout_risk', 0)),
                    _to_float(row.get('digital_addiction_score', 0)),
                    _to_float(row.get('productivity_index', 0)),
                    datetime.now()
                )
                
                cursor.execute(insert_query, values)
                rows_inserted += 1
            
            self.connection.commit()
            cursor.close()
            logger.info(f"✅ Inserted {rows_inserted} student records")
            return rows_inserted
        except Exception as e:
            self.connection.rollback()
            logger.error(f"❌ Data insertion failed: {e}")
            raise

    def get_all_students(self) -> pd.DataFrame:
        """Fetch all student records from database"""
        try:
            query = "SELECT * FROM students ORDER BY created_at DESC"
            results = self.execute_query(query)
            return pd.DataFrame(results) if results else pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Failed to fetch students: {e}")
            raise

    def get_student_by_roll(self, roll_no: str) -> Optional[Dict]:
        """Fetch specific student record"""
        try:
            query = "SELECT * FROM students WHERE roll_no = %s"
            results = self.execute_query(query, (roll_no,))
            return results[0] if results else None
        except Exception as e:
            logger.error(f"❌ Failed to fetch student: {e}")
            raise

    def log_ai_interaction(self, query_text: str, response: str, context: Dict, model_used: str):
        """Log AI agent interactions for audit trail"""
        try:
            insert_query = """
                INSERT INTO ai_logs (query, response, context, model_used, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """
            values = (query_text, response, Json(context), model_used, datetime.now())
            self.execute_update(insert_query, values)
            logger.info("✅ AI interaction logged")
        except Exception as e:
            logger.error(f"❌ Failed to log AI interaction: {e}")

    def save_model_metrics(self, model_name: str, metrics: Dict, timestamp: str):
        """Save model training metrics to database"""
        try:
            insert_query = """
                INSERT INTO model_metrics (model_name, metrics, created_at)
                VALUES (%s, %s, %s)
            """
            values = (model_name, Json(metrics), timestamp)
            self.execute_update(insert_query, values)
            logger.info(f"✅ Metrics saved for model: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to save metrics: {e}")

    def get_model_metrics(self, model_name: Optional[str] = None) -> List[Dict]:
        """Fetch model metrics from database"""
        try:
            if model_name:
                query = """
                    SELECT * FROM model_metrics 
                    WHERE model_name = %s 
                    ORDER BY created_at DESC LIMIT 1
                """
                return self.execute_query(query, (model_name,))
            else:
                query = """
                    SELECT DISTINCT ON (model_name) * FROM model_metrics 
                    ORDER BY model_name, created_at DESC
                """
                return self.execute_query(query)
        except Exception as e:
            logger.error(f"❌ Failed to fetch metrics: {e}")
            return []

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
