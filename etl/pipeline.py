import os
import sys
from pathlib import Path

import gspread
import pandas as pd
import polars as pl


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import DB_CONFIG  # noqa: E402
from db.db import DatabaseManager  # noqa: E402


def current_db_config() -> dict:
    return {
        "host": os.getenv("DB_HOST", DB_CONFIG.get("host", "localhost")),
        "port": int(os.getenv("DB_PORT", DB_CONFIG.get("port", 5432))),
        "database": os.getenv("DB_NAME", DB_CONFIG.get("database", "Student Life Analytics System")),
        "user": os.getenv("DB_USER", DB_CONFIG.get("user", "postgres")),
        "password": os.getenv("DB_PASSWORD", DB_CONFIG.get("password", "")),
        "sslmode": os.getenv("DB_SSL_MODE", DB_CONFIG.get("sslmode", "disable")),
    }


def ensure_schema(db: DatabaseManager) -> None:
    schema_path = BASE_DIR / "db" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    cursor = db.connection.cursor()
    cursor.execute(schema_sql)
    db.connection.commit()
    cursor.close()


def run_pipeline() -> int:
    cred_path = BASE_DIR / "etl" / "Credentials.json"
    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "student_lifestyle_raw_data")

    gc = gspread.service_account(filename=str(cred_path))
    sheet = gc.open(sheet_name).sheet1
    data = sheet.get_all_records()

    df_pd = pd.DataFrame(data)
    df_pd = df_pd.astype(str).replace("", None)
    df = pl.from_pandas(df_pd)

    print("Data loaded from Google Sheets")

    df = df.rename({col: col.strip() for col in df.columns})

    df = df.rename(
        {
            "Average sleep hours per day": "sleep_hours",
            "Sleep consistency": "sleep_consistency",
            "Exercise frequency": "exercise",
            "Average study hours per day": "study_hours",
            "How often do you feel tired during class?": "tired_during_class",
            "Daily screen time (hours)": "screen_time",
            "Attendance Percentage": "attendance",
            "Concentration during classes": "concentration",
            "Assignment submission behavior": "assignment_submission",
            "Self Productivity rating": "productivity_score",
            "Social media usage level": "social_media_usage",
            "How often do you feel overwhelmed?": "overwhelmed",
            "Time management ability": "time_management",
            "Current CGPA": "gpa",
            "Stress level": "stress_level",
            "Late night phone usage": "late_phone",
        }
    )

    df = df.with_columns(
        [
            pl.col("sleep_hours").cast(pl.Float64, strict=False),
            pl.col("study_hours").cast(pl.Float64, strict=False),
            pl.col("screen_time").cast(pl.Float64, strict=False),
            pl.col("attendance").cast(pl.Float64, strict=False),
            pl.col("productivity_score").cast(pl.Float64, strict=False),
            pl.col("gpa").cast(pl.Float64, strict=False),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("stress_level") == "Low")
            .then(1)
            .when(pl.col("stress_level") == "Moderate")
            .then(2)
            .when(pl.col("stress_level") == "High")
            .then(3)
            .alias("stress_level"),
            pl.when(pl.col("exercise") == "Daily")
            .then(4)
            .when(pl.col("exercise").is_in(["3-4 times/week", "3–4 times/week"]))
            .then(3)
            .when(pl.col("exercise") == "Once/week")
            .then(2)
            .when(pl.col("exercise") == "Rarely")
            .then(1)
            .when(pl.col("exercise") == "Never")
            .then(0)
            .alias("exercise"),
            pl.when(pl.col("late_phone") == "Never")
            .then(0)
            .when(pl.col("late_phone") == "Sometimes")
            .then(1)
            .when(pl.col("late_phone") == "Daily")
            .then(2)
            .alias("late_phone"),
            pl.when(pl.col("concentration") == "High")
            .then(3)
            .when(pl.col("concentration") == "Medium")
            .then(2)
            .when(pl.col("concentration") == "Low")
            .then(1)
            .alias("concentration"),
            pl.when(pl.col("social_media_usage") == "Low")
            .then(1)
            .when(pl.col("social_media_usage") == "Moderate")
            .then(2)
            .when(pl.col("social_media_usage") == "High")
            .then(3)
            .alias("social_media_usage"),
            pl.when(pl.col("overwhelmed") == "Never")
            .then(0)
            .when(pl.col("overwhelmed") == "Sometimes")
            .then(1)
            .when(pl.col("overwhelmed") == "Often")
            .then(2)
            .alias("overwhelmed"),
            pl.when(pl.col("time_management") == "Good")
            .then(3)
            .when(pl.col("time_management") == "Average")
            .then(2)
            .when(pl.col("time_management") == "Poor")
            .then(1)
            .alias("time_management"),
            pl.when(pl.col("assignment_submission") == "Always on time")
            .then(3)
            .when(pl.col("assignment_submission") == "Sometimes late")
            .then(2)
            .when(pl.col("assignment_submission") == "Often late")
            .then(1)
            .alias("assignment_submission"),
            pl.when(pl.col("tired_during_class") == "Never")
            .then(0)
            .when(pl.col("tired_during_class") == "Sometimes")
            .then(1)
            .when(pl.col("tired_during_class") == "Often")
            .then(2)
            .alias("tired_during_class"),
            pl.when(pl.col("sleep_consistency") == "Very consistent")
            .then(3)
            .when(pl.col("sleep_consistency") == "Somewhat consistent")
            .then(2)
            .when(pl.col("sleep_consistency") == "Irregular")
            .then(1)
            .alias("sleep_consistency"),
        ]
    )

    numeric_columns = [
        "sleep_hours",
        "sleep_consistency",
        "exercise",
        "tired_during_class",
        "study_hours",
        "attendance",
        "assignment_submission",
        "concentration",
        "screen_time",
        "late_phone",
        "social_media_usage",
        "stress_level",
        "overwhelmed",
        "time_management",
        "productivity_score",
        "gpa",
    ]
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in numeric_columns if c in df.columns]
    )

    df = df.with_columns(
        [
            (
                pl.col("sleep_hours") * 0.20
                + pl.col("study_hours") * 0.20
                - pl.col("screen_time") * 0.15
                + pl.col("exercise") * 0.10
                - pl.col("stress_level") * 0.10
                + pl.col("productivity_score") * 0.10
                + pl.col("concentration") * 0.05
                + pl.col("time_management") * 0.05
                - pl.col("overwhelmed") * 0.03
                - pl.col("tired_during_class") * 0.02
            ).alias("lifestyle_score"),
            (pl.col("stress_level") + pl.col("overwhelmed") + pl.col("tired_during_class")).alias(
                "burnout_risk"
            ),
            (pl.col("screen_time") + pl.col("social_media_usage") + pl.col("late_phone")).alias(
                "digital_addiction_score"
            ),
            (
                (pl.col("productivity_score") + pl.col("concentration") + pl.col("time_management")) / 3
            ).alias("productivity_index"),
        ]
    )

    engineered_columns = ["lifestyle_score", "burnout_risk", "digital_addiction_score", "productivity_index"]
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in engineered_columns if c in df.columns]
    )
    output_path = BASE_DIR / "data" / "cleaned_student_data.csv"
    df.write_csv(str(output_path))
    print(f"Cleaned CSV saved: {output_path}")

    db = DatabaseManager(current_db_config())
    if not db.is_connected:
        raise RuntimeError("Database connection failed from ETL pipeline.")

    ensure_schema(db)
    rows = db.insert_student_data(df)
    print(f"Rows loaded into PostgreSQL: {rows}")
    print("Pipeline executed successfully")
    db.close()
    return rows


if __name__ == "__main__":
    run_pipeline()
