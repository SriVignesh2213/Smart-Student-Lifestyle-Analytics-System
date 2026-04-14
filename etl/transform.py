import polars as pl
import os

# =========================
# LOAD DATA
# =========================
import gspread
import pandas as pd
import polars as pl


gc = gspread.service_account(filename="Credentials.json")


sheet = gc.open("student_lifestyle_raw_data").sheet1

data = sheet.get_all_records()

df_pd = pd.DataFrame(data)

# Convert everything to string first (safe step)
df_pd = df_pd.astype(str)

# Replace empty strings with None
df_pd = df_pd.replace("", None)

# Convert to Polars
df = pl.from_pandas(df_pd)

print("Data loaded from Google Sheets ✅")

# cleaning the column names by stripping whitespace

df = df.rename({col: col.strip() for col in df.columns})

print("Columns:", df.columns)

# =========================
# RENAME COLUMNS
# =========================
df = df.rename({
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
    "Late night phone usage": "late_phone"
})

# =========================
# NUMERIC CAST (ONLY REAL NUMBERS)
# =========================
df = df.with_columns([
    pl.col("sleep_hours").cast(pl.Float64, strict=False),
    pl.col("study_hours").cast(pl.Float64, strict=False),
    pl.col("screen_time").cast(pl.Float64, strict=False),
    pl.col("attendance").cast(pl.Float64, strict=False),
    pl.col("productivity_score").cast(pl.Float64, strict=False),
    pl.col("gpa").cast(pl.Float64, strict=False),
])

# =========================
# ENCODE CATEGORICAL (VERY IMPORTANT)
# =========================
df = df.with_columns([

    pl.when(pl.col("stress_level") == "Low").then(1)
    .when(pl.col("stress_level") == "Moderate").then(2)
    .when(pl.col("stress_level") == "High").then(3)
    .alias("stress_level"),

    pl.when(pl.col("exercise") == "Daily").then(4)
    .when(pl.col("exercise") == "3–4 times/week").then(3)
    .when(pl.col("exercise") == "Once/week").then(2)
    .when(pl.col("exercise") == "Rarely").then(1)
    .when(pl.col("exercise") == "Never").then(0)
    .alias("exercise"),

    pl.when(pl.col("late_phone") == "Never").then(0)
    .when(pl.col("late_phone") == "Sometimes").then(1)
    .when(pl.col("late_phone") == "Daily").then(2)
    .alias("late_phone"),

    pl.when(pl.col("concentration") == "High").then(3)
    .when(pl.col("concentration") == "Medium").then(2)
    .when(pl.col("concentration") == "Low").then(1)
    .alias("concentration"),

    pl.when(pl.col("social_media_usage") == "Low").then(1)
    .when(pl.col("social_media_usage") == "Moderate").then(2)
    .when(pl.col("social_media_usage") == "High").then(3)
    .alias("social_media_usage"),

    pl.when(pl.col("overwhelmed") == "Never").then(0)
    .when(pl.col("overwhelmed") == "Sometimes").then(1)
    .when(pl.col("overwhelmed") == "Often").then(2)
    .alias("overwhelmed"),

    pl.when(pl.col("time_management") == "Good").then(3)
    .when(pl.col("time_management") == "Average").then(2)
    .when(pl.col("time_management") == "Poor").then(1)
    .alias("time_management"),

    pl.when(pl.col("assignment_submission") == "Always on time").then(3)
    .when(pl.col("assignment_submission") == "Sometimes late").then(2)
    .when(pl.col("assignment_submission") == "Often late").then(1)
    .alias("assignment_submission"),

    pl.when(pl.col("tired_during_class") == "Never").then(0)
    .when(pl.col("tired_during_class") == "Sometimes").then(1)
    .when(pl.col("tired_during_class") == "Often").then(2)
    .alias("tired_during_class"),

    pl.when(pl.col("sleep_consistency") == "Very consistent").then(3)
    .when(pl.col("sleep_consistency") == "Somewhat consistent").then(2)
    .when(pl.col("sleep_consistency") == "Irregular").then(1)
    .alias("sleep_consistency"),

])

# =========================
# CAST AFTER ENCODING
# =========================
df = df.with_columns([
    pl.col("stress_level").cast(pl.Float64),
    pl.col("exercise").cast(pl.Float64),
    pl.col("late_phone").cast(pl.Float64),
    pl.col("concentration").cast(pl.Float64),
    pl.col("social_media_usage").cast(pl.Float64),
    pl.col("overwhelmed").cast(pl.Float64),
    pl.col("time_management").cast(pl.Float64),
    pl.col("assignment_submission").cast(pl.Float64),
    pl.col("tired_during_class").cast(pl.Float64),
    pl.col("sleep_consistency").cast(pl.Float64),
])

# =========================
# FEATURE ENGINEERING
# =========================
df = df.with_columns([

    (
        pl.col("sleep_hours") * 0.20 +
        pl.col("study_hours") * 0.20 -
        pl.col("screen_time") * 0.15 +
        pl.col("exercise") * 0.10 -
        pl.col("stress_level") * 0.10 +
        pl.col("productivity_score") * 0.10 +
        pl.col("concentration") * 0.05 +
        pl.col("time_management") * 0.05 -
        pl.col("overwhelmed") * 0.03 -
        pl.col("tired_during_class") * 0.02
    ).alias("lifestyle_score"),

    (
        pl.col("stress_level") +
        pl.col("overwhelmed") +
        pl.col("tired_during_class")
    ).alias("burnout_risk"),

    (
        pl.col("screen_time") +
        pl.col("social_media_usage") +
        pl.col("late_phone")
    ).alias("digital_addiction_score"),

    (
        (
            pl.col("productivity_score") +
            pl.col("concentration") +
            pl.col("time_management")
        ) / 3
    ).alias("productivity_index")

])

# Drop nulls
df = df.drop_nulls()

# =========================
# SAVE FILE
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
output_path = os.path.join(BASE_DIR, "data", "cleaned_student_data.csv")

df.write_csv(output_path)

print("ETL completed using Polars 🚀")