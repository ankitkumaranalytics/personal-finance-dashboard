"""
data_cleaning.py
================

Professional data-cleaning pipeline for the Personal Finance Dashboard project.

    "Personal Finance Analytics Dashboard using Python & Streamlit"

STEP 3 — DATA CLEANING & DATA QUALITY ANALYSIS

This script:
  * Reads the RAW dataset   : data/raw/personal_finance.csv
  * Cleans it               : removing duplicates, handling missing values,
                              standardizing text, validating dates, categories,
                              payment methods, amounts, budgets and business rules.
  * Writes the CLEAN dataset : data/cleaned/personal_finance_cleaned.csv
  * Writes quality reports   : reports/data_quality_report.csv
                               reports/cleaning_summary.txt

IMPORTANT
---------
* The original raw file in `data/raw/` is NEVER modified.
* The whole process is reproducible — running this script again produces the
  exact same cleaned output (no randomness is used in the cleaning logic).
* All paths are RELATIVE so the project works on any computer.

Usage:
    python src/data_cleaning.py

Required libraries:
    pandas, numpy   (both already in requirements.txt)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Paths & configuration (relative so the project is portable)
# ---------------------------------------------------------------------------
RAW_FILE = Path("data") / "raw" / "personal_finance.csv"
CLEANED_FILE = Path("data") / "cleaned" / "personal_finance_cleaned.csv"
QUALITY_REPORT_FILE = Path("reports") / "data_quality_report.csv"
CLEANING_SUMMARY_FILE = Path("reports") / "cleaning_summary.txt"

# Allowed values for validation.
ALLOWED_TRANSACTION_TYPES = {"Income", "Expense"}

INCOME_CATEGORIES = {
    "Salary",
    "Freelance",
    "Business",
    "Investment",
    "Other Income",
}

EXPENSE_CATEGORIES = {
    "Food",
    "Transportation",
    "Shopping",
    "Bills & Utilities",
    "Rent",
    "Healthcare",
    "Education",
    "Entertainment",
    "Travel",
    "Groceries",
    "Personal Care",
    "Insurance",
    "Other",
}

ALLOWED_PAYMENT_METHODS = {
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash",
    "Net Banking",
    "Bank Transfer",
    "Unknown",
}

# Expected date range for every transaction (2025 financial year).
DATE_MIN = pd.Timestamp("2025-01-01")
DATE_MAX = pd.Timestamp("2025-12-31")

# Standardization maps (obvious variations -> canonical value).
PAYMENT_METHOD_MAP = {
    "upi": "UPI",
    "credit card": "Credit Card",
    "debit card": "Debit Card",
    "cash": "Cash",
    "net banking": "Net Banking",
    "bank transfer": "Bank Transfer",
    "unknown": "Unknown",
}

CATEGORY_MAP = {
    "food": "Food",
    "foods": "Food",
    "transport": "Transportation",
    "transportation": "Transportation",
    "shopping": "Shopping",
    "grocery": "Groceries",
    "groceries": "Groceries",
    "entertainment": "Entertainment",
    "bills & utilities": "Bills & Utilities",
    "bills and utilities": "Bills & Utilities",
    "personal care": "Personal Care",
    "other income": "Other Income",
}

# Text columns that get whitespace-stripped and standardized.
TEXT_COLUMNS = [
    "Transaction_Type",
    "Category",
    "Sub_Category",
    "Payment_Method",
    "Account",
    "Income_Source",
    "Description",
]

# Monthly budget assigned to each expense category (used when Budget is missing).
EXPENSE_BUDGETS = {
    "Food": 8000, "Transportation": 5000, "Shopping": 10000,
    "Bills & Utilities": 6000, "Rent": 20000, "Healthcare": 5000,
    "Education": 15000, "Entertainment": 4000, "Travel": 15000,
    "Groceries": 8000, "Personal Care": 4000, "Insurance": 12000,
    "Other": 5000,
}

# Account standardization map (preserve proper-noun casing).
ACCOUNT_MAP = {
    "hdfc bank": "HDFC Bank",
    "icici bank": "ICICI Bank",
    "axis bank": "Axis Bank",
    "sbi": "SBI",
    "cash wallet": "Cash Wallet",
    "unknown": "Unknown",
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_cleaning")


# ---------------------------------------------------------------------------
# 1. Load the raw dataset
# ---------------------------------------------------------------------------
def load_data(filepath: Path) -> pd.DataFrame:
    """Load the raw CSV into a pandas DataFrame."""
    logger.info("Loading raw data from: %s", filepath)
    df = pd.read_csv(filepath)
    # Strip leading/trailing whitespace from column names.
    df.columns = [col.strip() for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# 2. Initial data inspection
# ---------------------------------------------------------------------------
def inspect_data(df: pd.DataFrame) -> None:
    """Print a full initial-inspection summary of the raw dataset."""
    print("=" * 72)
    print("INITIAL DATA INSPECTION (RAW DATA)")
    print("=" * 72)

    print(f"\n1. Number of rows        : {len(df)}")
    print(f"   Number of columns     : {df.shape[1]}")

    print("\n2. Column names:")
    for col in df.columns:
        print(f"   - {col}")

    print("\n3. Data types:")
    print(df.dtypes.to_string())

    print("\n4. First 5 rows:")
    print(df.head(5).to_string(index=False))

    print("\n5. Last 5 rows:")
    print(df.tail(5).to_string(index=False))

    print("\n6. Missing values by column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "   (none)")

    print("\n7. Duplicate rows (exact):", int(df.duplicated().sum()))
    print("   Duplicate Transaction_IDs:",
          int(df["Transaction_ID"].duplicated().sum()))

    print("\n8. Unique values for important categorical columns:")
    for col in ["Transaction_Type", "Category", "Payment_Method", "Account",
                "Income_Source"]:
        uniq = sorted(df[col].dropna().astype(str).unique())
        print(f"   {col} ({len(uniq)}): {uniq}")

    print("\n9. Transaction date range:")
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    if parsed.notna().any():
        print(f"   min = {parsed.min().date()} , max = {parsed.max().date()}")
    else:
        print("   (no valid dates)")

    print("\n10. Amount statistics (before cleaning):")
    amt = pd.to_numeric(df["Amount"], errors="coerce")
    print(f"    min = {amt.min()} , max = {amt.max()}")
    print(f"    mean = {amt.mean():.2f} , median = {amt.median():.2f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 3. Handle duplicate records
# ---------------------------------------------------------------------------
def handle_duplicates(df: pd.DataFrame) -> dict:
    """
    Remove exact duplicate rows and report duplicate Transaction_IDs.

    Notes:
      * A duplicate Transaction_ID does NOT automatically mean the whole row is
        duplicated. We only remove rows that are EXACT duplicates everywhere.
      * The number of duplicate Transaction_IDs is reported so the user can
        investigate; we do not drop by Transaction_ID alone.
    """
    n_duplicate_rows = int(df.duplicated().sum())
    n_dup_ids = int(df["Transaction_ID"].duplicated().sum())

    print("\n--- DUPLICATE HANDLING ---")
    print(f"Number of duplicate rows before cleaning : {n_duplicate_rows}")
    print(f"Duplicate Transaction_IDs found          : {n_dup_ids}")

    duplicates_removed = n_duplicate_rows
    logger.info("Removed %d exact duplicate row(s).", duplicates_removed)

    return {"duplicate_rows_removed": duplicates_removed,
            "duplicate_ids_found": n_dup_ids}


# ---------------------------------------------------------------------------
# 4. Handle missing values
# ---------------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Column / Missing_Count / Missing_Percentage."""
    missing = df.isna().sum()
    report = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing.values,
        "Missing_Percentage": (missing.values / len(df) * 100).round(2),
    })
    return report[report["Missing_Count"] > 0]


def handle_missing_values(df: pd.DataFrame):
    """
    Fill / handle missing values column by column in a logical way.

    Strategy:
      * Description      -> "Unknown"
      * Payment_Method   -> "Unknown"
      * Account          -> "Unknown"
      * Income_Source    -> "Not Applicable" for Expenses (and blank ones)
      * Category         -> investigated; if Sub_Category gives a hint we map
                            it, otherwise it is assigned "Other".
      * Sub_Category     -> "Other" when missing
      * Budget           -> 0 for Income, category budget for Expense (else 0)
      * Amount           -> critical; any remaining missing Amount rows are
                            dropped (investigated) because Amount is essential.
      * Date             -> handled separately in clean_dates()

    Returns:
        (cleaned_df, stats_dict)
    """
    print("\n--- MISSING VALUE HANDLING ---")
    print("Missing values BEFORE handling:")
    before_report = missing_value_report(df)
    print(before_report.to_string(index=False))
    n_before = int(df.isna().sum().sum())

    df = df.copy()

    # ---- Text fields that are safe to fill ----
    df["Description"] = df["Description"].fillna("Unknown")
    df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
    df["Account"] = df["Account"].fillna("Unknown")

    # ---- Income_Source ----
    # Expenses always get "Not Applicable"; blank Income rows get "Other Sources".
    df["Income_Source"] = df.apply(
        lambda r: "Not Applicable"
                  if r["Transaction_Type"] == "Expense"
                  else (r["Income_Source"] if not pd.isna(r["Income_Source"])
                        else "Other Sources"),
        axis=1,
    )

    # ---- Category (critical field - investigate first) ----
    # For missing Category, try to infer from Sub_Category.
    missing_cat = df["Category"].isna()
    if missing_cat.any():
        sub_to_cat = {}
        samples = df.dropna(subset=["Category"]).drop_duplicates("Sub_Category")
        for _, row in samples.iterrows():
            sub_to_cat.setdefault(str(row["Sub_Category"]).strip(),
                                  row["Category"])
        for idx in df.index[missing_cat]:
            sub = str(df.at[idx, "Sub_Category"] or "").strip()
            if sub in sub_to_cat:
                df.at[idx, "Category"] = sub_to_cat[sub]
            else:
                # Fallback: assign "Other" so the row is usable.
                df.at[idx, "Category"] = "Other"

    # ---- Sub_Category ----
    df["Sub_Category"] = df["Sub_Category"].fillna("Other")

    # ---- Budget ----
    # Income  -> 0 ; Expense -> category budget if known else 0.
    df["Budget"] = df.apply(
        lambda r: 0.0 if r["Transaction_Type"] == "Income"
                  else float(EXPENSE_BUDGETS.get(r["Category"], 0.0)),
        axis=1,
    )

    # ---- Amount (critical field - investigate then drop) ----
    # Any remaining missing Amount cannot be meaningfully filled, so we drop
    # those rows after reporting them. This is a deliberate, documented choice.
    n_missing_amount = int(df["Amount"].isna().sum())
    if n_missing_amount > 0:
        logger.warning("Dropping %d row(s) with missing Amount (critical).",
                       n_missing_amount)
        df = df.dropna(subset=["Amount"]).reset_index(drop=True)

    print("\nMissing values AFTER handling:")
    n_after = int(df.isna().sum().sum())
    print(f"   Total missing cells before : {n_before}")
    print(f"   Total missing cells after  : {n_after}")
    print(f"   Rows dropped for missing Amount : {n_missing_amount}")

    stats = {"missing_before": n_before, "missing_after": n_after,
             "amount_rows_dropped": n_missing_amount}
    return df, stats


# ---------------------------------------------------------------------------
# 5. Clean text columns
# ---------------------------------------------------------------------------
def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and standardise capitalisation across text columns.

    * Leading/trailing spaces are removed.
    * Values are standardised to Title Case (e.g. 'salary' -> 'Salary').
    * Specific maps fix known variations (e.g. 'upi' -> 'UPI').
    """
    print("\n--- TEXT CLEANING ---")
    df = df.copy()

    def _title(value):
        if pd.isna(value):
            return value
        s = str(value).strip()
        return " ".join(w.capitalize() for w in s.split())

    # Apply title-case to all text columns.
    for col in TEXT_COLUMNS:
        df[col] = df[col].map(_title)

# Fix known variations (e.g. 'Upi' -> 'UPI').
    df["Payment_Method"] = df["Payment_Method"].map(
        lambda v: PAYMENT_METHOD_MAP.get(str(v).strip().lower(), v)
        if not pd.isna(v) else v)

    # Preserve proper-noun casing for Account (e.g. 'Hdfc Bank' -> 'HDFC Bank').
    df["Account"] = df["Account"].map(
        lambda v: ACCOUNT_MAP.get(str(v).strip().lower(), v)
        if not pd.isna(v) else v)

    return df


# ---------------------------------------------------------------------------
# 6. Clean and validate dates
# ---------------------------------------------------------------------------
def clean_dates(df: pd.DataFrame):
    """
    Convert Date to proper datetime, handle invalid/missing/out-of-range dates.

    Reports counts. Valid dates are saved as YYYY-MM-DD. Rows with
    invalid/missing/out-of-range dates are dropped (Date is critical).

    Returns:
        (cleaned_df, stats_dict)
    """
    print("\n--- DATE CLEANING & VALIDATION ---")
    df = df.copy()

    df["Date"] = df["Date"].astype(str).str.strip()
    n_missing = int(df["Date"].isna().sum() + (df["Date"] == "nan").sum()
                    + (df["Date"] == "").sum())

    # Try multiple common formats, coercing failures to NaT (invalid).
    parsed = pd.to_datetime(df["Date"], format="mixed", errors="coerce")

    n_invalid = int(parsed.isna().sum()) - n_missing

    # Out-of-range dates (outside 2025).
    out_of_range = parsed.notna() & ((parsed < DATE_MIN) | (parsed > DATE_MAX))
    n_out_of_range = int(out_of_range.sum())

    print(f"Missing dates                     : {n_missing}")
    print(f"Invalid dates (unparseable)       : {n_invalid}")
    print(f"Dates outside 2025-01-01..12-31   : {n_out_of_range}")

    # Drop rows with invalid / missing / out-of-range dates (critical field).
    valid_mask = parsed.notna() & ~out_of_range
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        logger.warning(
            "Dropping %d row(s) with invalid/missing/out-of-range dates.",
            n_dropped)
        df = df[valid_mask].copy()

    # Format the surviving dates as YYYY-MM-DD.
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    print(f"Rows affected by date issues     : {n_dropped}")
    stats = {"missing_dates": n_missing, "invalid_dates": n_invalid,
             "out_of_range_dates": n_out_of_range,
             "date_rows_dropped": n_dropped}
    return df, stats


# ---------------------------------------------------------------------------
# 7. Validate Transaction_Type
# ---------------------------------------------------------------------------
def validate_transaction_type(df: pd.DataFrame):
    """Standardise Transaction_Type to Income/Expense and report invalid."""
    print("\n--- TRANSACTION TYPE VALIDATION ---")
    df = df.copy()

    df["Transaction_Type"] = df["Transaction_Type"].str.strip().str.capitalize()

    invalid = ~df["Transaction_Type"].isin(ALLOWED_TRANSACTION_TYPES)
    n_invalid = int(invalid.sum())
    if n_invalid:
        logger.warning("Found %d invalid Transaction_Type value(s).", n_invalid)

    print(f"Invalid Transaction_Type values : {n_invalid}")
    return df, {"invalid_transaction_types": n_invalid}


# ---------------------------------------------------------------------------
# 8. Validate Category
# ---------------------------------------------------------------------------
def validate_categories(df: pd.DataFrame):
    """Standardise categories and report invalid ones."""
    print("\n--- CATEGORY VALIDATION ---")
    df = df.copy()

    # Fix obvious variations (foods->Food, grocery->Groceries, transport->...).
    df["Category"] = df["Category"].map(
        lambda v: CATEGORY_MAP.get(str(v).strip().lower(), v)
        if not pd.isna(v) else v)

    all_categories = INCOME_CATEGORIES | EXPENSE_CATEGORIES
    invalid = ~df["Category"].isin(all_categories)
    n_invalid = int(invalid.sum())
    if n_invalid:
        logger.warning("Found %d invalid Category value(s): %s", n_invalid,
                       df.loc[invalid, "Category"].unique().tolist())

    print(f"Invalid Category values : {n_invalid}")
    return df, {"invalid_categories": n_invalid}


# ---------------------------------------------------------------------------
# 9. Validate Payment_Method
# ---------------------------------------------------------------------------
def validate_payment_method(df: pd.DataFrame):
    """Standardise Payment_Method and report invalid ones."""
    print("\n--- PAYMENT METHOD VALIDATION ---")
    df = df.copy()

    df["Payment_Method"] = df["Payment_Method"].map(
        lambda v: PAYMENT_METHOD_MAP.get(str(v).strip().lower(), v)
        if not pd.isna(v) else v)

    invalid = ~df["Payment_Method"].isin(ALLOWED_PAYMENT_METHODS)
    n_invalid = int(invalid.sum())
    if n_invalid:
        logger.warning("Found %d invalid Payment_Method value(s).", n_invalid)

    print(f"Invalid Payment_Method values : {n_invalid}")
    return df, {"invalid_payment_methods": n_invalid}


# ---------------------------------------------------------------------------
# 10. Validate numeric columns (Amount & Budget)
# ---------------------------------------------------------------------------
def clean_numeric_columns(df: pd.DataFrame):
    """
    Convert Amount & Budget to numeric and produce an outlier report (IQR).

    Amount must be numeric, without currency symbols/text, and positive.
    Rows with missing/invalid/zero/negative Amount are dropped (critical).

    Returns:
        (cleaned_df, stats_dict)
    """
    print("\n--- NUMERIC VALIDATION (Amount & Budget) ---")
    df = df.copy()

    # ---- Amount ----
    amount_raw = pd.to_numeric(df["Amount"], errors="coerce")
    n_missing_amount = int(amount_raw.isna().sum())
    n_zero = int((amount_raw == 0).sum())
    n_negative = int((amount_raw < 0).sum())
    n_invalid = n_missing_amount  # non-numeric values coerced to NaN

    # Drop rows with missing/invalid/zero/negative Amount (critical).
    before = len(df)
    df = df[amount_raw.notna() & (amount_raw > 0)].copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    n_dropped_amount = before - len(df)

    # ---- Outlier detection (IQR) on surviving Amount values ----
    q1 = df["Amount"].quantile(0.25)
    q3 = df["Amount"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = (df["Amount"] < lower) | (df["Amount"] > upper)
    n_outliers = int(outliers.sum())

    # ---- Budget ----
    budget_raw = pd.to_numeric(df["Budget"], errors="coerce")
    n_missing_budget = int(budget_raw.isna().sum())
    n_neg_budget = int((budget_raw < 0).sum())
    n_zero_budget = int((budget_raw == 0).sum())
    n_invalid_budget = n_missing_budget

    # Coerce Budget to numeric; fill NaN with 0 and clamp negatives to 0.
    df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce").fillna(0)
    df.loc[df["Budget"] < 0, "Budget"] = 0

    # ---- Budget consistency fix ----
    # Expense rows should carry the correct category budget. After category
    # standardization, assign the matching budget to any expense row whose
    # Budget is 0 (e.g. rows whose Category was missing/inferred).
    expense_mask = df["Transaction_Type"] == "Expense"
    zero_budget_expenses = expense_mask & (df["Budget"] == 0)
    if zero_budget_expenses.any():
        df.loc[zero_budget_expenses, "Budget"] = df.loc[
            zero_budget_expenses, "Category"].map(EXPENSE_BUDGETS)

    print(f"Amount -> missing: {n_missing_amount}, invalid: {n_invalid}, "
          f"zero: {n_zero}, negative: {n_negative}")
    print(f"Outliers detected (IQR) in Amount : {n_outliers}")
    print(f"Budget -> missing: {n_missing_budget}, invalid: {n_invalid_budget}, "
          f"zero: {n_zero_budget}, negative: {n_neg_budget}")
    stats = {"invalid_amounts": n_invalid, "zero_amounts": n_zero,
             "negative_amounts": n_negative, "outliers": n_outliers,
             "amount_rows_dropped": n_dropped_amount,
             "invalid_budgets": n_invalid_budget}
    return df, stats


# ---------------------------------------------------------------------------
# 11. Validate Budget (income/expense consistency)
# ---------------------------------------------------------------------------
def validate_budget(df: pd.DataFrame) -> None:
    """Check Budget consistency: income should be 0, expense should be > 0."""
    print("\n--- BUDGET CONSISTENCY CHECK ---")
    income_budget = df.loc[df["Transaction_Type"] == "Income", "Budget"]
    expense_budget = df.loc[df["Transaction_Type"] == "Expense", "Budget"]

    n_income_nonzero = int((income_budget != 0).sum())
    n_expense_zero = int((expense_budget == 0).sum())

    print(f"Income transactions with non-zero Budget : {n_income_nonzero}")
    print(f"Expense transactions with zero Budget     : {n_expense_zero}")


# ---------------------------------------------------------------------------
# 12. Business-rule validation (Rules 1-6)
# ---------------------------------------------------------------------------
def validate_business_rules(df: pd.DataFrame) -> dict:
    """
    Apply the six business rules and report any violations.

      Rule 1 : Income  -> Income_Source != "Not Applicable"
      Rule 2 : Expense -> Income_Source == "Not Applicable"
      Rule 3 : Income  -> must NOT use expense-only categories
      Rule 4 : Expense -> must NOT use income-only categories
      Rule 5 : Amount must be greater than 0
      Rule 6 : Transaction_ID matches TXN00001 pattern
    """
    print("\n--- BUSINESS RULE VALIDATION ---")
    income = df["Transaction_Type"] == "Income"
    expense = df["Transaction_Type"] == "Expense"

    # Rule 1
    r1 = int((income & (df["Income_Source"] == "Not Applicable")).sum())
    # Rule 2 (allow clearly justified exceptions -> none expected here)
    r2 = int((expense & (df["Income_Source"] != "Not Applicable")).sum())
    # Rule 3
    r3 = int((income & df["Category"].isin(EXPENSE_CATEGORIES
                                           - INCOME_CATEGORIES)).sum())
    # Rule 4
    r4 = int((expense & df["Category"].isin(INCOME_CATEGORIES
                                            - EXPENSE_CATEGORIES)).sum())
    # Rule 5
    r5 = int((df["Amount"] <= 0).sum())
    # Rule 6
    pattern = r"^TXN\d{5}$"
    r6 = int((~df["Transaction_ID"].astype(str).str.match(pattern)).sum())

    violations = {
        "rule1_income_source_violations": r1,
        "rule2_expense_source_violations": r2,
        "rule3_income_expense_category_violations": r3,
        "rule4_expense_income_category_violations": r4,
        "rule5_non_positive_amount_violations": r5,
        "rule6_bad_transaction_id_violations": r6,
    }

    for rule, count in violations.items():
        status = "OK" if count == 0 else "VIOLATION"
        print(f"  {rule:<46} : {count}  ({status})")

    return violations


# ---------------------------------------------------------------------------
# 13. Save cleaned dataset
# ---------------------------------------------------------------------------
def save_cleaned_data(df: pd.DataFrame, filepath: Path) -> None:
    """Write the cleaned dataset to CSV (creates parent dirs if needed)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Cleaned dataset saved to: %s (%d rows)", filepath, len(df))


# ---------------------------------------------------------------------------
# 14. Generate quality reports
# ---------------------------------------------------------------------------
def generate_quality_report(df_before: pd.DataFrame,
                            df_after: pd.DataFrame,
                            stats: dict) -> None:
    """
    Write reports/data_quality_report.csv and reports/cleaning_summary.txt.

    `stats` is a dict assembled by main() containing counts of every cleaning
    operation (duplicates, invalid values, missing values, etc).
    """
    QUALITY_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ---- Machine-readable quality report (CSV) ----
    quality_rows = {
        "Metric": [
            "Rows Before Cleaning",
            "Rows After Cleaning",
            "Columns",
            "Duplicate Rows",
            "Missing Values Before",
            "Missing Values After",
            "Invalid Dates",
            "Invalid Amounts",
            "Invalid Categories",
            "Invalid Payment Methods",
            "Invalid Transaction Types",
        ],
        "Value": [
            stats["rows_before"],
            stats["rows_after"],
            stats["columns"],
            stats["duplicate_rows_removed"],
            stats["missing_before"],
            stats["missing_after"],
            stats["invalid_dates"],
            stats["invalid_amounts"],
            stats["invalid_categories"],
            stats["invalid_payment_methods"],
            stats["invalid_transaction_types"],
        ],
    }
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(QUALITY_REPORT_FILE, index=False)
    logger.info("Quality report saved to: %s", QUALITY_REPORT_FILE)

    # ---- Human-readable cleaning summary (txt) ----
    lines = []
    lines.append("=" * 72)
    lines.append("PERSONAL FINANCE DATA CLEANING SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Source file      : {RAW_FILE}")
    lines.append(f"Cleaned file     : {CLEANED_FILE}")
    lines.append(f"Generated on     : {pd.Timestamp.now()}")
    lines.append("")
    lines.append("1. OVERALL COUNTS")
    lines.append(f"   Rows before cleaning           : {stats['rows_before']}")
    lines.append(f"   Rows after cleaning            : {stats['rows_after']}")
    lines.append(f"   Columns                        : {stats['columns']}")
    lines.append("")
    lines.append("2. DUPLICATES")
    lines.append(f"   Exact duplicate rows removed   : {stats['duplicate_rows_removed']}")
    lines.append(f"   Duplicate Transaction_IDs found: {stats['duplicate_ids_found']}")
    lines.append("")
    lines.append("3. MISSING VALUES")
    lines.append(f"   Missing cells before cleaning  : {stats['missing_before']}")
    lines.append(f"   Missing cells after cleaning   : {stats['missing_after']}")
    lines.append("   Fill strategy:")
    lines.append("     - Description      -> 'Unknown'")
    lines.append("     - Payment_Method   -> 'Unknown'")
    lines.append("     - Account          -> 'Unknown'")
    lines.append("     - Income_Source    -> 'Not Applicable' for expenses")
    lines.append("     - Budget           -> 0 (income) / category budget (expense)")
    lines.append("     - Category         -> inferred from Sub_Category, else 'Other'")
    lines.append("     - Sub_Category     -> 'Other'")
    lines.append("     - Amount           -> critical; missing Amount rows dropped")
    lines.append("")
    lines.append("4. DATE VALIDATION")
    lines.append(f"   Invalid dates                  : {stats['invalid_dates']}")
    lines.append(f"   Missing dates                  : {stats['missing_dates']}")
    lines.append(f"   Out-of-range dates (2025)      : {stats['out_of_range_dates']}")
    lines.append("   Action: rows with invalid/missing/out-of-range dates dropped")
    lines.append("")
    lines.append("5. NUMERIC VALIDATION")
    lines.append(f"   Invalid Amounts (non-numeric)  : {stats['invalid_amounts']}")
    lines.append(f"   Zero Amounts                   : {stats['zero_amounts']}")
    lines.append(f"   Negative Amounts               : {stats['negative_amounts']}")
    lines.append(f"   Outliers detected (IQR)        : {stats['outliers']}")
    lines.append(f"   Invalid Budgets                : {stats['invalid_budgets']}")
    lines.append("")
    lines.append("6. CATEGORY / PAYMENT / TYPE VALIDATION")
    lines.append(f"   Invalid Categories             : {stats['invalid_categories']}")
    lines.append(f"   Invalid Payment Methods        : {stats['invalid_payment_methods']}")
    lines.append(f"   Invalid Transaction Types      : {stats['invalid_transaction_types']}")
    lines.append("")
    lines.append("7. BUSINESS RULES")
    for rule, count in stats["business_rules"].items():
        lines.append(f"   {rule:<46}: {count}")
    lines.append("")
    lines.append("8. NOTE")
    lines.append("   The original raw dataset was never modified.")
    lines.append("   Outliers are detected but NOT removed (they are left in the")
    lines.append("   cleaned data since they may be legitimate high-value items).")
    lines.append("=" * 72)

    with open(CLEANING_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Cleaning summary saved to: %s", CLEANING_SUMMARY_FILE)


# ---------------------------------------------------------------------------
# 15. Final data-type report
# ---------------------------------------------------------------------------
def final_data_types(df: pd.DataFrame) -> None:
    """Print the final data types of the cleaned dataset."""
    print("\n--- FINAL DATA TYPES ---")
    print(df.dtypes.to_string())


# ---------------------------------------------------------------------------
# 16. Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the complete cleaning pipeline end-to-end."""
    logger.info("STEP 3 — Data cleaning pipeline started.")

    # 1. Load
    df_raw = load_data(RAW_FILE)
    rows_before = len(df_raw)
    n_columns = df_raw.shape[1]

    # 2. Inspect
    inspect_data(df_raw)

    # 3. Duplicates
    dup_stats = handle_duplicates(df_raw)
    df = df_raw.drop_duplicates(keep="first").reset_index(drop=True)

    # 4. Missing values
    df, missing_stats = handle_missing_values(df)

    # 5. Text cleaning
    df = clean_text_columns(df)

    # 6. Dates
    df, date_stats = clean_dates(df)

    # 7. Transaction type
    df, trans_stats = validate_transaction_type(df)

    # 8. Categories
    df, cat_stats = validate_categories(df)

    # 9. Payment methods
    df, pay_stats = validate_payment_method(df)

    # 10. Numeric (Amount & Budget)
    df, num_stats = clean_numeric_columns(df)

    # 11. Budget consistency
    validate_budget(df)

    # 12. Business rules
    rule_stats = validate_business_rules(df)

    # 13. Save cleaned data
    save_cleaned_data(df, CLEANED_FILE)

    # 14. Reports
    stats = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns": n_columns,
        "duplicate_rows_removed": dup_stats["duplicate_rows_removed"],
        "duplicate_ids_found": dup_stats["duplicate_ids_found"],
        "missing_before": missing_stats["missing_before"],
        "missing_after": missing_stats["missing_after"],
        "invalid_dates": date_stats["invalid_dates"],
        "missing_dates": date_stats["missing_dates"],
        "out_of_range_dates": date_stats["out_of_range_dates"],
        "invalid_amounts": num_stats["invalid_amounts"],
        "zero_amounts": num_stats["zero_amounts"],
        "negative_amounts": num_stats["negative_amounts"],
        "invalid_budgets": num_stats["invalid_budgets"],
        "invalid_categories": cat_stats["invalid_categories"],
        "invalid_payment_methods": pay_stats["invalid_payment_methods"],
        "invalid_transaction_types": trans_stats["invalid_transaction_types"],
        "outliers": num_stats["outliers"],
        "business_rules": rule_stats,
    }
    generate_quality_report(df_raw, df, stats)

    # 15. Final data types
    final_data_types(df)

    print("\n" + "=" * 72)
    print("CLEANING PIPELINE COMPLETE")
    print(f"Rows before : {rows_before} | Rows after : {len(df)}")
    print(f"Cleaned file: {CLEANED_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
