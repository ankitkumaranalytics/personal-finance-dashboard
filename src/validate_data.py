"""
validate_data.py
================

Validation script for the CLEANED Personal Finance dataset.

This script loads `data/cleaned/personal_finance_cleaned.csv` and verifies that
all the cleaning/validation rules from STEP 3 are satisfied:

  * No unexpected missing values
  * No exact duplicate rows
  * Valid dates (parseable, within 2025)
  * Valid Transaction_Type (Income / Expense)
  * Valid categories
  * Valid Payment_Method
  * Valid numeric Amount (positive)
  * Valid numeric Budget (income = 0, expense > 0)
  * Valid Transaction_ID format (TXN00001)
  * Valid Income/Expense business rules

It prints a final PASSED / FAILED result block.

Usage:
    python src/validate_data.py

Required libraries:
    pandas, numpy   (both already in requirements.txt)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
CLEANED_FILE = Path("data") / "cleaned" / "personal_finance_cleaned.csv"

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

ALLOWED_CATEGORIES = INCOME_CATEGORIES | EXPENSE_CATEGORIES

ALLOWED_PAYMENT_METHODS = {
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash",
    "Net Banking",
    "Bank Transfer",
    "Unknown",
}

DATE_MIN = pd.Timestamp("2025-01-01")
DATE_MAX = pd.Timestamp("2025-12-31")

TXN_ID_PATTERN = re.compile(r"^TXN\d{5}$")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def load_cleaned_data(filepath: Path) -> pd.DataFrame:
    """Load the cleaned CSV."""
    print(f"Loading cleaned data from: {filepath}")
    return pd.read_csv(filepath)


def validate_no_missing(df: pd.DataFrame) -> dict:
    """Check there are no missing values in the cleaned dataset."""
    missing_per_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    return {"total_missing": total_missing, "per_column": missing_per_col}


def validate_no_duplicates(df: pd.DataFrame) -> dict:
    """Check there are no exact duplicate rows."""
    n_duplicates = int(df.duplicated().sum())
    return {"exact_duplicates": n_duplicates}


def validate_dates(df: pd.DataFrame) -> dict:
    """Check dates are parseable and within 2025."""
    parsed = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    n_invalid = int(parsed.isna().sum())
    out_of_range = parsed.notna() & ((parsed < DATE_MIN) | (parsed > DATE_MAX))
    n_out_of_range = int(out_of_range.sum())
    return {"invalid_dates": n_invalid, "out_of_range_dates": n_out_of_range}


def validate_transaction_type(df: pd.DataFrame) -> dict:
    """Check Transaction_Type is only Income / Expense."""
    invalid = ~df["Transaction_Type"].isin(ALLOWED_TRANSACTION_TYPES)
    return {"invalid_transaction_types": int(invalid.sum())}


def validate_categories(df: pd.DataFrame) -> dict:
    """Check Category values are in the allowed set."""
    invalid = ~df["Category"].isin(ALLOWED_CATEGORIES)
    return {"invalid_categories": int(invalid.sum())}


def validate_payment_methods(df: pd.DataFrame) -> dict:
    """Check Payment_Method values are in the allowed set."""
    invalid = ~df["Payment_Method"].isin(ALLOWED_PAYMENT_METHODS)
    return {"invalid_payment_methods": int(invalid.sum())}


def validate_amounts(df: pd.DataFrame) -> dict:
    """Check Amount is numeric and greater than 0."""
    numeric = pd.to_numeric(df["Amount"], errors="coerce")
    n_invalid = int(numeric.isna().sum())
    n_non_positive = int((numeric <= 0).sum())
    return {"invalid_amounts": n_invalid, "non_positive_amounts": n_non_positive}


def validate_budgets(df: pd.DataFrame) -> dict:
    """Check Budget is numeric; income=0 and expense>0."""
    budget = pd.to_numeric(df["Budget"], errors="coerce")
    n_invalid = int(budget.isna().sum())

    income = df["Transaction_Type"] == "Income"
    expense = df["Transaction_Type"] == "Expense"

    n_income_nonzero = int((income & (budget != 0)).sum())
    n_expense_nonpositive = int((expense & (budget <= 0)).sum())

    return {
        "invalid_budgets": n_invalid,
        "income_with_nonzero_budget": n_income_nonzero,
        "expense_with_nonpositive_budget": n_expense_nonpositive,
    }


def validate_txn_ids(df: pd.DataFrame) -> dict:
    """Check Transaction_ID matches TXN00001 pattern and is unique."""
    n_duplicate_ids = int(df["Transaction_ID"].duplicated().sum())
    n_bad_format = int((~df["Transaction_ID"].astype(str).str.match(
        r"^TXN\d{5}$")).sum())
    return {"duplicate_txn_ids": n_duplicate_ids,
            "bad_format_txn_ids": n_bad_format}


def validate_business_rules(df: pd.DataFrame) -> dict:
    """Check Income/Expense business rules."""
    income = df["Transaction_Type"] == "Income"
    expense = df["Transaction_Type"] == "Expense"

    # Rule 1: Income -> Income_Source != "Not Applicable"
    r1 = int((income & (df["Income_Source"] == "Not Applicable")).sum())

    # Rule 2: Expense -> Income_Source == "Not Applicable"
    r2 = int((expense & (df["Income_Source"] != "Not Applicable")).sum())

    # Rule 3: Income -> not expense-only categories
    r3 = int((income & df["Category"].isin(EXPENSE_CATEGORIES
                                           - INCOME_CATEGORIES)).sum())

    # Rule 4: Expense -> not income-only categories
    r4 = int((expense & df["Category"].isin(INCOME_CATEGORIES
                                            - EXPENSE_CATEGORIES)).sum())

    return {
        "rule1_income_na_source": r1,
        "rule2_expense_non_na_source": r2,
        "rule3_income_expense_category": r3,
        "rule4_expense_income_category": r4,
    }


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all validation checks and print the final PASSED/FAILED result."""
    print("=" * 72)
    print("DATA VALIDATION")
    print("=" * 72)

    df = load_cleaned_data(CLEANED_FILE)

    # Run every check and collect results.
    checks = {
        "Missing Values": validate_no_missing(df),
        "Duplicates": validate_no_duplicates(df),
        "Dates": validate_dates(df),
        "Transaction Types": validate_transaction_type(df),
        "Categories": validate_categories(df),
        "Payment Methods": validate_payment_methods(df),
        "Amounts": validate_amounts(df),
        "Budgets": validate_budgets(df),
        "Transaction IDs": validate_txn_ids(df),
        "Business Rules": validate_business_rules(df),
    }

    # Determine overall pass/fail (any nonzero invalid count -> FAILED).
    failed = False
    for section, result in checks.items():
        for metric, value in result.items():
            if isinstance(value, int) and value != 0:
                failed = True

    # ---- Print the final result block ----
    print("\n" + "=" * 72)
    print("DATA VALIDATION RESULT")
    print("=" * 72)
    print(f"\nDataset Status: {'PASSED' if not failed else 'FAILED'}")
    print(f"{'Rows:':<18}{len(df)}")
    print(f"{'Columns:':<18}{df.shape[1]}")
    print(f"{'Duplicate Rows:':<18}{checks['Duplicates']['exact_duplicates']}")
    print(f"{'Missing Values:':<18}{checks['Missing Values']['total_missing']}")
    print(f"{'Invalid Dates:':<18}{checks['Dates']['invalid_dates']}")
    print(f"{'Invalid Amounts:':<18}{checks['Amounts']['invalid_amounts']}")
    print(f"{'Invalid Categories:':<18}{checks['Categories']['invalid_categories']}")
    print(f"{'Invalid Payment Methods:':<18}{checks['Payment Methods']['invalid_payment_methods']}")

    print("\nDetailed check breakdown:")
    for section, result in checks.items():
        print(f"  {section}:")
        for metric, value in result.items():
            status = "OK" if (isinstance(value, int) and value == 0) else "CHECK"
            print(f"    - {metric:<40}: {value}  ({status})")

    print("\n" + "=" * 50)
    print("All validation checks completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
