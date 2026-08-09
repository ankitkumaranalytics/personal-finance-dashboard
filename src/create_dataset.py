"""
create_dataset.py
==================

Generates a realistic Personal Finance dataset for the project:

    "Personal Finance Analytics Dashboard using Python & Streamlit"

The script creates `data/raw/personal_finance.csv` containing roughly 2,000
transaction records spanning January 2025 to December 2025.

The data is intentionally made "mostly clean" with a small number of realistic
data-quality issues (missing values, duplicates, inconsistent values and a few
invalid dates) so that later data-cleaning steps make sense.

The dataset is generated with a fixed random seed so that the output is
REPRODUCIBLE. Run this script again any time to regenerate the exact same CSV.

Usage:
    python src/create_dataset.py

Required libraries:
    pandas, numpy
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
# Fixing the seed ensures the same "random" data is generated every run.
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Where the CSV file will be saved.
RAW_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "personal_finance.csv")

# ---------------------------------------------------------------------------
# 1. Configuration: categories, sub-categories, descriptions and amounts
# ---------------------------------------------------------------------------

# Income categories and their sub-categories.
INCOME_CATEGORIES = [
    "Salary",
    "Freelance",
    "Business",
    "Investment",
    "Other Income",
]

# Expense categories mapped to their realistic sub-categories.
EXPENSE_CATEGORIES = {
    "Food": [
        "Restaurants",
        "Fast Food",
        "Cafe",
        "Food Delivery",
    ],
    "Transportation": [
        "Fuel",
        "Metro",
        "Bus",
        "Cab",
        "Vehicle Maintenance",
    ],
    "Shopping": [
        "Clothing",
        "Electronics",
        "Accessories",
        "Online Shopping",
    ],
    "Bills & Utilities": [
        "Electricity",
        "Internet",
        "Mobile",
        "Water",
    ],
    "Rent": ["Monthly Rent"],
    "Healthcare": [
        "Doctor Visit",
        "Medicine",
        "Lab Test",
        "Medical Consultation",
    ],
    "Education": [
        "College Fees",
        "Course Fee",
        "Books & Materials",
        "Online Course",
    ],
    "Entertainment": [
        "Movies",
        "Netflix Subscription",
        "Gaming",
        "Concerts",
        "Music",
    ],
    "Travel": [
        "Flight",
        "Hotel",
        "Train",
        "Trip Package",
    ],
    "Groceries": [
        "Supermarket",
        "Vegetables & Fruits",
        "Dairy & Bakery",
        "Monthly Grocery",
    ],
    "Personal Care": [
        "Salon",
        "Gym Membership",
        "Skincare",
        "Grooming",
    ],
    "Insurance": [
        "Health Insurance",
        "Life Insurance",
        "Vehicle Insurance",
        "Term Plan",
    ],
    "Other": ["Miscellaneous", "Gifts", "Donation", "Subscriptions"],
}

# Realistic descriptions for each expense category.
EXPENSE_DESCRIPTIONS = {
    "Food": [
        "Restaurant dinner",
        "Fast food lunch",
        "Coffee at cafe",
        "Food delivery order",
        "Weekend brunch",
        "Team lunch",
    ],
    "Transportation": [
        "Petrol refill",
        "Metro card recharge",
        "Bus ticket",
        "Uber ride",
        "Cab ride",
        "Car service",
        "Fuel top-up",
    ],
    "Shopping": [
        "Amazon purchase",
        "Flipkart order",
        "Clothing shopping",
        "New electronics",
        "Accessories purchase",
        "Online shopping",
    ],
    "Bills & Utilities": [
        "Electricity bill",
        "Internet bill",
        "Mobile recharge",
        "Water bill",
        "Broadband bill",
    ],
    "Rent": ["Monthly rent payment", "House rent"],
    "Healthcare": [
        "Doctor consultation",
        "Medicine purchase",
        "Lab test",
        "Medical checkup",
        "Pharmacy bill",
    ],
    "Education": [
        "College fees",
        "Online course fee",
        "Books purchase",
        "Certification fee",
        "Tuition fees",
    ],
    "Entertainment": [
        "Netflix subscription",
        "Movie ticket",
        "Gaming purchase",
        "Concert ticket",
        "Spotify subscription",
    ],
    "Travel": [
        "Flight ticket",
        "Hotel booking",
        "Train ticket",
        "Trip package",
        "Weekend getaway",
    ],
    "Groceries": [
        "Grocery shopping",
        "Vegetables & fruits",
        "Dairy products",
        "Monthly grocery",
        "Supermarket bill",
    ],
    "Personal Care": [
        "Haircut at salon",
        "Gym membership",
        "Skincare products",
        "Grooming kit",
    ],
    "Insurance": [
        "Health insurance premium",
        "Life insurance premium",
        "Vehicle insurance",
        "Term plan premium",
    ],
    "Other": [
        "Miscellaneous expense",
        "Gift purchase",
        "Donation",
        "Subscription fee",
    ],
}

# Realistic income descriptions per category.
INCOME_DESCRIPTIONS = {
    "Salary": [
        "Monthly salary",
        "Salary credit",
        "Salary - January",
        "Salary - February",
        "Salary - March",
        "Salary - April",
        "Salary - May",
        "Salary - June",
        "Salary - July",
        "Salary - August",
        "Salary - September",
        "Salary - October",
        "Salary - November",
        "Salary - December",
    ],
    "Freelance": [
        "Freelance project payment",
        "Freelance design work",
        "Freelance writing gig",
        "Freelance development payment",
        "Freelance consulting",
        "Freelance gig income",
    ],
    "Business": [
        "Business revenue",
        "Business sale income",
        "Business profit share",
        "Client payment",
        "Business earnings",
    ],
    "Investment": [
        "Dividend income",
        "Stock profit",
        "Mutual fund returns",
        "Interest income",
        "FD interest",
        "Investment returns",
        "SIP redemption",
    ],
    "Other Income": [
        "Cashback reward",
        "Gift received",
        "Refund received",
        "Bonus income",
        "Miscellaneous income",
    ],
}

# ---------------------------------------------------------------------------
# 2. Amount ranges (Indian Rupees)
# ---------------------------------------------------------------------------

# Income amount ranges per category: (min, max)
INCOME_AMOUNT_RANGES = {
    "Salary": (20000, 80000),
    "Freelance": (2000, 30000),
    "Business": (5000, 50000),
    "Investment": (500, 15000),
    "Other Income": (200, 8000),
}

# Expense amount ranges per category: (min, max)
EXPENSE_AMOUNT_RANGES = {
    "Food": (80, 1500),
    "Transportation": (50, 2500),
    "Shopping": (500, 12000),
    "Bills & Utilities": (300, 5000),
    "Rent": (12000, 25000),
    "Healthcare": (300, 8000),
    "Education": (1000, 25000),
    "Entertainment": (150, 3000),
    "Travel": (2000, 40000),
    "Groceries": (500, 6000),
    "Personal Care": (200, 3000),
    "Insurance": (2000, 15000),
    "Other": (100, 3000),
}

# Monthly budget assigned to each expense category (for the Budget column).
EXPENSE_BUDGET = {
    "Food": 8000,
    "Transportation": 5000,
    "Shopping": 10000,
    "Bills & Utilities": 6000,
    "Rent": 20000,
    "Healthcare": 5000,
    "Education": 15000,
    "Entertainment": 4000,
    "Travel": 15000,
    "Groceries": 8000,
    "Personal Care": 4000,
    "Insurance": 12000,
    "Other": 5000,
}

PAYMENT_METHODS = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash",
    "Net Banking",
    "Bank Transfer",
]

ACCOUNTS = [
    "HDFC Bank",
    "SBI",
    "ICICI Bank",
    "Axis Bank",
    "Cash Wallet",
]

# Income source mapped per income category.
INCOME_SOURCE = {
    "Salary": "Employer",
    "Freelance": "Freelance Clients",
    "Business": "Business Earnings",
    "Investment": "Investment Portfolio",
    "Other Income": "Other Sources",
}


# ---------------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------------

def random_date(start="2025-01-01", end="2025-12-31"):
    """Return a random date (as a string) between start and end."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    # Number of days between the two dates.
    days = (end_dt - start_dt).days
    random_offset = rng.integers(0, days + 1)
    return (start_dt + pd.Timedelta(days=int(random_offset))).strftime("%Y-%m-%d")


def build_fixed_categories():
    """Create a reusable pool of (type, category, sub_category) rows."""
    rows = []

    # Income rows.
    for cat in INCOME_CATEGORIES:
        # Salary only gets one sub-category slot each month but we keep it simple.
        sub = "Monthly Salary" if cat == "Salary" else cat
        rows.append(("Income", cat, sub))

    # Expense rows: one row per (category, subcategory).
    for cat, subs in EXPENSE_CATEGORIES.items():
        for sub in subs:
            rows.append(("Expense", cat, sub))

    return rows


# ---------------------------------------------------------------------------
# 4. Generate the dataset
# ---------------------------------------------------------------------------

def generate_transactions(n_total=2000):
    """
    Generate `n_total` realistic transaction records.

    Roughly 15% are Income and 85% are Expense.
    """
    records = []

    # Decide how many income vs expense transactions.
    n_income = int(n_total * 0.15)  # ~300 income
    n_expense = n_total - n_income  # ~1700 expense

    # ------------------------------------------------------------------
    # Income transactions
    # ------------------------------------------------------------------
    for i in range(n_income):
        category = rng.choice(INCOME_CATEGORIES)
        sub_category = (
            "Monthly Salary" if category == "Salary" else category
        )
        description = rng.choice(INCOME_DESCRIPTIONS[category])
        amount_min, amount_max = INCOME_AMOUNT_RANGES[category]
        # Round to nearest whole rupee.
        amount = round(rng.uniform(amount_min, amount_max))

        records.append(
            {
                "Transaction_Type": "Income",
                "Category": category,
                "Sub_Category": sub_category,
                "Description": description,
                "Amount": amount,
                "Payment_Method": rng.choice(PAYMENT_METHODS),
                "Account": rng.choice(ACCOUNTS),
                "Income_Source": INCOME_SOURCE[category],
                "Budget": 0,
            }
        )

    # ------------------------------------------------------------------
    # Expense transactions
    # ------------------------------------------------------------------
    # Build a weighted pool of categories so everyday categories appear more often.
    category_weights = {
        "Food": 8,
        "Transportation": 6,
        "Shopping": 5,
        "Bills & Utilities": 5,
        "Rent": 2,
        "Healthcare": 3,
        "Education": 2,
        "Entertainment": 4,
        "Travel": 2,
        "Groceries": 6,
        "Personal Care": 3,
        "Insurance": 2,
        "Other": 2,
    }
    expense_cats = list(category_weights.keys())
    weights = [category_weights[c] for c in expense_cats]

    for _ in range(n_expense):
        category = rng.choice(expense_cats, p=np.array(weights) / sum(weights))
        sub_category = rng.choice(EXPENSE_CATEGORIES[category])
        description = rng.choice(EXPENSE_DESCRIPTIONS[category])
        amount_min, amount_max = EXPENSE_AMOUNT_RANGES[category]
        amount = round(rng.uniform(amount_min, amount_max))

        records.append(
            {
                "Transaction_Type": "Expense",
                "Category": category,
                "Sub_Category": sub_category,
                "Description": description,
                "Amount": amount,
                "Payment_Method": rng.choice(PAYMENT_METHODS),
                "Account": rng.choice(ACCOUNTS),
                "Income_Source": "Not Applicable",
                "Budget": EXPENSE_BUDGET[category],
            }
        )

    # ------------------------------------------------------------------
    # Build the DataFrame and add date + Transaction_ID.
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)

    # Random dates for every transaction.
    df["Date"] = [random_date() for _ in range(len(df))]

    # Transaction_ID in the format TXN00001, TXN00002, ...
    df["Transaction_ID"] = [f"TXN{i:05d}" for i in range(1, len(df) + 1)]

    # Reorder columns to match the required specification.
    df = df[
        [
            "Transaction_ID",
            "Date",
            "Transaction_Type",
            "Category",
            "Sub_Category",
            "Description",
            "Amount",
            "Payment_Method",
            "Account",
            "Income_Source",
            "Budget",
        ]
    ]

    return df


def inject_quality_issues(df):
    """
    Intentionally add a small number of realistic data-quality issues:

    1. 10-15 missing values
    2. 5-10 duplicate transaction records
    3. 5 inconsistent category/subcategory values
    4. 5 inconsistent payment-method values (e.g. 'UPI ', 'upi')
    5. 3-5 invalid/incorrect date formats
    """
    df = df.copy()

    # --- 1. Missing values (12) -------------------------------------
    # Replace a handful of Amount / Category / Description values with NaN.
    missing_indices_amount = rng.choice(df.index, size=5, replace=False)
    df.loc[missing_indices_amount, "Amount"] = np.nan

    missing_indices_category = rng.choice(df.index, size=4, replace=False)
    df.loc[missing_indices_category, "Category"] = np.nan

    missing_indices_desc = rng.choice(df.index, size=3, replace=False)
    df.loc[missing_indices_desc, "Description"] = np.nan
    # Note: some indices may overlap; that is fine and realistic.

    # --- 2. Duplicate transaction records (7) -----------------------
    # Pick some rows and append exact copies of them at the end.
    duplicate_source = df.sample(n=7, random_state=RANDOM_SEED)
    duplicates = duplicate_source.copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # --- 3. Inconsistent category/subcategory values (5) ------------
    # Pick 5 random expense rows and change their category to a slightly
    # inconsistent label (e.g. 'food' instead of 'Food').
    expense_rows = df.index[df["Transaction_Type"] == "Expense"].tolist()
    change_idx = rng.choice(expense_rows, size=5, replace=False)
    inconsistent_cats = ["food", "TRansportation", "shopping", "grocery", "entertainment"]
    for k, idx in enumerate(change_idx):
        df.loc[idx, "Category"] = inconsistent_cats[k % len(inconsistent_cats)]

    # --- 4. Inconsistent payment-method values (5) ------------------
    # e.g. 'UPI ' (trailing space), 'upi', 'credit card', 'cash'.
    payment_idx = rng.choice(df.index, size=5, replace=False)
    inconsistent_payments = ["UPI ", "upi", "Credit card", "cash", "Net banking"]
    for k, idx in enumerate(payment_idx):
        df.loc[idx, "Payment_Method"] = inconsistent_payments[k % len(inconsistent_payments)]

    # --- 5. Invalid/incorrect date formats (4) ----------------------
    date_idx = rng.choice(df.index, size=4, replace=False)
    invalid_dates = ["2025-13-01", "31-01-2025", "2025/02/15", "2025-12-32"]
    for k, idx in enumerate(date_idx):
        df.loc[idx, "Date"] = invalid_dates[k % len(invalid_dates)]

    return df


# ---------------------------------------------------------------------------
# 5. Verification / reporting
# ---------------------------------------------------------------------------

def verify_dataset(df):
    """Print all the required verification checks for the dataset."""
    print("=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)

    # 1. Number of rows and columns.
    print(f"1. Shape (rows, columns): {df.shape}")

    # 2. Transaction_ID uniqueness (excluding intended duplicates).
    total = len(df)
    unique_ids = df["Transaction_ID"].nunique()
    print(
        f"2. Transaction_ID -> total records: {total}, "
        f"unique IDs: {unique_ids}, duplicates: {total - unique_ids}"
    )

    # 3. First 10 rows.
    print("\n3. First 10 rows:")
    print(df.head(10).to_string(index=False))

    # 4. Last 10 rows.
    print("\n4. Last 10 rows:")
    print(df.tail(10).to_string(index=False))

    # 5. All column names.
    print("\n5. Column names:")
    print(list(df.columns))

    # 6. Missing-value counts.
    print(f"\n6. Missing values per column:\n{df.isna().sum()}")

    # 7. Duplicate-row count (exact duplicates).
    print(f"\n7. Duplicate rows (exact): {df.duplicated().sum()}")

    # 8. Basic statistics for Amount.
    print(f"\n8. Amount statistics:\n{df['Amount'].describe()}")

    # 9. Number of Income and Expense transactions.
    print(f"\n9. Transaction type counts:\n{df['Transaction_Type'].value_counts()}")

    # 10. Category-wise transaction counts.
    print(f"\n10. Category-wise counts:\n{df['Category'].value_counts()}")

    # 11. Monthly transaction counts (parse Date where possible).
    #    We temporarily coerce invalid dates to NaT so they do not crash.
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    monthly = parsed.dt.to_period("M").value_counts().sort_index()
    print(f"\n11. Monthly counts:\n{monthly}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------------------------

def main():
    """Generate the dataset, save it, and verify it."""
    # Make sure the output directory exists.
    os.makedirs(RAW_DIR, exist_ok=True)

    # Generate the raw dataset.
    df = generate_transactions(n_total=2000)

    # Inject small, realistic data-quality issues.
    df = inject_quality_issues(df)

    # Save the raw CSV (original remains untouched after this).
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Total records written: {len(df)}")

    # Run the verification checks.
    verify_dataset(df)


if __name__ == "__main__":
    main()
