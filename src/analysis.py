"""Analysis functions for the Personal Finance Dashboard.

This module provides a collection of reusable, pure-Pandas analysis helpers
that the Streamlit dashboard (`dashboard/app.py`) uses to compute KPIs,
trends, category breakdowns, budget comparisons and more.

All functions expect a cleaned DataFrame that contains (at least) the columns:
    Transaction_Type, Category, Sub_Category, Description, Amount,
    Payment_Method, Account, Income_Source, Budget, Date, Transaction_ID

The `Date` column is expected to be parseable as a datetime.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Basic KPIs
# ---------------------------------------------------------------------------
def total_income(df: pd.DataFrame) -> float:
    """Return the total income across the filtered dataset."""
    return float(
        df.loc[df["Transaction_Type"] == "Income", "Amount"].sum()
    )


def total_expenses(df: pd.DataFrame) -> float:
    """Return the total expenses across the filtered dataset."""
    return float(
        df.loc[df["Transaction_Type"] == "Expense", "Amount"].sum()
    )


def net_balance(df: pd.DataFrame) -> float:
    """Return the net balance (income - expenses)."""
    return total_income(df) - total_expenses(df)


def savings_rate(df: pd.DataFrame) -> float:
    """Return the savings rate as a percentage (0-100)."""
    inc = total_income(df)
    if inc <= 0:
        return 0.0
    return ((inc - total_expenses(df)) / inc) * 100


def transaction_counts(df: pd.DataFrame) -> dict:
    """Return the number of income and expense transactions."""
    incomes = int((df["Transaction_Type"] == "Income").sum())
    expenses = int((df["Transaction_Type"] == "Expense").sum())
    return {"income_count": incomes, "expense_count": expenses}


def average_transaction(df: pd.DataFrame) -> float:
    """Return the average transaction amount."""
    if len(df) == 0:
        return 0.0
    return float(df["Amount"].mean())


# ---------------------------------------------------------------------------
# Time-based trends
# ---------------------------------------------------------------------------
def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with `Date` parsed to datetime."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with monthly income, expense and net values.

    Index/column layout:
        Date (period) | Income | Expense | Net
    """
    df = ensure_datetime(df)
    if df.empty:
        return pd.DataFrame(columns=["Month", "Income", "Expense", "Net"])

    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    income = (
        df[df["Transaction_Type"] == "Income"]
        .groupby("Month")["Amount"]
        .sum()
        .rename("Income")
    )
    expense = (
        df[df["Transaction_Type"] == "Expense"]
        .groupby("Month")["Amount"]
        .sum()
        .rename("Expense")
    )

    out = pd.DataFrame(index=income.index)
    out["Income"] = income
    out["Expense"] = expense.reindex(out.index).fillna(0.0)
    out["Net"] = out["Income"] - out["Expense"]
    out = out.fillna(0.0).reset_index()
    return out


def income_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Series (Month -> Income)."""
    df = ensure_datetime(df)
    df = df[df["Transaction_Type"] == "Income"]
    if df.empty:
        return pd.Series(dtype=float)
    return (
        df["Amount"]
        .groupby(df["Date"].dt.to_period("M").astype(str))
        .sum()
    )


def expense_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Series (Month -> Expense)."""
    df = ensure_datetime(df)
    df = df[df["Transaction_Type"] == "Expense"]
    if df.empty:
        return pd.Series(dtype=float)
    return (
        df["Amount"]
        .groupby(df["Date"].dt.to_period("M").astype(str))
        .sum()
    )


# ---------------------------------------------------------------------------
# Category breakdowns
# ---------------------------------------------------------------------------
def category_totals(df: pd.DataFrame,
                    transaction_type: str = "Expense") -> pd.DataFrame:
    """Return a DataFrame of total amount per Category for a transaction type."""
    df = df[df["Transaction_Type"] == transaction_type]
    if df.empty:
        return pd.DataFrame(columns=["Category", "Amount"])
    out = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    return out


def subcategory_totals(df: pd.DataFrame,
                       transaction_type: str = "Expense") -> pd.DataFrame:
    """Return a DataFrame of total amount per Sub_Category."""
    df = df[df["Transaction_Type"] == transaction_type]
    if df.empty:
        return pd.DataFrame(columns=["Sub_Category", "Amount"])
    out = (
        df.groupby("Sub_Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    return out


def top_categories(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top-N spending categories."""
    return category_totals(df, "Expense").head(n)


# ---------------------------------------------------------------------------
# Payment method & account breakdowns
# ---------------------------------------------------------------------------
def payment_method_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return totals per Payment_Method (all transactions)."""
    if df.empty:
        return pd.DataFrame(columns=["Payment_Method", "Amount"])
    return (
        df.groupby("Payment_Method")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def account_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return totals per Account (all transactions)."""
    if df.empty:
        return pd.DataFrame(columns=["Account", "Amount"])
    return (
        df.groupby("Account")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Budget vs actual
# ---------------------------------------------------------------------------
def budget_vs_actual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each expense category's total spend against its monthly budget.

    Parameters
    ----------
    df : the currently-filtered DataFrame (may span multiple months).

    Returns
    -------
    DataFrame with columns: Category, Budget, Actual, Utilization_Percent
    """
    expenses = df[df["Transaction_Type"] == "Expense"]
    if expenses.empty:
        return pd.DataFrame(
            columns=["Category", "Budget", "Actual", "Utilization_Percent"]
        )

    actual = expenses.groupby("Category")["Amount"].sum()
    # Budget is per-month; approximate by taking the mode/max budget per category.
    budget = (
        expenses.groupby("Category")["Budget"]
        .max()
    )

    out = pd.DataFrame({"Budget": budget, "Actual": actual}).fillna(0.0)
    out["Utilization_Percent"] = (
        out["Actual"] / out["Budget"].replace(0, pd.NA) * 100
    ).fillna(0.0)
    out = out.sort_values("Actual", ascending=False).reset_index()
    return out


def over_budget_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Return expense categories whose spend exceeds budget."""
    bv = budget_vs_actual(df)
    if bv.empty:
        return bv
    return bv[bv["Utilization_Percent"] > 100]


# ---------------------------------------------------------------------------
# Insights / highlights
# ---------------------------------------------------------------------------
def top_transactions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-N transactions by amount (whole dataset)."""
    if df.empty:
        return df
    return df.nlargest(n, "Amount")


def largest_expense(df: pd.DataFrame) -> pd.Series:
    """Return the single largest expense transaction."""
    expenses = df[df["Transaction_Type"] == "Expense"]
    if expenses.empty:
        return pd.Series(dtype=float)
    return expenses.loc[expenses["Amount"].idxmax()]


def largest_income(df: pd.DataFrame) -> pd.Series:
    """Return the single largest income transaction."""
    incomes = df[df["Transaction_Type"] == "Income"]
    if incomes.empty:
        return pd.Series(dtype=float)
    return incomes.loc[incomes["Amount"].idxmax()]


def busiest_payment_method(df: pd.DataFrame) -> str:
    """Return the payment method used most often (by count)."""
    if df.empty:
        return "N/A"
    return df["Payment_Method"].value_counts().idxmax()


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------
def summary(df: pd.DataFrame) -> dict:
    """Return a dictionary of headline KPIs for the dashboard."""
    return {
        "total_income": total_income(df),
        "total_expenses": total_expenses(df),
        "net_balance": net_balance(df),
        "savings_rate": savings_rate(df),
        "income_count": transaction_counts(df)["income_count"],
        "expense_count": transaction_counts(df)["expense_count"],
        "average_transaction": average_transaction(df),
    }
