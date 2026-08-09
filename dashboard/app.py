"""Personal Finance Dashboard — Streamlit application.

A full-featured, interactive dashboard built on the project's cleaned dataset.

Features
--------
* Sidebar filters (date range, transaction type, category, payment method, account)
* Headline KPI cards (income, expenses, net balance, savings rate)
* Interactive Plotly charts:
    - Monthly income vs expense trend (line / area)
    - Expense breakdown by category (bar + donut)
    - Payment method breakdown
    - Budget vs actual spend
* Top transactions table
* Raw data explorer with CSV download

Run the app with:
    python run.py
or
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import analysis  # noqa: E402

CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "personal_finance_cleaned.csv"

# Streamlit page config (must be the first Streamlit command).
st.set_page_config(
    page_title="Personal Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached for performance)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and lightly prepare the cleaned dataset."""
    df = pd.read_csv(CLEANED_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df


@st.cache_data
def get_filters(df: pd.DataFrame) -> dict:
    """Pre-compute the unique option lists for the filter widgets."""
    return {
        "categories": sorted(df["Category"].dropna().unique().tolist()),
        "payment_methods": sorted(df["Payment_Method"].dropna().unique().tolist()),
        "accounts": sorted(df["Account"].dropna().unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Theming / helpers
# ---------------------------------------------------------------------------
INCOME_COLOR = "#2e9e5b"
EXPENSE_COLOR = "#e0503f"
ACCENT_COLOR = "#1f77b4"


def kpi_card(label: str, value: str, delta: str = None,
             color: str = "#1f77b4") -> None:
    """Render a single KPI metric card with a colored value."""
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border-radius:12px;
            padding:18px 20px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
            border-left:6px solid {color};
            margin-bottom:8px;
        ">
            <div style="font-size:13px;color:#6b7280;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.5px;">
                {label}
            </div>
            <div style="font-size:26px;font-weight:700;color:#111827;
                        margin-top:6px;">
                {value}
            </div>
            <div style="font-size:13px;color:#6b7280;margin-top:4px;">
                {delta or ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_inr(amount: float) -> str:
    """Format a number as Indian Rupees."""
    return f"₹{amount:,.0f}"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line/area chart of monthly income, expense and net."""
    trend = analysis.monthly_trend(df)
    if trend.empty:
        return px.line(title="No data for the selected period")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["Month"], y=trend["Income"], name="Income",
        mode="lines+markers", line=dict(color=INCOME_COLOR, width=3),
        fill="tozeroy", fillcolor="rgba(46,158,91,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=trend["Month"], y=trend["Expense"], name="Expense",
        mode="lines+markers", line=dict(color=EXPENSE_COLOR, width=3),
        fill="tozeroy", fillcolor="rgba(224,80,63,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=trend["Month"], y=trend["Net"], name="Net",
        mode="lines+markers", line=dict(color=ACCENT_COLOR, width=2, dash="dot"),
    ))
    fig.update_layout(
        title="Monthly Income vs Expense",
        xaxis_title="Month",
        yaxis_title="Amount (₹)",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def category_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of expense by category."""
    cats = analysis.category_totals(df, "Expense")
    if cats.empty:
        return px.bar(title="No expense data")
    fig = px.bar(
        cats,
        x="Amount",
        y="Category",
        orientation="h",
        title="Expense by Category",
        color_discrete_sequence=[EXPENSE_COLOR],
    )
    fig.update_layout(
        height=480,
        xaxis_title="Amount (₹)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def category_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of expense share by category."""
    cats = analysis.category_totals(df, "Expense")
    if cats.empty:
        return px.pie(title="No expense data")
    fig = px.pie(
        cats,
        names="Category",
        values="Amount",
        title="Expense Share by Category",
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent",
                      hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def payment_method_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of spend by payment method."""
    pm = analysis.payment_method_breakdown(df)
    if pm.empty:
        return px.bar(title="No data")
    fig = px.bar(
        pm,
        x="Payment_Method",
        y="Amount",
        title="Spend by Payment Method",
        color="Amount",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=380,
        xaxis_title="",
        yaxis_title="Amount (₹)",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def budget_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing category budget vs actual spend."""
    bv = analysis.budget_vs_actual(df)
    if bv.empty:
        return px.bar(title="No budget data")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bv["Category"], y=bv["Budget"], name="Budget",
        marker_color="#9ca3af",
    ))
    fig.add_trace(go.Bar(
        x=bv["Category"], y=bv["Actual"], name="Actual",
        marker_color=EXPENSE_COLOR,
    ))
    fig.update_layout(
        title="Budget vs Actual Spend by Category",
        barmode="group",
        height=460,
        xaxis_title="",
        yaxis_title="Amount (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the dashboard UI."""
    # ---- Header ----
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            .block-container { padding-top: 1.5rem; }
            h1 { font-family:'Inter',sans-serif; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("💰 Personal Finance Dashboard")
    st.caption("Interactive analytics of your 2025 personal finances "
               "(Income, Expenses, Budgets & Trends).")

    # ---- Load data ----
    df = load_data()
    if df.empty:
        st.error("No cleaned data found. Please run the cleaning pipeline first.")
        return

    filters = get_filters(df)

    # ===================================================================
    # SIDEBAR — Filters
    # ===================================================================
    st.sidebar.header("🔍 Filters")
    st.sidebar.markdown("Narrow down the view to analyse specific slices.")

    # Date range
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Transaction type
    txn_type = st.sidebar.selectbox(
        "Transaction Type",
        options=["All", "Income", "Expense"],
        index=0,
    )

    # Category
    category = st.sidebar.selectbox(
        "Category",
        options=["All"] + filters["categories"],
        index=0,
    )

    # Payment method
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        options=["All"] + filters["payment_methods"],
        index=0,
    )

    # Account
    account = st.sidebar.selectbox(
        "Account",
        options=["All"] + filters["accounts"],
        index=0,
    )

    # Reset button
    if st.sidebar.button("🔄 Reset Filters"):
        start_date, end_date = min_date, max_date
        txn_type, category, payment_method, account = (
            "All", "All", "All", "All"
        )
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("_Data source: `data/cleaned/personal_finance_cleaned.csv`_")

    # ===================================================================
    # Apply filters
    # ===================================================================
    filtered = df.copy()
    filtered = filtered[
        (filtered["Date"].dt.date >= start_date)
        & (filtered["Date"].dt.date <= end_date)
    ]
    if txn_type != "All":
        filtered = filtered[filtered["Transaction_Type"] == txn_type]
    if category != "All":
        filtered = filtered[filtered["Category"] == category]
    if payment_method != "All":
        filtered = filtered[filtered["Payment_Method"] == payment_method]
    if account != "All":
        filtered = filtered[filtered["Account"] == account]

    # ===================================================================
    # KPI Cards
    # ===================================================================
    summ = analysis.summary(filtered)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Income", format_inr(summ["total_income"]),
                 f"{summ['income_count']:,} transactions", INCOME_COLOR)
    with col2:
        kpi_card("Total Expenses", format_inr(summ["total_expenses"]),
                 f"{summ['expense_count']:,} transactions", EXPENSE_COLOR)
    with col3:
        net_color = ACCENT_COLOR if summ["net_balance"] >= 0 else EXPENSE_COLOR
        kpi_card("Net Balance", format_inr(summ["net_balance"]),
                 "Income − Expenses", net_color)
    with col4:
        kpi_card("Savings Rate", f"{summ['savings_rate']:.1f}%",
                 f"Avg txn: {format_inr(summ['average_transaction'])}",
                 INCOME_COLOR)

    st.divider()

    # ===================================================================
    # Charts - Row 1: Trend + Category
    # ===================================================================
    st.subheader("📈 Trends & Category Analysis")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(monthly_trend_chart(filtered),
                        use_container_width=True)
    with right:
        st.plotly_chart(payment_method_chart(filtered),
                        use_container_width=True)

    st.divider()

    # ===================================================================
    # Charts - Row 2: Category bar + donut + budget
    # ===================================================================
    st.subheader("🧾 Spending & Budgets")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(category_bar_chart(filtered), use_container_width=True)
    with c2:
        st.plotly_chart(category_donut_chart(filtered), use_container_width=True)

    st.plotly_chart(budget_chart(filtered), use_container_width=True)

    st.divider()

    # ===================================================================
    # Top transactions & raw data
    # ===================================================================
    st.subheader("🏆 Notable Transactions")

    tab1, tab2 = st.tabs(["Top Transactions", "Raw Data Explorer"])

    with tab1:
        if filtered.empty:
            st.info("No transactions match the selected filters.")
        else:
            top = analysis.top_transactions(filtered, n=10)
            display_cols = [
                "Date", "Transaction_ID", "Transaction_Type", "Category",
                "Sub_Category", "Description", "Amount", "Payment_Method",
                "Account",
            ]
            st.dataframe(
                top[display_cols].style.format(
                    {"Amount": "₹{:,.0f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        if filtered.empty:
            st.info("No transactions match the selected filters.")
        else:
            st.dataframe(filtered, use_container_width=True, height=420)
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download filtered data (CSV)",
                data=csv,
                file_name="filtered_finance_data.csv",
                mime="text/csv",
            )

    # ===================================================================
    # Footer
    # ===================================================================
    st.divider()
    st.caption(
        f"Showing {len(filtered):,} of {len(df):,} transactions "
        f"({start_date} → {end_date}). "
        "Built with Streamlit, Pandas & Plotly."
    )


if __name__ == "__main__":
    main()
