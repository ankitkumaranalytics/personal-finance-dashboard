# 💰 Personal Finance Dashboard

An end-to-end **Personal Finance Analytics Dashboard** built with **Python, Pandas & Streamlit**. It ingests raw transaction data, cleans and validates it, performs insightful analysis, and visualizes everything in an interactive web dashboard.

---

## ✨ Features

- **Interactive Streamlit dashboard** with:
  - Sidebar filters (date range, transaction type, category, payment method, account)
  - KPI cards — total income, total expenses, net balance, savings rate
  - Monthly income vs expense trend chart
  - Expense breakdown by category (bar + donut)
  - Spend by payment method
  - Budget vs actual comparison
  - Top transactions & raw data explorer (with CSV download)

---

## 🗂️ Project Structure

```
Personal-Finance-Dashboard/
├── data/
│   ├── raw/                        # Raw financial data (generated)
│   └── cleaned/                    # Cleaned & validated data
├── src/
│   ├── create_dataset.py           # STEP 1 — Generate realistic dataset
│   ├── data_cleaning.py            # STEP 2 — Full cleaning pipeline
│   ├── validate_data.py            # STEP 3 — Validation & quality checks
│   └── analysis.py                 # STEP 4 — Analysis functions
├── dashboard/
│   └── app.py                      # STEP 5 — Streamlit dashboard
├── reports/                        # Cleaning & quality reports
├── assets/                         # Static assets
├── notebooks/                      # Exploration notebooks
├── requirements.txt
├── run.py                          # Launcher
└── README.md
```

---

## 🚀 Setup

1. **Create a virtual environment** (recommended):

   Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running the pipeline (end-to-end)

Run each stage in order:

```bash
# STEP 1 — Generate the raw dataset (~2000 transactions)
python src/create_dataset.py

# STEP 2 — Clean the data
python src/data_cleaning.py

# STEP 3 — Validate the cleaned data
python src/validate_data.py
```

---

## 📊 Launching the dashboard

```bash
python run.py
```

Or directly:

```bash
streamlit run dashboard/app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 🧠 What the dashboard answers

- **How much am I earning & spending?** → KPI cards & monthly trend
- **Where does my money go?** → Category bar & donut charts
- **Which payment method do I use most?** → Payment method breakdown
- **Am I staying within budget?** → Budget vs actual chart
- **Which single transactions matter most?** → Top transactions table

---

## 📦 Dependencies

- `pandas` — data manipulation
- `numpy` — numerical operations
- `streamlit` — interactive dashboard framework
- `plotly` — interactive charts

---

## 📄 Reports

After running the cleaning pipeline, quality reports are generated in `reports/`:

- `data_quality_report.csv` — machine-readable quality metrics
- `cleaning_summary.txt` — human-readable cleaning summary
