
# Create enhanced version with MORE GRAPHS
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dateutil.parser import parse
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Professional UI
# -----------------------------
st.markdown("""
<style>

/* ------------------ GLOBAL LAYOUT ------------------ */

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #101827 0%, #020617 50%, #020617 100%) !important;
    color: #e5e7eb !important;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Main app container */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 1.5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, rgba(15,23,42,0.98), rgba(15,23,42,0.96)) !important;
    border-right: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 0 40px rgba(15,23,42,0.9);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6366f1, #a855f7);
    border-radius: 999px;
}

/* ------------------ TITLES & TEXT ------------------ */

h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

h1 {
    font-size: 2.1rem !important;
    background: linear-gradient(90deg, #a855f7, #22d3ee);
    -webkit-background-clip: text;
    color: transparent !important;
}

h2 {
    color: #e5e7eb !important;
    margin-top: 1.5rem !important;
}

h3 {
    color: #cbd5f5 !important;
}

/* Sub headings / labels */
[data-testid="stMarkdown"] p {
    color: #e5e7eb;
}

/* ------------------ GLASS CARDS ------------------ */

/* Generic glass container for cards, expanders, etc. */
.block-container, .stTabs, .stDataFrame, .element-container {
    transition: all 300ms ease;
}

/* Streamlit metric cards */
[data-testid="stMetric"] {
    background: radial-gradient(circle at top left,
        rgba(148,163,253,0.12),
        rgba(15,23,42,0.96)
    );
    border-radius: 18px;
    padding: 1.2rem 1rem;
    margin: 0.4rem;
    border: 1px solid rgba(148,163,253,0.45);
    box-shadow:
        0 18px 45px rgba(15,23,42,0.95),
        0 0 0 0 rgba(129,140,248,0.0);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow:
        0 24px 65px rgba(15,23,42,0.95),
        0 0 0 1px rgba(129,140,248,0.25);
    border-color: rgba(129,140,248,0.9);
}

/* Metric label & value */
[data-testid="stMetric"] > div:nth-child(1) {
    color: #9ca3af !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

[data-testid="stMetric"] > div:nth-child(2) {
    color: #e5e7eb !important;
    font-size: 1.4rem;
    font-weight: 700;
}

/* ------------------ CUSTOM RECOMMENDATION CARDS ------------------ */

.custom-card {
    background: linear-gradient(135deg,
        rgba(15,23,42,0.95),
        rgba(15,23,42,0.90)
    );
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(148,163,253,0.45);
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    color: #e5e7eb;
    font-size: 0.95rem;
    position: relative;
    overflow: hidden;
    transition: transform 230ms ease, box-shadow 230ms ease, border-color 230ms ease;
}

/* Animated gradient border glow using pseudo-element */
.custom-card::before {
    content: "";
    position: absolute;
    inset: -40%;
    background: conic-gradient(
        from 90deg,
        #6366f1,
        #22d3ee,
        #a855f7,
        #6366f1
    );
    opacity: 0;
    z-index: -1;
    animation: spin-border 10s linear infinite;
}

/* Inner mask to create subtle border glow */
.custom-card::after {
    content: "";
    position: absolute;
    inset: 1px;
    background: radial-gradient(circle at top left,
        rgba(15,23,42,0.96),
        rgba(15,23,42,0.98)
    );
    border-radius: inherit;
    z-index: -1;
}

.custom-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 60px rgba(15,23,42,0.98);
    border-color: rgba(129,140,248,0.9);
}

.custom-card:hover::before {
    opacity: 0.9;
}

/* ------------------ EXPANDERS & DATA TABLES ------------------ */

details {
    border-radius: 18px !important;
    border: 1px solid rgba(148,163,253,0.45) !important;
    background: linear-gradient(145deg,
        rgba(15,23,42,0.97),
        rgba(15,23,42,0.93)
    ) !important;
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

details > summary {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* Dataframe container */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148,163,253,0.45);
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
}

/* ------------------ INPUTS & WIDGETS ------------------ */

input, textarea, select {
    background: rgba(15,23,42,0.95) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,253,0.45) !important;
    color: #e5e7eb !important;
    box-shadow: 0 0 0 0 rgba(129,140,248,0.0);
    transition: border-color 200ms ease, box-shadow 200ms ease, transform 150ms ease;
}

input:focus, textarea:focus, select:focus {
    outline: none !important;
    border-color: rgba(129,140,248,0.95) !important;
    box-shadow: 0 0 0 1px rgba(129,140,248,0.5);
    transform: translateY(-1px);
}

/* File uploader */
[data-testid="stFileUploader"] section {
    border-radius: 16px !important;
    border: 1px dashed rgba(148,163,253,0.7) !important;
    background: radial-gradient(circle at top left,
        rgba(30,64,175,0.3),
        rgba(15,23,42,0.95)
    ) !important;
}

/* Checkboxes & sliders */
[data-baseweb="checkbox"] > div {
    background: rgba(15,23,42,0.95) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(148,163,253,0.5) !important;
}

/* Slider track */
[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
}

/* ------------------ BUTTONS ------------------ */

button[kind="primary"], .stButton button {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: #f9fafb !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 0.55rem 1.3rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    box-shadow: 0 14px 35px rgba(79,70,229,0.55);
    transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
}

button[kind="primary"]:hover, .stButton button:hover {
    transform: translateY(-1.5px) scale(1.02);
    box-shadow: 0 20px 45px rgba(79,70,229,0.85);
    filter: brightness(1.05);
}

button[kind="primary"]:active, .stButton button:active {
    transform: translateY(0px) scale(0.99);
    box-shadow: 0 10px 25px rgba(79,70,229,0.6);
}

/* ------------------ CHART CONTAINERS ------------------ */

[data-testid="stPlotlyChart"] {
    background: radial-gradient(circle at top left,
        rgba(15,23,42,0.98),
        rgba(15,23,42,0.94)
    );
    border-radius: 20px;
    padding: 0.75rem;
    border: 1px solid rgba(148,163,253,0.5);
    box-shadow: 0 20px 55px rgba(15,23,42,0.98);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: transform 230ms ease, box-shadow 230ms ease, border-color 230ms ease;
}

[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 26px 75px rgba(15,23,42,1);
    border-color: rgba(129,140,248,0.9);
}

/* ------------------ ANIMATIONS ------------------ */

/* Subtle page fade-in */
@keyframes fade-in-up {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Animated gradient border around custom cards */
@keyframes spin-border {
    to {
        transform: rotate(360deg);
    }
}

/* Apply fade-in to main content blocks */
.block-container > * {
    animation: fade-in-up 420ms ease-out;
}

/* ------------------ BADGES / ALERTS ------------------ */

/* Success / warning messages */
div.stAlert {
    border-radius: 16px !important;
    border-width: 1px !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* ------------------ MOBILE TWEAKS ------------------ */

@media (max-width: 768px) {
    [data-testid="stAppViewContainer"] > .main {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 1.6rem !important;
    }

    [data-testid="stMetric"] {
        margin: 0.25rem 0;
    }
}

</style>
""", unsafe_allow_html=True)



# -----------------------------
# Helper Functions
# -----------------------------
DATE_CANDIDATES = ["date","transaction_date","trans_date","timestamp","Date","DATE"]
AMOUNT_CANDIDATES = ["amount","debit","expense","amt","spend","Amount","AMOUNT"]
CATEGORY_CANDIDATES = ["category","cat","type","Category","CATEGORY","merchant_category"]

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def parse_date_safe(x):
    try: return parse(str(x))
    except: return pd.NaT

def load_and_clean(df_raw):
    date_col = find_col(df_raw.columns, DATE_CANDIDATES)
    amount_col = find_col(df_raw.columns, AMOUNT_CANDIDATES)
    category_col = find_col(df_raw.columns, CATEGORY_CANDIDATES)

    if not date_col or not amount_col:
        raise ValueError("CSV must have at least a date and an amount column.")

    df = df_raw.copy()
    df.rename(columns={date_col:"date", amount_col:"amount"}, inplace=True)
    if category_col and category_col != "category":
        df.rename(columns={category_col:"category"}, inplace=True)

    df["date"] = df["date"].apply(parse_date_safe)
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.tz_localize(None)

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    df["category"] = df["category"].fillna("Uncategorized").astype(str).str.title()

    if "type" in df.columns:
        expense_mask = df["type"].astype(str).str.lower().str.contains("expense")
        income_mask = df["type"].astype(str).str.lower().str.contains("income")
        df.loc[expense_mask & (df["amount"] > 0), "amount"] = -df.loc[expense_mask & (df["amount"] > 0), "amount"]
        df.loc[income_mask & (df["amount"] < 0), "amount"] = -df.loc[income_mask & (df["amount"] < 0), "amount"]
    else:
        if (df["amount"] >= 0).all():
            df["amount"] = -df["amount"]

    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["day_of_week"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["week"] = df["date"].dt.isocalendar().week
    return df

def monthly_expense_series(df):
    expenses_only = df[df["amount"] < 0]
    monthly = expenses_only.groupby("year_month")["amount"].sum()
    return (-monthly).rename("spend")

def fit_arima(y, seasonal=True, horizon=12):
    import statsmodels.api as sm
    y = y.asfreq('MS')
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = sm.tsa.ARIMA(y, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        best_model = results
                except Exception:
                    continue

    fc = best_model.get_forecast(steps=horizon)
    fc_mean = pd.Series(fc.predicted_mean, index=pd.date_range(
        y.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon, freq="MS"
    ))
    fc_mean = fc_mean.clip(lower=0)
    if len(fc_mean) > 1:
        growth_factor = 0.01
        fc_mean = fc_mean * (1 + growth_factor) ** np.arange(1, len(fc_mean) + 1)

    conf_int = fc.conf_int()
    conf_int.index = fc_mean.index
    return best_model, fc_mean, conf_int, f"Best order={best_order}, AIC={best_aic:.2f}"

def budget_recommendations(df, monthly_spend, forecast, target_savings_rate=0.2):
    out = []
    recent_avg = monthly_spend.tail(3).mean() if len(monthly_spend)>=3 else monthly_spend.mean()
    next_month = forecast.iloc[0] if forecast is not None and len(forecast)>0 else np.nan

    out.append(f" **Recent 3-month average spend:** ₹{recent_avg:,.0f}")

    if not np.isnan(next_month):
        out.append(f" **Forecast next month:** ₹{next_month:,.0f}")
        change_pct = ((next_month - recent_avg)/recent_avg)*100
        if change_pct>5:
            out.append(f" **Alert:** Spending expected to increase by {change_pct:.1f}%")
        elif change_pct<-5:
            out.append(f" **Good news:** Spending expected to decrease by {abs(change_pct):.1f}%")

    income_data = df[df["amount"]>0]
    if len(income_data)>0:
        monthly_income = income_data.groupby("year_month")["amount"].sum().mean()
        out.append(f" **Average monthly income:** ₹{monthly_income:,.0f}")
        current_savings_rate = (monthly_income - recent_avg)/monthly_income
        out.append(f" **Current savings rate:** {current_savings_rate*100:.1f}%")

        baseline = next_month if not np.isnan(next_month) else recent_avg
        target_spend = monthly_income*(1-target_savings_rate)
        out.append(f" **To save {int(target_savings_rate*100)}%, limit monthly spend to:** ₹{target_spend:,.0f}")

        if baseline>target_spend:
            reduction_needed = baseline-target_spend
            out.append(f" **Reduction needed:** ₹{reduction_needed:,.0f}")

    recent_months = df[df["year_month"] >= df["year_month"].max() - pd.DateOffset(months=2)]
    expense_data = recent_months[recent_months["amount"] < 0]
    if len(expense_data) > 0:
        cat_spend = (-expense_data.groupby("category")["amount"].sum()).sort_values(ascending=False)
        out.append("\\n**🏆 Top spending categories (last 3 months):**")
        for i, (cat, amt) in enumerate(cat_spend.head(5).items()):
            pct = (amt / cat_spend.sum())*100
            emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            out.append(f"{emoji} **{cat}:** ₹{amt:,.0f} ({pct:.1f}%)")

    return out

# -----------------------------
# Streamlit App
# -----------------------------

st.markdown("#  Personal Finance Assistant")
st.markdown("###  Upload your transaction data and get smart insights on spending, forecasts, and budget recommendations")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    seasonal = st.checkbox("📅 Use Seasonal ARIMA", value=False)
    horizon = st.slider("🔮 Forecast Horizon (months)", 1, 12, 3)
    target_savings_rate = st.slider("🎯 Target Savings Rate", 0.0, 0.6, 0.2, step=0.05)

    st.markdown("---")
    st.markdown("### 📖 How to use:")
    st.markdown("""
    1. Upload a CSV file with transaction data
    2. Ensure it has columns for date, amount, and optionally category
    3. View insights and forecasts automatically
    4. Adjust settings in the sidebar as needed
    """)

# File upload
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "📁 Upload Your Transaction CSV",
        type=["csv"]
    )
    sample_checkbox = st.checkbox(
        "📋 Use Sample Data   ",
        value=False
    )

with col2:
    st.markdown("### ℹ️ Instructions")
    st.write(
        """
        • CSV must contain **date** and **amount**  
        • Category column is optional  
        • Negative amount = expense  
        • Positive amount = income
        """
    )


if uploaded or sample_checkbox:
    if sample_checkbox and not uploaded:
        df_raw = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=150, freq="D"),
            "amount": np.random.normal(-500, 300, 150),
            "category": np.random.choice(["Groceries","Rent","Transport","Dining","Utilities"], 150),
            "type": ["Expense"]*150
        })
    else:
        df_raw = pd.read_csv(uploaded)

    try:
        df = load_and_clean(df_raw)
    except Exception as e:
        st.error(f" **Error loading CSV:** {e}")
        st.info(" Make sure your CSV has at least 'date' and 'amount' columns")
        st.stop()

    st.success(f"✅ **Successfully loaded {len(df)} transactions** from {df['date'].min().date()} to {df['date'].max().date()}")

    with st.expander(" View Transaction Data", expanded=False):
        st.dataframe(df.style.format({"amount": "₹{:,.2f}"}), use_container_width=True)

    st.markdown("---")

    # Date filter
    st.markdown("###  Filter by Date Range")
    col1, col2 = st.columns(2)
    with col1:
        fmin = st.date_input("Start Date", df["date"].min().date())
    with col2:
        fmax = st.date_input("End Date", df["date"].max().date())

    fmin, fmax = pd.to_datetime(fmin), pd.to_datetime(fmax)
    df_filtered = df[(df["date"]>=fmin)&(df["date"]<=fmax)]

    if len(df_filtered)==0:
        st.warning(" No data available in the selected date range.")
        st.stop()

    st.markdown("---")

    # KPI Metrics
    st.markdown("##  Financial Overview")
    total_spend = (-df_filtered[df_filtered["amount"]<0]["amount"]).sum()
    total_income = df_filtered[df_filtered["amount"]>0]["amount"].sum()
    net_cashflow = total_income - total_spend

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(" Total Spend", f"₹{total_spend:,.0f}",
             delta=f"{len(df_filtered[df_filtered['amount']<0])} transactions",
             delta_color="inverse")
    k2.metric(" Total Income", f"₹{total_income:,.0f}",
             delta=f"{len(df_filtered[df_filtered['amount']>0])} transactions")
    k3.metric(" Net Cashflow", f"₹{net_cashflow:,.0f}",
             delta="Positive" if net_cashflow > 0 else "Negative",
             delta_color="normal" if net_cashflow > 0 else "inverse")

    savings_rate = (net_cashflow / total_income * 100) if total_income > 0 else 0
    k4.metric(" Savings Rate", f"{savings_rate:.1f}%",
             delta="Good" if savings_rate >= 20 else "Improve",
             delta_color="normal" if savings_rate >= 20 else "inverse")

    st.markdown("---")

    # PRIMARY CHARTS SECTION
    st.markdown("##  Spending Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("###  Monthly Spending Trend")
        monthly = monthly_expense_series(df_filtered)
        if len(monthly)>0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly.index,
                y=monthly.values,
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2'),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)',
                name='Monthly Spend'
            ))
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for monthly trend analysis")

    with chart_col2:
        st.markdown("###  Spending by Category")
        expense_data = df_filtered[df_filtered["amount"]<0]
        if len(expense_data)>0:
            cat_spend = (-expense_data.groupby("category")["amount"].sum()).sort_values(ascending=False)
            fig = px.pie(
                cat_spend.reset_index(),
                names="category",
                values="amount",
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ADDITIONAL ANALYSIS CHARTS
    st.markdown("##  Advanced Analytics")

    # Row 1: Daily Average & Category Bars
    acol1, acol2 = st.columns(2)

    with acol1:
        st.markdown("###  Daily Average Spending")
        expense_data = df_filtered[df_filtered["amount"]<0]
        if len(expense_data)>0:
            daily_avg = (-expense_data.groupby(expense_data["date"].dt.date)["amount"].sum())
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_avg.index,
                y=daily_avg.values,
                marker_color='#764ba2',
                name='Daily Spend'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with acol2:
        st.markdown("###  Category Comparison (Bar)")
        if len(expense_data)>0:
            cat_spend = (-expense_data.groupby("category")["amount"].sum()).sort_values(ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cat_spend.values,
                y=cat_spend.index,
                orientation='h',
                marker=dict(
                    color=cat_spend.values,
                    colorscale='Purples',
                    showscale=False
                ),
                text=cat_spend.values,
                texttemplate='₹%{text:,.0f}',
                textposition='auto'
            ))
            fig.update_layout(
                xaxis_title="Amount (₹)",
                yaxis_title="Category",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Row 2: Day of Week & Income vs Expense
    acol3, acol4 = st.columns(2)

    with acol3:
        st.markdown("###  Spending by Day of Week")
        expense_data = df_filtered[df_filtered["amount"]<0]
        if len(expense_data)>0:
            dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dow_spend = (-expense_data.groupby("day_of_week")["amount"].sum()).reindex(dow_order, fill_value=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dow_spend.index,
                y=dow_spend.values,
                marker_color=['#667eea', '#764ba2', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe'],
                text=dow_spend.values,
                texttemplate='₹%{text:,.0f}',
                textposition='auto'
            ))
            fig.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Amount (₹)",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with acol4:
        st.markdown("###  Income vs Expense (Monthly)")
        monthly_income = df_filtered[df_filtered["amount"]>0].groupby("year_month")["amount"].sum()
        monthly_expense = (-df_filtered[df_filtered["amount"]<0].groupby("year_month")["amount"].sum())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_income.index,
            y=monthly_income.values,
            name='Income',
            marker_color='#10b981'
        ))
        fig.add_trace(go.Bar(
            x=monthly_expense.index,
            y=monthly_expense.values,
            name='Expense',
            marker_color='#ef4444'
        ))
        fig.update_layout(
            barmode='group',
            xaxis_title="Month",
            yaxis_title="Amount (₹)",
            template='plotly_white',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Cumulative Spending & Transaction Distribution
    acol5, acol6 = st.columns(2)

    with acol5:
        st.markdown("###  Cumulative Spending Over Time")
        expense_data = df_filtered[df_filtered["amount"]<0].copy()
        if len(expense_data)>0:
            expense_data = expense_data.sort_values("date")
            expense_data["cumulative"] = (-expense_data["amount"]).cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=expense_data["date"],
                y=expense_data["cumulative"],
                mode='lines',
                fill='tozeroy',
                line=dict(color='#667eea', width=3),
                fillcolor='rgba(102, 126, 234, 0.2)',
                name='Cumulative Spend'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Cumulative Amount (₹)",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with acol6:
        st.markdown("###  Transaction Size Distribution")
        expense_data = df_filtered[df_filtered["amount"]<0]
        if len(expense_data)>0:
            amounts = -expense_data["amount"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=amounts,
                nbinsx=30,
                marker_color='#764ba2',
                name='Transactions'
            ))
            fig.update_layout(
                xaxis_title="Transaction Amount (₹)",
                yaxis_title="Frequency",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ARIMA Forecast
    st.markdown("##  Expense Forecast (ARIMA)")

    if len(monthly)>=3:
        with st.spinner(' Building forecast model...'):
            model, fc_mean, conf, note = fit_arima(monthly, seasonal=seasonal, horizon=horizon)

        st.info(f"ℹ Model Info: {note}")

        if model is not None and fc_mean is not None:
            fc_df = pd.DataFrame({
                "date":fc_mean.index,
                "forecast":fc_mean.values,
                "lower":conf.iloc[:,0].values,
                "upper":conf.iloc[:,1].values
            })

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=monthly.index,
                y=monthly.values,
                mode="lines+markers",
                name="Historical",
                line=dict(color="#667eea", width=3),
                marker=dict(size=8)
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["date"],
                y=fc_df["forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#ff6b6b", width=3, dash='dash'),
                marker=dict(size=8)
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["date"],
                y=fc_df["upper"],
                mode="lines",
                name="Upper 95% CI",
                line=dict(color="rgba(102, 126, 234, 0.3)", width=0),
                showlegend=False
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_df["date"],
                y=fc_df["lower"],
                mode="lines",
                name="Lower 95% CI",
                line=dict(color="rgba(102, 126, 234, 0.3)", width=0),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.2)',
                showlegend=True
            ))
            fig_fc.update_layout(
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            with st.expander(" View Detailed Forecast"):
                forecast_table = fc_df.copy()
                forecast_table['date'] = forecast_table['date'].dt.strftime('%B %Y')
                forecast_table = forecast_table.rename(columns={
                    'date': 'Month',
                    'forecast': 'Predicted Spend (₹)',
                    'lower': 'Lower Bound (₹)',
                    'upper': 'Upper Bound (₹)'
                })
                st.dataframe(
                    forecast_table.style.format({
                        'Predicted Spend (₹)': '₹{:,.0f}',
                        'Lower Bound (₹)': '₹{:,.0f}',
                        'Upper Bound (₹)': '₹{:,.0f}'
                    }),
                    use_container_width=True
                )

            st.markdown("---")

            # Budget recommendations
            st.markdown("##  Smart Budget Recommendations")
            recs = budget_recommendations(df_filtered, monthly, fc_mean, target_savings_rate)

            for r in recs:
                st.markdown(f"<div class='custom-card'>{r}</div>", unsafe_allow_html=True)

    else:
        st.warning(" Need at least 3 months of data for forecasting. Please upload more transaction history.")

else:

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ###  Smart Analytics
        - Automatic category detection
        - Monthly spending trends
        - Income vs expense tracking
        """)

    with col2:
        st.markdown("""
        ###  AI Forecasting
        - ARIMA time series models
        - Confidence intervals
        - Multi-month predictions
        """)

    with col3:
        st.markdown("""
        ###  Actionable Insights
        - Savings rate calculation
        - Budget recommendations
        - Spending alerts
        """)


print("✅ ENHANCED version created: enhanced_finance_app.py")
print("\n" + "="*70)
print(" NEW GRAPHS ADDED (Total: 8 visualizations)")
print("="*70)
print("\n1.  Monthly Spending Trend (Area Chart)")
print("2.  Spending by Category (Pie Chart)")
print("3.  Daily Average Spending (Bar Chart) - NEW!")
print("4.  Category Comparison (Horizontal Bar) - NEW!")
print("5.  Spending by Day of Week (Colored Bars) - NEW!")
print("6.  Income vs Expense Monthly (Grouped Bars) - NEW!")
print("7.  Cumulative Spending Over Time (Area) - NEW!")
print("8.  Transaction Size Distribution (Histogram) - NEW!")
print("9.  ARIMA Forecast with Confidence Intervals")
print("\n" + "="*70)
print("✨ FEATURES ADDED:")
print("="*70)
print("- Day of week analysis")
print("- Daily spending patterns")
print("- Income vs Expense comparison")
print("- Cumulative spending tracking")
print("- Transaction distribution analysis")
print("- Enhanced category visualization")
print("\nCopy the content from 'enhanced_finance_app.py' to your Colab notebook!")
