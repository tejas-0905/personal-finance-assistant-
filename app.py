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

# Import our modules
from styles import load_css
from data_processing import load_and_clean, monthly_expense_series
from analytics import calculate_kpis, create_category_insights
from forecasting import fit_arima
from recommendations import budget_recommendations
from charts import (
    monthly_trend_chart, category_pie_chart, daily_average_chart,
    category_bar_chart, day_of_week_chart, income_vs_expense_chart,
    cumulative_spending_chart, transaction_distribution_chart, forecast_chart
)
from pdf_report import generate_pdf_report

# -----------------------------
# Helper Functions
# -----------------------------
def display_recommendation_section(title, recommendations):
    """Display a recommendation section with professional formatting"""
    # Section header with icon
    icon_map = {
        "Spending Alerts": "🚨",
        "Savings Opportunities": "💰",
        "Category Insights": "📊",
        "Action Items": "✅",
        "Financial Goals": "🎯"
    }

    

    # Display each recommendation in a card
    for rec in recommendations:
        if rec.strip():  # Skip empty lines
            # Determine priority level from content
            if "HIGH PRIORITY" in rec.upper():
                card_class = "priority-high"
                priority_icon = "🔴"
            elif "MEDIUM PRIORITY" in rec.upper():
                card_class = "priority-medium"
                priority_icon = "🟡"
            elif "LOW PRIORITY" in rec.upper():
                card_class = "priority-low"
                priority_icon = "🟢"
            else:
                card_class = "custom-card"
                priority_icon = ""

            # Format the recommendation text
            formatted_rec = rec.replace("**", "").replace("*", "")  # Remove markdown formatting
            if priority_icon:
                formatted_rec = f"{priority_icon} {formatted_rec}"

            st.markdown(f"<div class='{card_class}'>{formatted_rec}</div>", unsafe_allow_html=True)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="MoneyMentor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Custom CSS
# -----------------------------
st.markdown(load_css(), unsafe_allow_html=True)

# -----------------------------
# Streamlit App
# -----------------------------
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
    kpis = calculate_kpis(df_filtered)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(" Total Spend", f"₹{kpis['total_spend']:,.0f}",
             delta=f"{len(df_filtered[df_filtered['amount']<0])} transactions",
             delta_color="inverse")
    k2.metric(" Total Income", f"₹{kpis['total_income']:,.0f}",
             delta=f"{len(df_filtered[df_filtered['amount']>0])} transactions")
    k3.metric(" Net Cashflow", f"₹{kpis['net_cashflow']:,.0f}",
             delta="Positive" if kpis['net_cashflow'] > 0 else "Negative",
             delta_color="normal" if kpis['net_cashflow'] > 0 else "inverse")

    k4.metric(" Savings Rate", f"{kpis['savings_rate']:.1f}%",
             delta="Good" if kpis['savings_rate'] >= 20 else "Improve",
             delta_color="normal" if kpis['savings_rate'] >= 20 else "inverse")

    st.markdown("---")

    # PRIMARY CHARTS SECTION
    charts_for_pdf = []
    st.markdown("##  Spending Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("###  Monthly Spending Trend")
        monthly = monthly_expense_series(df_filtered)
        if len(monthly)>0:
            fig = monthly_trend_chart(monthly)
            st.plotly_chart(fig, use_container_width=True)
            charts_for_pdf.append(("Monthly Spending Trend", fig))
        else:
            st.info("Not enough data for monthly trend analysis")

    with chart_col2:
        st.markdown("###  Spending by Category")
        fig = category_pie_chart(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            charts_for_pdf.append(("Spending by Category", fig))

    st.markdown("---")

    # ADDITIONAL ANALYSIS CHARTS
    st.markdown("##  Advanced Analytics")

    # Row 1: Daily Average & Category Bars
    acol1, acol2 = st.columns(2)

    with acol1:
        st.markdown("###  Daily Average Spending")
        fig = daily_average_chart(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            charts_for_pdf.append(("Daily Average Spending", fig))

    with acol2:
        st.markdown("###  Category Comparison (Bar)")
        fig = category_bar_chart(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            charts_for_pdf.append(("Category Comparison", fig))

    # Row 2: Day of Week & Income vs Expense
    acol3, acol4 = st.columns(2)

    with acol3:
        st.markdown("###  Spending by Day of Week")
        fig = day_of_week_chart(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            charts_for_pdf.append(("Spending by Day of Week", fig))

    with acol4:
        st.markdown("###  Income vs Expense (Monthly)")
        fig = income_vs_expense_chart(df_filtered)
        st.plotly_chart(fig, use_container_width=True)
        charts_for_pdf.append(("Income vs Expense", fig))

    # Row 3: Cumulative Spending & Transaction Distribution
    acol5, acol6 = st.columns(2)

    with acol5:
        st.markdown("###  Cumulative Spending Over Time")
        fig = cumulative_spending_chart(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            charts_for_pdf.append(("Cumulative Spending", fig))

    with acol6:
        st.markdown("###  Transaction Size Distribution")
        fig = transaction_distribution_chart(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            charts_for_pdf.append(("Transaction Distribution", fig))

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

            fig_fc = forecast_chart(monthly, fc_df)
            st.plotly_chart(fig_fc, use_container_width=True)
            forecast_fig = fig_fc

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
            st.markdown("## 🎯 Smart Budget Recommendations")

            recs = budget_recommendations(df_filtered, monthly, fc_mean, target_savings_rate)
            recommendations_for_pdf = recs

            # Group recommendations by sections
            current_section = None
            section_content = []

            for rec in recs:
                if rec.startswith("## "):  # New section
                    if current_section and section_content:
                        # Display previous section
                        display_recommendation_section(current_section, section_content)
                    current_section = rec.replace("## ", "")
                    section_content = []
                elif rec.strip() == "":  # Empty line (section separator)
                    if current_section and section_content:
                        display_recommendation_section(current_section, section_content)
                        current_section = None
                        section_content = []
                else:
                    section_content.append(rec)

            # Display final section if any
            if current_section and section_content:
                display_recommendation_section(current_section, section_content)

    else:
        st.warning(" Need at least 3 months of data for forecasting. Please upload more transaction history.")
        
    st.markdown("---")
    st.markdown("## 📄 Download Full Report")


    pdf_data = {
    "total_spend": kpis['total_spend'],
    "total_income": kpis['total_income'],
    "net_cashflow": kpis['net_cashflow'],
    "savings_rate": kpis['savings_rate'],
    "charts": charts_for_pdf,
    "forecast_fig": forecast_fig if 'forecast_fig' in locals() else None,
    "recommendations": recommendations_for_pdf if 'recommendations_for_pdf' in locals() else []
}

    pdf_file = generate_pdf_report(pdf_data)

    st.download_button(
    label="📥 Download Report",
    data=pdf_file,
    file_name="Finance_Report.pdf",
    mime="application/pdf"
)

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
