import pandas as pd
from data_processing import monthly_expense_series


def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_spend = (-df[df["amount"]<0]["amount"]).sum()
    total_income = df[df["amount"]>0]["amount"].sum()
    net_cashflow = total_income - total_spend
    savings_rate = (net_cashflow / total_income * 100) if total_income > 0 else 0

    return {
        "total_spend": total_spend,
        "total_income": total_income,
        "net_cashflow": net_cashflow,
        "savings_rate": savings_rate
    }


def get_top_spending_categories(df, months=3):
    """Get top spending categories for recent months"""
    recent_months = df[df["year_month"] >= df["year_month"].max() - pd.DateOffset(months=months)]
    expense_data = recent_months[recent_months["amount"] < 0]

    if len(expense_data) == 0:
        return []

    cat_spend = (-expense_data.groupby("category")["amount"].sum()).sort_values(ascending=False)
    return cat_spend.head(5)


def create_category_insights(df):
    """Create insights about spending categories"""
    insights = []

    recent_months = df[df["year_month"] >= df["year_month"].max() - pd.DateOffset(months=2)]
    expense_data = recent_months[recent_months["amount"] < 0]

    if len(expense_data) > 0:
        cat_spend = (-expense_data.groupby("category")["amount"].sum()).sort_values(ascending=False)
        insights.append("**🏆 Top spending categories (last 3 months):**")
        for i, (cat, amt) in enumerate(cat_spend.head(5).items()):
            pct = (amt / cat_spend.sum())*100
            emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            insights.append(".1f")

    return insights