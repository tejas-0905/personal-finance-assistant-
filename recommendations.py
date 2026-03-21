import numpy as np
import pandas as pd


def budget_recommendations(df, monthly_spend, forecast, target_savings_rate=0.2):
    """
    Generate comprehensive, professional budget recommendations with actionable insights
    """
    # Month name mapping
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }

    recommendations = {
        "executive_summary": [],
        "spending_analysis": [],
        "forecast_insights": [],
        "savings_assessment": [],
        "actionable_recommendations": [],
        "risk_alerts": [],
        "goal_based_targets": []
    }

    # Calculate key metrics
    recent_avg = monthly_spend.tail(3).mean() if len(monthly_spend) >= 3 else monthly_spend.mean()
    next_month = forecast.iloc[0] if forecast is not None and len(forecast) > 0 else np.nan

    # Executive Summary
    recommendations["executive_summary"].append("## 📊 Executive Summary")
    recommendations["executive_summary"].append(f"**Current Monthly Spending:** ₹{recent_avg:,.0f}")

    if not np.isnan(next_month):
        change_pct = ((next_month - recent_avg) / recent_avg) * 100
        trend_icon = "📈" if change_pct > 0 else "📉"
        recommendations["executive_summary"].append(f"**Next Month Forecast:** ₹{next_month:,.0f} {trend_icon} ({change_pct:+.1f}%)")

    # Income Analysis
    income_data = df[df["amount"] > 0]
    if len(income_data) > 0:
        monthly_income = income_data.groupby("year_month")["amount"].sum().mean()
        recommendations["executive_summary"].append(f"**Average Monthly Income:** ₹{monthly_income:,.0f}")

        current_savings_rate = (monthly_income - recent_avg) / monthly_income
        recommendations["executive_summary"].append(f"**Current Savings Rate:** {current_savings_rate*100:.1f}%")

    # Spending Analysis
    recommendations["spending_analysis"].append("## 💰 Spending Analysis")

    # Seasonal patterns
    if len(monthly_spend) >= 6:
        seasonal_avg = monthly_spend.groupby(monthly_spend.index.month).mean()
        peak_month = seasonal_avg.idxmax()
        low_month = seasonal_avg.idxmin()
        recommendations["spending_analysis"].append(f"**Seasonal Pattern:** Highest spending in {month_names[peak_month]} (₹{seasonal_avg[peak_month]:,.0f}), lowest in {month_names[low_month]} (₹{seasonal_avg[low_month]:,.0f})")

    # Spending volatility
    if len(monthly_spend) >= 3:
        volatility = monthly_spend.std() / monthly_spend.mean()
        if volatility > 0.3:
            recommendations["spending_analysis"].append("**⚠️ High Volatility:** Your spending varies significantly month-to-month. Consider building an emergency buffer.")
        elif volatility < 0.1:
            recommendations["spending_analysis"].append("**✅ Stable Spending:** Your monthly expenses are consistent and predictable.")

    # Forecast Insights
    if not np.isnan(next_month):
        recommendations["forecast_insights"].append("## 🔮 Forecast Insights")

        change_pct = ((next_month - recent_avg) / recent_avg) * 100

        if change_pct > 10:
            recommendations["forecast_insights"].append(f"**🚨 Critical Alert:** Expected spending surge of {change_pct:.1f}%. Immediate budget review recommended.")
        elif change_pct > 5:
            recommendations["forecast_insights"].append(f"**⚠️ Warning:** Spending projected to increase by {change_pct:.1f}%. Monitor closely.")
        elif change_pct < -5:
            recommendations["forecast_insights"].append(f"**✅ Positive Trend:** Spending expected to decrease by {abs(change_pct):.1f}%. Great job on cost control!")
        else:
            recommendations["forecast_insights"].append("**📊 Stable:** Spending forecast is within normal range.")

    # Savings Assessment
    if len(income_data) > 0:
        recommendations["savings_assessment"].append("## 🎯 Savings Assessment")

        current_savings_rate = (monthly_income - recent_avg) / monthly_income
        target_savings = monthly_income * target_savings_rate
        baseline = next_month if not np.isnan(next_month) else recent_avg

        recommendations["savings_assessment"].append(f"**Target Savings Rate:** {target_savings_rate*100:.0f}% (₹{target_savings:,.0f}/month)")
        recommendations["savings_assessment"].append(f"**Current Performance:** {current_savings_rate*100:.1f}%")

        if current_savings_rate >= target_savings_rate:
            recommendations["savings_assessment"].append("**✅ On Track:** You're meeting or exceeding your savings goal!")
        else:
            gap = target_savings - (monthly_income - baseline)
            recommendations["savings_assessment"].append(f"**📈 Gap to Target:** ₹{gap:,.0f}/month")

            # Emergency fund assessment
            emergency_fund_months = 6  # Recommended 6 months of expenses
            emergency_fund_target = recent_avg * emergency_fund_months
            recommendations["savings_assessment"].append(f"**Emergency Fund Target:** ₹{emergency_fund_target:,.0f} ({emergency_fund_months} months of expenses)")

    # Actionable Recommendations
    recommendations["actionable_recommendations"].append("## 🎯 Actionable Recommendations")

    # Category-based recommendations
    recent_months = df[df["year_month"] >= df["year_month"].max() - pd.DateOffset(months=2)]
    expense_data = recent_months[recent_months["amount"] < 0]

    if len(expense_data) > 0:
        cat_spend = (-expense_data.groupby("category")["amount"].sum()).sort_values(ascending=False)
        total_recent_spend = cat_spend.sum()

        # Top spending categories
        top_categories = cat_spend.head(3)
        recommendations["actionable_recommendations"].append("**Top Spending Categories (Last 3 Months):**")
        for cat, amt in top_categories.items():
            pct = (amt / total_recent_spend) * 100
            recommendations["actionable_recommendations"].append(f"• **{cat}:** ₹{amt:,.0f} ({pct:.1f}%)")

        # Optimization suggestions
        if len(income_data) > 0:
            affordable_spend = monthly_income * (1 - target_savings_rate)
            monthly_budget_per_category = affordable_spend / len(cat_spend)

            recommendations["actionable_recommendations"].append(f"\n**Budget Allocation:** With ₹{affordable_spend:,.0f} available for expenses:")
            for cat, amt in top_categories.items():
                if amt > monthly_budget_per_category:
                    excess = amt - monthly_budget_per_category
                    recommendations["actionable_recommendations"].append(f"• **{cat}** could save ₹{excess:,.0f}/month")

    # Goal-based Targets
    if len(income_data) > 0:
        recommendations["goal_based_targets"].append("## 🎯 Goal-Based Financial Targets")

        # Short-term goals (3-6 months)
        short_term_savings = monthly_income * 0.15  # 15% for short-term goals
        recommendations["goal_based_targets"].append(f"**Short-term Goals (3-6 months):** ₹{short_term_savings:,.0f}/month")

        # Medium-term goals (1-3 years)
        medium_term_savings = monthly_income * 0.20  # 20% for medium-term goals
        recommendations["goal_based_targets"].append(f"**Medium-term Goals (1-3 years):** ₹{medium_term_savings:,.0f}/month")

        # Retirement/Investment
        retirement_savings = monthly_income * 0.25  # 25% for long-term goals
        recommendations["goal_based_targets"].append(f"**Long-term/Retirement:** ₹{retirement_savings:,.0f}/month")

    # Risk Alerts
    recommendations["risk_alerts"].append("## ⚠️ Risk Assessment")

    risk_score = 0
    risk_factors = []

    # High spending relative to income
    if len(income_data) > 0:
        spend_to_income_ratio = recent_avg / monthly_income
        if spend_to_income_ratio > 0.9:
            risk_score += 3
            risk_factors.append("Critical: Spending exceeds 90% of income")
        elif spend_to_income_ratio > 0.8:
            risk_score += 2
            risk_factors.append("High: Spending exceeds 80% of income")

    # Low savings rate
    if len(income_data) > 0 and current_savings_rate < 0.1:
        risk_score += 2
        risk_factors.append("Low savings rate (<10%)")

    # High spending volatility
    if len(monthly_spend) >= 3:
        volatility = monthly_spend.std() / monthly_spend.mean()
        if volatility > 0.4:
            risk_score += 1
            risk_factors.append("High spending volatility")

    # Determine risk level
    if risk_score >= 4:
        recommendations["risk_alerts"].append("**🔴 HIGH RISK** - Immediate attention required")
    elif risk_score >= 2:
        recommendations["risk_alerts"].append("**🟡 MEDIUM RISK** - Monitor closely")
    else:
        recommendations["risk_alerts"].append("**🟢 LOW RISK** - Good financial health")

    if risk_factors:
        recommendations["risk_alerts"].append("**Key Risk Factors:**")
        for factor in risk_factors:
            recommendations["risk_alerts"].append(f"• {factor}")

    # Compile all recommendations into a flat list
    final_recommendations = []
    for section, items in recommendations.items():
        if items:  # Only add non-empty sections
            final_recommendations.extend(items)
            final_recommendations.append("")  # Add spacing between sections

    return final_recommendations