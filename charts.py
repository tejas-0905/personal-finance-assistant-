import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def monthly_trend_chart(monthly):
    """Create monthly spending trend chart"""
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
    return fig


def category_pie_chart(df):
    """Create spending by category pie chart"""
    expense_data = df[df["amount"]<0]
    if len(expense_data) == 0:
        return None

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
    return fig


def daily_average_chart(df):
    """Create daily average spending chart"""
    expense_data = df[df["amount"]<0]
    if len(expense_data) == 0:
        return None

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
    return fig


def category_bar_chart(df):
    """Create category comparison bar chart"""
    expense_data = df[df["amount"]<0]
    if len(expense_data) == 0:
        return None

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
    return fig


def day_of_week_chart(df):
    """Create spending by day of week chart"""
    expense_data = df[df["amount"]<0]
    if len(expense_data) == 0:
        return None

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
    return fig


def income_vs_expense_chart(df):
    """Create income vs expense monthly chart"""
    monthly_income = df[df["amount"]>0].groupby("year_month")["amount"].sum()
    monthly_expense = (-df[df["amount"]<0].groupby("year_month")["amount"].sum())

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
    return fig


def cumulative_spending_chart(df):
    """Create cumulative spending over time chart"""
    expense_data = df[df["amount"]<0].copy()
    if len(expense_data) == 0:
        return None

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
    return fig


def transaction_distribution_chart(df):
    """Create transaction size distribution chart"""
    expense_data = df[df["amount"]<0]
    if len(expense_data) == 0:
        return None

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
    return fig


def forecast_chart(monthly, fc_df):
    """Create ARIMA forecast chart with confidence intervals"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly.index,
        y=monthly.values,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#667eea", width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"],
        y=fc_df["forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#ff6b6b", width=3, dash='dash'),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"],
        y=fc_df["upper"],
        mode="lines",
        name="Upper 95% CI",
        line=dict(color="rgba(102, 126, 234, 0.3)", width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"],
        y=fc_df["lower"],
        mode="lines",
        name="Lower 95% CI",
        line=dict(color="rgba(102, 126, 234, 0.3)", width=0),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.2)',
        showlegend=True
    ))
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount (₹)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    return fig