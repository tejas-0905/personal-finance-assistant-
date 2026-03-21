import numpy as np
import pandas as pd
from dateutil.parser import parse

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
    try:
        return parse(str(x))
    except:
        return pd.NaT


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