import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# =========================================================
# FRESHFLOW - SINGLE FILE STREAMLIT APPLICATION
# =========================================================

st.set_page_config(
    page_title="FreshFlow | Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "notebooks" / "data" / "processed" / "cleaned_data.csv"
MODEL_PATH = BASE_DIR / "notebooks" / "models" / "random_forest_baseline.pkl"

# ------------------------- Styling -------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0b1220; }
    [data-testid="stSidebar"] { background-color: #101a2d; }
    [data-testid="stMetric"] {
        background-color: #162238;
        border: 1px solid #2b3b55;
        border-radius: 12px;
        padding: 14px;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Data / Model -------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    for col in ["transaction_date", "expiration_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    numeric_cols = [
        "daily_demand", "units_sold", "initial_quantity", "units_wasted",
        "waste_pct", "base_price", "selling_price", "discount_pct",
        "spoilage_risk", "days_remaining_at_purchase", "shelf_life_days",
        "storage_temp", "temp_deviation", "cost_price", "spoilage_sensitivity",
        "day_of_week", "is_weekend", "month", "temp_abuse_events",
        "distribution_hours", "handling_score", "packaging_score",
        "supplier_score", "is_promoted",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["transaction_date", "daily_demand", "category"])
    data["potential_shortage"] = data["initial_quantity"] < data["daily_demand"]
    return data


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def display_category(value):
    return {"Ready_to_Eat": "Ready-to-Eat", "Frozen_Meals": "Frozen Meals"}.get(value, value)


def fmt(value):
    return f"{value:,.0f}"


def classify_expiry_risk(days_remaining):
    if days_remaining <= 3:
        return "Critical"
    if days_remaining <= 7:
        return "High"
    if days_remaining <= 14:
        return "Moderate"
    return "Low"


def combined_inventory_recommendation(predicted_demand, available_stock, days_remaining):
    shortage = max(0.0, float(predicted_demand) - float(available_stock))
    expiry_risk = classify_expiry_risk(int(days_remaining))
    if shortage > 0 and expiry_risk in ["Critical", "High"]:
        status = "Review Replenishment + Expiry"
        action = "Check existing stock and near-expiry batches before ordering."
    elif shortage > 0:
        status = "Reorder Required"
        action = "Consider replenishing the potential demand gap."
    elif expiry_risk in ["Critical", "High"]:
        status = "Expiry Risk"
        action = "Prioritize existing stock for sale and review markdown options."
    else:
        status = "Stock Sufficient"
        action = "Continue monitoring inventory."
    return {
        "Predicted Demand": round(float(predicted_demand), 2),
        "Available Stock": int(available_stock),
        "Potential Shortage": round(shortage, 2),
        "Potential Reorder Quantity": int(np.ceil(shortage)),
        "Days Remaining": int(days_remaining),
        "Expiry Risk": expiry_risk,
        "Final Status": status,
        "Recommended Action": action,
    }


def page_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)
    st.divider()


# ------------------------- Validate required files -------------------------
if not DATA_PATH.exists():
    st.error(f"Cleaned dataset not found:\n\n{DATA_PATH}")
    st.info("Check that cleaned_data.csv is in notebooks/data/processed/")
    st.stop()

df = load_data(DATA_PATH)

# ------------------------- Sidebar / Filters -------------------------
st.sidebar.title("📦 FreshFlow")
st.sidebar.caption("Predictive Supply Chain Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Historical Analytics",
        "Demand Prediction",
        "Inventory Recommendations",
        "Expiry & Waste Analysis",
    ],
)

st.sidebar.subheader("Analysis Filters")
min_date = df["transaction_date"].min().date()
max_date = df["transaction_date"].max().date()
date_range = st.sidebar.date_input(
    "Transaction period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

all_categories = sorted(df["category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Product category", all_categories, default=all_categories,
    format_func=display_category,
)
all_regions = sorted(df["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Region", all_regions, default=all_regions,
)

filtered = df[
    df["transaction_date"].dt.date.between(start_date, end_date)
    & df["category"].isin(selected_categories)
    & df["region"].isin(selected_regions)
].copy()

if filtered.empty:
    st.warning("No records match these filters. Please widen the selected period or filters.")
    st.stop()

# =========================================================
# PAGE 1 - EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":
    page_header(
        "📦 FreshFlow | Executive Overview",
        "A concise view of historical demand, sales, waste and potential inventory shortage.",
    )
    total_records = len(filtered)
    avg_demand = filtered["daily_demand"].mean()
    total_demand = filtered["daily_demand"].sum()
    total_sold = filtered["units_sold"].sum()
    total_waste = filtered["units_wasted"].sum()
    waste_pct = filtered["waste_pct"].mean()
    shortage_rate = filtered["potential_shortage"].mean() * 100

    k = st.columns(6)
    metrics = [
        ("Records", fmt(total_records)),
        ("Average Demand", f"{avg_demand:,.1f}"),
        ("Total Demand", fmt(total_demand)),
        ("Units Sold", fmt(total_sold)),
        ("Units Wasted", fmt(total_waste)),
        ("Potential Shortage", f"{shortage_rate:.2f}%"),
    ]
    for col, (label, value) in zip(k, metrics):
        col.metric(label, value)

    st.caption(
        "Potential shortage is flagged when initial_quantity is below observed daily_demand. "
        "It is a screening indicator, not a confirmed stockout."
    )

    left, right = st.columns(2)
    monthly = (
        filtered.set_index("transaction_date")["daily_demand"]
        .resample("MS").mean().reset_index(name="avg_daily_demand")
    )
    monthly["moving_avg_3m"] = monthly["avg_daily_demand"].rolling(3).mean()
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["transaction_date"], y=monthly["avg_daily_demand"],
            mode="lines+markers", name="Monthly average",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["transaction_date"], y=monthly["moving_avg_3m"],
            mode="lines", name="3-month moving average",
            line=dict(dash="dash"),
        ))
        fig.update_layout(
            title="Average Daily Demand Over Time",
            template="plotly_dark", height=380,
            xaxis_title="Month", yaxis_title="Average demand",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cat = filtered.groupby("category", as_index=False)["daily_demand"].sum()
        cat["Category"] = cat["category"].map(display_category)
        fig = px.bar(
            cat.sort_values("daily_demand"), x="daily_demand", y="Category",
            orientation="h", title="Total Observed Demand by Category",
            template="plotly_dark",
        )
        fig.update_layout(height=380, xaxis_title="Total demand", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key business signals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average waste percentage", f"{waste_pct:.1f}%")
    c2.metric("Potential shortage records", fmt(filtered["potential_shortage"].sum()))
    c3.metric("Categories represented", filtered["category"].nunique())
    st.info(
        "FreshFlow combines historical analysis, demand prediction and scenario-based "
        "inventory recommendations to support better replenishment and waste decisions."
    )

# =========================================================
# PAGE 2 - HISTORICAL ANALYTICS
# =========================================================
elif page == "Historical Analytics":
    page_header(
        "📊 Historical Analytics",
        "Explore how demand, category mix and inventory pressure vary across the selected data.",
    )
    col1, col2 = st.columns(2)
    with col1:
        monthly = (
            filtered.set_index("transaction_date")["daily_demand"]
            .resample("MS").mean().reset_index(name="Average Demand")
        )
        fig = px.line(
            monthly, x="transaction_date", y="Average Demand",
            markers=True, title="Monthly Average Daily Demand",
            template="plotly_dark",
        )
        fig.update_layout(height=390, xaxis_title="Month")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        weekday = filtered.groupby("is_weekend", as_index=False)["daily_demand"].mean()
        weekday["Day Type"] = weekday["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        fig = px.bar(
            weekday, x="Day Type", y="daily_demand",
            title="Average Demand: Weekday vs Weekend",
            template="plotly_dark", text_auto=".1f",
        )
        fig.update_layout(height=390, yaxis_title="Average daily demand")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cat = filtered.groupby("category", as_index=False)["daily_demand"].sum()
        cat["Category"] = cat["category"].map(display_category)
        fig = px.bar(
            cat.sort_values("daily_demand"), x="daily_demand", y="Category",
            orientation="h", title="Total Demand by Category",
            template="plotly_dark",
        )
        fig.update_layout(height=430, xaxis_title="Total demand", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        waste = filtered.groupby("category", as_index=False)["waste_pct"].mean()
        waste["Category"] = waste["category"].map(display_category)
        fig = px.bar(
            waste.sort_values("waste_pct"), x="waste_pct", y="Category",
            orientation="h", title="Average Waste Percentage by Category",
            template="plotly_dark", text_auto=".1f",
        )
        fig.update_layout(height=430, xaxis_title="Average waste (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    shortage = (
        filtered.groupby("category")
        .agg(records=("potential_shortage", "size"),
             shortage_records=("potential_shortage", "sum"))
        .reset_index()
    )
    shortage["Shortage Rate (%)"] = shortage["shortage_records"] / shortage["records"] * 100
    shortage["Category"] = shortage["category"].map(display_category)
    fig = px.bar(
        shortage.sort_values("Shortage Rate (%)"),
        x="Shortage Rate (%)", y="Category", orientation="h",
        title="Potential Shortage Rate by Category",
        template="plotly_dark", text_auto=".2f",
    )
    fig.update_layout(height=430, xaxis_title="Potential shortage rate (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These charts describe historical records. The potential-shortage measure is not "
        "a confirmed stockout measure."
    )

# =========================================================
# PAGE 3 - DEMAND PREDICTION
# =========================================================
elif page == "Demand Prediction":
    page_header(
        "🤖 Demand Prediction",
        "Run the trained Random Forest pipeline on a selected historical record as a model demonstration.",
    )
    if not MODEL_PATH.exists():
        st.error(f"Trained model file not found:\n\n{MODEL_PATH}")
        st.info("Check that random_forest_baseline.pkl is in notebooks/models/")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    st.warning(
        "Important: this dataset does not provide a reliable product-store daily time series. "
        "This page demonstrates the model on a selected historical record; it is not a "
        "validated next-day forecast."
    )

    if "record_id" in filtered.columns:
        record_options = filtered["record_id"].astype(str).tolist()
        chosen_id = st.selectbox("Select a historical record", record_options)
        selected_row = filtered.loc[filtered["record_id"].astype(str) == chosen_id].iloc[0]
    else:
        row_index = st.selectbox("Select row", filtered.index.tolist())
        selected_row = filtered.loc[row_index]

    feature_columns = [
        "shelf_life_days", "days_remaining_at_purchase", "storage_temp",
        "temp_deviation", "base_price", "cost_price", "initial_quantity",
        "spoilage_sensitivity", "day_of_week", "is_weekend", "month",
        "temp_abuse_events", "distribution_hours", "handling_score",
        "packaging_score", "supplier_score", "is_promoted",
        "product_name", "category", "store_id", "region",
    ]
    missing = [c for c in feature_columns if c not in filtered.columns]
    if missing:
        st.error(f"Required model features are missing from the dataset: {missing}")
        st.stop()

    input_df = pd.DataFrame([{c: selected_row[c] for c in feature_columns}])
    if st.button("Predict demand for selected record", type="primary"):
        try:
            prediction = float(model.predict(input_df)[0])
            actual = float(selected_row["daily_demand"])
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Demand", f"{prediction:,.1f}")
            p2.metric("Actual Recorded Demand", f"{actual:,.1f}")
            p3.metric("Absolute Error", f"{abs(prediction - actual):,.1f}")
            st.write("Selected record details")
            st.dataframe(
                pd.DataFrame({
                    "Field": ["Record ID", "Product", "Category", "Transaction Date"],
                    "Value": [
                        selected_row.get("record_id", "N/A"),
                        selected_row.get("product_name", "N/A"),
                        display_category(selected_row.get("category", "N/A")),
                        selected_row["transaction_date"].strftime("%d-%b-%Y"),
                    ],
                }),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# =========================================================
# PAGE 4 - INVENTORY RECOMMENDATIONS
# =========================================================
elif page == "Inventory Recommendations":
    page_header(
        "📦 Inventory Recommendations",
        "Test a replenishment scenario using predicted demand and manually supplied available stock.",
    )
    if not MODEL_PATH.exists():
        st.error(f"Trained model file not found:\n\n{MODEL_PATH}")
        st.stop()
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    st.warning(
        "The dataset's initial_quantity equals units_sold + units_wasted in every row, "
        "so it does not provide a separate closing-stock value. Enter available stock "
        "manually for this scenario; do not treat it as observed dataset stock."
    )

    if "record_id" in filtered.columns:
        options = filtered["record_id"].astype(str).tolist()
        chosen_id = st.selectbox("Choose a historical product record", options, key="inventory_record")
        row = filtered.loc[filtered["record_id"].astype(str) == chosen_id].iloc[0]
    else:
        row = filtered.iloc[0]

    feature_columns = [
        "shelf_life_days", "days_remaining_at_purchase", "storage_temp",
        "temp_deviation", "base_price", "cost_price", "initial_quantity",
        "spoilage_sensitivity", "day_of_week", "is_weekend", "month",
        "temp_abuse_events", "distribution_hours", "handling_score",
        "packaging_score", "supplier_score", "is_promoted",
        "product_name", "category", "store_id", "region",
    ]
    input_df = pd.DataFrame([{c: row[c] for c in feature_columns}])
    try:
        predicted = float(model.predict(input_df)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Product", str(row.get("product_name", "Selected product")))
    b.metric("Predicted Demand", f"{predicted:,.1f}")
    c.metric("Days Remaining", f"{int(row['days_remaining_at_purchase'])}")

    st.markdown("### Enter scenario values")
    left, right = st.columns(2)
    with left:
        available_stock = st.number_input(
            "Available stock (units)", min_value=0, value=60, step=1
        )
    with right:
        days_remaining = st.number_input(
            "Days remaining until expiry", min_value=0,
            max_value=1000,
            value=int(max(0, row["days_remaining_at_purchase"])),
            step=1,
        )

    result = combined_inventory_recommendation(predicted, available_stock, days_remaining)
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Potential Shortage", f"{result['Potential Shortage']:,.1f}")
    r2.metric("Potential Reorder Quantity", f"{result['Potential Reorder Quantity']:,}")
    r3.metric("Expiry Risk", result["Expiry Risk"])
    r4.metric("Recommendation", result["Final Status"])
    if result["Potential Shortage"] > 0:
        st.warning(result["Recommended Action"])
    else:
        st.success(result["Recommended Action"])
    st.caption(
        "This is a planning scenario, not an automatic purchase order. The simple rule "
        "compares predicted demand with the stock value entered above and does not yet "
        "account for supplier lead time or a calibrated safety-stock policy."
    )

# =========================================================
# PAGE 5 - EXPIRY & WASTE ANALYSIS
# =========================================================
elif page == "Expiry & Waste Analysis":
    page_header(
        "🗑️ Expiry & Waste Analysis",
        "Identify categories with shorter remaining shelf life and higher recorded waste.",
    )
    if "days_remaining_at_purchase" not in filtered.columns:
        st.error("days_remaining_at_purchase is missing from the dataset.")
        st.stop()

    expiry_df = filtered.dropna(subset=["days_remaining_at_purchase"]).copy()
    expiry_df["expiry_risk"] = expiry_df["days_remaining_at_purchase"].apply(classify_expiry_risk)
    risk_order = ["Critical", "High", "Moderate", "Low"]

    risk_counts = (
        expiry_df["expiry_risk"].value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Expiry Risk")
        .reset_index(name="Records")
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            risk_counts, x="Expiry Risk", y="Records",
            title="Records by Provisional Expiry-Risk Band",
            category_orders={"Expiry Risk": risk_order},
            template="plotly_dark", text_auto=True,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        category_waste = (
            expiry_df.groupby("category", as_index=False)
            .agg(avg_waste_pct=("waste_pct", "mean"),
                 avg_days_remaining=("days_remaining_at_purchase", "mean"))
        )
        category_waste["Category"] = category_waste["category"].map(display_category)
        fig = px.scatter(
            category_waste, x="avg_days_remaining", y="avg_waste_pct",
            text="Category", size="avg_waste_pct",
            title="Average Remaining Shelf Life vs Average Waste",
            labels={
                "avg_days_remaining": "Average days remaining",
                "avg_waste_pct": "Average waste (%)",
            },
            template="plotly_dark",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    cat_risk = (
        expiry_df.groupby(["category", "expiry_risk"]).size()
        .unstack(fill_value=0)
        .reindex(columns=risk_order, fill_value=0)
    )
    cat_risk_pct = cat_risk.div(cat_risk.sum(axis=1), axis=0) * 100
    cat_risk_pct.index = [display_category(x) for x in cat_risk_pct.index]
    st.subheader("Expiry-risk distribution by category (%)")
    st.dataframe(cat_risk_pct.round(1), use_container_width=True)

    st.info(
        "Provisional bands used here: Critical = 0–3 days, High = 4–7 days, "
        "Moderate = 8–14 days, Low = more than 14 days. The calculation uses "
        "days_remaining_at_purchase. The dataset's days_until_expiry field showed "
        "inconsistencies during validation, so it is not used here."
    )
    st.caption(
        "Expiry risk and waste risk are related business concerns, but they are not "
        "the same measure. Categories with longer shelf life can still have high waste."
    )

# ------------------------- Footer -------------------------
st.sidebar.divider()
st.sidebar.caption("FreshFlow | M.Tech Business Data Analytics Project")
