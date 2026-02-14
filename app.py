import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Apple Sales Analytics", layout="wide")

st.title("📊 Apple Product Sales Analytics Dashboard")

# -------------------------------------------------
# 1️⃣ Generate Synthetic Dataset (1000 rows)
# -------------------------------------------------
@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)

    dates = pd.date_range(start="2021-01-01", periods=n, freq="D")
    regions = ["Americas", "Europe", "Greater China", "Japan", "Rest of Asia-Pacific"]
    products = ["iPhone", "iPad", "Mac"]
    models = ["Pro", "Air", "Base"]

    data = pd.DataFrame({
        "Date": np.random.choice(dates, n),
        "Region": np.random.choice(regions, n),
        "Product Category": np.random.choice(products, n),
        "Product Model": np.random.choice(models, n),
        "Units Sold": np.random.randint(50, 500, n)
    })

    base_price = {"iPhone": 900, "iPad": 600, "Mac": 1500}

    data["Revenue"] = data.apply(
        lambda x: x["Units Sold"] * base_price[x["Product Category"]] * np.random.uniform(0.9, 1.1),
        axis=1
    )

    data["COGS"] = data["Revenue"] * np.random.uniform(0.6, 0.8, n)
    data["Gross Margin"] = data["Revenue"] - data["COGS"]
    data["ASP"] = data["Revenue"] / data["Units Sold"]

    return data


# -------------------------------------------------
# 2️⃣ Upload or Generate
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_data()

st.subheader("Raw Data Preview")
st.dataframe(df.head())


# -------------------------------------------------
# 3️⃣ Data Cleaning & Transformation
# -------------------------------------------------
st.subheader("Data Cleaning & Feature Engineering")

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Quarter"] = df["Date"].dt.quarter

# Regex Example: Extract Model Type
df["Model_Type"] = df["Product Model"].apply(
    lambda x: re.findall(r"(Pro|Air|Base)", str(x))[0]
)

# Normalization
scaler = MinMaxScaler()
df[["Revenue_Norm", "Units_Norm"]] = scaler.fit_transform(
    df[["Revenue", "Units Sold"]]
)

st.success("Data cleaned, transformed, normalized & engineered successfully.")


# -------------------------------------------------
# 4️⃣ KPI Cards
# -------------------------------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"${df['Revenue'].sum():,.0f}")
col2.metric("Total Units Sold", f"{df['Units Sold'].sum():,.0f}")
col3.metric("Average Gross Margin", f"${df['Gross Margin'].mean():,.0f}")


# -------------------------------------------------
# 5️⃣ Regional Performance Analysis
# -------------------------------------------------
st.subheader("🌍 Revenue by Region & Product")

region_product = df.groupby(["Region", "Product Category"])["Revenue"].sum().reset_index()

fig1 = plt.figure(figsize=(10,5))
sns.barplot(data=region_product, x="Region", y="Revenue", hue="Product Category")
plt.xticks(rotation=45)
st.pyplot(fig1)


# -------------------------------------------------
# 6️⃣ Pie Chart – Revenue Share
# -------------------------------------------------
st.subheader("📊 Revenue Distribution by Product")

product_revenue = df.groupby("Product Category")["Revenue"].sum()

fig2 = plt.figure()
plt.pie(product_revenue, labels=product_revenue.index, autopct="%1.1f%%")
plt.title("Revenue Share")
st.pyplot(fig2)


# -------------------------------------------------
# 7️⃣ Time Trend Analysis
# -------------------------------------------------
st.subheader("📈 Revenue Trend Over Time")

time_trend = df.groupby("Year")["Revenue"].sum()

fig3 = plt.figure()
plt.plot(time_trend.index, time_trend.values, marker="o")
plt.title("Yearly Revenue Trend")
plt.xlabel("Year")
plt.ylabel("Revenue")
st.pyplot(fig3)


# -------------------------------------------------
# 8️⃣ Hypothesis Testing
# Example: Does iPhone generate more revenue than iPad?
# -------------------------------------------------
st.subheader("🧪 Hypothesis Testing")

iphone_rev = df[df["Product Category"] == "iPhone"]["Revenue"]
ipad_rev = df[df["Product Category"] == "iPad"]["Revenue"]

t_stat, p_val = stats.ttest_ind(iphone_rev, ipad_rev)

st.write(f"T-Statistic: {t_stat:.2f}")
st.write(f"P-Value: {p_val:.4f}")

if p_val < 0.05:
    st.success("Statistically significant difference in revenue between iPhone and iPad.")
else:
    st.warning("No statistically significant difference found.")


# -------------------------------------------------
# 9️⃣ Growth Rate Analysis
# -------------------------------------------------
st.subheader("🚀 Fastest Growing Region")

region_year = df.groupby(["Region", "Year"])["Revenue"].sum().reset_index()
region_growth = region_year.groupby("Region")["Revenue"].pct_change()

region_year["Growth"] = region_growth
growth_summary = region_year.groupby("Region")["Growth"].mean().sort_values(ascending=False)

st.write("Average Growth Rate by Region")
st.dataframe(growth_summary)


# -------------------------------------------------
# 🔟 Underperforming Product Detection
# -------------------------------------------------
st.subheader("⚠️ Underperforming Products")

avg_revenue = df.groupby(["Region", "Product Category"])["Revenue"].mean().reset_index()

threshold = avg_revenue["Revenue"].mean()
underperform = avg_revenue[avg_revenue["Revenue"] < threshold]

st.dataframe(underperform)


# -------------------------------------------------
# 📊 Correlation Heatmap
# -------------------------------------------------
st.subheader("🔎 Correlation Analysis")

fig4 = plt.figure()
sns.heatmap(df[["Revenue", "Units Sold", "COGS", "Gross Margin"]].corr(), annot=True)
st.pyplot(fig4)


st.success("Business Questions Answered via Visual & Statistical Insights.")
