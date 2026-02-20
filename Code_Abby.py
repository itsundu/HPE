import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

#print("Libraries loaded successfully.")

# Define date columns
date_cols = {
    "firmographics": ["Relationship_Start_Date"],
    "campaigns": ["Start_Date", "End_Date"],
    "leads": ["Date_Created"],
    "opportunities": ["Created_Date", "Close_Date"],
    "digital_events": ["Date"],
    "email_interactions": ["Date"],
    "web_traffic": ["Month"],
    "bu_performance": ["Quarter_End"]
}
#print("Date Columns defined.")

# Load CSV files
firmographics = pd.read_csv("firmographics.csv", parse_dates=date_cols["firmographics"])
campaigns = pd.read_csv("campaigns.csv", parse_dates=date_cols["campaigns"])
leads = pd.read_csv("leads.csv", parse_dates=date_cols["leads"])
opportunities = pd.read_csv("opportunities.csv", parse_dates=date_cols["opportunities"])
digital_events = pd.read_csv("digital_events.csv", parse_dates=date_cols["digital_events"])
email_interactions = pd.read_csv("email_interactions.csv", parse_dates=date_cols["email_interactions"])
web_traffic = pd.read_csv("web_traffic.csv", parse_dates=date_cols["web_traffic"])
bu_performance = pd.read_csv("bu_performance.csv", parse_dates=date_cols["bu_performance"])

contacts = pd.read_csv("contacts.csv")
products = pd.read_csv("products.csv")
sales_roster = pd.read_csv("sales_roster.csv")
competitors = pd.read_csv("competitors.csv")

#print("All datasets loaded with date parsing.")

# Create dictionary
datasets = {
    "firmographics": firmographics,
    "contacts": contacts,
    "products": products,
    "campaigns": campaigns,
    "leads": leads,
    "opportunities": opportunities,
    "sales_roster": sales_roster,
    "web_traffic": web_traffic,
    "digital_events": digital_events,
    "email_interactions": email_interactions,
    "bu_performance": bu_performance,
    "competitors": competitors
}
#print("Dictionary created.")

# Filter firmographics to North America (case‑insensitive)
na_firmographics = firmographics[
    firmographics["Region"].str.strip().str.lower() == "north america"
]

# Columns to display
cols = [
    "Customer_ID","Company_Name","Industry","Region","Company_Size",
    "Annual_Revenue_Band","Tech_Budget","Ownership_Type","Account_Status",
    "Relationship_Start_Date","Technology_Leadership","Country",
    "State_Province","Relationship_Length_Years"
]

# Print all rows
#print(na_firmographics[cols])


def basic_validation(df, name):
    #print(f"\n{name.upper()}")
    #print("Shape:", df.shape)
    #print("\nMissing Values:")
    #print(df.isnull().sum().sort_values(ascending=False).head(10))
    #print("\nDuplicate Rows:", df.duplicated().sum())
    #print("\nData Types:")
    #print(df.dtypes)
    return
for name, df in datasets.items():
    basic_validation(df, name)

    # Primary Key Validation

def check_unique(df, col, name):
    duplicates = df[col].duplicated().sum()
    #print(f"{name} - {col} duplicates:", duplicates)

check_unique(na_firmographics, "Customer_ID", "Firmographics")
check_unique(contacts, "Contact_ID", "Contacts")
check_unique(products, "Product_ID", "Products")
check_unique(campaigns, "Campaign_ID", "Campaigns")
check_unique(leads, "Lead_ID", "Leads")
check_unique(opportunities, "Opportunity_ID", "Opportunities")
check_unique(sales_roster, "Sales_Rep_ID", "Sales Roster")

opportunities.groupby("Opportunity_ID").size().sort_values(ascending=False).head()

# Confirm opportunity is line-level
opportunities.loc[
    opportunities["Opportunity_ID"] == "OPP-012530"
]

#1. ICP segmentation model (who to target)
#What & why
#Cluster accounts into high‑value vs low‑value segments using firmographics + realized revenue. This explains where Phantom should focus and why current spend is diluted.

# Example: aggregate revenue per customer from opportunities
print("#1. ICP segmentation model (who to target");
opp_na = opportunities.merge(firmographics, on="Customer_ID", how="inner")
opp_na = opp_na[opp_na["Region"].str.lower() == "north america"]

cust_rev = (
    opp_na.groupby("Customer_ID")["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Amount": "Total_Revenue"})
)

# Join with firmographics
seg = cust_rev.merge(
    firmographics[
        ["Customer_ID", "Industry", "Company_Size", "Annual_Revenue_Band", "Tech_Budget"]
    ],
    on="Customer_ID",
    how="left"
)

# Encode simple numeric features (you can one‑hot categorical in practice)
seg_num = seg[["Total_Revenue", "Tech_Budget"]].fillna(0)

scaler = StandardScaler()
X = scaler.fit_transform(seg_num)

kmeans = KMeans(n_clusters=3, random_state=42)
seg["Segment"] = kmeans.fit_predict(X)

# Inspect segment profiles
segment_summary = ( seg.groupby("Segment")[["Total_Revenue", "Tech_Budget"]] .mean() )
print(segment_summary)
print("SEG COLUMNS:", seg.columns.tolist())

# ---------------------------------------------------
# 1. Segment Size
# ---------------------------------------------------
segment_size = seg.groupby("Segment")["Customer_ID"].nunique().rename("Num_Customers")

# ---------------------------------------------------
# 2. Revenue Contribution
# ---------------------------------------------------
opp_seg = opportunities.merge(seg[["Customer_ID", "Segment"]], on="Customer_ID", how="left")

revenue_by_segment = opp_seg.groupby("Segment")["Amount"].sum().rename("Total_Revenue")
revenue_share = (revenue_by_segment / revenue_by_segment.sum()).rename("Revenue_Share")

# ---------------------------------------------------
# 3. Top Industry per Segment
# ---------------------------------------------------
industry_mix = (
    seg.groupby(["Segment", "Industry"])["Customer_ID"]
       .count()
       .reset_index()
)

top_industry = (
    industry_mix.sort_values(["Segment", "Customer_ID"], ascending=[True, False])
                .groupby("Segment")
                .first()["Industry"]
                .rename("Top_Industry")
)

# ---------------------------------------------------
# 4. Top Product Category per Segment (Correct column = Category)
# ---------------------------------------------------
prod_cat_col = "Category"   # your actual column

opp_prod = (
    opportunities
    .merge(products[["Product_ID", prod_cat_col]], on="Product_ID", how="left")
    .merge(seg[["Customer_ID", "Segment"]], on="Customer_ID", how="left")
)

product_mix = (
    opp_prod.groupby(["Segment", prod_cat_col])["Amount"]
            .sum()
            .reset_index()
)

top_product = (
    product_mix.sort_values(["Segment", "Amount"], ascending=[True, False])
               .groupby("Segment")
               .first()[prod_cat_col]
               .rename("Top_Product_Category")
)

# ---------------------------------------------------
# 5. Win Rate per Segment
# ---------------------------------------------------
opp_wl = opp_seg[opp_seg["Stage"].isin(["Won", "Lost"])].copy()
opp_wl["Won_Flag"] = (opp_wl["Stage"] == "Won").astype(int)

win_rate = opp_wl.groupby("Segment")["Won_Flag"].mean().rename("Win_Rate")

# ---------------------------------------------------
# 6. LTV per Segment (Annualized Revenue Proxy)
# ---------------------------------------------------
cust_ltv = (
    opp_seg.groupby("Customer_ID")["Amount"].sum().reset_index()
           .merge(firmographics[["Customer_ID", "Relationship_Length_Years"]], 
                  on="Customer_ID", how="left")
           .merge(seg[["Customer_ID", "Segment"]], on="Customer_ID", how="left")
)

cust_ltv["Annualized_Revenue"] = (
    cust_ltv["Amount"] / cust_ltv["Relationship_Length_Years"].clip(lower=0.5)
)

ltv_by_segment = cust_ltv.groupby("Segment")["Annualized_Revenue"].mean().rename("Avg_LTV")

# ---------------------------------------------------
# 7. Combine All Metrics into One Table
# ---------------------------------------------------
segment_summary = pd.concat(
    [
        segment_size,
        revenue_by_segment,
        revenue_share,
        seg.groupby("Segment")["Tech_Budget"].mean().rename("Avg_Tech_Budget"),
        top_industry,
        top_product,
        win_rate,
        ltv_by_segment
    ],
    axis=1
)

print(segment_summary)

#A bar chart for revenue share
#A bar chart for average LTV
#A bar chart for average tech budget
#A bar chart for number of customers

# Ensure segment_summary index is numeric
segment_summary = segment_summary.reset_index()

plt.style.use("seaborn-v0_8")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Phantom ICP Segmentation Summary", fontsize=20, fontweight="bold")

# -----------------------------------------
# 1. Revenue Share
# -----------------------------------------
sns.barplot(
    data=segment_summary,
    x="Segment",
    y="Revenue_Share",
    ax=axes[0, 0],
    palette="Blues_d"
)
axes[0, 0].set_title("Revenue Share by Segment")
axes[0, 0].set_ylabel("Revenue Share (%)")
axes[0, 0].set_xlabel("Segment")
axes[0, 0].set_ylim(0, segment_summary["Revenue_Share"].max() * 1.2)
axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%.2f")

# -----------------------------------------
# 2. Average LTV
# -----------------------------------------
sns.barplot(
    data=segment_summary,
    x="Segment",
    y="Avg_LTV",
    ax=axes[0, 1],
    palette="Greens_d"
)
axes[0, 1].set_title("Average LTV by Segment")
axes[0, 1].set_ylabel("Avg LTV ($)")
axes[0, 1].set_xlabel("Segment")
axes[0, 1].bar_label(axes[0, 1].containers[0], fmt="%.0f")

# -----------------------------------------
# 3. Average Tech Budget
# -----------------------------------------
sns.barplot(
    data=segment_summary,
    x="Segment",
    y="Avg_Tech_Budget",
    ax=axes[1, 0],
    palette="Purples_d"
)
axes[1, 0].set_title("Average Tech Budget by Segment")
axes[1, 0].set_ylabel("Avg Tech Budget ($)")
axes[1, 0].set_xlabel("Segment")
axes[1, 0].bar_label(axes[1, 0].containers[0], fmt="%.0f")

# -----------------------------------------
# 4. Number of Customers
# -----------------------------------------
sns.barplot(
    data=segment_summary,
    x="Segment",
    y="Num_Customers",
    ax=axes[1, 1],
    palette="Oranges_d"
)
axes[1, 1].set_title("Number of Customers by Segment")
axes[1, 1].set_ylabel("# Customers")
axes[1, 1].set_xlabel("Segment")
axes[1, 1].bar_label(axes[1, 1].containers[0], fmt="%.0f")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
