# web-traffic-Anayatics
# ================================================================
#                    WEB TRAFFIC ANALYTICS
# ================================================================
#
# OBJECTIVE:
# Analyze website traffic channels, bounce rates, session duration,
# pageviews, conversion funnels and drop-off points.
#
# ANALYSIS INCLUDED:
# 1. Load web traffic dataset
# 2. Data cleaning
# 3. Missing-value handling
# 4. Bounce rate calculation
# 5. Average session duration
# 6. Pageviews per user
# 7. Traffic-source analysis
# 8. Conversion rate by traffic source
# 9. Conversion funnel analysis
# 10. Funnel drop-off analysis
# 11. Cohort analysis
# 12. Statistical summary
# 13. Bar charts
# 14. Pie chart
# 15. Boxplots
# 16. Heatmap
# 17. Traffic performance dashboard
# 18. Business recommendations
# 19. Export results to CSV files
#
# DATASET:
# Keep "web_traffic.csv" in the same folder.
#
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# Plotting style
sns.set_theme(style="whitegrid")

# Display all columns
pd.set_option("display.max_columns", None)

# Display more rows
pd.set_option("display.max_rows", 100)

# ================================================================
# 2. PROJECT TITLE
# ================================================================

print("\n")
print("=" * 80)
print("                    WEB TRAFFIC ANALYTICS")
print("=" * 80)

print("""
Objective:
Analyze website traffic data to understand traffic sources,
bounce rates, session duration, pageviews, conversion performance,
conversion funnels and user drop-off points.
""")

# ================================================================
# 3. LOAD DATASET
# ================================================================

FILE_NAME = "web_traffic.csv"

if not os.path.exists(FILE_NAME):
    print("\nERROR:")
    print("web_traffic.csv was not found.")
    print("Place the dataset in the same folder as this Python program.")
    raise SystemExit

df = pd.read_csv(FILE_NAME)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nFirst five records:")
print(df.head())


# ================================================================
# 4. DATASET INFORMATION
# ================================================================

print("\n")
print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# ================================================================
# 5. REMOVE DUPLICATES
# ================================================================

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print("Rows:", len(df))


# ================================================================
# 6. CLEAN COLUMN NAMES
# ================================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())


# ================================================================
# 7. FUNCTION TO FIND COLUMNS
# ================================================================

def find_column(possible_names):

    for name in possible_names:

        name = name.lower()

        if name in df.columns:
            return name

    return None


# ================================================================
# 8. IDENTIFY IMPORTANT COLUMNS
# ================================================================

user_col = find_column([
    "user_id",
    "userid",
    "visitor_id",
    "visitorid",
    "customer_id",
    "customerid"
])

session_col = find_column([
    "session_id",
    "sessionid",
    "visit_id",
    "visitid"
])

source_col = find_column([
    "traffic_source",
    "traffic_source_name",
    "source",
    "channel",
    "traffic_channel",
    "marketing_channel"
])

duration_col = find_column([
    "session_duration",
    "session_duration_seconds",
    "duration",
    "session_time",
    "time_on_site"
])

pageview_col = find_column([
    "pageviews",
    "page_views",
    "pages_viewed",
    "number_of_pages"
])

bounce_col = find_column([
    "bounce",
    "bounced",
    "is_bounce",
    "bounce_flag"
])

conversion_col = find_column([
    "conversion",
    "converted",
    "conversion_flag",
    "is_converted",
    "purchase",
    "converted_user"
])

cohort_col = find_column([
    "cohort",
    "cohort_month",
    "user_cohort",
    "signup_month"
])

page_col = find_column([
    "page",
    "page_name",
    "landing_page",
    "current_page"
])


# ================================================================
# 9. PRINT DETECTED COLUMNS
# ================================================================

print("\n")
print("=" * 80)
print("DETECTED DATASET COLUMNS")
print("=" * 80)

print("User column          :", user_col)
print("Session column       :", session_col)
print("Traffic source       :", source_col)
print("Session duration     :", duration_col)
print("Pageviews            :", pageview_col)
print("Bounce column        :", bounce_col)
print("Conversion column    :", conversion_col)
print("Cohort column        :", cohort_col)
print("Page column          :", page_col)


# ================================================================
# 10. HANDLE MISSING VALUES
# ================================================================

print("\n")
print("=" * 80)
print("DATA CLEANING")
print("=" * 80)

# Numeric columns
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


# Categorical columns
categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()

        if len(mode_value) > 0:

            df[column] = df[column].fillna(
                mode_value[0]
            )

print("Missing-value treatment completed.")


# ================================================================
# 11. CONVERT NUMERIC COLUMNS
# ================================================================

if duration_col is not None:

    df[duration_col] = pd.to_numeric(
        df[duration_col],
        errors="coerce"
    )

    df[duration_col] = df[duration_col].fillna(
        df[duration_col].median()
    )


if pageview_col is not None:

    df[pageview_col] = pd.to_numeric(
        df[pageview_col],
        errors="coerce"
    )

    df[pageview_col] = df[pageview_col].fillna(
        df[pageview_col].median()
    )


# ================================================================
# 12. CONVERT BOUNCE COLUMN INTO 0 AND 1
# ================================================================

if bounce_col is not None:

    def convert_binary(value):

        value = str(value).strip().lower()

        if value in [
            "yes",
            "y",
            "true",
            "1",
            "bounce",
            "bounced"
        ]:
            return 1

        return 0


    df["bounce_flag"] = df[bounce_col].apply(
        convert_binary
    )

else:

    # If bounce column doesn't exist,
    # calculate bounce using one pageview where possible.

    if pageview_col is not None:

        df["bounce_flag"] = (
            df[pageview_col] <= 1
        ).astype(int)

    else:

        df["bounce_flag"] = 0


# ================================================================
# 13. CONVERT CONVERSION COLUMN
# ================================================================

if conversion_col is not None:

    df["conversion_flag"] = df[
        conversion_col
    ].apply(convert_binary)

else:

    df["conversion_flag"] = 0


# ================================================================
# 14. OVERALL TRAFFIC METRICS
# ================================================================

print("\n")
print("=" * 80)
print("OVERALL WEB TRAFFIC METRICS")
print("=" * 80)

total_records = len(df)

if user_col is not None:

    total_users = df[
        user_col
    ].nunique()

else:

    total_users = total_records


if session_col is not None:

    total_sessions = df[
        session_col
    ].nunique()

else:

    total_sessions = total_records


total_bounces = df[
    "bounce_flag"
].sum()

total_conversions = df[
    "conversion_flag"
].sum()


if total_sessions > 0:

    bounce_rate = (
        total_bounces /
        total_sessions
    ) * 100

else:

    bounce_rate = 0


if total_sessions > 0:

    conversion_rate = (
        total_conversions /
        total_sessions
    ) * 100

else:

    conversion_rate = 0


print("Total records        :", total_records)
print("Total users          :", total_users)
print("Total sessions       :", total_sessions)
print("Total bounces        :", total_bounces)
print("Total conversions    :", total_conversions)

print(
    "Overall bounce rate  : {:.2f}%".format(
        bounce_rate
    )
)

print(
    "Overall conversion   : {:.2f}%".format(
        conversion_rate
    )
)


# ================================================================
# 15. AVERAGE SESSION DURATION
# ================================================================

print("\n")
print("=" * 80)
print("SESSION DURATION ANALYSIS")
print("=" * 80)

if duration_col is not None:

    average_session_duration = df[
        duration_col
    ].mean()

    median_session_duration = df[
        duration_col
    ].median()

    maximum_session_duration = df[
        duration_col
    ].max()

    minimum_session_duration = df[
        duration_col
    ].min()

    print(
        "Average session duration : {:.2f}".format(
            average_session_duration
        )
    )

    print(
        "Median session duration  : {:.2f}".format(
            median_session_duration
        )
    )

    print(
        "Maximum session duration : {:.2f}".format(
            maximum_session_duration
        )
    )

    print(
        "Minimum session duration : {:.2f}".format(
            minimum_session_duration
        )
    )

else:

    average_session_duration = 0

    print(
        "Session duration column was not found."
    )


# ================================================================
# 16. PAGEVIEWS PER USER
# ================================================================

print("\n")
print("=" * 80)
print("PAGEVIEWS ANALYSIS")
print("=" * 80)

if pageview_col is not None:

    total_pageviews = df[
        pageview_col
    ].sum()

    average_pageviews = df[
        pageview_col
    ].mean()

    if total_users > 0:

        pageviews_per_user = (
            total_pageviews /
            total_users
        )

    else:

        pageviews_per_user = 0

    print(
        "Total pageviews       : {:.0f}".format(
            total_pageviews
        )
    )

    print(
        "Average pageviews     : {:.2f}".format(
            average_pageviews
        )
    )

    print(
        "Pageviews per user    : {:.2f}".format(
            pageviews_per_user
        )
    )

else:

    total_pageviews = 0
    pageviews_per_user = 0

    print(
        "Pageviews column was not found."
    )


# ================================================================
# 17. TRAFFIC SOURCE ANALYSIS
# ================================================================

print("\n")
print("=" * 80)
print("TRAFFIC SOURCE ANALYSIS")
print("=" * 80)

if source_col is not None:

    source_summary = df.groupby(
        source_col
    ).agg(

        Sessions=(
            source_col,
            "count"
        ),

        Bounces=(
            "bounce_flag",
            "sum"
        ),

        Conversions=(
            "conversion_flag",
            "sum"
        )
    )

    if duration_col is not None:

        duration_summary = df.groupby(
            source_col
        )[duration_col].mean()

        source_summary[
            "Average_Session_Duration"
        ] = duration_summary


    if pageview_col is not None:

        pageview_summary = df.groupby(
            source_col
        )[pageview_col].mean()

        source_summary[
            "Average_Pageviews"
        ] = pageview_summary


    source_summary[
        "Bounce_Rate_Percent"
    ] = (
        source_summary["Bounces"] /
        source_summary["Sessions"] *
        100
    )


    source_summary[
        "Conversion_Rate_Percent"
    ] = (
        source_summary["Conversions"] /
        source_summary["Sessions"] *
        100
    )


    source_summary = source_summary.sort_values(
        "Conversion_Rate_Percent",
        ascending=False
    )


    print(
        source_summary
    )

else:

    source_summary = None

    print(
        "Traffic source column was not found."
    )


# ================================================================
# 18. TRAFFIC SOURCE BAR CHART
# ================================================================

if source_summary is not None:

    plt.figure(
        figsize=(10, 6)
    )

    source_summary[
        "Sessions"
    ].sort_values().plot(
        kind="bar"
    )

    plt.title(
        "Website Sessions by Traffic Source"
    )

    plt.xlabel(
        "Traffic Source"
    )

    plt.ylabel(
        "Number of Sessions"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 19. BOUNCE RATE BY TRAFFIC SOURCE
# ================================================================

if source_summary is not None:

    plt.figure(
        figsize=(10, 6)
    )

    source_summary[
        "Bounce_Rate_Percent"
    ].sort_values().plot(
        kind="bar"
    )

    plt.title(
        "Bounce Rate by Traffic Source"
    )

    plt.xlabel(
        "Traffic Source"
    )

    plt.ylabel(
        "Bounce Rate (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 20. CONVERSION RATE BY TRAFFIC SOURCE
# ================================================================

if source_summary is not None:

    plt.figure(
        figsize=(10, 6)
    )

    source_summary[
        "Conversion_Rate_Percent"
    ].sort_values().plot(
        kind="bar"
    )

    plt.title(
        "Conversion Rate by Traffic Source"
    )

    plt.xlabel(
        "Traffic Source"
    )

    plt.ylabel(
        "Conversion Rate (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 21. PIE CHART - TRAFFIC DISTRIBUTION
# ================================================================

if source_col is not None:

    traffic_distribution = (
        df[source_col]
        .value_counts()
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.pie(
        traffic_distribution.values,
        labels=traffic_distribution.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Traffic Distribution by Source"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 22. CONVERSION DISTRIBUTION
# ================================================================

conversion_counts = df[
    "conversion_flag"
].value_counts()

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Not Converted", "Converted"],
    [
        conversion_counts.get(0, 0),
        conversion_counts.get(1, 0)
    ]
)

plt.title(
    "User Conversion Distribution"
)

plt.xlabel(
    "Conversion Status"
)

plt.ylabel(
    "Number of Users/Sessions"
)

plt.tight_layout()

plt.show()


# ================================================================
# 23. SESSION DURATION BOXPLOT
# ================================================================

if duration_col is not None:

    plt.figure(
        figsize=(8, 6)
    )

    sns.boxplot(
        data=df,
        x="conversion_flag",
        y=duration_col
    )

    plt.title(
        "Session Duration vs Conversion"
    )

    plt.xlabel(
        "Conversion Status"
    )

    plt.ylabel(
        "Session Duration"
    )

    plt.xticks(
        [0, 1],
        ["Not Converted", "Converted"]
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 24. PAGEVIEWS BOXPLOT
# ================================================================

if pageview_col is not None:

    plt.figure(
        figsize=(8, 6)
    )

    sns.boxplot(
        data=df,
        x="conversion_flag",
        y=pageview_col
    )

    plt.title(
        "Pageviews vs Conversion"
    )

    plt.xlabel(
        "Conversion Status"
    )

    plt.ylabel(
        "Pageviews"
    )

    plt.xticks(
        [0, 1],
        ["Not Converted", "Converted"]
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 25. BOUNCE RATE ANALYSIS
# ================================================================

print("\n")
print("=" * 80)
print("BOUNCE RATE ANALYSIS")
print("=" * 80)

print(
    "Total sessions :", total_sessions
)

print(
    "Bounced sessions:", total_bounces
)

print(
    "Bounce rate     : {:.2f}%".format(
        bounce_rate
    )
)


# ================================================================
# 26. COHORT ANALYSIS
# ================================================================

print("\n")
print("=" * 80)
print("USER COHORT ANALYSIS")
print("=" * 80)

if cohort_col is not None:

    cohort_summary = df.groupby(
        cohort_col
    ).agg(

        Users=(
            user_col if user_col else cohort_col,
            "nunique"
        ),

        Sessions=(
            cohort_col,
            "count"
        ),

        Conversions=(
            "conversion_flag",
            "sum"
        ),

        Bounces=(
            "bounce_flag",
            "sum"
        )
    )

    cohort_summary[
        "Conversion_Rate_Percent"
    ] = (
        cohort_summary["Conversions"] /
        cohort_summary["Sessions"] *
        100
    )

    cohort_summary[
        "Bounce_Rate_Percent"
    ] = (
        cohort_summary["Bounces"] /
        cohort_summary["Sessions"] *
        100
    )

    print(
        cohort_summary
    )

else:

    cohort_summary = None

    print(
        "Cohort column was not found."
    )


# ================================================================
# 27. COHORT CONVERSION CHART
# ================================================================

if cohort_summary is not None:

    plt.figure(
        figsize=(10, 6)
    )

    cohort_summary[
        "Conversion_Rate_Percent"
    ].plot(
        kind="bar"
    )

    plt.title(
        "Conversion Rate by User Cohort"
    )

    plt.xlabel(
        "User Cohort"
    )

    plt.ylabel(
        "Conversion Rate (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 28. CONVERSION FUNNEL
# ================================================================
#
# The dataset may contain funnel columns such as:
# landing_page
# product_view
# add_to_cart
# checkout
# purchase
#
# The code automatically searches for common funnel columns.
# ================================================================

print("\n")
print("=" * 80)
print("CONVERSION FUNNEL ANALYSIS")
print("=" * 80)


funnel_candidates = {

    "Landing Page": [
        "landing_page",
        "landing",
        "visited",
        "visit"
    ],

    "Product View": [
        "product_view",
        "product_viewed",
        "view_product",
        "product_page"
    ],

    "Add to Cart": [
        "add_to_cart",
        "added_to_cart",
        "cart",
        "add_cart"
    ],

    "Checkout": [
        "checkout",
        "checkout_started",
        "started_checkout"
    ],

    "Purchase": [
        "purchase",
        "purchased",
        "order",
        "transaction"
    ]
}


funnel_columns = {}


for stage, possible_columns in funnel_candidates.items():

    found = find_column(
        possible_columns
    )

    if found is not None:

        funnel_columns[
            stage
        ] = found


print("\nDetected funnel stages:")

for stage, column in funnel_columns.items():

    print(
        stage,
        "->",
        column
    )


# ================================================================
# 29. CALCULATE FUNNEL COUNTS
# ================================================================

funnel_results = []


for stage, column in funnel_columns.items():

    try:

        count = df[column].apply(
            convert_binary
        ).sum()

    except:

        count = 0

    funnel_results.append({
        "Stage": stage,
        "Users": count
    })


if len(funnel_results) > 0:

    funnel_df = pd.DataFrame(
        funnel_results
    )

    print("\nFunnel results:")
    print(
        funnel_df
    )

else:

    funnel_df = pd.DataFrame(
        columns=["Stage", "Users"]
    )

    print(
        "\nNo standard funnel columns were detected."
    )


# ================================================================
# 30. FUNNEL DROP-OFF
# ================================================================

if len(funnel_df) >= 2:

    funnel_df[
        "Drop_Off"
    ] = funnel_df[
        "Users"
    ].shift(1) - funnel_df[
        "Users"
    ]

    funnel_df[
        "Drop_Off_Percent"
    ] = (
        funnel_df["Drop_Off"] /
        funnel_df["Users"].shift(1)
    ) * 100

    funnel_df[
        "Drop_Off_Percent"
    ] = funnel_df[
        "Drop_Off_Percent"
    ].fillna(0)

    print("\n")
    print("=" * 80)
    print("FUNNEL DROP-OFF ANALYSIS")
    print("=" * 80)

    print(
        funnel_df
    )


# ================================================================
# 31. FUNNEL VISUALIZATION
# ================================================================

if len(funnel_df) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        funnel_df["Stage"],
        funnel_df["Users"]
    )

    plt.title(
        "Website Conversion Funnel"
    )

    plt.xlabel(
        "Funnel Stage"
    )

    plt.ylabel(
        "Number of Users"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 32. DROP-OFF VISUALIZATION
# ================================================================

if len(funnel_df) >= 2:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        funnel_df["Stage"],
        funnel_df["Drop_Off_Percent"]
    )

    plt.title(
        "Conversion Funnel Drop-Off Rate"
    )

    plt.xlabel(
        "Funnel Stage"
    )

    plt.ylabel(
        "Drop-Off Rate (%)"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 33. TOP DROP-OFF POINT
# ================================================================

if len(funnel_df) >= 2:

    valid_dropoffs = funnel_df.iloc[1:]

    if len(valid_dropoffs) > 0:

        highest_dropoff_index = (
            valid_dropoffs[
                "Drop_Off_Percent"
            ].idxmax()
        )

        highest_dropoff_stage = (
            funnel_df.loc[
                highest_dropoff_index,
                "Stage"
            ]
        )

        highest_dropoff_rate = (
            funnel_df.loc[
                highest_dropoff_index,
                "Drop_Off_Percent"
            ]
        )

        print("\n")
        print("=" * 80)
        print("TOP FUNNEL DROP-OFF")
        print("=" * 80)

        print(
            "Stage              :",
            highest_dropoff_stage
        )

        print(
            "Drop-off rate      : {:.2f}%".format(
                highest_dropoff_rate
            )
        )


# ================================================================
# 34. PAGE ANALYSIS
# ================================================================

print("\n")
print("=" * 80)
print("PAGE PERFORMANCE ANALYSIS")
print("=" * 80)

if page_col is not None:

    page_summary = df.groupby(
        page_col
    ).agg(

        Visits=(
            page_col,
            "count"
        ),

        Bounces=(
            "bounce_flag",
            "sum"
        ),

        Conversions=(
            "conversion_flag",
            "sum"
        )
    )

    page_summary[
        "Bounce_Rate_Percent"
    ] = (
        page_summary["Bounces"] /
        page_summary["Visits"] *
        100
    )

    page_summary[
        "Conversion_Rate_Percent"
    ] = (
        page_summary["Conversions"] /
        page_summary["Visits"] *
        100
    )

    page_summary = page_summary.sort_values(
        "Bounce_Rate_Percent",
        ascending=False
    )

    print(
        page_summary.head(20)
    )

else:

    page_summary = None

    print(
        "Page column was not found."
    )


# ================================================================
# 35. TOP PAGES WITH HIGH BOUNCE
# ================================================================

if page_summary is not None:

    plt.figure(
        figsize=(10, 7)
    )

    page_summary.head(10)[
        "Bounce_Rate_Percent"
    ].sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top Pages by Bounce Rate"
    )

    plt.xlabel(
        "Bounce Rate (%)"
    )

    plt.ylabel(
        "Page"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 36. CORRELATION HEATMAP
# ================================================================

numeric_data = df.select_dtypes(
    include=np.number
)

if numeric_data.shape[1] >= 2:

    plt.figure(
        figsize=(12, 8)
    )

    correlation_matrix = (
        numeric_data.corr()
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title(
        "Web Traffic Correlation Heatmap"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 37. SESSION DURATION DISTRIBUTION
# ================================================================

if duration_col is not None:

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        df[duration_col],
        bins=30
    )

    plt.title(
        "Session Duration Distribution"
    )

    plt.xlabel(
        "Session Duration"
    )

    plt.ylabel(
        "Number of Sessions"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 38. PAGEVIEWS DISTRIBUTION
# ================================================================

if pageview_col is not None:

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        df[pageview_col],
        bins=30
    )

    plt.title(
        "Pageviews Distribution"
    )

    plt.xlabel(
        "Pageviews"
    )

    plt.ylabel(
        "Number of Sessions"
    )

    plt.tight_layout()

    plt.show()


# ================================================================
# 39. TRAFFIC SOURCE PERFORMANCE TABLE
# ================================================================

if source_summary is not None:

    print("\n")
    print("=" * 80)
    print("TRAFFIC SOURCE PERFORMANCE")
    print("=" * 80)

    display_columns = [
        "Sessions",
        "Bounces",
        "Conversions",
        "Bounce_Rate_Percent",
        "Conversion_Rate_Percent"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in source_summary.columns
    ]

    print(
        source_summary[
            display_columns
        ].round(2)
    )


# ================================================================
# 40. IDENTIFY BEST PERFORMING TRAFFIC SOURCE
# ================================================================

if source_summary is not None:

    best_source = (
        source_summary[
            "Conversion_Rate_Percent"
        ].idxmax()
    )

    best_source_rate = (
        source_summary.loc[
            best_source,
            "Conversion_Rate_Percent"
        ]
    )

    highest_bounce_source = (
        source_summary[
            "Bounce_Rate_Percent"
        ].idxmax()
    )

    highest_bounce_rate = (
        source_summary.loc[
            highest_bounce_source,
            "Bounce_Rate_Percent"
        ]
    )

    print("\n")
    print("=" * 80)
    print("TRAFFIC SOURCE FINDINGS")
    print("=" * 80)

    print(
        "Highest conversion source :",
        best_source
    )

    print(
        "Conversion rate            : {:.2f}%".format(
            best_source_rate
        )
    )

    print(
        "\nHighest bounce source     :",
        highest_bounce_source
    )

    print(
        "Bounce rate                : {:.2f}%".format(
            highest_bounce_rate
        )
    )


# ================================================================
# 41. FINAL KPI DASHBOARD
# ================================================================

print("\n")
print("=" * 80)
print("FINAL WEB TRAFFIC KPI DASHBOARD")
print("=" * 80)

print(
    "\nTotal Users              :",
    total_users
)

print(
    "Total Sessions           :",
    total_sessions
)

print(
    "Total Pageviews          :",
    total_pageviews
)

print(
    "Bounce Rate              : {:.2f}%".format(
        bounce_rate
    )
)

print(
    "Average Session Duration : {:.2f}".format(
        average_session_duration
    )
)

print(
    "Pageviews per User       : {:.2f}".format(
        pageviews_per_user
    )
)

print(
    "Total Conversions        :",
    total_conversions
)

print(
    "Conversion Rate          : {:.2f}%".format(
        conversion_rate
    )
)


# ================================================================
# 42. FINAL SUMMARY DATAFRAME
# ================================================================

final_summary = pd.DataFrame({

    "KPI": [
        "Total Users",
        "Total Sessions",
        "Total Pageviews",
        "Bounce Rate",
        "Average Session Duration",
        "Pageviews per User",
        "Total Conversions",
        "Conversion Rate"
    ],

    "Value": [
        total_users,
        total_sessions,
        total_pageviews,
        "{:.2f}%".format(
            bounce_rate
        ),
        "{:.2f}".format(
            average_session_duration
        ),
        "{:.2f}".format(
            pageviews_per_user
        ),
        total_conversions,
        "{:.2f}%".format(
            conversion_rate
        )
    ]
})

print("\nFinal KPI table:")
print(
    final_summary.to_string(
        index=False
    )
)


# ================================================================
# 43. BUSINESS RECOMMENDATIONS
# ================================================================

print("\n")
print("=" * 80)
print("BUSINESS RECOMMENDATIONS")
print("=" * 80)

print("""
1. Improve high-bounce landing pages.
   Pages with high bounce rates should be checked for page speed,
   content quality, navigation and mobile responsiveness.

2. Improve conversion funnel pages.
   The largest drop-off stage should be investigated to identify
   usability problems and unnecessary steps.

3. Analyze traffic sources separately.
   Organic, advertising, social and direct traffic may behave
   differently, so performance should be monitored by source.

4. Improve session engagement.
   Increase relevant content, internal links and clear calls-to-action
   to encourage users to visit additional pages.

5. Optimize high-performing traffic sources.
   Traffic sources with strong conversion performance can be analyzed
   further to understand which campaigns and audiences work well.

6. Reduce unnecessary checkout steps.
   If users drop significantly during checkout, simplify the process,
   reduce form fields and make important information clear.

7. Improve mobile experience.
   Website performance should be checked across mobile, tablet and
   desktop devices.

8. Monitor website performance continuously.
   Track sessions, bounce rate, session duration, pageviews and
   conversion rate on a regular basis.

9. Use A/B testing.
   Test different headlines, buttons, layouts and calls-to-action
   to measure their effect on user conversion.

10. Build a recurring analytics dashboard.
    A dashboard can help the business monitor traffic performance,
    user engagement and conversion trends over time.
""")


# ================================================================
# 44. EXPORT RESULTS
# ================================================================

print("\n")
print("=" * 80)
print("EXPORTING ANALYSIS RESULTS")
print("=" * 80)


# Save cleaned dataset
df.to_csv(
    "cleaned_web_traffic.csv",
    index=False
)

print(
    "Saved: cleaned_web_traffic.csv"
)


# Save final KPI summary
final_summary.to_csv(
    "web_traffic_kpi_summary.csv",
    index=False
)

print(
    "Saved: web_traffic_kpi_summary.csv"
)


# Save traffic source analysis
if source_summary is not None:

    source_summary.to_csv(
        "traffic_source_analysis.csv"
    )

    print(
        "Saved: traffic_source_analysis.csv"
    )


# Save cohort analysis
if cohort_summary is not None:

    cohort_summary.to_csv(
        "cohort_analysis.csv"
    )

    print(
        "Saved: cohort_analysis.csv"
    )


# Save funnel analysis
if len(funnel_df) > 0:

    funnel_df.to_csv(
        "conversion_funnel_analysis.csv",
        index=False
    )

    print(
        "Saved: conversion_funnel_analysis.csv"
    )


# Save page analysis
if page_summary is not None:

    page_summary.to_csv(
        "page_performance_analysis.csv"
    )

    print(
        "Saved: page_performance_analysis.csv"
    )


# ================================================================
# 45. COMPLETION MESSAGE
# ================================================================

print("\n")
print("=" * 80)
print("                 ANALYSIS COMPLETED")
print("=" * 80)

print("""
The Web Traffic Analytics project has been completed.

The program analyzed:

✓ Website traffic
✓ Traffic sources
✓ Bounce rate
✓ Average session duration
✓ Pageviews per user
✓ Conversion rate
✓ User cohorts
✓ Conversion funnel
✓ Funnel drop-off points
✓ Page performance
✓ Traffic source performance
✓ Correlation between numerical variables
✓ Visual charts
✓ Heatmap
✓ Business recommendations

Generated output files can be used for your
Web Analytics Dashboard PDF or presentation.
""")

print("=" * 80)