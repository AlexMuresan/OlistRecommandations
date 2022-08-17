from operator import itemgetter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from recommender import RecommendationEngine
from utils import *


@st.cache
def load_data(path: str):
    # Loading necessary csvs into Pandas
    # data_folder = "/Users/alex/Workspace/Datasets/OlistEcommercePublicDataset"
    data_folder = path

    df_names = [
        "orders_df",
        "reviews_df",
        "products_df",
        "order_items_df",
        "customer_df",
        "cat_name_translation",
    ]
    df_files = [
        "olist_orders_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_products_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_customers_dataset.csv",
        "product_category_name_translation.csv",
    ]

    # Loading dataframes
    dataframes = preprocess_dataframes(read_dataframes(df_names, df_files, data_folder))

    # Filtering final dataframe by most active users and bought items
    final_df = filter_dataframe(
        join_dataframes(dataframes), item_number=500, user_number=1000
    )

    # Loading translation dictionary [Portugues -> English]
    translate_dict = get_translation_dict(dataframes["cat_name_translation"])

    # Initializing our custom recommendation engine
    recommendationengine = RecommendationEngine(
        final_df,
        dataframes["products_df"],
        dataframes["order_items_df"],
        translate_dict,
    )

    return recommendationengine, final_df


@st.cache
def generate_id_dict(recommendationengine):
    # Generating a ID to index dictionary so we can retrieve each user's ID
    id_to_idx = {}
    for customer_idx in range(min_id, max_id):
        recos, customer_id = recommendationengine.get_recommendation(
            customer_idx=customer_idx, nr_of_items=2, cluster=True
        )
        id_to_idx[customer_id] = customer_idx

    return id_to_idx


@st.cache
def get_sales_percentage(price_list, percentage):
    # Returns a percentage of products out of all the recommended products.
    # The products are selected randomly.
    total = len(price_list)
    needed_rows = round(total * percentage / 100)

    selected_prices = []
    selected_idxs = []

    for i in range(round(needed_rows)):
        selected_idx = np.random.randint(0, total)
        while selected_idx in selected_idxs:
            selected_idx = np.random.randint(0, total)

        selected_idxs.append(selected_idx)
        selected_prices.append(price_list[selected_idx])

    return selected_prices


@st.cache
def get_increased_profit(
    recommendationengine,
    profit_df,
    sales_df,
    users_dict,
    id_to_idx,
    sales_percentage,
    printing=False,
):
    # Calculates the increase in profit based on users buying a percentage of
    # the recommended products
    increased_monthly_profit = profit_df.copy()
    increased_monthly_precentage = {}

    for month in sorted(sales_df["order_purchase_timestamp"].dt.month.unique()):
        ids = users_dict[month]
        # This is to prevent a corner case where one of the user IDs didn't
        # match up with any of the indexes for some reason
        try:
            idx = itemgetter(*ids)(id_to_idx)
        except KeyError as e:
            ids = list(ids)
            ids.remove(e.args[0])
            ids = itemgetter(*ids)(id_to_idx)

        item_prices = []

        for customer_idx in idx:
            # Get recommendations for a certain user
            recos, customer_id = recommendationengine.get_recommendation(
                customer_idx=customer_idx, nr_of_items=2, cluster=True
            )

            # If clustered colab filtering doesn't return any recommendations we fall back
            # on the basic colab filtering
            if recos.empty:
                recos, customer_id = recommendationengine.get_recommendation(
                    customer_idx=customer_idx, nr_of_items=2, cluster=False
                )

            item_prices.append(recos.iloc[0]["price"])

        # Since we use the 'round()' function we need more than 5 products when
        # the selection percentage is under 10%, otherwise it rounds down to 0
        # e.g: 10% of 4 products = 0.4 -> round(0.4) = 0
        # When this happens we just generate a random integer and use that as a
        if (len(item_prices) < 5) and (sales_percentage <= 10):
            random_idx = np.random.randint(0, len(item_prices))
            item_prices_percentage = [item_prices[random_idx]]
        else:
            item_prices_percentage = get_sales_percentage(item_prices, sales_percentage)

        profit_increase_percentage = (
            sum(item_prices_percentage) * 100 / profit_df[month]
        )

        if printing:
            print(f"Month: {month}")
            print(f"{profit_increase_percentage:.2f}% increase in profit")

        increased_monthly_profit[month] += sum(item_prices_percentage)
        increased_monthly_precentage[month] = profit_increase_percentage

    return increased_monthly_profit, increased_monthly_precentage


if __name__ == "__main__":
    min_id = 0
    max_id = 999
    users = {}

    st.title("Recommendation Engine Demo")

    data_load_state = st.text("Loading and processing data...")

    dataset_path = "/Users/alex/Workspace/Datasets/OlistEcommercePublicDataset"
    recommendationengine, final_df = load_data(dataset_path)

    # Converting 'order_purchase_timestamp' column to Pandas datetime
    time_df = final_df.drop_duplicates(
        subset=["customer_unique_id", "product_id"]
    ).reset_index(drop=True)
    time_df["order_purchase_timestamp"] = pd.to_datetime(
        time_df["order_purchase_timestamp"]
    )

    id_to_idx = generate_id_dict(recommendationengine)

    data_load_state.text("Loading and processing data... done!")

    year = st.selectbox("Select year", [2017, 2018])
    df = time_df[time_df["order_purchase_timestamp"].dt.year == year]
    monthly_profit = df.groupby(df["order_purchase_timestamp"].dt.month)["price"].sum()

    # Finding the users that have bought items in each month
    for month in sorted(df["order_purchase_timestamp"].dt.month.unique()):
        users[month] = df[df["order_purchase_timestamp"].dt.month == month][
            "customer_unique_id"
        ].values

    sales_percentage = st.slider("Sales percentage", 0, 100, 10)
    (increased_monthly_profit, increased_monthly_percentage,) = get_increased_profit(
        recommendationengine,
        monthly_profit,
        df,
        users,
        id_to_idx,
        sales_percentage,
        printing=False,
    )

    percentage = pd.Series(increased_monthly_percentage).round()
    percentage = (
        pd.DataFrame(percentage, columns=["Precentage of increase"])
        .reset_index()
        .rename(columns={"index": "Month"})
    )
    percentage["Precentage of increase"] = (
        percentage["Precentage of increase"].astype(int).astype(str) + "%"
    )

    increased_profit = pd.DataFrame(increased_monthly_profit).reset_index()
    increased_profit["type"] = "profit"

    monthly_profit = pd.DataFrame(monthly_profit).reset_index()
    monthly_profit["type"] = "revenue"

    combined = pd.concat([increased_profit, monthly_profit])
    combined = combined.rename(
        columns={"price": "Ammount", "order_purchase_timestamp": "Month"}
    )

    bars = (
        alt.Chart(combined)
        .mark_bar(size=30)
        .encode(x="Month", y="Ammount", color="type")
    )

    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.altair_chart(bars, use_container_width=True)
    st.table(percentage)
    # st.bar_chart(data=monthly_profit)
    # st.bar_chart(data=increased_monthly_profit)
