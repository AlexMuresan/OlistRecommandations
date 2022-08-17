from json import load

import names
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

    return recommendationengine


@st.cache
def generate_names(min_id, max_id):
    random_names = {}

    for i in range(min_id, max_id + 1):
        random_name = names.get_full_name()

        while random_name in random_names.values():
            random_name = names.get_full_name()

        random_names[i] = random_name

    return random_names


if __name__ == "__main__":
    pd.set_option("display.precision", 1)

    min_id = 0
    max_id = 999

    st.title("Recommendation Engine Demo")

    data_load_state = st.text("Loading data...")

    dataset_path = "/Users/alex/Workspace/Datasets/OlistEcommercePublicDataset"
    recommendationengine = load_data(dataset_path)
    generated_names = generate_names(min_id, max_id)

    data_load_state.text("Loading data... done!")

    st.subheader("Recommendations for website users")
    customer_idx = st.slider("User index", min_id, max_id, 10)

    user_bought_items = recommendationengine.get_bought_items(
        customer_idx=customer_idx, nr_of_items=2
    )
    user_recommendations, user_id = recommendationengine.get_recommendation(
        customer_idx=customer_idx, nr_of_items=2
    )
    user_recommendations_cluster, user_id = recommendationengine.get_recommendation(
        customer_idx=customer_idx, nr_of_items=2, cluster=True
    )

    user_bought_items = user_bought_items.rename(
        columns={
            "product_id": "Product ID",
            "rating": "Rating",
            "product_category_name": "Category Name",
            "price": "Price",
        }
    )
    user_bought_items["Rating"] = user_bought_items["Rating"].astype(int)
    user_bought_items["Price"] = user_bought_items["Price"].astype(int)

    user_recommendations = user_recommendations.rename(
        columns={
            "product_id": "Product ID",
            "score": "Score",
            "product_category_name": "Category Name",
            "price": "Price",
        }
    )
    user_recommendations["Score"] = user_recommendations["Score"].astype(int)
    user_recommendations["Price"] = user_recommendations["Price"].astype(int)

    user_recommendations_cluster = user_recommendations_cluster.rename(
        columns={
            "product_id": "Product ID",
            "score": "Score",
            "product_category_name": "Category Name",
            "price": "Price",
        }
    )
    user_recommendations_cluster["Score"] = (
        user_recommendations_cluster["Score"].round().astype(int)
    )
    user_recommendations_cluster["Price"] = (
        user_recommendations_cluster["Price"].round().astype(int)
    )

    st.markdown(f"#### Items bought by user {generated_names[customer_idx]}")
    st.table(user_bought_items)

    st.markdown(f"#### Recommended items")
    st.table(user_recommendations)

    st.markdown(f"#### Recommended items - clustering")
    st.table(user_recommendations_cluster)
