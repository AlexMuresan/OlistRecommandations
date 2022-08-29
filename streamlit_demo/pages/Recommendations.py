from json import load

import names
import numpy as np
import pandas as pd
import streamlit as st
from recommender import RecommendationEngine
from utils import *


def to_gender(number):
    if number == 1:
        return "male"
    if number == 2:
        return "female"


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
    recommendationengine_cosine = RecommendationEngine(
        final_df,
        dataframes["products_df"],
        dataframes["order_items_df"],
        translate_dict,
        sim_method="cosine",
    )

    # Initializing our custom recommendation engine
    recommendationengine_itr = RecommendationEngine(
        final_df,
        dataframes["products_df"],
        dataframes["order_items_df"],
        translate_dict,
        sim_method="itr",
    )

    return recommendationengine_cosine, recommendationengine_itr


@st.cache
def generate_names(min_id, max_id):
    gender_list = list(map(to_gender, np.random.randint(1, 3, user_size + 1)))
    age_list = np.random.randint(18, 75, user_size + 1)
    random_names = {}
    random_images = {}

    for i in range(min_id, max_id + 1):
        gender = gender_list[i]
        age = age_list[i]
        url_string = f"https://fakeface.rest/thumb/view/json?minimum_age={age}&maximum_age={age}&gender={gender}"

        random_name = names.get_full_name(gender=gender)

        while random_name in random_names.values():
            random_name = names.get_full_name(gender=gender)

        random_names[i] = random_name
        random_images[i] = url_string

    return random_names, random_images


if __name__ == "__main__":
    pd.set_option("display.precision", 1)

    min_id = 0
    max_id = 999
    user_size = max_id - min_id

    st.title("Recommendation Engine Demo")

    data_load_state = st.text("Loading data...")

    dataset_path = "/Users/alex/Workspace/Datasets/OlistEcommercePublicDataset"
    recommendationengine_cosine, recommendationengine_itr = load_data(dataset_path)
    generated_names, generated_faces = generate_names(
        min_id,
        max_id,
    )

    data_load_state.text("Loading data... done!")

    st.subheader("Recommendations for website users")
    customer_idx = st.slider("User index", min_id, max_id, 10)
    similarity_method = st.selectbox(
        "Method used for user similarity calculation", ("Cosine", "ITR")
    )

    if similarity_method == "Cosine":
        recommendationengine = recommendationengine_cosine
    if similarity_method == "ITR":
        recommendationengine = recommendationengine_itr

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

    col1, col2 = st.columns([1, 3])
    # st.markdown(f"![Alt Text]({generated_faces[customer_idx]})")
    with col1:
        st.image(f"{generated_faces[customer_idx]}", width=125)
    with col2:
        st.markdown(f"#### Items bought by user {generated_names[customer_idx]}")

    st.table(user_bought_items)

    st.markdown(f"#### Recommended items")
    st.table(user_recommendations)

    st.markdown(f"#### Recommended items - clustering")
    st.table(user_recommendations_cluster)
