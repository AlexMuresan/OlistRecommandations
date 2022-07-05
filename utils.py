import os

import pandas as pd


def get_translation_dict(cat_name_translation):
    portuguese_cat_names = cat_name_translation.to_dict()["product_category_name"]
    english_cat_names = cat_name_translation.to_dict()["product_category_name_english"]
    translate_dict = {}

    for p_key in portuguese_cat_names:
        if portuguese_cat_names[p_key] not in translate_dict:
            translate_dict[portuguese_cat_names[p_key]] = english_cat_names[p_key]

    return translate_dict


def read_dataframes(df_names: list, df_files: list, root_folder: str) -> dict:
    data_folder = os.path.abspath(root_folder)

    dataframes = {}

    for name, file in zip(df_names, df_files):
        dataframes[name] = pd.read_csv(os.path.join(data_folder, file))

    return dataframes


def preprocess_dataframes(dataframes: dict) -> dict:
    # Converting ID columns from 'object' type to string
    for df_name in dataframes.keys():
        for column, ctype in zip(
            dataframes[df_name].columns, dataframes[df_name].dtypes
        ):
            if ctype == "object":
                dataframes[df_name][column] = dataframes[df_name][column].astype(
                    "string"
                )

    return dataframes


def join_dataframes(dataframes: dict) -> pd.core.frame.DataFrame:
    unique_id_df = pd.merge(
        dataframes["orders_df"][
            ["order_id", "customer_id", "order_purchase_timestamp"]
        ],
        dataframes["customer_df"][["customer_id", "customer_unique_id"]],
        on=["customer_id"],
        how="inner",
    )

    product_and_order_id_df = pd.merge(
        dataframes["orders_df"][["order_id", "customer_id"]],
        dataframes["order_items_df"][["order_id", "product_id", "price"]],
        on=["order_id"],
        how="inner",
    )
    user_product_order_id_df = pd.merge(
        unique_id_df,
        product_and_order_id_df,
        on=["order_id", "customer_id"],
        how="inner",
    )

    final_df = pd.merge(
        user_product_order_id_df,
        dataframes["reviews_df"][["order_id", "review_score"]],
        on=["order_id"],
        how="inner",
    ).drop(["customer_id", "order_id"], axis=1)

    return final_df


def filter_dataframe(df: pd.core.frame.DataFrame, item_number: int, user_number: int):
    df["count"] = df.groupby("product_id").transform("count")["customer_unique_id"]

    # fetch top items based on count
    product_id = (
        df.drop_duplicates("product_id")
        .sort_values("count", ascending=False)
        .iloc[:item_number]["product_id"]
    )

    # filter out data as per the product_id
    df = df[df["product_id"].isin(product_id)].reset_index(drop=True)

    # get total counts of no. of occurence of customer
    df["count"] = df.groupby("customer_unique_id").transform("count")["product_id"]

    # fetch top 1000 users based on count
    customer_id = (
        df.drop_duplicates("customer_unique_id")
        .sort_values("count", ascending=False)
        .iloc[:user_number]["customer_unique_id"]
    )

    df = df[df["customer_unique_id"].isin(customer_id)].reset_index(drop=True)

    return df
