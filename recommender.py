import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:
    def __init__(self, df, products_metadata, order_information, translate_dict):
        self.df = df
        self.products_metadata = products_metadata
        self.order_information = order_information
        self.df_imputed = self.df.fillna(self.df.mean(axis=0))
        self.similarity_matrix = cosine_similarity(self.df_imputed.values)
        self.translate_dict = translate_dict

    def get_recommendation(self, customer_idx, nr_of_items=2):
        # Saving the customer id for later refference
        customer_id = self.df.iloc[customer_idx].name

        # Retrieving similarity of current customer with other customers
        similarity_scores = list(enumerate(self.similarity_matrix[customer_idx]))

        # Getting the products that were not rated by the current customer
        unrated_products = self.df.iloc[customer_idx][
            self.df.iloc[customer_idx].isna()
        ].index

        # We're using the similarity scores as weights for the collaborative filtering
        weights = [x[1] for x in similarity_scores]

        # Calculating inferred ratings as such:
        # 1. Multiply scores of unrated products with similarity scores (weights)
        # 2. Get the mean of the result for each product
        # 3. Sort the values based on this mean
        product_ratings = (self.df[unrated_products].T * weights).T
        product_ratings = product_ratings.iloc[[x[0] for x in similarity_scores]].mean()
        recommendations = product_ratings.sort_values(ascending=False)[:nr_of_items]

        # Getting the  category name for each product
        recommendations_tmp = self.products_metadata[
            self.products_metadata["product_id"].isin(
                recommendations.reset_index().sort_values(0, ascending=False)[
                    "product_id"
                ]
            )
        ][["product_category_name", "product_id"]]

        # Using the translation dict to get english category names
        recommendations_tmp["product_category_name"] = recommendations_tmp[
            "product_category_name"
        ].map(self.translate_dict)

        # Placing our recommendations in a DataFrame so we can merge with other DataFrames
        recommendations_df = pd.DataFrame(
            {"product_id": recommendations.index, "score": recommendations.values}
        )

        # Merging recommendations with category nams
        recommendations_final = pd.merge(
            recommendations_tmp, recommendations_df, on="product_id", how="inner"
        )

        # # Adding each recommendet item's price
        # recommendations_final = (
        #     pd.merge(
        #         recommendations_final,
        #         self.order_information,
        #         on="product_id",
        #         how="inner",
        #     )
        #     .drop_duplicates(subset="product_id")[
        #         ["product_id", "product_category_name", "score", "price"]
        #     ]
        #     .sort_values(by="score", ascending=False)
        # )
        recommendations_final = (
            pd.merge(
                recommendations_final,
                self.order_information,
                on="product_id",
                how="inner",
            )[["product_id", "product_category_name", "score", "price"]]
            .groupby(["product_id", "product_category_name"])
            .mean()
            .reset_index()
            .sort_values(by="score", ascending=False)
        )

        return recommendations_final

    def get_bought_items(self, customer_idx, nr_of_items=2):
        rated_items_df = self.df.iloc[customer_idx][
            self.df.iloc[customer_idx].notnull()
        ].reset_index()
        rated_items_df.columns = ["product_id", "rating"]
        rated_items_df = rated_items_df.sort_values(by="rating", ascending=False)

        filtered_products = self.products_metadata[
            self.products_metadata["product_id"].isin(
                rated_items_df["product_id"].values
            )
        ][["product_id", "product_category_name"]]

        filtered_products["product_category_name"] = filtered_products[
            "product_category_name"
        ].map(self.translate_dict)

        rated_items_df = pd.merge(
            rated_items_df, filtered_products, on="product_id", how="inner"
        )
        rated_items_df = (
            pd.merge(
                rated_items_df,
                self.order_information[["product_id", "price"]],
                on="product_id",
                how="inner",
            )
            .drop_duplicates(subset=["product_id"], keep="first")
            .reset_index(drop=True)
        )

        return rated_items_df[:nr_of_items]


def get_translation_dict(cat_name_translation):
    portuguese_cat_names = cat_name_translation.to_dict()["product_category_name"]
    english_cat_names = cat_name_translation.to_dict()["product_category_name_english"]
    translate_dict = {}

    for p_key in portuguese_cat_names:
        if portuguese_cat_names[p_key] not in translate_dict:
            translate_dict[portuguese_cat_names[p_key]] = english_cat_names[p_key]

    return translate_dict
