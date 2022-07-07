# Olist Website Recommendation Engine

This repository contains a **collabrative filtering** implementation created for the [Olist Brazilian E-Comerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Structure

### Notebooks

* `collaborative_filtering.ipynb` - contains examples on how the collaborative fitlering works and it's results
* `filtering_accuracy.ipynb` - shows a comparison between using naive clustering and not using any clustering at all.
* `profit_improvement.ipynb` - a short simulation that can be changed in order to showcase the improvements a simple recommender system can bring
* `dataframes_examples.ipynb` - contains examples of how some of the dataframes look, including the pivoted dataframe that we use for collaborative filtering

### Scripts

* `recommendery.py` - contains the class in which everything related to collaborative filtering is implemented
* `utils.py` - various utility and helper functions (e.g. reading csv into dataframes, dataframe preprocessing)

## Implementation Details

### Dataset

The Olist dataset is spread across various `.csv` files that can be cross-refference using unique identifiers such as `order_id`, `product_id`, `customer_id` and others.

<br>
<p align="center"><b><i>Olist Dataset Structure</i></b></p>
<p align="center"><img src="images/OlistDatasetStructure.png" width="800"></p>
<br>

### Dataset Filtering

Due to the size of the dataset it had to be reduced in order to prevent memory issues.
I decided that the best course of action is to select the top most bought products and top most active customers.

The resulting *user-item matrix* is still sparse enough, but the dataset is now easier to work with for demonstrative purposes.

### Recommendations via Collaborative Filtering

This specific implementation of the collaborative filtering uses *cosine similarity* as a similarity metric.

The similarity is calculated between users (User-based collaborative filtering) meaning that we look at each user's rated items and see how similar they are to all the other users in terms of how they rated the same item.
