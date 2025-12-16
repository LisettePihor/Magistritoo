import pandas as pd
from src.get_activity_data_chembl import create_query
from src.data_analysis import get_important, unique_molecules
from src.info_from_description import extract_info_from_description
from src.data_preprocessing import create_data, plot_dist
from src.model_training import random_forest
import matplotlib.pyplot as plt

def main():
    create_query()

    df = get_important()
    df_new = extract_info_from_description(df)

    X_train1, y_train1, X_test1, y_test1 = create_data(0)
    test1 = create_data(0)
    plot_dist(y_train1, y_test1, "random")

    r2, mse = random_forest(X_train1, y_train1, X_test1, y_test1)

    X_train1_ordered, y_train1_ordered, X_test1_ordered, y_test1_ordered = create_data(0, ordered=True)
    plot_dist(y_train1_ordered, y_test1_ordered, "ordered")
    r2_ordered, mse_ordered = random_forest(X_train1_ordered, y_train1_ordered, X_test1_ordered, y_test1_ordered)
    #train2, test2 = create_data(1)
    




if __name__ == "__main__":
    main()