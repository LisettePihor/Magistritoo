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

    X_train0, y_train0, X_test0, y_test0 = create_data(0)
    plot_dist(y_train0, y_test0, "random_combo_0")
    print('Cytotoxicity prediction with random split:')
    r2, mse = random_forest(X_train0, y_train0, X_test0, y_test0)

    X_train0_ordered, y_train0_ordered, X_test0_ordered, y_test0_ordered = create_data(0, ordered=True)
    plot_dist(y_train0_ordered, y_test0_ordered, "ordered_combo_0")
    print('Cytotoxicity prediction with ordered split:')
    r2_ordered, mse_ordered = random_forest(X_train0_ordered, y_train0_ordered, X_test0_ordered, y_test0_ordered)

    X_train1, y_train1, X_test1, y_test1 = create_data(1)
    plot_dist(y_train1, y_test1, "random_combo_1")
    print('Antiproliferative activity prediction with random split:')
    r2_1, mse_1 = random_forest(X_train1, y_train1, X_test1, y_test1)

    X_train1_ordered, y_train1_ordered, X_test1_ordered, y_test1_ordered = create_data(1, ordered=True)
    plot_dist(y_train1_ordered, y_test1_ordered, "ordered_combo_1")
    print('Antiproliferative activity prediction with ordered split:')
    r2_1_ordered, mse_1_ordered = random_forest(X_train1_ordered, y_train1_ordered, X_test1_ordered, y_test1_ordered)
    




if __name__ == "__main__":
    main()