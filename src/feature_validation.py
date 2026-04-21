import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def feature_correlations(X, file_name,selected_features=None):
    if os.path.exists(f'{file_name}.csv'):
        return pd.read_csv(f'{file_name}.csv', index_col=0)
    else:
        if selected_features is None:
            raise ValueError("Selected features must be provided if the correlation file does not exist.")
        else:
            corr = X.corr()
            corr_selected = corr[selected_features]
            for feature in selected_features:
                if feature in corr_selected.index:
                    corr_selected.at[feature, feature] = np.nan
            mask = corr_selected.abs().max(axis=1) > 0.8
            corr_filtered = corr_selected[mask]
            corr_filtered.to_csv(f'{file_name}.csv')
            plt.figure(figsize=(10, 14))
            ax = sns.heatmap(corr_filtered, 
                    annot=True,
                    fmt=".2f",
                    annot_kws={"size": 8},
                    cmap='coolwarm', 
                    center=0, 
                    linewidths=0.5,
                    cbar_kws={"shrink": .5},
                    yticklabels=True)
            plt.xticks(rotation=45, ha='right') 
            plt.yticks(rotation=0)              
                
            plt.title('High Correlation Features (|r| > 0.8)', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f'{file_name}.png')
            plt.close()
            return corr_filtered

def feature_vs_feature_plots(X, file_name):
    feature_correlations = feature_correlations(X, file_name)
