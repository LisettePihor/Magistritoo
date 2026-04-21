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

def feature_vs_feature_plots(X, file_name, selected_features=None):
    corr_df = feature_correlations(X, file_name, selected_features)
    plot_dir = f"{file_name}_scatter_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plotted_pairs = set()
    for target_col in corr_df.columns:
        high_corr_rows = corr_df[target_col][corr_df[target_col].abs() > 0.8]
        for feature_col in high_corr_rows.index:
            if feature_col != target_col and (feature_col, target_col) not in plotted_pairs:
                plt.figure(figsize=(6, 6))
                sns.scatterplot(x=X[feature_col], y=X[target_col], alpha=0.5)
                plt.xlabel(feature_col)
                plt.ylabel(target_col)
                plt.title(f'Scatter Plot: {target_col} vs {feature_col}')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{target_col}_vs_{feature_col}.png'))
                plt.close()
                plotted_pairs.add((target_col, feature_col))
    return None