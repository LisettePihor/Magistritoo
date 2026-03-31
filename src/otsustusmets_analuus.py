import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



def tunnus_vs_tunnus(X, tunnused, fail):
    graafik_fail = os.path.join(os.getcwd(), f'{fail}_tunnused_vs_tunnused.png')
    korrellatsioon_fail = os.path.join(os.getcwd(), f'{fail}_korrellatsioon.csv')
    if os.path.exists(graafik_fail) and os.path.exists(korrellatsioon_fail):
        return  
    
    plt.figure(figsize=(12, 12))
    pair_plot = sns.pairplot(X[tunnused], diag_kind='kde', plot_kws={'alpha': 0.6})
    pair_plot.figure.suptitle("Tunnuste korrellatsioonid", y=1.02)
    pair_plot.savefig(graafik_fail, dpi=300)
    plt.close()

    corr_matrix = X[tunnused].corr()
    corr_matrix.to_csv(korrellatsioon_fail)
    mask = np.eye(len(corr_matrix), dtype=bool) 

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Tunnuste korrellatsioonid")
    plt.savefig(os.path.join(os.getcwd(), f'{fail}_korrellatsioon.png'), dpi=300)
    plt.close()


def tunnus_vs_y(X, y, tunnused, fail):
    fail = os.path.join(os.getcwd(), f'{fail}_tunnused_vs_y.png')
    if os.path.exists(fail):
        return  
    andmestik = pd.DataFrame(X, columns=tunnused)
    andmestik['pIC50'] = y['pChEMBL Value']
    nr_tunnused = len(tunnused)
    fig, axes = plt.subplots(nrows=(nr_tunnused // 3) + 1, ncols=3, figsize=(15, nr_tunnused * 1.5))
    axes = axes.flatten()
    
    for i, col in enumerate(tunnused):
        sns.regplot(data=andmestik, x=col, y='pIC50', ax=axes[i], scatter_kws={'alpha':0.5})
        axes[i].set_title(f'{col} vs pIC50')
    
    plt.tight_layout()
    plt.savefig(fail, dpi=300)
    plt.close()

def influence_plot(X, y, fail):
    fail = os.path.join(os.getcwd(), f'{fail}_influence_plot.png')
    if os.path.exists(fail):
        return  
    X = X.set_index('Molecule ChEMBL ID')
    y.index = X.index
    X_with_const = sm.add_constant(X)
    results = sm.OLS(y, X_with_const).fit()
    fig, ax = plt.subplots(figsize=(12, 8))
    sm.graphics.influence_plot(results, ax=ax, criterion="cooks")
    plt.tight_layout()
    plt.savefig(fail, dpi=300)
    plt.close()


def shap_summary(mudel, X, fail):
    fail = os.path.join(os.getcwd(), f'{fail}_shap_summary.png')
    if os.path.exists(fail):
        return
    explainer = shap.Explainer(mudel, seed=42)
    shap_values = explainer(X)
    
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(fail, dpi=300)
    plt.close()

def applicability_domain(mudel, X_treening, ennustatud):
    ad_sees = []
    
    return ad_sees