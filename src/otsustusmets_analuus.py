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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class ApplicabilityDomain:
    def __init__(self):
        self.h_star = None
        self.X_train = None
        self.XTX_inv = None
        self.train_knn_dist = None
        self.train_vars = None
        self.nn = None

    def fit(self, X, y, rf_mudel):
        self.X_train = np.array(X)
        n, p = self.X_train.shape
        self.h_star = (3 * p) / n
        XTX = self.X_train.T @ self.X_train
        self.XTX_inv = np.linalg.pinv(XTX) 
        
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        self.nn.fit(self.X_train)
        distances, _ = self.nn.kneighbors(self.X_train)
        self.train_knn_dist_limit = np.percentile(distances[:, -1], 95) # 95th percentile as cutoff
        
        # 3. Ensemble Variance (Uncertainty)
        # Calculate variance across all trees for training data
        tree_preds = np.array([tree.predict(self.X_train) for tree in rf_mudel.estimators_])
        self.train_vars = np.var(tree_preds, axis=0)
        self.var_limit = np.percentile(self.train_vars, 95)

    def calculate_leverage(self, X_new):
        # h_i = x_i^T * (X^T * X)^-1 * x_i
        leverages = np.array([x @ self.XTX_inv @ x.T for x in X_new])
        return leverages

    def check_ad(self, X_new, rf_model):
        X_new = np.array(X_new)
        
        # Compute individual metrics
        leverages = self.calculate_leverage(X_new)
        knn_dist, _ = self.nn.kneighbors(X_new)
        knn_dist = knn_dist[:, -1]
        
        tree_preds = np.array([tree.predict(X_new) for tree in rf_model.estimators_])
        variances = np.var(tree_preds, axis=0)
        
        # Combine into flags (True = In Domain)
        lev_flag = leverages <= self.h_star
        knn_flag = knn_dist <= self.train_knn_dist_limit
        var_flag = variances <= self.var_limit
        
        # Final decision: In-Domain if all (or most) criteria are met
        combined_flag = lev_flag & knn_flag & var_flag
        
        results = pd.DataFrame({
            'leverage': leverages,
            'knn_dist': knn_dist,
            'variance': variances,
            'in_domain': combined_flag
        })
        
        return results