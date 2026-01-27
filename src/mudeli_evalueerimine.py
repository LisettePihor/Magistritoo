from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
import os
import matplotlib.pyplot as plt

def r2_ja_mse(mudel, X_treening, y_treening, X_test=None, y_test=None, ristvalideerimine=False):
    if ristvalideerimine:
        scores = cross_validate(mudel, X_treening, y_treening, cv=10, scoring=('neg_mean_squared_error','r2'), return_train_score=True)
        r2_treening = scores["train_r2"].mean()
        mse_treening = -scores["train_neg_mean_squared_error"].mean()
        r2 = scores["test_r2"].mean()
        mse = -scores["test_neg_mean_squared_error"].mean()
        '''print(f'Treening MSE väärtused: {scores["train_neg_mean_squared_error"]}, keskmine: {mse_treening}')
        print(f'Test MSE väärtused: {scores["test_neg_mean_squared_error"]}, keskmine: {mse}')
        print(f'Treening R^2 väärtused: {scores["train_r2"]}, keskmine: {r2_treening}')
        print(f'Test R^2 väärtused: {scores["test_r2"]}, keskmine: {r2}')'''
    else:
        mudel.fit(X_treening, y_treening)
        ennustatud = mudel.predict(X_test)
        mse_treening = mean_squared_error(y_treening, mudel.predict(X_treening))
        r2_treening = r2_score(y_treening, mudel.predict(X_treening))
        mse = mean_squared_error(y_test, ennustatud)
        r2 = r2_score(y_test, ennustatud)
    return mse_treening, r2_treening, mse, r2

def ennustuste_graafik(ennustatud, tegelikud, pealkiri, mse, r2):
    output_file = os.path.join(os.getcwd(),"plots/ennustatud_" + pealkiri + ".png")
    if not os.path.exists(output_file):
        plt.scatter(tegelikud, ennustatud, alpha=0.6)
        plt.xlabel("Tegelikud pChEMBL väärtused")
        plt.ylabel("Ennustatud pChEMBL väärtused")
        plt.title(f"Ennustatud vs tegelikud pChEMBL väärtused {pealkiri}\nMSE: {mse:.2f}, R²: {r2:.2f}")
        plt.savefig(output_file)
        plt.clf()
        
    return None