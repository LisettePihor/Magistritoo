from sklearn.ensemble import RandomForestRegressor
from src.graafikud import ennustuste_graafik
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def tunnuste_olulisus(mudel, X_treening, X_test, y_test):
    count = 0
    parimad_tunnused = X_treening.columns.tolist()
    mse_parim = mean_squared_error(y_test, mudel.predict(X_test))
    uus_X_test = X_test.copy()
    uus_X_treening= X_treening.copy()
    while True:
        importances = pd.Series(mudel.feature_importances_, index=uus_X_treening.columns).sort_values(ascending=False)
        nr_eemaldada = int(importances.shape[0]*0.1)
        uued_tunnused = importances.index[:-nr_eemaldada].tolist()
        uus_X_treening = X_treening[uued_tunnused].copy()
        uus_X_test = X_test[uued_tunnused].copy()
        mse = mean_squared_error(y_test, mudel.predict(uus_X_test))
        if mse < mse_parim:
            parimad_tunnused = uued_tunnused
            mse_parim = mse
        else:
            if count >= 10 or len(parimad_tunnused) <= 10:
                break
            else:
                count += 1
    return parimad_tunnused

def otsustusmets(X_treening, y_treening, X_test, y_test, kombo_ja_jaotus):
    mudel = RandomForestRegressor(n_estimators=100, random_state=42)
    mudel.fit(X_treening, y_treening)
    ennustatud_treening = mudel.predict(X_treening)
    ennustatud_test = mudel.predict(X_test)

    mse_treening = mean_squared_error(y_treening, ennustatud_treening)
    mse = mean_squared_error(y_test, ennustatud_test)
    r2_treening = r2_score(y_treening, ennustatud_treening)
    r2 = r2_score(y_test, ennustatud_test)
    ennustuste_graafik(ennustatud_treening, y_treening, ennustatud_test, y_test,mse_treening, 
                       r2_treening, mse, r2, f"{kombo_ja_jaotus}_otsustusmets")
    print(f'Treening MSE: {mse_treening}')
    print(f'Test MSE: {mse}')

    print(f'Treening R^2: {r2_treening}')
    print(f'Test R^2: {r2}')    

    return None
