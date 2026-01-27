from sklearn.ensemble import RandomForestRegressor
from src.andmete_tootlus import kombo_koos_tunnustega, jaota_andmestik
from src.graafikud import ennustuste_graafik
import pandas as pd
from src.mudeli_evalueerimine import r2_ja_mse, ennustuste_graafik

def tunnuste_olulisus(mudel, X_treening, y_treening, X_test, y_test):
    count = 0
    parimad_tunnused = X_treening.columns.tolist()
    _,_,mse_parim,_ = r2_ja_mse(mudel, X_treening, y_treening, X_test, y_test)
    uus_X_test = X_test.copy()
    uus_X_treening= X_treening.copy()
    while True:
        importances = pd.Series(mudel.feature_importances_, index=uus_X_treening.columns).sort_values(ascending=False)
        nr_eemaldada = int(importances.shape[0]*0.1)
        uued_tunnused = importances.index[:-nr_eemaldada].tolist()
        uus_X_treening = X_treening[uued_tunnused].copy()
        uus_X_test = X_test[uued_tunnused].copy()
        _,_,mse,_ = r2_ja_mse(mudel, uus_X_treening, y_treening, uus_X_test, y_test)
        if mse < mse_parim:
            parimad_tunnused = uued_tunnused
            mse_parim = mse
        else:
            if count >= 10 or len(parimad_tunnused) <= 10:
                break
            else:
                count += 1
    return parimad_tunnused

def otsustusmets(kombo_nr, jarjestatud=False, juhuarv=42, ristvalideerimine=False, tunnuste_valik=False):
    andmestik = kombo_koos_tunnustega(kombo_nr)
    mudel = RandomForestRegressor(n_estimators=100, random_state=42)
    if ristvalideerimine:
        X, y = jaota_andmestik(andmestik, jarjestatud, testita=True)
        mse_treening, r2_treening, mse, r2 = r2_ja_mse(mudel, X, y, ristvalideerimine=True)
    else:
        X, y, X_test, y_test = jaota_andmestik(andmestik, jarjestatud, juhuarv=juhuarv)
        if tunnuste_valik:
            parimad_tunnused = tunnuste_olulisus(mudel, X, y, X_test, y_test)
            X_parim = X[parimad_tunnused].copy()
            X_test_parim = X_test[parimad_tunnused].copy()
            print(f"Mudel parimate tunnustega ({len(parimad_tunnused)} tunnust):")
            mse_treening, r2_treening, mse, r2 = r2_ja_mse(mudel, X_parim, y, X_test_parim, y_test)
        else:
            mse_treening, r2_treening, mse, r2 = r2_ja_mse(mudel, X, y, X_test, y_test)
            mudel.fit(X, y)
            ennustus = mudel.predict(X_test)
            if jarjestatud: pealkiri = 'kombo_nr_' + kombo_nr + '_jarjestatud_otsustusmets'
            else: pealkiri = f'kombo_nr_{kombo_nr}_juhuarv_{juhuarv}_otsustusmets'
            ennustuste_graafik(ennustus, y_test, pealkiri, mse, r2)

    print(f'Treening MSE: {mse_treening}')
    print(f'Test MSE: {mse}')

    print(f'Treening R^2: {r2_treening}')
    print(f'Test R^2: {r2}')    

    return None
