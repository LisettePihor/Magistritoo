from sklearn.ensemble import RandomForestRegressor
from src.graafikud import ennustuste_graafik, loo_ennustuste_notebook
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score
import os

def tunnuste_olulisus_otsustusmets(mudel, X_treening, y_treening):
    count = 0
    parimad_tunnused = X_treening.columns.tolist()
    mse_parim = round(cross_val_score(mudel, X_treening, y_treening, cv=5, scoring='neg_mean_squared_error').mean(),3)
    uus_X_treening= X_treening.copy()
    while True:
        mudel.fit(uus_X_treening, y_treening)
        importances = pd.Series(mudel.feature_importances_, index=uus_X_treening.columns).sort_values(ascending=False)
        nr_eemaldada = int(importances.shape[0]*0.1)
        uued_tunnused = importances.index[:-nr_eemaldada].tolist()
        uus_X_treening = X_treening[uued_tunnused].copy()
        mse = round(cross_val_score(mudel, uus_X_treening, y_treening, cv=5, scoring='neg_mean_squared_error').mean(),3)
        if mse > mse_parim:
            parimad_tunnused = uued_tunnused
            mse_parim = mse
            print(f'Parem MSE leitud: {mse_parim}, Tunnuseid: {len(parimad_tunnused)}')
            count = 0
        else:
            if count >= 15 or len(parimad_tunnused) <= 5:
                break
            else:
                count += 1
                print(f'MSE: {mse}, Tunnuseid: {len(uued_tunnused)}')
    return parimad_tunnused
    


def otsustusmets(X_treening_idga, y_treening_idga, X_test_idga, y_test_idga, kombo_nr, jaotus):
    fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}/kombo_nr_{kombo_nr}_{jaotus}_otsustusmets.csv')
    if os.path.exists(fail):
        with open(fail, 'r') as f:
            tulemused = f.readlines()
    else:
        X_treening = X_treening_idga.drop('Molecule ChEMBL ID', axis=1)
        X_test = X_test_idga.drop('Molecule ChEMBL ID', axis=1)
        y_treening = y_treening_idga['pChEMBL Value']
        y_test = y_test_idga['pChEMBL Value']
        mudel = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        parimad_tunnused = tunnuste_olulisus_otsustusmets(mudel, X_treening, y_treening)
        X_treening = X_treening[parimad_tunnused]
        X_test = X_test[parimad_tunnused]
        y_treening = y_treening_idga['pChEMBL Value']
        y_test = y_test_idga['pChEMBL Value']
        mudel.fit(X_treening, y_treening)

        ennustatud_treening = mudel.predict(X_treening)
        ennustatud_test = mudel.predict(X_test)

        mse_treening = mean_squared_error(y_treening, ennustatud_treening)
        mse = mean_squared_error(y_test, ennustatud_test)
        r2_treening = r2_score(y_treening, ennustatud_treening)
        r2 = r2_score(y_test, ennustatud_test)
        ennustuste_graafik(ennustatud_treening, y_treening, ennustatud_test, y_test,mse_treening, 
                        r2_treening, mse, r2, f"kombo_nr_{kombo_nr}_{jaotus}_otsustusmets", kombo_nr)
        
        loo_ennustuste_notebook(ennustatud_treening, y_treening_idga, ennustatud_test, y_test_idga, f"kombo_nr_{kombo_nr}_{jaotus}_otsustusmets", kombo_nr)
        oob_skoor = mudel.oob_score_
        tulemused = [mse_treening, mse,oob_skoor,r2_treening, r2, parimad_tunnused]
        with open(fail, 'w') as f:
            for item in tulemused:
                f.write(f"{item}\n")

    print(f'Treening MSE: {tulemused[0]}')
    print(f'Test MSE: {tulemused[1]}')
    print(f'Out of bag: {tulemused[2]}')

    print(f'Treening R^2: {tulemused[3]}')
    print(f'Test R^2: {tulemused[4]}')    

    return None

def narvivork():
    return None

