from sklearn.ensemble import RandomForestRegressor
from src.graafikud import ennustuste_graafik
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score

def tunnuste_olulisus(mudel, X_treening, y_treening):
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
    


def otsustusmets(andmestik, X_treening, y_treening, X_test, y_test, kombo_nr, jaotus):
    mudel = RandomForestRegressor(n_estimators=100, random_state=42)
    parimad_tunnused = tunnuste_olulisus(mudel, X_treening, y_treening)
    X_treening = X_treening[parimad_tunnused]
    X_test = X_test[parimad_tunnused]
    mudel.fit(X_treening, y_treening)

    ennustatud_treening = mudel.predict(X_treening)
    ennustatud_test = mudel.predict(X_test)

    jaagid_treening = y_treening - ennustatud_treening
    jaagid_test = y_test - ennustatud_test
    piir = 3 * np.std(jaagid_treening)
    outlier_indeksid_treening = y_treening.index[np.abs(jaagid_treening) > piir]
    outlier_indeksid_test = y_test.index[np.abs(jaagid_test) > piir]
    outlier_ids_treening = andmestik.loc[outlier_indeksid_treening, 'Molecule ChEMBL ID'].tolist()
    outlier_ids_test = andmestik.loc[outlier_indeksid_test, 'Molecule ChEMBL ID'].tolist()
    outlier_df = pd.DataFrame({
        'ChEMBL_ID': outlier_ids_treening + outlier_ids_test,
        'Set': ['Train'] * len(outlier_ids_treening) + ['Test'] * len(outlier_ids_test),
        'Residual': list(jaagid_treening[outlier_indeksid_treening]) + list(jaagid_test[outlier_indeksid_test])
    })
    outlier_df.to_csv(f'andmed/kombo_nr_{kombo_nr}/kombo_nr_{kombo_nr}_{jaotus}_outlierid.csv', index=False)

    mse_treening = mean_squared_error(y_treening, ennustatud_treening)
    mse = mean_squared_error(y_test, ennustatud_test)
    r2_treening = r2_score(y_treening, ennustatud_treening)
    r2 = r2_score(y_test, ennustatud_test)
    ennustuste_graafik(ennustatud_treening, y_treening, ennustatud_test, y_test,mse_treening, 
                       r2_treening, mse, r2, f"kombo_nr_{kombo_nr}_{jaotus}_otsustusmets", kombo_nr)
    print(f'Treening MSE: {mse_treening}')
    print(f'Test MSE: {mse}')

    print(f'Treening R^2: {r2_treening}')
    print(f'Test R^2: {r2}')    

    return None

def narvivork():
    return None

