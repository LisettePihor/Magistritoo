import itertools
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from src.graafikud import ennustuste_graafik, loo_ennustuste_notebook
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score
import os
import gc
import math
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors


def optimeeri_mets(X,y, kombo_nr, jaotus, tunnuste_algo):
    fail = os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo_nr}/{tunnuste_algo}/mudelid/{jaotus}_otsustusmetsa_parameetrid.csv')
    if os.path.exists(fail):
        params_df = pd.read_csv(fail)
        params_df = params_df.iloc[0].to_dict()
        for key, value in params_df.items():
            if pd.isna(value):
                params_df[key] = None
            elif isinstance(value, float) and value.is_integer():
                params_df[key] = int(value)
        params_df.pop('Unnamed: 0', None)
        print(params_df)
    else:
        parameetrid = {'n_estimators': [100, 300, 500, 600],
                    'max_depth': [None, 10, 20, 30, 50], 
                    'max_features': ['sqrt', 'log2', 0.5, 0.8],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'oob_score': [True],
                    'n_jobs': [-1],
                    'random_state': [42]}
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf, param_grid=parameetrid, scoring='neg_mean_squared_error', verbose=1, cv=5)
        grid_search.fit(X, y)
        params = grid_search.best_params_
        params_df = pd.DataFrame([params])
        params_df.to_csv(fail)
    
    return params_df

def ennusta(mudel, smiles, parimad_tunnused, kombo, jaotus,mudeli_tuup):
    fail = os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo}/{jaotus}_{mudeli_tuup}_ennustatud_smiles.csv')
    if os.path.exists(fail):
        return pd.read_csv(fail)
    else:
        andmestik = []
        kalkulaator = MoleculeDescriptors.MolecularDescriptorCalculator(parimad_tunnused)

        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                tunnused = list(kalkulaator.CalcDescriptors(mol))
                tunnused_df = pd.DataFrame([tunnused], columns=parimad_tunnused)
                ennustus = mudel.predict(tunnused_df)
                andmestik.append({'SMILES':smile, 'Ennustus':ennustus[0]})
                del mol
                del tunnused
                del tunnused_df
                del ennustus
            else:
                print(f"Vigane SMILES: {smile}")
                andmestik.append({'SMILES':smile, 'Ennustus':None})
            gc.collect()
        andmestik = pd.DataFrame(andmestik)
        andmestik.sort_values(by='Ennustus', ascending=False, inplace=True)
        andmestik.to_csv(os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo}/{mudeli_tuup}_{jaotus}_ennustatud_smiles.csv'), index=False)
    return andmestik

def viimaste_tunnuste_eemaldamine(X, y, tunnused, kombo_nr, jaotus, tunnuste_algo, params):
    X = X[tunnused].copy()
    tulemused = {}
    mets = RandomForestRegressor(**params)
    mets.fit(X, y)
    mse = round(cross_val_score(mets, X, y, cv=5, scoring='neg_mean_squared_error').mean(), 2)
    tulemused[tuple(tunnused)] = (len(tunnused), mse, mets.oob_score_)
    kombinatsioonid = []
    for i in range(1, len(tunnused)):
        kombinatsioonid.extend(itertools.combinations(tunnused, i))
    for kombinatsioon in kombinatsioonid:
        kombinatsioon = list(kombinatsioon)
        X_kombinatsioon = X.drop(columns=kombinatsioon).copy()
        mets = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        mets.fit(X_kombinatsioon, y)
        mse = round(cross_val_score(mets, X_kombinatsioon, y, cv=5, scoring='neg_mean_squared_error').mean(), 3)
        tulemused[tuple(X_kombinatsioon.columns.tolist())] = (len(X_kombinatsioon.columns),mse, round(mets.oob_score_, 3))
        del mets
        gc.collect()
    tulemused_df = pd.DataFrame([(v[0], v[1], v[2]) for k, v in tulemused.items()], columns=['nr_tunnused', 'mse', 'oob'], index=[str(k) for k in tulemused.keys()])
    tulemused_df = tulemused_df.sort_values(by=['oob','mse', 'nr_tunnused'], ascending=[False,False,True])
    tulemused_df.to_csv(os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo_nr}/{tunnuste_algo}/mudelid/{jaotus}_viimaste_tunnuste_eemaldamine.csv'))
    parim_kombinatsioon = eval(tulemused_df.index[0])
    print(f"Parim kombinatsioon: {parim_kombinatsioon}, MSE: {tulemused[parim_kombinatsioon][1]}, OOB: {tulemused[parim_kombinatsioon][2]}")
    return list(parim_kombinatsioon)


def eemalda_kollineaarsed_tunnused(X, k=0.95):
    korrelatsioonid = X.corr().abs()
    upper = korrelatsioonid.where(np.triu(np.ones(korrelatsioonid.shape), k=1).astype(bool))
    eemaldatavad = [col for col in upper.columns if any(upper[col] > k)]
    return X.drop(columns=eemaldatavad), eemaldatavad

def tunnuste_olulisus_otsustusmets(mudel, X_treening, y_treening, kombo_nr, jaotus, tunnuste_algo, params):
    fail = os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo_nr}/{tunnuste_algo}/mudelid/{jaotus}_otsustusmetsa_proovitud.csv')
    if os.path.exists(fail):
        with open(fail, 'r') as f:
            parimad_tunnused = pd.read_csv(fail)['tunnused'].sort_values(ascending=False).tolist()[0]
        return parimad_tunnused
    else:
        print("Parima otsustusmetsa leidmine, algseid tunnuseid on:", len(X_treening.columns))
        #X_treening, eemaldatud = eemalda_kollineaarsed_tunnused(X_treening)
        #print("Eemaldatud tunnused:", len(eemaldatud), ",", eemaldatud)
        proovitud = []
        parimad_tunnused = X_treening.columns.tolist()
        mse_parim = float('inf')
        oob_parim = 0
        uus_X_treening= X_treening.copy()
        uued_tunnused = X_treening.columns.tolist()
        while True:
            mudel.fit(uus_X_treening, y_treening)
            mse = round(cross_val_score(mudel, uus_X_treening, y_treening, cv=5, scoring='neg_mean_squared_error').mean(),3)
            r2 = r2_score(y_treening, mudel.predict(uus_X_treening))
            oob = round(mudel.oob_score_,3)
            proovitud.append((mse, r2, oob, len(uued_tunnused), uued_tunnused))
            if len(uued_tunnused) <= 10:
                if oob > oob_parim:
                    parimad_tunnused = uued_tunnused
                    mse_parim = mse
                    oob_parim = oob
                    print(f'Parem leitud, OOB: {oob_parim}, MSE: {mse_parim}, Tunnuseid: {len(parimad_tunnused)}')
                else:
                    if len(uued_tunnused) <= 2:
                        break
            importances = pd.Series(mudel.feature_importances_, index=uus_X_treening.columns).sort_values(ascending=False)
            nr_eemaldada = math.ceil(importances.shape[0]*0.1)
            uued_tunnused = importances.index[:-nr_eemaldada].tolist()
            uus_X_treening = X_treening[uued_tunnused].copy()
        if len(parimad_tunnused) <= 10:
            parimad_tunnused = viimaste_tunnuste_eemaldamine(X_treening, y_treening, parimad_tunnused, kombo_nr, jaotus, tunnuste_algo,params)
        proovitud_df = pd.DataFrame(proovitud, columns=['mse', 'r2', 'oob', 'tunnuste_arv', 'tunnused'])
        proovitud_df.to_csv(fail, index=False)
    return parimad_tunnused
    

def otsustusmets(X_treening_idga, y_treening_idga, X_test_idga, y_test_idga, kombo_nr, 
                 jaotus, tunnuste_algo, params, tunnused=None):
    print("-" * 30)
    print("OTSUSTUSMETSA MUDEL:\n")
    fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}/{tunnuste_algo}/mudelid/{jaotus}_otsustusmets.csv')
    mudeli_fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}/{tunnuste_algo}/mudelid/{jaotus}_otsustusmets.joblib')
    if os.path.exists(fail):
        tulemused_df = pd.read_csv(fail)
    else:
        os.makedirs(f'andmed/kombo_nr_{kombo_nr}/{tunnuste_algo}/mudelid', exist_ok=True)
        X_treening = X_treening_idga.drop('Molecule ChEMBL ID', axis=1)
        X_test = X_test_idga.drop('Molecule ChEMBL ID', axis=1)
        y_treening = y_treening_idga['pChEMBL Value']
        y_test = y_test_idga['pChEMBL Value']
        mudel = RandomForestRegressor(**params)
        if tunnused is None:
            parimad_tunnused = tunnuste_olulisus_otsustusmets(mudel, X_treening, y_treening, kombo_nr, jaotus, tunnuste_algo, params)
        else:
            parimad_tunnused = tunnused
        if isinstance(parimad_tunnused, str):
            parimad_tunnused = parimad_tunnused.replace("[", "").replace("]", "").replace("'", "").split(", ")
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
                        r2_treening, mse, r2, f"kombo_nr_{kombo_nr}_{jaotus}_otsustusmets", kombo_nr, tunnuste_algo)
        
        loo_ennustuste_notebook(ennustatud_treening, y_treening_idga, ennustatud_test, y_test_idga, 
                                f"kombo_nr_{kombo_nr}_{jaotus}_otsustusmets", kombo_nr, tunnuste_algo)
        oob_skoor = mudel.oob_score_
        tulemused = [mse_treening, mse,oob_skoor,r2_treening, r2, parimad_tunnused]
        tulemused_df = pd.DataFrame([tulemused], columns=['mse_treening', 'mse_test', 'oob', 'r2_treening', 'r2_test', 'tunnused'])
        tulemused_df.to_csv(fail, index=False)
        joblib.dump(mudel, mudeli_fail)

    print(f'Treening MSE: {tulemused_df["mse_treening"].iloc[0]}')
    print(f'Test MSE: {tulemused_df["mse_test"].iloc[0]}')
    print(f'Out of bag: {tulemused_df["oob"].iloc[0]}')

    print(f'Treening R^2: {tulemused_df["r2_treening"].iloc[0]}')
    print(f'Test R^2: {tulemused_df["r2_test"].iloc[0]}')  
    parimad_tunnused = tulemused_df["tunnused"].iloc[0]
    if isinstance(parimad_tunnused, str):
            parimad_tunnused = parimad_tunnused.replace("[", "").replace("]", "").replace("'", "").split(", ")
    print(f'Parimaid tunnuseid: {len(parimad_tunnused)}, {parimad_tunnused}')  
    print("-" * 30)

    return parimad_tunnused