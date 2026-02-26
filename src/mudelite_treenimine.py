import joblib
from sklearn.ensemble import RandomForestRegressor
from src.graafikud import ennustuste_graafik, loo_ennustuste_notebook
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score
import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import optuna
from sklearn.preprocessing import StandardScaler
import math





def tunnuste_olulisus_otsustusmets(mudel, X_treening, y_treening, kombo_nr, jaotus):
    fail = os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo_nr}/mudelid/{jaotus}_otsustusmetsa_proovitud.csv')
    if os.path.exists(fail):
        with open(fail, 'r') as f:
            parimad_tunnused = pd.read_csv(fail)['tunnused'].sort_values(ascending=False).tolist()[0]
        return parimad_tunnused
    else:
        print("Parima otsustusmetsa leidmine, algseid tunnuseid on:", len(X_treening.columns))
        proovitud = []
        parimad_tunnused = X_treening.columns.tolist()
        mse_parim = round(cross_val_score(mudel, X_treening, y_treening, cv=5, scoring='neg_mean_squared_error').mean(),3)
        oob_parim = 0
        uus_X_treening= X_treening.copy()
        uued_tunnused = X_treening.columns.tolist()
        while True:
            mudel.fit(uus_X_treening, y_treening)
            mse = round(cross_val_score(mudel, uus_X_treening, y_treening, cv=5, scoring='neg_mean_squared_error').mean(),3)
            r2 = r2_score(y_treening, mudel.predict(uus_X_treening))
            oob = mudel.oob_score_
            proovitud.append((mse, r2, oob, len(uued_tunnused), uued_tunnused))
            if len(uued_tunnused) <= 10:
                if round(oob,1) > round(oob_parim,1):
                    parimad_tunnused = uued_tunnused
                    mse_parim = mse
                    oob_parim = oob
                    print(f'Parem OOB leitud: {oob_parim}, MSE: {mse_parim}, Tunnuseid: {len(parimad_tunnused)}')
                else:
                    if len(uued_tunnused) <= 2:
                        break
            importances = pd.Series(mudel.feature_importances_, index=uus_X_treening.columns).sort_values(ascending=False)
            nr_eemaldada = math.ceil(importances.shape[0]*0.1)
            uued_tunnused = importances.index[:-nr_eemaldada].tolist()
            uus_X_treening = X_treening[uued_tunnused].copy()
        proovitud_df = pd.DataFrame(proovitud, columns=['mse', 'r2', 'oob', 'tunnuste_arv', 'tunnused'])
        proovitud_df.to_csv(fail, index=False)
    return parimad_tunnused
    


def otsustusmets(X_treening_idga, y_treening_idga, X_test_idga, y_test_idga, kombo_nr, jaotus):
    print("-" * 30)
    print("OTSUSTUSPUU MUDEL:\n")
    fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}/mudelid/{jaotus}_otsustusmets.csv')
    mudeli_fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}/mudelid/{jaotus}_otsustusmets.joblib')
    if os.path.exists(fail):
        tulemused_df = pd.read_csv(fail)
    else:
        os.makedirs(f'andmed/kombo_nr_{kombo_nr}/mudelid', exist_ok=True)
        X_treening = X_treening_idga.drop('Molecule ChEMBL ID', axis=1)
        X_test = X_test_idga.drop('Molecule ChEMBL ID', axis=1)
        y_treening = y_treening_idga['pChEMBL Value']
        y_test = y_test_idga['pChEMBL Value']
        mudel = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        parimad_tunnused = tunnuste_olulisus_otsustusmets(mudel, X_treening, y_treening, kombo_nr, jaotus)
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
        tulemused_df = pd.DataFrame([tulemused], columns=['mse_treening', 'mse_test', 'oob', 'r2_treening', 'r2_test', 'tunnused'])
        tulemused_df.to_csv(fail, index=False)
        joblib.dump(mudel, mudeli_fail)

    print(f'Treening MSE: {tulemused_df["mse_treening"].iloc[0]}')
    print(f'Test MSE: {tulemused_df["mse_test"].iloc[0]}')
    print(f'Out of bag: {tulemused_df["oob"].iloc[0]}')

    print(f'Treening R^2: {tulemused_df["r2_treening"].iloc[0]}')
    print(f'Test R^2: {tulemused_df["r2_test"].iloc[0]}')  
    parimad_tunnused = tulemused_df["tunnused"].iloc[0]
    print(f'Parimaid tunnuseid: {len(parimad_tunnused)}, {parimad_tunnused}')  
    print("-" * 30)

    return parimad_tunnused

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Narvivork(nn.Module):
    def __init__(self, sisend_dim, konfiguratsioonid, dropout_rate=0.3):
        super(Narvivork, self).__init__()
        kihid = []
        sisend_tunnuseid = sisend_dim
        for neuronid, funktsioon in konfiguratsioonid:
            kihid.append(nn.Linear(sisend_tunnuseid, neuronid))
            if funktsioon == 'relu':
                kihid.append(nn.ReLU())
            elif funktsioon == 'sigmoid':
                kihid.append(nn.Sigmoid())
            elif funktsioon == 'tanh':
                kihid.append(nn.Tanh())
            else:
                raise ValueError(f"Tundmatu aktivatsiooni funktsioon: {funktsioon}")
            sisend_tunnuseid = neuronid
            kihid.append(nn.Dropout(dropout_rate))
        kihid.append(nn.Linear(sisend_tunnuseid, 1))
        self.model = nn.Sequential(*kihid)
    
    def forward(self, x):
        return self.model(x)

def objective(trial, X_tr, y_tr, X_val, y_val, masin):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pikkus = trial.suggest_int('pikkus', 1, 3)
    konfig = []
    for i in range(pikkus):
        n = trial.suggest_categorical(f'n_l{i}', [4, 8, 16, 32])
        f = trial.suggest_categorical(f'f_l{i}', ['relu', 'sigmoid', 'tanh'])
        konfig.append((n, f))
    
    mudel = Narvivork(X_tr.shape[1], konfig).to(masin)
    optimiseerija = optim.Adam(mudel.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiseerija, mode='min', factor=0.5, patience=20)
    kriteerium = nn.MSELoss()
    early_stopping = EarlyStopping(patience=20) 
    parim_loss = float('inf')
    ajalugu_mse_treening = []
    ajalugu_mse_val = []

    for epoch in range(500): 
        mudel.train()
        optimiseerija.zero_grad()
        loss = kriteerium(mudel(X_tr), y_tr)
        loss.backward()
        optimiseerija.step()
        loss_item = loss.item()

        if loss_item < parim_loss:
            parim_loss = loss_item
            parim_state = mudel.state_dict()

        mudel.eval()
        with torch.no_grad():
            v_loss = kriteerium(mudel(X_val), y_val).item()
            scheduler.step(v_loss)
            early_stopping(v_loss, mudel)
            if early_stopping.early_stop:
                break

        ajalugu_mse_treening.append(loss_item)
        ajalugu_mse_val.append(v_loss)
    trial.set_user_attr("ajalugu_mse_treening", ajalugu_mse_treening)
    trial.set_user_attr("ajalugu_mse_val", ajalugu_mse_val)
    trial.set_user_attr("parim_state", parim_state)
    del mudel
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return parim_loss

def narvivork(X_idga, y_idga, X_test_idga, y_test_idga, kombo_nr, jaotus):
    print("-" * 30)
    print(f"NÄRVIVÕRK (Bayes): Kombo {kombo_nr}, Jaotus {jaotus}\n")
    
    fail = os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo_nr}/mudelid/{jaotus}_narvivork.pth')
    X = X_idga.drop('Molecule ChEMBL ID', axis=1)
    X_test = X_test_idga.drop('Molecule ChEMBL ID', axis=1)
    
    val_indeksid = range(9, len(X), 10)
    treening_indeksid = [i for i in range(len(X)) if i not in val_indeksid]
    
    y_treening_idga = y_idga.iloc[treening_indeksid]
    y = y_idga['pChEMBL Value']
    y_test = y_test_idga['pChEMBL Value']

    skaleeria = StandardScaler()
    X_treening_sk = skaleeria.fit_transform(X.iloc[treening_indeksid].values)
    X_val_sk = skaleeria.transform(X.iloc[val_indeksid].values)
    X_test_sk = skaleeria.transform(X_test.values)

    X_treening_t = torch.tensor(X_treening_sk).float()
    y_treening = y.iloc[treening_indeksid].values
    y_treening_t = torch.tensor(y_treening).float().view(-1, 1)
    X_val_t = torch.tensor(X_val_sk).float()
    y_val = y.iloc[val_indeksid].values
    y_val_t = torch.tensor(y_val).float().view(-1, 1)
    X_test_t = torch.tensor(X_test_sk).float()

    if os.path.exists(fail):
        seisund = torch.load(fail)
        parim_mudel = Narvivork(seisund['sisend_dim'], seisund['konfig'])
        parim_mudel.load_state_dict(seisund['state'])
    else:
        os.makedirs(f'andmed/kombo_nr_{kombo_nr}/mudelid', exist_ok=True)
        os.makedirs(f'andmed/kombo_nr_{kombo_nr}/graafikud', exist_ok=True)
        
        masin = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        X_tr_m, y_tr_m = X_treening_t.to(masin), y_treening_t.to(masin)
        X_val_m, y_val_m = X_val_t.to(masin), y_val_t.to(masin)

        n_trials = 50
        pbar = tqdm(total=n_trials, desc="Optimeerimine", unit="katse")

        def pbar_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Parim MSE": f"{study.best_value:.4f}"})

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=optuna.pruners.MedianPruner())
        study.optimize(
            lambda trial: objective(trial, X_tr_m, y_tr_m, X_val_m, y_val_m, masin), 
            n_trials=n_trials,
            callbacks=[pbar_callback]
        )
        pbar.close()

        bp = study.best_params
        parim_konfig = []
        for i in range(bp['pikkus']):
            parim_konfig.append((bp[f'n_l{i}'], bp[f'f_l{i}']))
        print(f"\nParim struktuur: {parim_konfig}, Parim MSE: {study.best_value:.4f}\n")

        parim_mudel = Narvivork(X_treening_t.shape[1], parim_konfig).to(masin)
        parim_mudel.load_state_dict(study.best_trial.user_attrs["parim_state"])
        seisund = {
            'sisend_dim': X_treening_t.shape[1],
            'konfig': parim_konfig,
            'state': parim_mudel.state_dict()
        }
        torch.save(seisund, fail)
        
        ajalugu_mse_treening = study.best_trial.user_attrs["ajalugu_mse_treening"]
        ajalugu_mse_val = study.best_trial.user_attrs["ajalugu_mse_val"]
        plt.figure()
        plt.plot(ajalugu_mse_treening, label='Train')
        plt.plot(ajalugu_mse_val, label='Val')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), f"andmed/kombo_nr_{kombo_nr}/graafikud/narvivork_{jaotus}.png"))
        plt.close()

    parim_mudel.eval()
    with torch.no_grad():
        enn_tr = parim_mudel(X_treening_t).numpy().flatten()
        enn_test = parim_mudel(X_test_t).numpy().flatten()

    loo_ennustuste_notebook(enn_tr, y_treening_idga, enn_test, y_test_idga, f"kombo_nr_{kombo_nr}_{jaotus}_narvivork", kombo_nr)
    
    print(f'TREENING MSE: {mean_squared_error(y_treening, enn_tr):.4f}, TREENING R^2: {r2_score(y_treening, enn_tr):.4f}')
    print(f'VALIDEERIMISE MSE: {mean_squared_error(y_val, parim_mudel(X_val_t).detach().numpy().flatten()):.4f}, VALIDEERIMISE R^2: {r2_score(y_val, parim_mudel(X_val_t).detach().numpy().flatten()):.4f}')
    print(f"TEST MSE: {mean_squared_error(y_test, enn_test):.4f}, TEST R^2: {r2_score(y_test, enn_test):.4f}")
    print("-" * 30)

    return None