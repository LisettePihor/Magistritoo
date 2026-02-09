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
import itertools



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
    return parimad_tunnused
    


def otsustusmets(X_treening_idga, y_treening_idga, X_test_idga, y_test_idga, kombo_nr, jaotus):
    print("-" * 30)
    print("OTSUSTUSPUU MUDEL:\n")
    fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}/mudelid/{jaotus}_otsustusmets.csv')
    if os.path.exists(fail):
        with open(fail, 'r') as f:
            tulemused = f.readlines()
    else:
        os.makedirs(f'andmed/kombo_nr_{kombo_nr}/mudelid', exist_ok=True)
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
                f.write(f"{item}")

    print(f'Treening MSE: {tulemused[0]}')
    print(f'Test MSE: {tulemused[1]}')
    print(f'Out of bag: {tulemused[2]}')

    print(f'Treening R^2: {tulemused[3]}')
    print(f'Test R^2: {tulemused[4]}')    
    print("-" * 30)
    if isinstance(tulemused[5], list):
        parimad_tunnused = tulemused[5]
    else:
        parimad_tunnused = ast.literal_eval(tulemused[5])

    return parimad_tunnused



def narvivork(X_idga, y_idga, X_test_idga, y_test_idga, kombo_nr, jaotus):
    print("-" * 30)
    print("NÄRVIVÕRGU MUDEL:\n")
    fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}/mudelid/{jaotus}_narvivork.pth')
    X = X_idga.drop('Molecule ChEMBL ID', axis=1)
    X_test = X_test_idga.drop('Molecule ChEMBL ID', axis=1)
    val_indeksid = range(9, len(X), 10)
    treening_indeksid = [i for i in range(len(X)) if i not in val_indeksid]
    y_treening_idga = y_idga.iloc[treening_indeksid]
    y = y_idga['pChEMBL Value']
    y_test = y_test_idga['pChEMBL Value']

    X_treening_t = torch.tensor(X.iloc[treening_indeksid].values).float()
    y_treening = y.iloc[treening_indeksid].values
    y_treening_t = torch.tensor(y_treening).float().view(-1, 1)
    X_val_t = torch.tensor(X.iloc[val_indeksid].values).float()
    y_val = y.iloc[val_indeksid].values
    y_val_t = torch.tensor(y_val).float().view(-1, 1)
    X_test_t = torch.tensor(X_test.values).float()


    if os.path.exists(fail):
        seisund = torch.load(fail)
        parim_mudel = Narvivork(seisund['sisend_dim'], seisund['konfig'])
        parim_mudel.load_state_dict(seisund['state'])
        parim_mudel.eval()
    else:
        os.makedirs(f'andmed/kombo_nr_{kombo_nr}/mudelid', exist_ok=True)
        arhitektuurid = []
        neuronid = [4,8,16,32,64]
        pikkused = [1,2,3,4,5]
        funktsioonid = ['relu', 'sigmoid','tanh']
        kihi_valikud = list(itertools.product(neuronid, funktsioonid))
        for p in pikkused:
            kombod = list(itertools.product(kihi_valikud, repeat=p))
            arhitektuurid.extend(kombod)
        parim_val_loss = float('inf')
        parim_tulemus = {}

        print("Alustan parameetrite otsingut...")
        masin = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f'Kasutan {masin} masinat\n')

        for kihid in arhitektuurid:
            algne_lr = 0.01
            mudel = Narvivork(X.shape[1], kihid).to(masin)
            optimiseerija = optim.Adam(mudel.parameters(), lr=algne_lr)
            kriteerium = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiseerija, mode='min', factor=0.5, patience=20)
            early_stopping = EarlyStopping(patience=40)

            ajalugu_mse_treening, ajalugu_mse_val, lrs = [], [], []

            for epoch in range(500):
                mudel.train()
                optimiseerija.zero_grad()
                loss = kriteerium(mudel(X_treening_t), y_treening_t)
                loss.backward()
                optimiseerija.step()
                mudel.eval()
                with torch.no_grad():
                    t_pred = mudel(X_val_t)
                    t_loss = kriteerium(t_pred, y_val_t)
                    
                    scheduler.step(t_loss)
                    early_stopping(t_loss)
                    
                    ajalugu_mse_treening.append(loss.item())
                    ajalugu_mse_val.append(t_loss.item())
                    lrs.append(optimiseerija.param_groups[0]['lr'])
                if early_stopping.early_stop:
                    break
            
            hetkel_val_loss = min(ajalugu_mse_val)
            if hetkel_val_loss < parim_val_loss:
                parim_val_loss = hetkel_val_loss
                parim_tulemus = {
                    'state': mudel.state_dict(),
                    'konfig': kihid,
                    'h_train': ajalugu_mse_treening,
                    'h_val': ajalugu_mse_val
                }
            
        seisund = {
            'sisend_dim': X_treening_t.shape[1],
            'konfig': parim_tulemus['konfig'],
            'state': parim_tulemus['state']
        }
        torch.save(seisund, fail)
        print(f"Uus parim mudel salvestatud faili.")

        
        parim_mudel = Narvivork(seisund['sisend_dim'], seisund['konfig'])
        parim_mudel.load_state_dict(seisund['state'])
        parim_mudel.eval()
        
        plt.plot(parim_tulemus['h_train'], label='Train Loss')
        plt.plot(parim_tulemus['h_val'], label='Val Loss')
        plt.yscale('log')
        plt.legend()
        #plt.show()

        print("-" * 30)
        print(f"PARIM STRUKTUUR: {parim_tulemus['konfig']}")
        print("-" * 30)

    with torch.no_grad():
        ennustatud_treening = parim_mudel(X_treening_t).numpy().flatten()
        ennustatud_val = parim_mudel(X_val_t).numpy().flatten()
        ennustatud_test = parim_mudel(X_test_t).numpy().flatten()

    loo_ennustuste_notebook(ennustatud_treening, y_treening_idga, ennustatud_test, y_test_idga, f"kombo_nr_{kombo_nr}_{jaotus}_narvivork", kombo_nr)
    print(f"TRAIN: R2 = {r2_score(y_treening, ennustatud_treening):.4f}, MSE = {mean_squared_error(y_treening, ennustatud_treening):.4f}")
    print(f"VAL:   R2 = {r2_score(y_val, ennustatud_val):.4f}, MSE = {mean_squared_error(y_val, ennustatud_val):.4f}")
    print(f"TEST:  R2 = {r2_score(y_test, ennustatud_test):.4f}, MSE = {mean_squared_error(y_test, ennustatud_test):.4f}")
    print("-" * 30)

    return None

class Narvivork(nn.Module):
    def __init__(self, sisend_dim, konfiguratsioonid):
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
            
        kihid.append(nn.Linear(sisend_tunnuseid, 1))
        self.model = nn.Sequential(*kihid)
    
    def forward(self, x):
        return self.model(x)
    
class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
