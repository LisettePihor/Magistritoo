
import ast
import os
import random
import numpy as np
import torch
from src.andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega
from src.otsustusmets import otsustusmets, ennusta, optimeeri_mets
from src.graafikud import jaotus_hist
from src.otsustusmets_analuus import tunnus_vs_tunnus, tunnus_vs_y, influence_plot, shap_summary
import joblib
import pandas as pd

seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
'''torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)'''
from src.chembl_aktiivuse_andmed import loo_otsing

#cellosaurus database search -> chembl id cellosaurusest -> ChEMBL otsing -> andmed ainult vajalik -> 
#leia info kirjeldusest -> andmed duplikaatideta -> parimad kombinatsioonid -> kombo tunnustega
andmed_0 = kombo_koos_tunnustega(0)
X = pd.read_csv('andmed/kombo_nr_0/Dragon/dragon.csv', sep='\t')
y_ja_muu = andmed_0[['pChEMBL Value','Molecule ChEMBL ID','Smiles','Molecule Name']]
X.set_index('NAME', inplace=True)
y_ja_muu.set_index('Molecule ChEMBL ID', inplace=True)
eemaldatavad = ['CHEMBL4438947', 'CHEMBL4513992', 'CHEMBL4877449', 'CHEMBL1256616']
X.drop(index=eemaldatavad, inplace=True)
X.drop(columns=['No.'], inplace=True)
X.dropna(axis=1, inplace=True)
y_ja_muu.drop(index=eemaldatavad, inplace=True)
andmestik_alg = pd.concat([X, y_ja_muu], axis=1)
andmestik = andmestik_alg.copy()
andmestik.reset_index(inplace=True)
andmestik.rename(columns={'index':'Molecule ChEMBL ID'}, inplace=True)

X_treening, y_treening, X_test, y_test = jaota_andmestik(andmestik, 0, True, 'Dragon')
#params = optimeeri_mets(X_treening.drop('Molecule ChEMBL ID', axis=1), y_treening['pChEMBL Value'], 0, 'jarjestatud', 'Dragon')
params = {'oob_score': True, 'random_state': 42}
tunnused = otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'jarjestatud_optimeerimata', 'Dragon',params)