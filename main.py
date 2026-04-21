
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
from src.feature_validation import feature_correlations

#cellosaurus database search -> chembl id cellosaurusest -> ChEMBL otsing -> andmed ainult vajalik -> 
#leia info kirjeldusest -> andmed duplikaatideta -> parimad kombinatsioonid -> kombo tunnustega
andmed_0 = kombo_koos_tunnustega(0)
X_treening, y_treening, X_test, y_test = jaota_andmestik(andmed_0, 0, True, 'RDKit')

params = {'oob_score': True}
features = otsustusmets(X_treening, y_treening, X_test, y_test, 0, '14.04_optimeerimata', 'RDKit',params)
'''mudel = joblib.load(os.path.join(os.getcwd(), f'andmed/kombo_nr_0/RDKit/mudelid/14.04_optimeerimata_otsustusmets.joblib'))
CNS_andmed = pd.read_csv(os.path.join(os.getcwd(), 'andmed/Molport_CNS_Focused_Library.csv'))
ennustatud = ennusta(mudel,CNS_andmed['SMILES Canonical'], tunnused, 0, '14.04_optimeerimata_RDKit')'''
wd = os.getcwd()
X_train_clean = X_treening.drop(columns=['Molecule ChEMBL ID'])
feature_correlations(X_train_clean, features, os.path.join(wd, 'andmed/kombo_nr_0/RDKit/graafikud/feature_correlations'))
