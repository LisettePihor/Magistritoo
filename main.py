import os
import random
import numpy as np
import torch
from src.andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega, RDKit_tunnused
from src.mudelite_treenimine import otsustusmets, narvivork
import joblib
import pandas as pd

seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

andmed_0 = kombo_koos_tunnustega(0)
X_treening, y_treening, X_test, y_test = jaota_andmestik(andmed_0, 0, jarjestatud=True)
tunnused = otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'jarjestatud')

otsustusmets = joblib.load(f'andmed/kombo_nr_0/mudelid/jarjestatud_otsustusmets.joblib')
CNS_andmed = pd.read_csv('andmed/Molport_CNS_Focused_Library.csv', sep=',')
#CNS_tunnustega = RDKit_tunnused(CNS_andmed['SMILES Canonical'], 'andmed/Molport_CNS_Tunnustega.csv')
