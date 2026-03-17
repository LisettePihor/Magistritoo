
import ast
import os
import random
import numpy as np
import torch
from src.andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega
from src.mudelite_treenimine import ennusta, otsustusmets, narvivork
from src.graafikud import jaotus_hist
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
from src.chembl_aktiivuse_andmed import loo_otsing

#cellosaurus database search -> chembl id cellosaurusest -> ChEMBL otsing -> andmed ainult vajalik -> 
#leia info kirjeldusest -> andmed duplikaatideta -> parimad kombinatsioonid -> kombo tunnustega

andmed_0 = kombo_koos_tunnustega(0)
X_treening, y_treening, X_test, y_test = jaota_andmestik(andmed_0, 0, jarjestatud=True)

'''parimad_tunnused = ['AvgIpc', 'TPSA', 'SMR_VSA3', 'SlogP_VSA8', 'SPS', 'VSA_EState1']
otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'jarjestatud')
otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'jarjestatud_7_tunnust')
otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'jarjestatud_viimaste_tunnuste_eemaldamine')
tunnused = otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust')
otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'jarjestatud_viimaste_tunnuste_eemaldamine_5_tunnust')
mets = joblib.load(os.path.join(os.getcwd(), 'andmed/kombo_nr_0/mudelid/jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets.joblib'))

CNS_andmed = pd.read_csv('andmed/Molport_CNS_Focused_Library.csv', sep=',')
ennustatud = ennusta(mets, CNS_andmed['SMILES Canonical'], parimad_tunnused, 0, 'jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust', 'otsustusmets')
#infuence plot
#correlation matrix
#feature vs pIC50
#SHAP
#applicability domain'''