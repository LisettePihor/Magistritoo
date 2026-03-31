
import ast
import os
import random
import numpy as np
import torch
from andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega
from src.otsustusmets import otsustusmets, ennusta, optimeeri_mets
from src.graafikud import jaotus_hist
from src.otsustusmets_analuus import tunnus_vs_tunnus, tunnus_vs_y, influence_plot, shap_summary
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

