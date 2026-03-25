
import ast
import os
import random
import numpy as np
import torch
from src.andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega
from src.otsustusmets import otsustusmets, ennusta, optimeeri_mets
from src.graafikud import jaotus_hist
from otsustusmets_analuus import ApplicabilityDomain, tunnus_vs_tunnus, tunnus_vs_y, influence_plot, shap_summary
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
parimad_tunnused = otsustusmets(X_treening, y_treening, X_test, y_test, 0, 'tunnused_optimeeritud', ['TPSA', 'SlogP_VSA8', 'BertzCT', 'VSA_EState4', 'AvgIpc'])

mets = joblib.load(os.path.join(os.getcwd(), f'andmed/kombo_nr_0/mudelid/jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets.joblib'))
parimad_tunnused = ['AvgIpc', 'TPSA', 'SMR_VSA3', 'SlogP_VSA8', 'SPS', 'VSA_EState1']

tunnus_vs_tunnus(X_treening, parimad_tunnused, 'andmed/kombo_nr_0/graafikud/jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets')
tunnus_vs_y(X_treening, y_treening, parimad_tunnused, 'andmed/kombo_nr_0/graafikud/jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets')
influence_plot(X_treening.drop('Molecule ChEMBL ID', axis=1),y_treening['pChEMBL Value'], 
               'andmed/kombo_nr_0/graafikud/jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets')
shap_summary(mets, X_treening[parimad_tunnused], 
             'andmed/kombo_nr_0/graafikud/jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets')

CNS_andmed = pd.read_csv('andmed/Molport_CNS_Focused_Library.csv', sep=',')
ad = ApplicabilityDomain()
ad.fit(X_treening[parimad_tunnused], y_treening['pChEMBL Value'], mets)
molekulide_ad = ad.check_ad(CNS_andmed['SMILES Canonical'], mets, parimad_tunnused, 
                            'andmed/kombo_nr_0/graafikud/jarjestatud_viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets')

ennustatud = ennusta(mets, CNS_andmed['SMILES Canonical'], parimad_tunnused, 0, 'jarjestatud', 'viimaste_tunnuste_eemaldamine_6_tunnust_otsustusmets')
