import os
import random
import numpy as np
import torch
from src.andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega
from src.mudelite_treenimine import otsustusmets, narvivork

def main():
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
    tunnused.append('Molecule ChEMBL ID')
    narvivork(X_treening[tunnused], y_treening, X_test[tunnused], y_test, 0, 'jarjestatud')

    andmed_1 = kombo_koos_tunnustega(1)
    X_treening, y_treening, X_test, y_test = jaota_andmestik(andmed_1, 1, jarjestatud=True)
    tunnused = otsustusmets(X_treening, y_treening, X_test, y_test, 1, 'jarjestatud')
    tunnused.append('Molecule ChEMBL ID')
    narvivork(X_treening[tunnused], y_treening, X_test[tunnused], y_test, 1, 'jarjestatud')

    

if __name__ == "__main__":
    main()