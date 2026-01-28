from src.andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega
from src.mudelite_treenimine import otsustusmets

def main():

    andmed_0 = kombo_koos_tunnustega(0)
    X_treening, y_treening, X_test, y_test = jaota_andmestik(andmed_0, 0, jarjestatud=True)
    otsustusmets(andmed_0, X_treening, y_treening, X_test, y_test, "kombo_0_jarjestatud")

    andmed_1 = kombo_koos_tunnustega(1)
    X_treening, y_treening, X_test, y_test = jaota_andmestik(andmed_1, 1, jarjestatud=True)
    otsustusmets(andmed_1, X_treening, y_treening, X_test, y_test, "kombo_1_jarjestatud")

    

if __name__ == "__main__":
    main()