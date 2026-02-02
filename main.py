from src.andmete_tootlus import jaota_andmestik, kombo_koos_tunnustega
from src.mudelite_treenimine import otsustusmets

def main():
    andmed_1 = kombo_koos_tunnustega(1)
    X_treening, y_treening, X_test, y_test = jaota_andmestik(andmed_1, 1, jarjestatud=True)
    otsustusmets(X_treening, y_treening, X_test, y_test, 1, 'jarjestatud')

    

if __name__ == "__main__":
    main()