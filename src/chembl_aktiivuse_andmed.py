import pickle
import os
from src.cellosaurus_rakuliinid import cellosaurus_database_search


def chembl_id_from_cellosaurus():
    andmed = cellosaurus_database_search()
    fail = os.path.join(os.getcwd(), 'andmed/chembl_id.pkl')
    if not os.path.exists(fail):
        rakuliinide_list = andmed['Cellosaurus']['cell-line-list']
        if rakuliinide_list is None:
            print("Error: Ei leitud 'cell-line-list'i cellosaurus JSON andmetest")

        print(f"Leiti{len(rakuliinide_list)} rakuliini. Otsitakse ChEMBL linke...")
        chembl_lingid = {}
        for rakuliin in rakuliinide_list:
            
            identikaator = next(
                (item['value'] for item in rakuliin.get('name-list', []) if item.get('type') == 'identifier'), 
                "UNKNOWN_IDENTIFIER"
            )
            
            primary_accession = "UNKNOWN_ACCESSION"
            if "accession-list" in rakuliin:
                for acc in rakuliin.get("accession-list", []):
                    if acc.get("type") == "primary":
                        primary_accession = acc.get("value")
                        break
            
            if "xref-list" in rakuliin:
                for xref in rakuliin.get("xref-list", []):
                    andmebaas = xref.get("database")
                    
                    if andmebaas in ("ChEMBL", "ChEMBL-Cells"):
                        raku_chembl_id = xref.get("accession")
                        if raku_chembl_id:
                            chembl_lingid[raku_chembl_id] = identikaator
                            break 
        with open(fail, 'wb') as f:
            pickle.dump(chembl_lingid, f)
    else:
        with open(fail, 'rb') as f:
            chembl_lingid = pickle.load(f)
    print(f'Leiti{len(chembl_lingid)} rakuliini chembl ID')
    return chembl_lingid

def loo_otsing():
    chembl_id = chembl_id_from_cellosaurus()
    or_jada = " OR ".join(f'"{cid}"' for cid in chembl_id)
    otsing_str = f'(_metadata.assay_data.cell_chembl_id:({or_jada}) AND (standard_relation:=)'
    return otsing_str

