import pickle
import os
from src.cell_lines_cellosaurus import cellosaurus_database_search


def chembl_id_from_cellosaurus():
    data = cellosaurus_database_search()
    file_path = os.path.join(os.getcwd(), 'data/chembl_ids.pkl')
    if not os.path.exists(file_path):
        list_of_cell_lines = data['Cellosaurus']['cell-line-list']
        if list_of_cell_lines is None:
            print("Error: Could not find the 'cell-line-list' inside the cellosaurus json data")

        print(f"Found {len(list_of_cell_lines)} cell lines. Scanning for ChEMBL links...")
        chembl_links = {}
        for cell_line in list_of_cell_lines:
            
            identifier_value = next(
                (item['value'] for item in cell_line.get('name-list', []) if item.get('type') == 'identifier'), 
                "UNKNOWN_IDENTIFIER"
            )
            
            primary_accession = "UNKNOWN_ACCESSION"
            if "accession-list" in cell_line:
                for acc in cell_line.get("accession-list", []):
                    if acc.get("type") == "primary":
                        primary_accession = acc.get("value")
                        break
            
            if "xref-list" in cell_line:
                for xref in cell_line.get("xref-list", []):
                    db_name = xref.get("database")
                    
                    if db_name in ("ChEMBL", "ChEMBL-Cells"):
                        cell_chembl_id = xref.get("accession")
                        if cell_chembl_id:
                            chembl_links[cell_chembl_id] = identifier_value
                            break 
        with open(file_path, 'wb') as f:
            pickle.dump(chembl_links, f)
    else:
        with open(file_path, 'rb') as f:
            chembl_links = pickle.load(f)
    print(f'Found {len(chembl_links)} cell line chembl IDs')
    return chembl_links

def create_query():
    chembl_ids = chembl_id_from_cellosaurus()
    or_query = " OR ".join(f'"{cid}"' for cid in chembl_ids)
    query_str = f'(_metadata.assay_data.cell_chembl_id:({or_query}) AND (standard_relation:=)'
    return query_str

