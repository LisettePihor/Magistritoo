
import requests
from cell_lines_cellosaurus import cellosaurus_database_search


def chembl_id_from_cellosaurus(base_dir):
    data = cellosaurus_database_search(base_dir)
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
                        #chembl_links[primary_accession] = (cell_chembl_id, identifier_value)
                        chembl_links[cell_chembl_id] = identifier_value
                        break 
    return chembl_links

def get_cell_ids(chembl_ids):
    #cell line can be in assay_organism, assay_cell_type, assay_tissue?, cell_chembl_id
    #official connection through variant_id = cell_id
    cell_base_url = "https://www.ebi.ac.uk/chembl/api/data/activity"
    for cell_line in chembl_ids.keys():
        params = {'cell_chembl_id': cell_line,'format': 'json', 'limit': 100}
        try:
            while cell_base_url:
                    session = requests.Session()
                    cell_response = session.get(cell_base_url, params=params, timeout=60)
                    cell_response.raise_for_status()
                    cell_json = cell_response.json()

                    assay_chembl_ids.extend(a["assay_chembl_id"] for a in cell_json.get("assays", []))
                    url = assay_json.get("page_meta", {}).get("next")
                    if url:
                        if url.startswith("/"):
                            url = urllib.parse.urljoin("https://www.ebi.ac.uk/chembl/api/data/assay", url)
                    else:
                        url = None
                    sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"  > Error fetching assays for {chembl_ids[cell_line]: {e}}")


#assay connected with activities through doc_id of assay to assay_id of activities
def get_activity_data(base_dir):
    chembl_ids = chembl_id_from_cellosaurus(base_dir)
    assay_cell_ids = get_cell_ids(chembl_ids)
