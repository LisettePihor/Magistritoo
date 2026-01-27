import pandas as pd
import re
import os
from src.chembl_aktiivuse_andmed import chembl_id_from_cellosaurus, loo_otsing
from rdkit import Chem

def extract_time_with_minutes(text):
    if not isinstance(text, str):
        return None

    mins_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mins?|minutes?|min)\b', text, re.IGNORECASE)
    if mins_match:
        return float(mins_match.group(1)) / 60.0
    
    hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hrs?|hours?|h)\b', text, re.IGNORECASE)
    if hours_match:
        return float(hours_match.group(1))
    
    days_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:days?|d)\b', text, re.IGNORECASE)
    if days_match:
        return float(days_match.group(1)) * 24.0
        
    return None

def extract_property(text):
    text_lower = text.lower()
    
    if "cytotox" in text_lower:
        return "Cytotoxicity"
    elif "antiprolif" in text_lower or "anti-prolif" in text_lower:
        return "Antiproliferative activity"
    elif "anticancer" in text_lower:
        return "Anticancer activity"
    elif "antiviral" in text_lower:
        return "Antiviral activity"
    elif "growth inhibition" in text_lower or ("growth" in text_lower and "inhibition" in text_lower):
        return "Growth inhibition"
    
    else:
        return "Unspecified"

def extract_assay_type(text):
    if not isinstance(text, str):
        return "Unspecified"
    
    match = re.search(r'by\s+(.*?)\s+(?:assay|method)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    common_assays = ['MTT', 'MTS', 'SRB', 'CellTiter-Glo', 'Alamar Blue', 'Resazurin', 'ELISA', 'Western Blot', 'Luciferase', 'Trypan Blue', 'CellTiter-Blue']
    for assay in common_assays:
        if assay.lower() in text.lower():
            return assay
            
    return "Unspecified"

def leia_info_kirjeldusest():
    fail = os.path.join(os.getcwd(), 'andmed/aktiivsused_kirjelduse_detailidega.csv')
    if os.path.exists(fail):
        return pd.read_csv(fail)
    else:
        andmed = andmed_ainult_vajalik()
        andmed['Incubation Time Hours'] = andmed['Assay Description'].apply(extract_time_with_minutes)
        andmed['Property Measured'] = andmed['Assay Description'].apply(extract_property)
        andmed['Assay'] = andmed['Assay Description'].apply(extract_assay_type)
        
        andmed.to_csv(fail, index=False)
        print(f"Kirjeldustega fail salvestatud: {fail}")
        return andmed
       
def andmed_ainult_vajalik():
    input_path = os.path.join(os.getcwd(), 'andmed/aktiivsused.csv')
    output_path = os.path.join(os.getcwd(), 'andmed/aktiivsused_oluline.csv')
    if not os.path.exists(output_path):
        if not os.path.exists(input_path):
            otsing = loo_otsing()
            raise FileNotFoundError(f"Koosta ChEMBL andmebaasis kohandatud otsing {otsing}\nLae alla ChEMBL andmefail ja salvesta see asukohta: {input_path}")
        data = pd.read_csv(input_path, sep=';', low_memory=False)
        
        id_dict = chembl_id_from_cellosaurus()
        data['Cell Name'] = data['Cell ChEMBL ID'].map(id_dict)

        data_by_cell = data.groupby('Cell ChEMBL ID')['Molecule ChEMBL ID'].nunique().reset_index()
        data_by_cell = data_by_cell.sort_values(by='Molecule ChEMBL ID', ascending=False)
        top_cells = data_by_cell.head(10)['Cell ChEMBL ID'].tolist()
        data = data[data['Cell ChEMBL ID'].isin(top_cells)]

        important_columns = ['Molecule ChEMBL ID', 'Molecule Name', 'Smiles', 'Standard Type', 'pChEMBL Value', 'Assay ChEMBL ID',
                            'Assay Description', 'Cell Name']
        important_data = data[important_columns]
        important_data = important_data.dropna(subset=['pChEMBL Value'])
        important_data.to_csv(output_path, index=False)
    else:
        important_data = pd.read_csv(output_path)
    return important_data

def smilemolekuliks(smile):
    if pd.isna(smile):
        return None
    else:
        try:
            mol = Chem.MolFromSmiles(smile)
            return mol
        except Exception as e:
            print(f"Error muutes SMILES: {smile} molekuliks: {e}")
            return None
 