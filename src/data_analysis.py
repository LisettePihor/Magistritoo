import pandas as pd
import os
from rdkit import Chem
import pickle

def smiletoinchikey(smile):
    if pd.isna(smile):
        return None
    else:
        try:
            mol = Chem.MolFromSmiles(smile)
            inchikey = Chem.inchi.MolToInchiKey(mol)
            return inchikey
        except Exception as e:
            print(f"Error converting SMILES: {smile} to InChIKey: {e}")
            return None
    
def get_important():
    input_path = os.path.join(os.getcwd(), 'data/activities.csv')
    output_path = os.path.join(os.getcwd(), 'data/activities_small.csv')
    if not os.path.exists(output_path):
        data = pd.read_csv(input_path, sep=';', low_memory=False)
        
        id_dict = pickle.load(open(os.path.join(os.getcwd(), 'data/chembl_ids.pkl'), 'rb'))
        data['Cell Name'] = data['Cell ChEMBL ID'].map(id_dict)

        data_by_cell = data.groupby('Cell ChEMBL ID')['Molecule ChEMBL ID'].nunique().reset_index()
        data_by_cell = data_by_cell.sort_values(by='Molecule ChEMBL ID', ascending=False)
        top_cells = data_by_cell.head(5)['Cell ChEMBL ID'].tolist()
        data = data[data['Cell ChEMBL ID'].isin(top_cells)]

        important_columns = ['Molecule ChEMBL ID', 'Molecule Name', 'Smiles', 'Standard Type', 'pChEMBL Value', 'Assay ChEMBL ID',
                            'Assay Description', 'Cell Name']
        important_data = data[important_columns]
        important_data = important_data.dropna(subset=['pChEMBL Value'])
        important_data['InChIKey'] = important_data['Smiles'].apply(smiletoinchikey)
        important_data.to_csv(output_path, index=False)
    else:
        important_data = pd.read_csv(output_path)
    return important_data

def analyze_data():
    important_data = get_important()
    '''if 'Measurement Type' and 'Assay Type' and 'Time' in important_data.columns:
    else:
        print("Required columns are missing in the dataset. Conduct analysis before continuing!")
        return'''
    
