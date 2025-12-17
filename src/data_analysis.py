import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from rdkit import Chem
import pickle
import json
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
        top_cells = data_by_cell.head(10)['Cell ChEMBL ID'].tolist()
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

def best_combination(df):
    grouped = df.groupby(['Cell Name', 'Property Measured', 'Standard Type','Assay', 'Incubation Time Hours'])['Molecule ChEMBL ID'].nunique().reset_index()

    # Rename the count column
    grouped.rename(columns={'Molecule ChEMBL ID': 'Unique Molecule Count'}, inplace=True)

    # Find the row with the maximum unique molecule count
    top_10 = grouped.sort_values(by='Unique Molecule Count', ascending=False).head(10)

    top_10_dict = top_10.reset_index(drop=True).to_dict(orient='index')

    # Save to file
    output_file = os.path.join(os.getcwd(), 'data/top_10_combinations.json')
    with open(output_file, 'w') as f:
        json.dump(top_10_dict, f, indent=4)
    return top_10_dict


def unique_molecules(df, filename):
    cell_lines = df['Cell Name'].unique()
    cell_lines.sort()
    n = len(cell_lines)
    matrix = np.zeros((n, n), dtype=int)

    cell_to_mols = df.groupby('Cell Name')['Molecule ChEMBL ID'].apply(set).to_dict()

    for i in range(n):
        for j in range(n):
            cell_i = cell_lines[i]
            cell_j = cell_lines[j]
            
            mols_i = cell_to_mols.get(cell_i, set())
            mols_j = cell_to_mols.get(cell_j, set())
            
            # Intersection count
            shared_count = len(mols_i.intersection(mols_j))
            matrix[i, j] = shared_count

    # Create DataFrame for better labelling if needed (optional, mainly for plotting)
    heatmap_df = pd.DataFrame(matrix, index=cell_lines, columns=cell_lines)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='viridis')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cell_lines)
    ax.set_yticklabels(cell_lines)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, matrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Unikaalsed molekulid rakuliinides")
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(os.path.join(os.getcwd(), filename))
    plt.clf()
    return n

    
