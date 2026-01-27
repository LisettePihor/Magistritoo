import os
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import matplotlib.pyplot as plt
import numpy as np
import json
from src.info_chembl_failist import leia_info_kirjeldusest

def kombo_koos_tunnustega(kombo_nr):
    '''
    Loo andmestik koos molekulaartunnustega andtud kombinatisooni jaoks. 
    Andmestikku luues salvestab ka info kombole vastavate duplikaatide kohta.
    
    :param kombo_nr: Kombinatsiooni number (int või str)
    '''
    kombo_nr = str(kombo_nr)
    fail = os.path.join(os.getcwd(),'andmed/kombo_nr_' + kombo_nr + '.csv')
    if os.path.exists(fail):
        andmestik_tunnustega = pd.read_csv(fail)
    else:
        andmed_alg = andmed_ilma_duplikaatideta()
        kombo = parimad_kombinatsioonid(andmed_alg)[kombo_nr]
        andmed_kombo = andmed_alg[(andmed_alg['Cell Name'] == kombo['Cell Name']) & 
                    (andmed_alg['Standard Type'] == kombo['Standard Type']) & 
                    (andmed_alg['Assay'] == kombo['Assay']) & 
                    (andmed_alg['Property Measured'] == kombo['Property Measured'])  & 
                    (andmed_alg['Incubation Time Hours'] == kombo['Incubation Time Hours'])].copy()

        andmed_kombo['Mol'] = andmed_kombo['Smiles'].apply(smiles_to_mol)
        andmed_kombo = andmed_kombo[andmed_kombo['Mol'].notnull()].reset_index(drop=True)
        tunnused = [desc_name[0] for desc_name in Descriptors._descList]
        kalkulaator = MoleculeDescriptors.MolecularDescriptorCalculator(tunnused)
        def arvuta_tunnused(mol):
            return list(kalkulaator.CalcDescriptors(mol))
        tunnuste_andmed = pd.DataFrame( andmed_kombo['Mol'].apply(arvuta_tunnused).tolist(), columns=tunnused)
        andmestik_tunnustega = pd.concat([tunnuste_andmed.reset_index(drop=True), andmed_kombo[['pChEMBL Value', 'Molecule ChEMBL ID', 'InChIKey']].reset_index(drop=True)],axis=1)
        andmestik_tunnustega.to_csv(fail, index=False)

        fail_duplikaadid_info = os.path.join(os.getcwd(),'andmed/duplikaatide_info.csv')
        fail_kombo_duplikaadid = os.path.join(os.getcwd(),'andmed/kombo_nr_' + kombo_nr + '_duplikaatide_info.csv')
        if not os.path.exists(fail_duplikaadid_info):
            raise FileNotFoundError("Duplikaatide info faili ei leitud.")
        else:
            duplikaadid_info = pd.read_csv(fail_duplikaadid_info)
            duplikaadid_kombo = duplikaadid_info[(duplikaadid_info['Cell Name'] == kombo['Cell Name']) & 
                        (duplikaadid_info['Standard Type'] == kombo['Standard Type']) & 
                        (duplikaadid_info['Assay'] == kombo['Assay']) & 
                        (duplikaadid_info['Property Measured'] == kombo['Property Measured'])  & 
                        (duplikaadid_info['Incubation Time Hours'] == kombo['Incubation Time Hours'])]
            duplikaadid_kombo.sort_values(by='count', ascending=False, inplace=True)
            duplikaadid_kombo.to_csv(fail_kombo_duplikaadid, index=False)
    return andmestik_tunnustega

def andmed_ilma_duplikaatideta():
    '''
    Lae andmed ilma duplikaatideta või loo need, kui faili pole olemas. 
    Andmeid luues koostab ka analüüsi duplikaatide kohta ja salvestab selle info.
    '''
    fail = os.path.join(os.getcwd(), 'andmed/aktiivsused_duplikaatideta.csv')
    fail_duplikaatide_info = os.path.join(os.getcwd(), 'andmed/duplikaatide_info.csv')
    if os.path.exists(fail):
        andmed = pd.read_csv(fail)
    else:
        alg_andmed = leia_info_kirjeldusest()
        grupi_veerud = ['Molecule ChEMBL ID', 'Cell Name','Standard Type','Assay','Incubation Time Hours', 'Property Measured']
        kirjeldus = alg_andmed.groupby(grupi_veerud, dropna=False)['pChEMBL Value'].agg(['max', 'min','mean', 'median', 'count']).reset_index()
        kirjeldus.to_csv(fail_duplikaatide_info, index=False)
        andmed = alg_andmed.copy()
        andmed['pChEMBL Value'] = andmed.groupby(grupi_veerud, dropna=False)['pChEMBL Value'].transform('median')
        andmed = andmed.drop_duplicates(subset=grupi_veerud).reset_index(drop=True)
        andmed.to_csv('andmed/aktiivsused_duplikaatideta.csv', index=False)
    return andmed

def parimad_kombinatsioonid(andmed):
    output_file = os.path.join(os.getcwd(), 'andmed/top_10_combinations.json')
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            top_10_dict = json.load(f)
    else:
        grouped = andmed.groupby(['Cell Name', 'Property Measured', 'Standard Type','Assay', 'Incubation Time Hours'])['Molecule ChEMBL ID'].nunique().reset_index()
        grouped.rename(columns={'Molecule ChEMBL ID': 'Unique Molecule Count'}, inplace=True)
        top_10 = grouped.sort_values(by='Unique Molecule Count', ascending=False).head(10)
        top_10_dict = top_10.reset_index(drop=True).to_dict(orient='index')
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
            
            shared_count = len(mols_i.intersection(mols_j))
            matrix[i, j] = shared_count

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

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def split_data(df, jarjestatud, juhuarv=42, testita=False):
    X = df.drop(['Molecule ChEMBL ID', 'InChIKey', 'pChEMBL Value'], axis=1)
    y = df['pChEMBL Value']    
    if testita:
        return X, y
    else:
        if jarjestatud:
            import numpy as np

            order = np.argsort(y)
            X = X.to_numpy()
            y = y.to_numpy()

            X_sorted = X[order]
            y_sorted = y[order]

            test_mask = np.zeros(len(y_sorted), dtype=bool)
            test_mask[::5] = True   # every 5th point → 20% test

            X_test  = X_sorted[test_mask]
            y_test  = y_sorted[test_mask]

            X_train = X_sorted[~test_mask]
            y_train = y_sorted[~test_mask]

            pealkiri = f'jarjestatud'
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=juhuarv)
            pealkiri = f'juhuarv_{juhuarv}'
        
        pealkiri = f'seed_{juhuarv}_ordered_{jarjestatud}'
        plot_dist(y_train, y_test, pealkiri)
    return X_train, y_train, X_test, y_test

def plot_dist(train_y, test_y, nimi):
    output_path = os.path.join(os.getcwd(),"plots/jaotus_" + nimi + ".png")
    if os.path.exists(output_path):
        return
    else:
        plt.hist(train_y, bins=30, alpha=0.6, label='Train')
        plt.hist(test_y, bins=30, alpha=0.6, label='Test')
        plt.xlabel("pChEMBL")
        plt.ylabel("Count")
        plt.title("Treening ja test andmete pChEMBL väärtused " + nimi)
        plt.legend()
        plt.savefig(output_path)
        plt.clf()
    return