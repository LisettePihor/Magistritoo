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
    
    :param kombo_nr: Kombinatsiooni number (int või str)
    '''
    fail = os.path.join(os.getcwd(),'andmed/andmed_kombo_nr_' + str(kombo_nr) + '.csv')
    if os.path.exists(fail):
        andmestik_tunnustega = pd.read_csv(fail)
    else:
        andmed_alg = andmed_ilma_duplikaatideta()
        parimad_kombod = parimad_kombinatsioonid(andmed_alg)
        andmed_kombo = andmed_alg[(andmed_alg['Cell Name'] == parimad_kombod[kombo_nr]['Cell Name']) & 
                    (andmed_alg['Standard Type'] == parimad_kombod[kombo_nr]['Standard Type']) & 
                    (andmed_alg['Assay'] == parimad_kombod[kombo_nr]['Assay']) & 
                    (andmed_alg['Property Measured'] == parimad_kombod[kombo_nr]['Property Measured'])  & 
                    (andmed_alg['Incubation Time Hours'] == parimad_kombod[kombo_nr]['Incubation Time Hours'])].copy()

        andmed_kombo['Mol'] = andmed_kombo['Smiles'].apply(smiles_to_mol)
        andmed_kombo = andmed_kombo[andmed_kombo['Mol'].notnull()].reset_index(drop=True)
        tunnused = [desc_name[0] for desc_name in Descriptors._descList]
        kalkulaator = MoleculeDescriptors.MolecularDescriptorCalculator(tunnused)
        def arvuta_tunnused(mol):
            return list(kalkulaator.CalcDescriptors(mol))
        tunnuste_andmed = pd.DataFrame( andmed_kombo['Mol'].apply(arvuta_tunnused).tolist(), columns=tunnused)
        andmestik_tunnustega = pd.concat([tunnuste_andmed.reset_index(drop=True), andmed_kombo[['pChEMBL Value', 'Molecule ChEMBL ID', 'InChIKey']].reset_index(drop=True)],axis=1)

        #andmestik_tunnustega.to_csv(fail, index=False)
    return andmestik_tunnustega

def andmed_ilma_duplikaatideta():
    '''
    Lae andmed ilma duplikaatideta või loo need, kui faili pole olemas. 
    Loob analüüsi duplikaatide kohta ja salvestab selle info, samuti salvestab duplikaadid eraldi faili.
    '''
    fail = os.path.join(os.getcwd(), 'andmed/aktiivsused_duplikaatideta_uus.csv')
    fail_duplikaadid = os.path.join(os.getcwd(), 'andmed/duplikaadid.csv')
    fail_duplikaatide_info = os.path.join(os.getcwd(), 'andmed/duplikaatide_info.csv')
    if os.path.exists(fail):
        andmed = pd.read_csv(fail)
    else:
        alg_andmed = leia_info_kirjeldusest()
        print(f"Algandmetes on {len(alg_andmed)} kirjet.")
        kirjeldus = alg_andmed.groupby(['Molecule ChEMBL ID', 'Standard Type','Incubation Time Hours', 'Property Measured'])['pChEMBL Value'].agg(['max', 'min','mean', 'median', 'count']).reset_index()
        kirjeldus.columns = ['Molekuli ChEMBL ID', 'Standard', 'Aeg', 'Omadus','Max pChEMBL', 'Min pChEMBL', 'Keskmine pChEMBL', 'Mediaan pChEMBL', 'Duplikaatide arv']
        print(kirjeldus)
    
        #andmed.to_csv('data/activities_wo_duplicates.csv', index=False)
    return None


def parimad_kombinatsioonid(andmed):
    output_file = os.path.join(os.getcwd(), 'data/top_10_combinations.json')
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