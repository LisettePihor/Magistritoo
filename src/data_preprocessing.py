import os
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from src.data_analysis import best_combination
import matplotlib.pyplot as plt

def remove_duplicates():
    df = pd.read_csv('data/activities_with_assay_details.csv')
    df.drop_duplicates(subset=['Molecule ChEMBL ID'], keep='first', inplace=True)
    df.to_csv('data/activities_wo_duplicates.csv', index=False)

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def create_data(combo_nr, ordered=False):
    filename = os.path.join(os.getcwd(),'data/df_combo_nr_' + str(combo_nr) + '.csv')
    if os.path.exists(filename):
        model_df_desc = pd.read_csv(filename)
    else:
        remove_duplicates()
        df = pd.read_csv('data/activities_wo_duplicates.csv')
        best_combos = best_combination(df)
        df_combo = df[(df['Cell Name'] == best_combos[combo_nr]['Cell Name']) & 
                    (df['Standard Type'] == best_combos[combo_nr]['Standard Type']) & 
                    (df['Assay'] == best_combos[combo_nr]['Assay']) & 
                    (df['Property Measured'] == best_combos[combo_nr]['Property Measured'])  & 
                    (df['Incubation Time Hours'] == best_combos[combo_nr]['Incubation Time Hours'])]


        df_combo['Mol'] = df_combo['Smiles'].apply(smiles_to_mol)
        # Remove invalid molecules
        df_combo = df_combo[df_combo['Mol'].notnull()].reset_index(drop=True)
        descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        def calc_descriptors(mol):
            return list(calculator.CalcDescriptors(mol))
        X_desc = pd.DataFrame( df_combo['Mol'].apply(calc_descriptors).tolist(), columns=descriptor_names)
        # Remove constant / invalid columns
        X_desc = X_desc.loc[:, X_desc.nunique() > 1]
        model_df_desc = pd.concat([X_desc.reset_index(drop=True), df_combo[['pChEMBL Value', 'Molecule ChEMBL ID', 'InChIKey']].reset_index(drop=True)],axis=1)

        model_df_desc.to_csv(filename, index=False)


    X = model_df_desc.drop(['Molecule ChEMBL ID', 'InChIKey', 'pChEMBL Value'], axis=1)
    y = model_df_desc['pChEMBL Value']    

    if ordered:
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
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

def plot_dist(train_y, test_y, name):
    plt.hist(train_y, bins=30, alpha=0.6, label='Train')
    plt.hist(test_y, bins=30, alpha=0.6, label='Test')
    plt.xlabel("pChEMBL")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(),"data/distribution_" + name + ".png"))