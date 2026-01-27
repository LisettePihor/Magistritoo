import os
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
import json
from src.info_chembl_failist import leia_info_kirjeldusest
from src.graafikud import loo_analuusi_tabel

def kombo_koos_tunnustega(kombo_nr):
    '''
    Loo andmestik koos molekulaartunnustega andtud kombinatisooni jaoks. 
    Andmestikku luues salvestab ka info kombole vastavate duplikaatide kohta.
    
    :param kombo_nr: Kombinatsiooni number (int või str)
    '''
    fail = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}.csv')
    if os.path.exists(fail):
        andmestik_tunnustega = pd.read_csv(fail)
    else:
        andmed_alg = andmed_ilma_duplikaatideta()
        kombo = parimad_kombinatsioonid(andmed_alg)[str(kombo_nr)]
        andmed_kombo = andmed_alg[(andmed_alg['Cell Name'] == kombo['Cell Name']) & 
                    (andmed_alg['Standard Type'] == kombo['Standard Type']) & 
                    (andmed_alg['Assay'] == kombo['Assay']) & 
                    (andmed_alg['Property Measured'] == kombo['Property Measured'])  & 
                    (andmed_alg['Incubation Time Hours'] == kombo['Incubation Time Hours'])].copy()

        andmed_kombo['ROMol'] = andmed_kombo['Smiles'].apply(smiles_to_mol)
        andmed_kombo = andmed_kombo[andmed_kombo['ROMol'].notnull()].reset_index(drop=True)
        tunnused = [desc_name[0] for desc_name in Descriptors._descList]
        kalkulaator = MoleculeDescriptors.MolecularDescriptorCalculator(tunnused)
        def arvuta_tunnused(mol):
            return list(kalkulaator.CalcDescriptors(mol))
        tunnuste_andmed = pd.DataFrame( andmed_kombo['ROMol'].apply(arvuta_tunnused).tolist(), columns=tunnused)
        andmestik_tunnustega = pd.concat([tunnuste_andmed.reset_index(drop=True), andmed_kombo[['pChEMBL Value', 'Molecule ChEMBL ID', 'Smiles', 'Molecule Name']].reset_index(drop=True)],axis=1)
        andmestik_tunnustega.to_csv(fail, index=False)

        fail_duplikaadid_info = os.path.join(os.getcwd(),'andmed/duplikaatide_info.csv')
        fail_kombo_duplikaadid = os.path.join(os.getcwd(),f'andmed/kombo_nr_{kombo_nr}_duplikaatide_info.csv')
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
    print(f"Kombinatsioon {kombo_nr} leitud - Molekulide arv: {andmestik_tunnustega.shape[0]}")
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
    print(f"Duplikaatideta andmestik laetud - Molekulide arv: {andmed.shape[0]}")
    return andmed

def parimad_kombinatsioonid(andmed):
    fail = os.path.join(os.getcwd(), 'andmed/top_10_kombinatsiooni.json')
    if os.path.exists(fail):
        with open(fail, 'r') as f:
            top_10 = json.load(f)
    else:
        grupeeritud = andmed.groupby(['Cell Name', 'Property Measured', 'Standard Type','Assay', 'Incubation Time Hours'])['Molecule ChEMBL ID'].nunique().reset_index()
        grupeeritud.rename(columns={'Molecule ChEMBL ID': 'Unique Molecule Count'}, inplace=True)
        top_10 = grupeeritud.sort_values(by='Unique Molecule Count', ascending=False).head(10)
        top_10 = top_10.reset_index(drop=True).to_dict(orient='index')
        with open(fail, 'w') as f:
            json.dump(top_10, f, indent=4)
    print("Top 10 kombinatsiooni leitud")
    return top_10

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def jaota_andmestik(algandmestik, kombo_nr, jarjestatud, juhuarv=42):
    fail_csv = os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo_nr}_jaotus.csv')
    andmestik = algandmestik.copy()
    andmestik['Set'] = 'Train' 

    if jarjestatud:
        sorteeritud_indeksid = andmestik.sort_values('pChEMBL Value').index
        test_indeksid = sorteeritud_indeksid[::5]
        andmestik.loc[test_indeksid, 'Set'] = 'Test'
    else:
        X = andmestik.drop(['pChEMBL Value'], axis=1)
        y = andmestik['pChEMBL Value']
        _, X_test_tmp, _, _ = train_test_split(X, y, test_size=0.2, random_state=juhuarv)
        andmestik.loc[X_test_tmp.index, 'Set'] = 'Test'

    valitud_veerud = ['pChEMBL Value','Molecule ChEMBL ID','Smiles','Molecule Name','Set']
    andmestik_valitud = andmestik[valitud_veerud]
    andmestik_valitud.sort_values(by=['Set', 'pChEMBL Value'], inplace=True)
    andmestik_valitud.to_csv(fail_csv, index=False)
    molekulide_pildid = os.path.join(os.getcwd(), f'andmed/kombo_nr_{kombo_nr}_molekulid')
    if not os.path.exists(molekulide_pildid):
        os.makedirs(molekulide_pildid)
        for idx, rida in andmestik_valitud.iterrows():
            mol = Chem.MolFromSmiles(rida['Smiles'])
            if mol:
                pilt_fail = os.path.join(molekulide_pildid, f"{rida['Molecule ChEMBL ID']}.png")
                Draw.MolToFile(mol, pilt_fail)
        loo_analuusi_tabel(andmestik, molekulide_pildid)

    test_mask = andmestik['Set'] == 'Test'
    treening_mask = andmestik['Set'] == 'Train'

    mittevajalikud = ['Molecule ChEMBL ID', 'pChEMBL Value', 'Set', 'Smiles', 'Molecule Name']

    X_treening = andmestik[treening_mask].drop(mittevajalikud, axis=1)
    y_treening = andmestik[treening_mask]['pChEMBL Value']
    X_test = andmestik[test_mask].drop(mittevajalikud, axis=1)
    y_test = andmestik[test_mask]['pChEMBL Value']
    
    print(f"Andmestik jagatud - Treening: {X_treening.shape[0]} molekuli, Test: {X_test.shape[0]} molekuli")
    return X_treening, y_treening, X_test, y_test