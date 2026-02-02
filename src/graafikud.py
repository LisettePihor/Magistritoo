
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import json

import os
import json
import pandas as pd

def loo_ennustuste_notebook(ennustatud_treening, y_treening_idga, ennustatud_test, y_test_idga, pealkiri, kombo_nr):
    asukoht = os.path.join(os.getcwd(), f"andmed/kombo_nr_{kombo_nr}/graafikud")
    faili_nimi = os.path.join(asukoht, f"ennustatud_{pealkiri}.ipynb")
    
    if not os.path.exists(faili_nimi):
        os.makedirs(asukoht, exist_ok=True)
        
        df_tr = pd.DataFrame({
            'Molecule ChEMBL ID': y_treening_idga['Molecule ChEMBL ID'],
            'Tegelik väärtus': y_treening_idga['pChEMBL Value'],
            'Ennustatud väärtus': ennustatud_treening,
            'Andmestik': 'Treening'
        })
        
        df_te = pd.DataFrame({
            'Molecule ChEMBL ID': y_test_idga['Molecule ChEMBL ID'],
            'Tegelik väärtus': y_test_idga['pChEMBL Value'],
            'Ennustatud väärtus': ennustatud_test,
            'Andmestik': 'Test'
        })
        
        df_koond = pd.concat([df_tr, df_te], ignore_index=True)
        df_koond['Viga'] = (df_koond['Ennustatud väärtus'] - df_koond['Tegelik väärtus']).abs()
        vead_treening = df_koond[df_koond['Andmestik'] == 'Treening']['Viga']
        std_treening = np.std(vead_treening)
        
        koodi_read = [
            "import plotly.express as px\n",
            "import plotly.graph_objects as go\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "# Andmete laadimine\n",
            f"andmed = pd.DataFrame({df_koond.to_dict(orient='list')})\n",
            f"std_treening = {std_treening}\n",
            "\n",
            "# Graafiku loomine\n",
            "fig = px.scatter(andmed, x='Ennustatud väärtus', y='Tegelik väärtus', color='Andmestik', \n",
            "                 hover_data={'Molecule ChEMBL ID': True, 'Tegelik väärtus': ':.2f', \n",
            "                             'Ennustatud väärtus': ':.2f', 'Viga': ':.2f', 'Andmestik': False},\n",
            f"                 title=f'Ennustatud vs Tegelik: {pealkiri}')\n",
            "\n",
            "# 3-standardvea ala ja ideaaljoone lisamine\n",
            "x_min = andmed['Ennustatud väärtus'].min() - 0.5\n",
            "x_max = andmed['Ennustatud väärtus'].max() + 0.5\n",
            "x_range = np.linspace(x_min, x_max, 100)\n",
            "\n",
            "fig.add_trace(go.Scatter(x=x_range, y=x_range + (3 * std_treening), \n",
            "                         mode='lines', line=dict(width=0), showlegend=False))\n",
            "fig.add_trace(go.Scatter(x=x_range, y=x_range - (3 * std_treening), \n",
            "                         mode='lines', line=dict(width=0), fill='tonexty', \n",
            "                         fillcolor='rgba(128, 128, 128, 0.2)', name='3 STD ala'))\n",
            "\n",
            "fig.add_trace(go.Scatter(x=[x_min, x_max], y=[x_min, x_max], \n",
            "                         mode='lines', name='Ideaalne', line=dict(color='red', dash='dash')))\n",
            "\n",
            "fig.update_layout(xaxis_title='Ennustatud', yaxis_title='Tegelik')\n",
            "fig.show()"
        ]

        notebook_json = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": koodi_read}
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}
            },
            "nbformat": 4, "nbformat_minor": 4
        }

        with open(faili_nimi, 'w', encoding='utf-8') as f:
            json.dump(notebook_json, f, indent=1)

    return None

def loo_analuusi_tabel(andmestik, piltide_kaust):
    failinimi = os.path.join(piltide_kaust, 'analüüsi_tabel.html')
    if 'Mol' in andmestik.columns:
        andmestik = andmestik.drop('Mol', axis=1)

    with open(failinimi, 'w', encoding='utf-8') as f:
        f.write("""
        <html>
        <head>
            <title>Molekulide Analüüs</title>
            <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.css">
            <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
            <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                img { max-width: 150px; height: auto; border: 1px solid #ddd; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h2>Andmestiku visuaalne kontroll (491 molekuli)</h2>
            <table id="molekuli_tabel" class="display">
                <thead>
                    <tr>
        """)
        columns = andmestik.columns.tolist() + ['Struktuur']
        for col in columns:
            f.write(f"<th>{col}</th>")
        
        f.write("</tr></thead><tbody>")

        for _, row in andmestik.iterrows():
            f.write("<tr>")
            for col in andmestik.columns:
                f.write(f"<td>{row[col]}</td>")
            
            pildi_nimi = f"{row['Molecule ChEMBL ID']}.png"
            pildi_tee = os.path.join(piltide_kaust, pildi_nimi)
            
            if os.path.exists(pildi_tee):
                f.write(f'<td><img src="{pildi_tee}"></td>')
            else:
                f.write("<td>Pilt puudub</td>")
            
            f.write("</tr>")

        f.write("""
                </tbody>
            </table>
            <script>
                $(document).ready( function () {
                    $('#molekuli_tabel').DataTable({
                        "pageLength": 25,
                        "order": [[ 1, "desc" ]] // Sorteeri teise veeru järgi (nt pChEMBL)
                    });
                } );
            </script>
        </body>
        </html>
        """)
    print(f"Fail '{failinimi}' on valmis ja ootab avamist brauseris!")
    return None

def ennustuste_graafik(ennustatud_treening, tegelikud_treening, ennustatud_test, tegelikud_test, mse_treening, r2_treening, mse_test, r2_test, pealkiri, kombo_nr):
    fail = os.path.join(os.getcwd(),f"andmed/kombo_nr_{kombo_nr}/graafikud/ennustatud_" + pealkiri + ".png")
    if not os.path.exists(fail):
        os.makedirs(os.path.join(os.getcwd(),f"andmed/kombo_nr_{kombo_nr}/graafikud"), exist_ok=True)
        jaagid_treening = ennustatud_treening - tegelikud_treening
        piir = 3 * np.std(jaagid_treening)
        min_v, max_v = tegelikud_treening.min() - 0.5, tegelikud_treening.max() + 0.5
        x_telg = np.linspace(min_v, max_v, 100)
        plt.fill_between(x_telg, x_telg - piir, x_telg + piir, color='gray', alpha=0.1)
        plt.scatter(tegelikud_treening, ennustatud_treening, alpha=0.6, label='Treening', color='blue')
        plt.scatter(tegelikud_test, ennustatud_test, alpha=0.6, label='Test', color='orange')
        stat = (f'R² Treening: {r2_treening:.2f}\nMSE Treening: {mse_treening:.2f}\nR² Test: {r2_test:.2f}\nMSE Test: {mse_test:.2f}\n')
        plt.text(min_v + 0.2, max_v - 1.5, stat, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.xlabel("Tegelikud pChEMBL väärtused")
        plt.ylabel("Ennustatud pChEMBL väärtused")
        plt.title(f"Ennustatud vs tegelikud pChEMBL väärtused\n{pealkiri}")
        plt.savefig(fail)
        plt.clf()
        
    return None

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