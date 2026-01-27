import os
import requests
import json

def cellosaurus_database_search():
    base_dir = os.getcwd()
    output_file = os.path.join(base_dir,'andmed/cellosaurus_rakuliinid.json')
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            andmed = json.load(file)
            print(f"Andmed laetud olemasolevast failist: {output_file}")
    else:
        database_url = "https://api.cellosaurus.org/search/cell-line"
        params = {
            'q': 'di:glioblastoma ox:Homo sapiens',
            'format': 'json',
            'rows': 10000
        }

        print(f"Otsitakse andmeid '{params['q']}'...")

        try:
            vastus = requests.get(database_url, params=params)
            
            vastus.raise_for_status()
            
            andmed = vastus.json()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(andmed, f, ensure_ascii=False, indent=4)
                
            print(f"JSON andmed salvestati faili {output_file}")

        except requests.exceptions.RequestException as e:
            print(f"Esines viga {e}")
        except json.JSONDecodeError:
            print("Viga: Faili ei õnnestunud JSON-ina laadida.")
            print("Vastus:", vastus.text)

    return andmed