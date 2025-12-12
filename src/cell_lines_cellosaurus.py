import os
import requests
import json

def cellosaurus_database_search(directory):
    output_file = os.path.join(directory,'data/cellosaurus_cell_lines.json')
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            data = json.load(file)
            print(f"Loaded data from existing file: {output_file}")
    else:
        database_url = "https://api.cellosaurus.org/search/cell-line"
        params = {
            'q': 'di:glioblastoma ox:Homo sapiens',
            'format': 'json',
            'rows': 10000
        }

        print(f"Fetching data for '{params['q']}'...")

        try:
            response = requests.get(database_url, params=params)
            
            response.raise_for_status()
            
            data = response.json()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            print(f"Successfully saved JSON data to {output_file}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from response.")
            print("Response text:", response.text)

    return data