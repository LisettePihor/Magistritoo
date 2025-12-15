import pandas as pd
from src.get_activity_data_chembl import create_query
from src.data_analysis import analyze_data, get_important
from src.info_from_description import extract_info_from_description

def main():
    create_query()
    analyze_data()

    
    df = get_important()
    df_new = extract_info_from_description(df)
    print(df_new.head())

if __name__ == "__main__":
    main()