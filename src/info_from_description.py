import pandas as pd
import re
import os

# Load the file
# Define the extraction functions
def extract_time_with_minutes(text):
    if not isinstance(text, str):
        return None
    
    # Check for minutes first (to avoid confusion if multiple times are present, though usually it's one)
    # or just simple regex checks.
    
    # Pattern for minutes
    mins_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mins?|minutes?|min)\b', text, re.IGNORECASE)
    if mins_match:
        return float(mins_match.group(1)) / 60.0
    
    # Pattern for hours
    hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hrs?|hours?|h)\b', text, re.IGNORECASE)
    if hours_match:
        return float(hours_match.group(1))
    
    # Pattern for days
    days_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:days?|d)\b', text, re.IGNORECASE)
    if days_match:
        return float(days_match.group(1)) * 24.0
        
    return None

def extract_property(text):
    text_lower = text.lower()
    
    if "cytotox" in text_lower:
        return "Cytotoxicity"
    elif "antiprolif" in text_lower or "anti-prolif" in text_lower:
        return "Antiproliferative activity"
    elif "anticancer" in text_lower:
        return "Anticancer activity"
    elif "antiviral" in text_lower:
        return "Antiviral activity"
    elif "growth inhibition" in text_lower or ("growth" in text_lower and "inhibition" in text_lower):
        return "Growth inhibition"
    
    else:
        return "Unspecified"

def extract_assay_type(text):
    if not isinstance(text, str):
        return "Unspecified"
    
    # Pattern: "by ... assay/method"
    match = re.search(r'by\s+(.*?)\s+(?:assay|method)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback to keywords
    common_assays = ['MTT', 'MTS', 'SRB', 'CellTiter-Glo', 'Alamar Blue', 'Resazurin', 'ELISA', 'Western Blot', 'Luciferase', 'Trypan Blue', 'CellTiter-Blue']
    for assay in common_assays:
        if assay.lower() in text.lower():
            return assay
            
    return "Unspecified"

def extract_info_from_description(df):
    output_filename = os.path.join(os.getcwd(), 'data/activities_with_assay_details.csv')
    if os.path.exists(output_filename):
        return pd.read_csv(output_filename)
    else:
        # Apply functions
        df['Incubation Time Hours'] = df['Assay Description'].apply(extract_time_with_minutes)
        df['Property Measured'] = df['Assay Description'].apply(extract_property)
        df['Assay'] = df['Assay Description'].apply(extract_assay_type)
        # Save
        df.to_csv(output_filename, index=False)
        print(f"File saved to {output_filename}")
        return df