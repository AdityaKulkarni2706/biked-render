import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from GA_Clip_utils import BikeCAD

def row_to_bcad(row_df, template_path, output_path):

    # Essentially ensure that row_series is of pd.Series dtype

    if isinstance(row_df, pd.DataFrame):
        row_series = row_df.iloc[0]
    else:
        row_series = row_df

    row_dict = row_series.to_dict()

    tree = ET.parse(template_path)
    root = tree.getroot()

    updated_count = 0

    for entry in root.findall('entry'):
        key = entry.get('key')

        # text = 1 sets mmInch unit to be true

        if key and key.endswith("mmInch"):
            entry.text = "1"
            continue

        # if XML key matches one of the 96 features, swap the data
        if key in row_dict:
            val = row_dict[key]

            # if value contains any NaNs, skip writing
            if pd.isna(val):
                print("NaN detected, writing skipped.")
                continue
            
            # java requires lowercase booleans

            if isinstance(val, (bool, np.bool_)):
                formatted_val = "true" if val else "false"
            
            # Convert numbers like 34.0 to 34 so BikeCAD doesn't complain about floats

            elif isinstance(val, (float, np.float64)) and val.is_integer():
                
                formatted_val = str(int(val))

            # For strings
            else:
                formatted_val = str(val)
            
            #inject it into the copy of the template

            entry.text = formatted_val
            updated_count+=1

    xml_content = ET.tostring(root, encoding='utf-8', xml_declaration=False).decode('utf-8')

    java_header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">\n'
    )
    final_file_content = java_header + xml_content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_file_content)

    print(f"Success: Injected {updated_count} features into '{Path(output_path).name}'")


if __name__ == "__main__":
    template_path = r"Biked_Reference_Data\PlainRoadbikestandardized.txt"
    processed_bikes_path = r"Biked_Reference_Data\clip_sBIKED_processed.csv"
    
    df = pd.read_csv(r"Biked_Reference_Data\clip_sBIKED_processed.csv")
    row = df.iloc[1]
    output_path = rf"Biked_Reference_Data\output\{row.iloc[0]}_clean.bcad"

    row_to_bcad(row, template_path, output_path)
    cad_engine = BikeCAD()
    cad_engine.export_svg_from_list([output_path])
    cad_engine.kill()