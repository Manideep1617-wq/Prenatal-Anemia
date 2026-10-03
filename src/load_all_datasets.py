import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

def load_datasets(data_dir):
    print("Loading datasets...")
    try:
        cbc = pd.read_csv(os.path.join(data_dir, 'cbc_dataframe.csv'))
        print(f"CBC Data: {cbc.shape}")
    except Exception as e:
        print(f"CBC Data error: {e}")
        
    try:
        mh = pd.read_csv(os.path.join(data_dir, 'Dataset - Updated.csv'))
        print(f"Maternal Health Data: {mh.shape}")
    except Exception as e:
        print(f"Maternal Health error: {e}")
        
    try:
        xl = pd.read_excel(os.path.join(data_dir, 'Book2.xlsx'), header=1)
        print(f"Excel Data: {xl.shape}")
    except Exception as e:
        print(f"Excel Data error: {e}")
        
    print("Skipping large DHS data load to save time.")
        
    valid_xpts = ['ALB_CR_L.xpt', 'BAX_L.xpt']
    for xpt in valid_xpts:
        try:
            df = pd.read_sas(os.path.join(data_dir, xpt))
            print(f"{xpt}: {df.shape}")
        except Exception as e:
            print(f"Error loading {xpt}: {e}")

if __name__ == '__main__':
    load_datasets('C:/Users/gshan/OneDrive/Documents/PrenatalAnemia/data')
