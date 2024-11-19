import os
import shutil
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_contract_targets(csv_path):
    """
    Load the contract name to target mappings from a CSV file.
    Assumes the first column is 'filename' and the second column is 'Reentrancy'.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df['filename'], df['Reentrancy'].astype(str)))

def split_data(source_dir, train_dir, val_dir, csv_path, train_size=0.8, val_size=0.2):
    # Load contract targets from CSV file
    contract_targets = load_contract_targets(csv_path)
    
    # List all .sol files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.sol')]
    train_files, val_files = train_test_split(files, train_size=train_size, random_state=42)

    # Create directories if they don’t exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Prepare JSON data lists
    train_json_data = []
    val_json_data = []

    # Copy files and prepare JSON content for train set
    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
        train_json_data.append({"filename": f, "Reentrancy": contract_targets.get(f, "0")})  # default to "0" if missing

    # Copy files and prepare JSON content for validation set
    for f in val_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))
        val_json_data.append({"contract_name": f, "targets": contract_targets.get(f, "0")})  # default to "0" if missing

    # Save JSON files
    with open(os.path.join(train_dir, 'train.json'), 'w') as train_json_file:
        json.dump(train_json_data, train_json_file, indent=4)
    
    with open(os.path.join(val_dir, 'valid.json'), 'w') as val_json_file:
        json.dump(val_json_data, val_json_file, indent=4)

    print("Data split complete and JSON files generated successfully.")

# Example usage
csv_path = 'ICSE_New.csv'  # Path to the uploaded CSV file

split_data(
    source_dir='data_example/reentrancy/source_code/',
    train_dir='data_example/reentrancy/source_code/train/',
    val_dir='data_example/reentrancy/source_code/validation/',
    csv_path=csv_path,
    train_size=0.8,
    val_size=0.2
)
