import os
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler
import argparse
import joblib

def parse_attributes(attr_str):
    try:
        # Safely evaluate the string to a list
        return ast.literal_eval(attr_str)
    except:
        return []

def preprocess_csv(file_path):
    """
    Reads and preprocesses a single CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    
    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Drop 'Node_ID' as it's an identifier
        if 'Node_ID' in df.columns:
            df = df.drop('Node_ID', axis=1)
        
        # Parse 'Attributes' column
        df['Attributes'] = df['Attributes'].apply(parse_attributes)
        
        # Explode the 'Attributes' list to have one attribute per row
        df_exploded = df.explode('Attributes')
        
        # One-Hot Encode 'Attributes'
        df_attributes_encoded = pd.get_dummies(df_exploded, columns=['Attributes'])
        
        # Group by original index and aggregate (max) to combine one-hot encodings
        df_attributes_final = df_attributes_encoded.groupby(df_attributes_encoded.index).max()
        
        # Drop original 'Attributes' column from the exploded DataFrame
        df_combined = df_exploded.drop('Attributes', axis=1)
        
        # One-Hot Encode 'Type', 'Related_Node', and 'Label'
        df_encoded = pd.get_dummies(df_combined, columns=['Type', 'Related_Node', 'Label'])
        
        # Combine with the attributes encoded DataFrame
        df_final = df_encoded.join(df_attributes_final.filter(regex='Attributes_'))
        
        # Drop duplicate columns if any
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        
        # Convert 'Flag' to numeric
        if 'Flag' in df_final.columns:
            df_final['Flag'] = pd.to_numeric(df_final['Flag'], errors='coerce').fillna(0)
        
        return df_final
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()

def first_pass_collect_features(feature_dir):
    """
    First pass to collect all unique feature columns across all CSV files.
    
    Parameters:
    - feature_dir (str): Path to the directory containing node feature CSV files.
    
    Returns:
    - set: A set containing all unique feature column names.
    """
    unique_features = set()
    for file in sorted(os.listdir(feature_dir)):
        if file.endswith('_nodes.csv'):
            file_path = os.path.join(feature_dir, file)
            print(f"First Pass - Processing file: {file_path}")
            df = preprocess_csv(file_path)
            if not df.empty:
                unique_features.update(df.columns.tolist())
    return unique_features

def second_pass_aggregate_features(feature_dir, output_file, unique_features, scaler=None):
    """
    Second pass to process each CSV file, align features, aggregate, and normalize.
    
    Parameters:
    - feature_dir (str): Path to the directory containing node feature CSV files.
    - output_file (str): Path to the output aggregated feature file.
    - unique_features (set): Set of all unique feature column names.
    - scaler (object, optional): A fitted scaler to apply to the features.
    
    Returns:
    - scaler (object): The fitted scaler (only if scaler was None).
    """
    aggregated_features = []
    for file in sorted(os.listdir(feature_dir)):
        if file.endswith('_nodes.csv'):
            file_path = os.path.join(feature_dir, file)
            print(f"Second Pass - Processing file: {file_path}")
            df = preprocess_csv(file_path)
            if not df.empty:
                # Aggregate by averaging across nodes
                contract_features = df.mean().reindex(sorted(unique_features), fill_value=0).values
                aggregated_features.append(contract_features)
            else:
                # If the dataframe is empty, append a zero vector
                contract_features = np.zeros(len(unique_features))
                aggregated_features.append(contract_features)
    
    if aggregated_features:
        aggregated_features = np.array(aggregated_features)
        print(f"Aggregated features shape before normalization: {aggregated_features.shape}")
        
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_features = scaler.fit_transform(aggregated_features)
            print("Applied Min-Max scaling to training data.")
        else:
            normalized_features = scaler.transform(aggregated_features)
            print("Applied existing scaler to validation/test data.")
        
        # Save to the output file
        np.savetxt(output_file, normalized_features, delimiter=',', fmt='%.6f')
        print(f"Aggregated features saved to {output_file}")
        return scaler
    else:
        print("No features were aggregated.")
        return scaler

def main():
    parser = argparse.ArgumentParser(description='Aggregate Node Features with Two-Pass Approach')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing node feature CSV files')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save aggregated features')
    parser.add_argument('--scaler_path', type=str, help='Path to save/load the scaler (optional)')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize features')
    args = parser.parse_args()
    
    unique_features = first_pass_collect_features(args.feature_dir)
    print(f"Total unique features collected: {len(unique_features)}")
    
    scaler = None
    if args.normalize and args.scaler_path:
        # If scaler path is provided and normalization is desired, load existing scaler
        if os.path.exists(args.scaler_path):
            scaler = joblib.load(args.scaler_path)
            print(f"Loaded existing scaler from {args.scaler_path}")
        else:
            print(f"Scaler path {args.scaler_path} does not exist. A new scaler will be created.")
    
    scaler = second_pass_aggregate_features(args.feature_dir, args.output_file, unique_features, scaler=scaler)
    
    if args.normalize and args.scaler_path and scaler is not None:
        # Save the scaler for future use
        joblib.dump(scaler, args.scaler_path)
        print(f"Scaler saved to {args.scaler_path}")

if __name__ == "__main__":
    main()