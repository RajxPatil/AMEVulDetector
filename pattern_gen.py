import os
import networkx as nx
import numpy as np
import csv

def parse_nodes_csv(file_path):
    """Parse nodes from a CSV file."""
    nodes = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row['Node_ID']
                try:
                    flag = int(row['Flag'])  # Attempt to parse the flag as an integer
                except ValueError:
                    flag = None  # Set flag to None if parsing fails
                    print(f"Warning: Non-integer value in Flag column for Node {node_id}")

                nodes[node_id] = {
                    'type': row['Type'],
                    'related_node': row['Related_Node'],
                    'attributes': row['Attributes'],
                    'flag': flag,  # Store the parsed or fallback value
                    'label': row['Label']
                }
    else:
        print(f"Node CSV file not found: {file_path}")
    return nodes

def parse_edges_csv(file_path):
    """Parse edges from a CSV file."""
    edges = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                edges.append((
                    row['Source'],
                    row['Target'],
                    {
                        'type': row['Edge_Type'],
                        'feature': row['Edge_Feature']
                    }
                ))
    else:
        print(f"Edge CSV file not found: {file_path}")
    return edges

def create_graph(nodes, edges):
    """Create a graph using nodes and edges."""
    G = nx.Graph()
    for node_id, node_info in nodes.items():
        G.add_node(node_id, **node_info)
    G.add_edges_from(edges)
    return G

def extract_graph_features(G):
    """Extract features from the graph."""
    features = []
    degree_centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G)
    
    try:
        eigenvector = nx.eigenvector_centrality(G)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: 0 for node in G.nodes}
    
    for node in G.nodes:
        node_features = [
            degree_centrality.get(node, 0),
            clustering.get(node, 0),
            eigenvector.get(node, 0)
        ]
        features.append(node_features)
    
    flattened_features = np.array(features).flatten()
    if flattened_features.size < 250:
        flattened_features = np.pad(flattened_features, (0, 250 - flattened_features.size), 'constant')
    
    return flattened_features[:250]

def process_solidity_file(file_path, nodes_folder, edges_folder):
    """Process a single Solidity file and return the graph features."""
    contract_name = os.path.splitext(os.path.basename(file_path))[0]
    
    node_file_path = os.path.join(nodes_folder, f"{contract_name}_nodes.csv")
    edge_file_path = os.path.join(edges_folder, f"{contract_name}_edges.csv")
    
    nodes = parse_nodes_csv(node_file_path)
    edges = parse_edges_csv(edge_file_path)
    
    if not nodes:
        print(f"No nodes parsed for contract: {contract_name}")
    if not edges:
        print(f"No edges parsed for contract: {contract_name}")
    
    G = create_graph(nodes, edges)
    return extract_graph_features(G)

def process_solidity_files_in_folder(folder_path, nodes_folder, edges_folder, output_file):
    """Process all Solidity files in a given folder and extract features for each."""
    with open(output_file, 'w') as f:
        sol_files = [file for file in os.listdir(folder_path) if file.endswith('.sol')]
        
        for sol_file in sol_files:
            file_path = os.path.join(folder_path, sol_file)
            print(f"Processing {sol_file}...")
            
            features = process_solidity_file(file_path, nodes_folder, edges_folder)
            f.write(" ".join(map(str, features)) + "\n")

# Input paths
folder_path = './data_example/reentrancy/source_code/validation'  # Path to the folder containing Solidity (.sol) files
nodes_folder ='./data_example/reentrancy/node/validation'   # Path to the folder containing node CSV files
edges_folder = './data_example/reentrancy/edge/validation'   # Path to the folder containing edge CSV files
output_file = 'graph_valid_features.txt'

# Process the files
process_solidity_files_in_folder(folder_path, nodes_folder, edges_folder, output_file)
