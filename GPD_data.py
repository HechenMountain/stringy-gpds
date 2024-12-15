# Dependencies
import numpy as np
import re
import os

base_path = "/mnt/c/Users/flori/Documents/PostDoc/Data/GPD/"

# Set up dictionary
# Add some colors
saturated_pink = (1.0, 0.1, 0.6)  # Higher red, some blue, minimal green
# Define a dictionary that maps publication IDs to a color
def initialize_data_dictionary():
    global publication_mapping 
    publication_mapping = {
        "2305.11117": "cyan",
        "0705.4295": "orange",
        "1908.10706": saturated_pink
    # Add more publication IDs and corresponding colors here
    }

initialize_data_dictionary()

def load_lattice_data(moment_type, moment_label, pub_id):
    """
    Load data from a .dat file, extracting 'n' values from the header and associating them with rows.

    Args:
        moment_type (str): The type of moment (e.g., "Isovector").
        moment_label (str): The label of the moment (e.g., "A").
        pub_id (str): The publication ID.

    Returns:
        tuple: A tuple containing the data, and a dictionary mapping 'n' values to row indices.
    """

    filename = f"{moment_type}Moments{moment_label}{pub_id}.dat"
    file_path = f"{base_path}{filename}"

    # Check if the file exists
    if not os.path.exists(file_path):
        #print(f"No data available for {moment_type}{moment_label} in {pub_id}")
        return None, None

    with open(file_path, 'r') as f:
        header = f.readline().strip()
        data = np.loadtxt(f)

    # Extract 'n' values from the header
    n_values = []
    for col_name in header.split():
        match = re.search(rf'{moment_label}(\d+)0_', col_name)
        if match:
            n_values.append(int(match.group(1)))

    # Create a dictionary mapping 'n' values to row indices
    n_to_row_map = {n: i for i, n in enumerate(n_values)}

    return data, n_to_row_map

def Fn0_values(n, moment_type, moment_label, pub_id):
    """
    Return central values for An0 for a given n, moment type, label, and publication ID.
    """
    data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)

    if data is None and n_to_row_map is None:
        #print(f"No data found for {moment_type} {moment_label} {pub_id}. Skipping.")
        return None
    
    # Check if the requested n is available in the data
    if n not in n_to_row_map:
        return None  # Requested n does not exist
    
    # Get the corresponding column index for An0_val
    col_idx = n_to_row_map[n]
    
    # Return the column data
    return data[:, col_idx]

def Fn0_errors(n, moment_type, moment_label, pub_id):
    """
    Return errors for An0 for a given n, moment type, label, and publication ID.
    """
    data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)

    if data is None and n_to_row_map is None:
        #print(f"No data found for {moment_type} {moment_label} {pub_id}. Skipping.")
        return None 
    
    # Check if the requested n is available in the data
    if n not in n_to_row_map:
        return None  # Requested n does not exist
    
    # Get the corresponding column index for An0_err
    col_idx = n_to_row_map[n]+1
    
    # Return the column data
    return data[:, col_idx]