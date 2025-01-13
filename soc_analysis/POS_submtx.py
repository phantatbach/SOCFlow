import pandas as pd
import os

# Get the submatrix of only the POS tags
# Input: SOC matrix
# Output: SOC matrix with columns = only the POS tags in POS_list
def get_POS_submtx(SenSOC_file, sub_SenSOC, POS_list, input_folder, sep=','):
    """
    Extracts a submatrix from a SOC matrix containing only the columns that match the specified POS tags.

    Args:
        SenSOC_file (str): Path to the SOC matrix file to be filtered.
        sub_SenSOC (str): Name for the output submatrix file.
        POS_list (list of str): List of Part-Of-Speech tags to filter the columns by.
        input_folder (str): Directory where the output file will be saved.
        sep (str): Separator used in the CSV file. Defaults to ','.
    
    Raises:
        FileNotFoundError: If the specified SOC matrix file does not exist.
        ValueError: If no columns match the specified POS tags.

    Returns:
        None: The function saves the filtered submatrix to a CSV file in the specified directory.
    """
    # Ensure input file exists
    if not os.path.exists(SenSOC_file):
        raise FileNotFoundError(f"Input file not found: {SenSOC_file}")
    
    # Load the SOC matrix
    SenSOC_matrix = pd.read_csv(SenSOC_file, sep=sep, header=0)

    # If POS_list is empty, log a warning and return the original matrix
    if not POS_list:
        print("Warning: POS_list is empty. No filtering applied.")
        return

    # Filter columns based on POS tags
    column_names = [
        column for POS in POS_list for column in SenSOC_matrix.columns if POS in column
    ]

    # Ensure at least one column matches
    if not column_names:
        raise ValueError(f"No columns found matching POS tags: {POS_list}")
    
    output_file = os.path.join(input_folder, f'{sub_SenSOC}_SOCs.csv')

    # Save the filtered submatrix
    SenSOC_matrix[column_names].to_csv(output_file, sep=',')
    print(f"Filtered SOC matrix saved to: {output_file}")