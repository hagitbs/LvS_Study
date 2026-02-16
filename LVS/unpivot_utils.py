from __future__ import annotations

from importlib import import_module
from pathlib import Path
import pandas as pd
import os
 

 

def unpivot_wide_dataframe(df_wide,id_vars,value_name='frequency_in_document', var_name='document'):
    """
    Unpivots a wide DataFrame into a long format.

    This function is a wrapper around pandas.melt for clear, reusable unpivot operations.

    Args:
        df_wide (pd.DataFrame): The wide DataFrame to unpivot.
        id_vars (list): A list of one or more column names to use as identifier variables.
                        These columns will not be unpivoted.
        value_name (str, optional): The name for the new 'value' column. Defaults to 'Value'.
        var_name (str, optional): The name for the new 'variable' column. Defaults to 'Variable'.

    Returns:
        pd.DataFrame: A new DataFrame in the long (unpivoted) format.
    """
    if not isinstance(id_vars, list):
        raise TypeError("id_vars must be a list of column names.")
        
    print("Unpivoting the DataFrame...")
    
    long_df = df_wide.melt(id_vars=id_vars,
                           var_name=var_name,
                           value_name=value_name)
    print("Unpivot process complete.")
    return long_df 

def unpivot_wide_csv_by_row(input_file_path,output_file_path, id_vars='element',value_name='frequency_in_document', var_name='document'): 
    """
    Unpivots a very wide CSV file in a memory-efficient manner by processing it row by row.

    This function reads the CSV file one row at a time to avoid loading the entire
    dataset into memory, which is crucial for files with a large number of columns.

    Args:
        input_file_path (str): The path to the input CSV file.
        output_file_path (str): The path where the unpivoted CSV will be saved.
        id_vars (list): A list of column names to be used as identifier variables.
                        These columns will not be unpivoted.
        value_name (str, optional): The name for the new column that will store the values
                                    from the unpivoted columns. Defaults to 'Value'.
        var_name (str, optional): The name for the new column that will store the original
                                  column headers. Defaults to 'document'.
    """
    try:
        # Define a chunk size of 1 to process the file row by row.
        chunk_iterator = pd.read_csv(input_file_path, chunksize=1)

        is_first_chunk = True
        #print("Starting the unpivot process...")

        # Iterate over each row (as a chunk) in the input file.
        for i, chunk in enumerate(chunk_iterator):
            # The pd.melt function transforms the chunk from wide to long format.
            melted_chunk = pd.melt(chunk,
                                   id_vars=id_vars,
                                   var_name=var_name,
                                   value_name=value_name)
            
            # For the first row, write the header to the new file.
            # For all subsequent rows, append without writing the header again.
            if is_first_chunk:
                melted_chunk.to_csv(output_file_path, index=False, mode='w', header=True)
                is_first_chunk = False
            else:
                melted_chunk.to_csv(output_file_path, index=False, mode='a', header=False)
            
            #print(f"Processed row {i+1}...")

        print("\nProcess complete.")
        print(f"Successfully unpivoted the file and saved it to '{output_file_path}'")

    except FileNotFoundError:
        print(f"Error: The input file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def unpivot_dataframe(df, ignore_columns, agg_column='Year', var_name='Cause', value_name='Deaths', file_path2=None):  
    try:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None) 
        # Drop the extra columns
        df = df.drop(columns=ignore_columns, errors='ignore') # errors='ignore' prevents errors if columns don't exist 
        # Melt the DataFrame, using only 'Year' as the id_vars
        df_melted = pd.melt(df, id_vars=[agg_column], var_name=var_name, value_name=value_name, value_vars=df.columns[1:])   
        # Group by 'Year' and 'Cause' to sum the deaths across all entities
        print(f"Unpivoting data...")
        print(agg_column , var_name , value_name) 
        print(ignore_columns) 
        print(file_path2) 
        print(f"DataFrame full shape: {df_melted.shape}") 
        #print 5 rows of the DataFrame
        print(df_melted.head(5)) 
    except Exception as e:
        print(f"An error occurred: {e}")
    return df_melted

