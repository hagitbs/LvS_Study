# %%
#
#Always use 3.11.9 
import pandas as pd  
import altair as alt
import numpy as np
from pathlib import Path
import sys, os
import configparser
import argparse 
from helpers import read
 
import bottleneck as bn
from LPA import Corpus, sockpuppet_distance
import lvs_per_document
from math import floor
from scipy.spatial.distance import cdist, cityblock
import matplotlib.pyplot as plt
from visualize import sockpuppet_matrix, timeline
alt.data_transformers.disable_max_rows()
from unpivot_utils import unpivot_wide_dataframe
from unpivot_utils import unpivot_dataframe 
import seaborn as sns
from sklearn.decomposition import PCA
 

# %%
def load_data(file_path):

    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}")
        print(f"DataFrame shape: {df.shape}") 
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None  # Important: Return None on error
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

    
def clean_data(df,short_names, dataset):
    """Cleans the DataFrame (e.g., keep only relevant columns , handles missing values, data type conversions)."""
    if df is None:
        print("Error: Input DataFrame is None. Skipping clean_data.")
        return None, None
    try:
        #clean elements that always = 0 in all the documents 
        print("Cleaning data...")
        # Group by 'element' and sum the frequency across all documents
        non_zero_elements = df.groupby('element')['frequency_in_document'].sum()
        out_dir="results/"+dataset
        os.makedirs(out_dir, exist_ok=True)
        
        # save the zero elements to a file"results/{dataset}/zero_elements.csv"
        non_zero_elements[non_zero_elements == 0].to_csv(out_dir+"/zero_elements.csv")
        #save the non zero elements to a file
        non_zero_elements[non_zero_elements > 0].to_csv(f"results/{dataset}/non_zero_elements.csv")
        # Keep only elements with a non-zero total frequency
        non_zero_elements = non_zero_elements[non_zero_elements > 0].index
        # Filter the original DataFrame to keep only those elements
        filtered_df = df[df['element'].isin(non_zero_elements)]
        df= filtered_df
        df_cleaned = df.dropna()
        # Keep only the relevant columns
        entity_code_df = None
        # Shorten the element names
        unique_elements = df['element'].unique()
        #print(f"Unique elements: {unique_elements}")
        if short_names=='True':
            element_to_code = { element: f'E{i}' for i,  element  in enumerate(unique_elements) }
        else:
            # Create a mapping from element names to codes
            element_to_code = {element: element for i, element in enumerate(unique_elements)}
            
        df_cleaned['element'] = df_cleaned['element'].map(element_to_code)  
        # Create a DataFrame from the dictionary
        entity_code_df = pd.DataFrame(list(element_to_code.items()), columns=['element_name', 'element']) 
        return df_cleaned, entity_code_df
    except KeyError as e:
        print(f"Error: Column not found: {e}. Check your 'columns_to_keep' parameter.")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None, None
 
def transform_names(df,agg_column,var_name,value_name):
    print(agg_column,var_name,value_name)
    # cast dynamically
    df[agg_column] = df[agg_column].astype("string")  # keeps <NA> nicely
    df[var_name] = df[var_name].astype("string")  # keeps <NA> nicely
    df[value_name]=df[value_name].astype("float")
    # Rename
    df = df.rename(columns={agg_column: 'document',
                            var_name  : 'element',
                            value_name: 'frequency_in_document'}) 
    # keep only 3 columns in the dataframe 
    df = df[['document', 'element', 'frequency_in_document']]
    return df  

def save_results(df,entity_code_df, output_path,output_dic):
    if df is None:
        print("Error: Input DataFrame is None. Skipping save_results.")
        return

    """Saves the processed DataFrames to a CSV file."""
    try: 
        print(f"Saving results to: {output_path} and {output_dic}")
        df.to_csv(output_path, index=False)  # Don't include the index
        print(f"Data successfully saved to {output_path}")

        if output_dic:
            #  Create a DataFrame from the dictionary and save it.  Important for consistent structure.
            entity_code_df.to_csv(output_dic, index=False)            
            print(f"Dictionary successfully saved to {output_path.replace('.csv', '_dict.csv')}")
    except Exception as e:
        print(f"Error saving results: {e}")



# %%



def generate_signatures(df, entity_code_df, sig_file, dataset,graph,top,sig_length,var_name,value_name):
    """
    Generates and saves document signatures, along with related analyses and visualizations.

    Args:
        df (pd.DataFrame): Input DataFrame containing document data.
        entity_code_df (pd.DataFrame, optional): DataFrame mapping entity codes to names.
        sig_file (str, optional): Path to save the signature DataFrame.
        dataset (str): Name of the dataset for output directory.
    """
    if df is None:
        print("Error: Input DataFrame is None. Skipping generate_signatures.")
        return

    try:
        # Create results directory if it doesn't exist
        import os
        os.makedirs(f"results/{dataset}", exist_ok=True)
        print (f"sig_file: {sig_file}, dataset: {dataset}, graph: {graph}, top: {top}, sig_length: {sig_length}, var_name: {var_name}, value_name: {value_name}")
        # aggregate lines in df 
        df = df.groupby(['document', 'element'])['frequency_in_document'].sum().reset_index()
        freq=df
        corpus = Corpus(df, "document", "element", "frequency_in_document")
        dvr = corpus.create_dvr(equally_weighted=True) # Create Document Vector Representation (DVR)
        dvr.to_csv(f"results/{dataset}/dvr.csv")
        top = int(top)
        sig_length = int(sig_length)

        sigs = corpus.create_signatures(distance="JSD",sig_length=sig_length, most_significant=top,prevalent=0.1) #Hagit check if this is the right distance

        #  Saving top N changed elements
        sigs[1].to_csv(f"results/{dataset}/top_{top}_most_changed.csv")
        sig = pd.DataFrame(sigs[1])

        # Rename columns based on entity_code_df if provided
        if entity_code_df is not None:
            entity_code_to_name = entity_code_df.set_index("element")["element_name"].to_dict()
            new_columns = [
                entity_code_to_name.get(col, col) for col in sig.columns
            ]  # Use get() for safety
            sig.columns = new_columns
            sig.to_csv(f"results/{dataset}/top_{top}_most_changed_real_names.csv")

        # Save signatures if sig_file is provided
        if sig_file:
            ndf = pd.DataFrame(sigs[0])
            ndf.to_csv(sig_file, index=True)
            print(f"Signatures successfully saved to {sig_file}")
            #save the signatures with real names
            if entity_code_df is not None:
                ndf.columns = [entity_code_to_name.get(col, col) for col in ndf.columns]
                ndf.to_csv(sig_file.replace('.csv', '_real_names.csv'), index=True)
                print(f"Signatures with real names successfully saved to {sig_file.replace('.csv', '_real_names.csv')}")
        else:
            print("No signature file provided, skipping signature saving.")

        # split the ndf DataFrame  to several dataframes   , by the column name   
        output_dir = f"results/{dataset}/split_dataframes"
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist 
        print("-" * 30)

        # Iterate over each row of the DataFrame using iterrows()
        # This method yields both the index and the row (as a Series)
        for index, row in ndf.iterrows():
            # Convert the row (which is a pandas Series) to a DataFrame
            # .to_frame() converts the Series to a DataFrame with the original Series index as the new DataFrame's index
            # We can provide a column name, for instance, using the original index
            
            row = row[row.notnull()]

            row_df = row.to_frame(name=f'row_{index}_data')

            # Pivot the result. For a single row DataFrame, transposing it achieves the desired pivoted effect.
            pivoted_df = row_df 


            # Define a unique filename for each new CSV file
            file_name = os.path.join(output_dir, f'row_{index}.csv') 

            # If the DataFrame is empty after dropping NaN columns, skip saving
            if pivoted_df.empty:    
                print(f"Row {index} has no data to save, skipping.")
                continue    

            # Save the pivoted DataFrame to a new CSV file.
            # The column names from the original DataFrame will be preserved as the header.
            # Select rows where the second column (index 1) is not null

            pivoted_df = pivoted_df[pivoted_df.iloc[:, 0].notna()]
            pivoted_sorted_desc = pivoted_df.sort_values(by=pivoted_df.columns[0], ascending=False)
            #pivoted_sorted_desc['col1_numeric'] = pd.to_numeric(pivoted_sorted_desc[pivoted_sorted_desc.columns[0]], errors='coerce')
            #pivoted_sorted_desc= pivoted_sorted_desc.sort_values(by=pivoted_sorted_desc.columns[1].abs(), ascending=False)
            pivoted_sorted_desc.to_csv(file_name, index=True)
            
            # Convert column[0] to numeric. 'coerce' will turn invalid parsing into NaN.
            ''' 
            pivoted_df['col1_numeric'] = pd.to_numeric(pivoted_df[pivoted_df.columns[0]], errors='coerce')

            # Drop rows where conversion failed (optional, depending on how you want to handle errors)
            pivoted_df.dropna(subset=['col1_numeric'], inplace=True)

            # Sort rx_filtered by the absolute value of the newly created numeric column in descending order
            rx_sorted_abs_desc = pivoted_df.sort_values(by=pivoted_df['col1_numeric'].abs(), ascending=False)

            # Drop the temporary numeric column if you don't need it
            rx_sorted_abs_desc = rx_sorted_abs_desc.drop(columns=['col1_numeric'])
            rx_sorted_abs_desc.to_csv(file_name, index=True)
            '''

            print(f"Saved pivoted data for row {index} to '{file_name}'") 

           # Save element list
        with open(f"results/{dataset}/list.txt", "w") as f:
            for item in sigs[0]:
                f.write(f"{item}\n")
        # save list into dataframe  
        df_list = pd.DataFrame(sigs[0])
        print(f"Element list saved to results/{dataset}/list.txt")
        #print(df_list.head(5))
        #pivot_the_list = df_list.melt(var_name='element', value_name='value', ignore_index=False)
        print(df_list.columns)
        #print(df_list.head(5))
        pivot_the_list = df_list.melt(var_name=var_name, value_name=value_name, ignore_index=False)
        pivot_the_list = pivot_the_list.reset_index().rename(columns={'index': 'element'})
        df_list = pivot_the_list.dropna().reset_index(drop=True)   
        df_list.to_csv(f"results/{dataset}/list.csv", index=True)
        print(f"Element list saved to results/{dataset}/list.csv")      

        # expected vs observed
        print("Columns in df_list:", df_list.columns.tolist())
        print("Columns in dvr:", dvr.columns.tolist()) 
        # Sockpuppet analysis
        freq['doc_total'] = freq.groupby('document')['frequency_in_document'].transform('sum')
        freq['freq_norm'] = freq['frequency_in_document'] / freq['doc_total']
        print(freq.head(10))
               
        if graph == 'True':
            df_observed=df_list 
            df_observed = df_observed.rename(columns={ 
                            var_name  : 'element_observed',
                            value_name: 'LvS'}) 
            
            df_merged = pd.merge(
            df_observed, 
            dvr, 
            left_on=['element_observed'],   # Columns in the first DF (sig)
            right_on=['element'],  # Columns in the second DF (dvr)
            how='outer'  , 
            suffixes=('_ob', '_expected')
            )
            #document         element  frequency_in_document     doc_total  freq_norm
            df_merged = pd.merge(
            df_merged, 
            freq, 
            left_on=['element_ob','element_observed'],   # Columns in the first DF (sig)
            right_on=['document','element'],  # Columns in the second DF (dvr)
            how='outer'  , 
            suffixes=('_m', '_base')
            )
            df_merged = df_merged.rename(columns={ 
                            'element_ob'  : 'key',
                            'freq_norm'   : 'observed',
                            'global_weight': 'expected' }) 
            df_merged['gap_val']=df_merged['observed']-df_merged['expected']
            
            print (df_merged.head(10) )  
            docs = df_merged[['document']]

            lvs_per_document.plot_document (df_merged,dataset,docs) 

            ''' GIGI
            ecorpus = Corpus(df)
            ecorpus_dvr = ecorpus.create_dvr(equally_weighted=True)  # Corrected variable name
            esigs = ecorpus.create_signatures(distance="JSD")
            #
            espd = sockpuppet_distance(ecorpus, ecorpus, heuristic=False, distance="euclidean")
            chart = sockpuppet_matrix(espd)
            if chart is not None:
                try:
                    chart.save(f"results/{dataset}/sockpuppet_distance_matrix.png", scale_factor=4.0)
                    print(f"Sockpuppet distance matrix chart saved to results/{dataset}/sockpuppet_distance_matrix.png")
                except Exception as e:
                    print(f"Error saving sockpuppet distance matrix chart: {e}")

            espd.to_csv(f"results/{dataset}/sockpuppet_distance_matrix.csv", index=False)
            #
            # Top 10 distances chart
            '''
            try:
                top_changing = sig[sig.sum(0).abs().sort_values(ascending=False).head(10).index]
                chart = (
                    alt.Chart(
                        top_changing.reset_index()
                        .melt(id_vars="index")
                        .rename(
                            columns={
                                "index": "document",
                                "variable": "element",
                                "value": "Distance from expected",
                            }
                        )
                    )
                    .mark_line()
                    .encode(x="document:N", y="Distance from expected", color="element")
                    .properties(width=300, height=300, title="")
                )
                chart.save(f"results/{dataset}/top_10_distances.png", scale_factor=4.0)
                print(f"Top 10 distances chart saved to results/{dataset}/top_10_distances.png")
            except Exception as e:
                print(f"Error generating or saving top 10 distances chart: {e}")
 

    except Exception as e:
        print(f"Failure in generate_signatures: {e}")
        return None


# %%
# Define the pipeline  
# Reuse the functions from the basic example
# clean_data, filter_data, calculate_summary, save_results

def     process_data(file_path,agg_column,var_name,value_name,output_path,output_dic,processing_type,sig_file,dataset,graph,top,sig_length,short_names):
    """
    Pipeline function to load, unpivot, clean, and save data.

    Args:
        file_path (str): Path to the input CSV file.
        file_path2 (str): Path to the second input CSV file.
        ignore_columns (list): List of columns to ignore during unpivoting.
        columns_to_keep (dict): Columns to keep and their new names.
        agg_column (str): Column to aggregate by during unpivoting.
        var_name (str): Name for the variable column after unpivoting.
        value_name (str): Name for the value column after unpivoting.
        output_path (str): Path to save the processed CSV file.
        output_dic (dict, optional): Dictionary to save as a CSV file.
    """
    df = load_data(file_path) 
    if df is None:
        print("Pipeline aborted due to error in load_data.")
        return  # Stop the pipeline

    df_unpivoted = transform_names(df, agg_column, var_name, value_name)
    df=df_unpivoted
    if df_unpivoted is None:
        print("Pipeline aborted due to error in unpivot_data.")
        return
    
    df_cleaned ,entity_code_df = clean_data(df_unpivoted,short_names, dataset) 

    # print(df_cleaned  ) 
    if df_cleaned is None:
        print("Pipeline aborted due to error in clean_data.")  
        return
 
    save_results(df_cleaned,entity_code_df, output_path, output_dic)
    print("Pipeline execution complete!")

    print ("Generating signatures...")
    print (f"sig_file: {sig_file}, dataset: {dataset}, graph: {graph}, top: {top}, sig_length: {sig_length}, var_name: {var_name}, value_name: {value_name}")
    generate_signatures(df_cleaned,entity_code_df,sig_file,dataset,graph,top,sig_length,var_name,value_name)  
    print("signatures execution complete!")



# %%
def main():
    # 1. Set up argument parser
    #parser = argparse.ArgumentParser(description="Process data from a CSV file.")
    #parser.add_argument("--config", help="Path to the config file", default="config.toml")
    #args = parser.parse_args()
    config_file_path = 'LVS_for_datasets/config_demo.toml'  # Replace with your actual path

    # 2. Read the config file
    config = configparser.ConfigParser()
    #config.read(args.config)
    config.read(config_file_path)
    # 3. Get parameters from the config
    file_path = config.get("data", "file_path")
    file_path2 = config.get("data", "file_path2")    
    agg_column=config.get("proc","agg_column")
    var_name=config.get("proc","var_name") 
    value_name=config.get("proc","value_name")  
    processing_type = config.get("proc","processing_type")
    columns_to_remove = config.get("proc", "columns_to_remove").split(',') if config.has_option("proc", "columns_to_remove") else []
    # Convert to list if it's a comma-separated string
    columns_to_remove = [col.strip() for col in columns_to_remove if col.strip()]  # Remove empty strings
    # If the config file has no columns to remove, it will be an empty list 
    if not columns_to_remove:
        print("No columns to remove specified in the config file.")
    else:
        print(f"Columns to remove: {columns_to_remove}")    
    #   
    output_path = config.get("output", "output_path") 
    output_dic = config.get("output", "output_dic")  
    sig_file = config.get("output", "sig_file") 
    dataset = config.get("data", "dataset")
    graph = config.get("output", "graph")
    top = config.get("output", "top")
    sig_length = config.get("output", "sig_length")
    short_names  = config.get("output", "short_names")
    # constants 
    #ignore_columns = ['Entity','Code']
    #columns_to_keep = ['entity', 'element', 'frequency_in_document']  
    ignore_columns = []
    columns_to_keep = []
    # 4. Call the processing function
    process_data(file_path,agg_column,var_name,value_name,output_path,output_dic,processing_type,sig_file,dataset,graph,top,sig_length,short_names) 
    
if __name__ == "__main__":
    main() 