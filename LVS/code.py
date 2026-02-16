# %%
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
from math import floor
from scipy.spatial.distance import cdist, cityblock
import matplotlib.pyplot as plt
from visualize import sockpuppet_matrix, timeline

alt.data_transformers.disable_max_rows()
 

# %%
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}")
        print(f"DataFrame shape: {df.shape}") 
        #print 5 rows of the DataFrame
        print(df.head(5))
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None  # Important: Return None on error
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def unpivot_data(df, agg_column , var_name , value_name ,ignore_columns, processing_type,file_path2):  
    if df is None:
        print("Error: Input DataFrame is None. Skipping unpivot_data.")
        return None 
    try:
        if processing_type == 'full':
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
        elif processing_type == 'fullpivot':
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
            print(f"DataFrame fullpivot  shape: {df_melted.shape}") 
            #print 5 rows of the DataFrame
            print(df_melted.head(5)) 
            # Pivot the DataFrame to have 'Year' as index and 'Cause' as columns    
            df_melted = df_melted.pivot(index=agg_column, columns=var_name, values=value_name).reset_index()
            # Reset the index to make 'Year' a column again
            df_melted = df_melted.reset_index()
            # Rename the columns to have 'Year' as a column
            df_melted.columns.name = None  # Remove the name of the columns
            df_melted = df_melted.rename(columns={agg_column: 'symbol', var_name: 'element', value_name: 'frequency_in_document'})
            # Print the shape of the DataFrame
            print(f"DataFrame after pivot shape: {df_melted.shape}")
            # Print the first 5 rows of the DataFrame
            print(df_melted.head(5))         
        else:
            df_melted = df
            df_melted
            # Melt the DataFrame, using only 'Year' as the id_vars


        df_melted_grouped = df_melted.groupby([agg_column, var_name])[value_name].sum().reset_index()

        # Calculate the total deaths per year
        df_melted_grouped['Total_Per_Agg'] = df_melted_grouped.groupby(agg_column)[value_name].transform('sum')
        # Calculate the relative deaths
        # additional column to calculate the relative [Optional]  

        if file_path2 == 'None':
                            
            df_melted_grouped['frequency_in_document'] = df_melted_grouped[value_name] / df_melted_grouped['Total_Per_Agg']
        else:   
            
            df2 = pd.read_csv(file_path2)   
            # Merge with the population data
            df_melted_grouped  = pd.merge(df_melted_grouped, df2, left_on=agg_column, right_on='Year', how='inner') 
            df_melted_grouped['frequency_in_document'] = df_melted_grouped[value_name] / df_melted_grouped['Population']

        # Rename
        df_melted_grouped = df_melted_grouped.rename(columns={agg_column:'document',
                                var_name: 'element'}) 
        print(f"Unpivoted data shape: {df_melted_grouped.head(5)}")
        return df_melted_grouped  
    except KeyError as e:
        print(f"Error: Column not found: {e}.  Check your 'agg_column', 'var_name', 'value_name', and 'ignore_columns' parameters.")
        return None
    except Exception as e:
        print(f"Error during unpivoting: {e}")
        return None
def clean_data(df,columns_to_keep, short_names):
    """Cleans the DataFrame (e.g., keep only relevant columns , handles missing values, data type conversions)."""
    if df is None:
        print("Error: Input DataFrame is None. Skipping clean_data.")
        return None, None
    try:
        #clean elements that always = 0 in all the documents 
        print("Cleaning data...")
        # Group by 'element' and sum the frequency across all documents
        non_zero_elements = df.groupby('element')['frequency_in_document'].sum()
        print (f"Non-zero elements: {non_zero_elements}")
        #print zero elements
        print (f"Zero elements: {non_zero_elements[non_zero_elements == 0]}")
        # save the zero elements to a file
        non_zero_elements[non_zero_elements == 0].to_csv('zero_elements.csv')
        #save the non zero elements to a file
        non_zero_elements[non_zero_elements > 0].to_csv('non_zero_elements.csv')
        # Keep only elements with a non-zero total frequency
        non_zero_elements = non_zero_elements[non_zero_elements > 0].index
        # Filter the original DataFrame to keep only those elements
        filtered_df = df[df['element'].isin(non_zero_elements)]
        df= filtered_df
        #print(f"Filtered df 50: {df.head(50)}")
        df_cleaned = df.dropna()
        # Keep only the relevant columns
        df_cleaned = df_cleaned[columns_to_keep] 
        entity_code_df = None
        
            # Shorten the element names
        unique_elements = df['element'].unique()
        print(f"Unique elements: {unique_elements}")
        if short_names=='True':
            element_to_code = { element: f'E{i}' for i,  element  in enumerate(unique_elements) }
        else:
            # Create a mapping from element names to codes
            element_to_code = {element: element for i, element in enumerate(unique_elements)}
            
        df_cleaned['element'] = df_cleaned['element'].map(element_to_code)  
        # Create a DataFrame from the dictionary
        entity_code_df = pd.DataFrame(list(element_to_code.items()), columns=['element_name', 'element']) 
        #df_cleaned['amount'] = pd.to_numeric(df_cleaned['amount'], errors='coerce') 
        return df_cleaned, entity_code_df
    except KeyError as e:
        print(f"Error: Column not found: {e}. Check your 'columns_to_keep' parameter.")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None, None
    
def filter_data(df, column, condition):
    """Filters the DataFrame based on a condition."""
    print(f"Filtering data where {column} {condition}")
    return df[df[column] > condition]

def calculate_summary(df, group_by_column, aggregation):
    """Calculates summary statistics on the DataFrame."""
    print(f"Calculating summary by {group_by_column}...")
    return df.groupby(group_by_column).agg(aggregation)

def save_results(df,entity_code_df, output_path,output_dic):
    if df is None:
        print("Error: Input DataFrame is None. Skipping save_results.")
        return

    """Saves the processed DataFrames to a CSV file."""
    print(f"Saving results to: {output_path} and {output_dic}") 
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



def generate_signatures(df, entity_code_df, sig_file, dataset,graph,top,sig_length):
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

        corpus = Corpus(df, "document", "element", "frequency_in_document")
        dvr = corpus.create_dvr(equally_weighted=True)
        dvr
        dvr.to_csv(f"results/{dataset}/dvr.csv")
        top = int(top)
        sig_length = int(sig_length)

        sigs = corpus.create_signatures(distance="JSD",sig_length=sig_length, most_significant=top,prevalent=0.1)

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


           # Save element list
        with open(f"results/{dataset}/list.txt", "w") as f:
            for item in sigs[0]:
                f.write(f"{item}\n")
        # save list into dataframe  
        df_list = pd.DataFrame(sigs[0])
        pivot_the_list = df_list.melt(var_name='element', value_name='value', ignore_index=False)
        pivot_the_list = pivot_the_list.reset_index().rename(columns={'index': 'document'})
        df_list = pivot_the_list.dropna().reset_index(drop=True)   
        df_list.to_csv(f"results/{dataset}/list.csv", index=True)
        print(f"Element list saved to results/{dataset}/list.csv")      

        # Sockpuppet analysis
        if graph == 'True':
            ecorpus = Corpus(df)
            ecorpus_dvr = ecorpus.create_dvr(equally_weighted=True)  # Corrected variable name
            esigs = ecorpus.create_signatures(distance="JSD")
            espd = sockpuppet_distance(ecorpus, ecorpus, heuristic=False, distance="euclidean")
            chart = sockpuppet_matrix(espd)
            if chart is not None:
                try:
                    chart.save(f"results/{dataset}/sockpuppet_distance_matrix.png", scale_factor=4.0)
                    print(f"Sockpuppet distance matrix chart saved to results/{dataset}/sockpuppet_distance_matrix.png")
                except Exception as e:
                    print(f"Error saving sockpuppet distance matrix chart: {e}")

            espd.to_csv(f"results/{dataset}/sockpuppet_distance_matrix.csv", index=False)

            # Top 10 distances chart
            try:
                top_changing = sig[sig.sum(0).abs().sort_values(ascending=False).head(10).index]
                chart = (
                    alt.Chart(
                        top_changing.reset_index()
                        .melt(id_vars="index")
                        .rename(
                            columns={
                                "index": "Year",
                                "variable": "Element",
                                "value": "Distance from PM",
                            }
                        )
                    )
                    .mark_line()
                    .encode(x="Year:N", y="Distance from PM", color="Element")
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

def process_data(file_path, file_path2, ignore_columns, columns_to_keep, agg_column, var_name, value_name, output_path, output_dic, processing_type, sig_file,dataset,graph,top,sig_length,short_names):
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

    df_unpivoted = unpivot_data(df, agg_column, var_name, value_name, ignore_columns,processing_type, file_path2)
    if df_unpivoted is None:
        print("Pipeline aborted due to error in unpivot_data.")
        return

    df_cleaned ,entity_code_df = clean_data(df_unpivoted, columns_to_keep,short_names)
    # print(df_cleaned  ) 
    if df_cleaned is None:
        print("Pipeline aborted due to error in clean_data.")  
        return
 
    save_results(df_cleaned,entity_code_df, output_path, output_dic)
    print("Pipeline execution complete!")

    generate_signatures(df_cleaned,entity_code_df,sig_file,dataset,graph,top,sig_length)  
    print("signatures execution complete!")



# %%
def main():
    # 1. Set up argument parser
    #parser = argparse.ArgumentParser(description="Process data from a CSV file.")
    #parser.add_argument("--config", help="Path to the config file", default="config.toml")
    #args = parser.parse_args()
    config_file_path = '/Users/hagitbenshoshan/Documents/PHD/LVS_Code/LVS/config_alon1.toml'  # Replace with your actual path

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
    output_path = config.get("output", "output_path") 
    output_dic = config.get("output", "output_dic")  
    sig_file = config.get("output", "sig_file") 
    dataset = config.get("data", "dataset")
    graph = config.get("output", "graph")
    top = config.get("output", "top")
    sig_length = config.get("output", "sig_length")
    short_names  = config.get("output", "short_names")
    # constants 
    ignore_columns = ['Entity','Code']
    columns_to_keep = ['document', 'element', 'frequency_in_document']  
    #ignore_columns = []
    # 4. Call the processing function
    process_data(file_path,file_path2,ignore_columns,columns_to_keep,agg_column,var_name,value_name,output_path,output_dic,processing_type,sig_file,dataset,graph,top,sig_length,short_names) 
    
if __name__ == "__main__":
    main() 

# %%



