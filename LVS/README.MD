# LVS

This repository contains the code for the LVS project.

## Input data 
* We accept 3 types of csv : 
* 1. full  , the features (elements) appears as columns , the documents appears as rows
For example : 

|Entity|Code|Year|Meningitis|Alzheimer|Parkinson|Malaria|HIV/AIDS|
|---|---|---|---|---|---|---|---| 
|Afghanistan|AFG|2007|2933|1402|450|2488|393| 
|Afghanistan|AFG|2008|2731|1424|455|2277|255|


The code is unpivotting the raw data and calculate frequencies per Year 

output : 

| element | frequency_in_document| document | 
|---|---|---|
|HIV/AIDS|0.0033616386602331435|2008|
|Cardiovascular diseases|0.3039135355246545|2008|
|Neoplasms|0.15561816876720513|2007|  

* 2. wide_unpivot, the features (elements) appears as rows , the documents appears as columns
For example : 

|element|F8|D8|E7|A1|E2|F2|
|---|---|---|---|---|---|---|
|0610007P14RIK|185.89362|182.45213|143.86440000000002|158.9239|154.0955|173.22320000000002|
|POLR2K|30.173896|32.088989|25.374459|30.583817|26.888652999999998|33.595935000000004|
|ADAM19|0.548329|0.909437|0.711346|0.151137|0.985998|0|

The code is unpivotting the raw data and calculate frequencies per feature in 2 phases : 

|document|0610007P14RIK|POLR2K|ADAM19|
|---|---|---|---|  
|F8|0.045|0.098|0.087|
|D8|0|0.0084|0.00054|
|E7|0.34|0.231|0|  

| element | frequency_in_document| document | 
|---|---|---|
|0610007P14RIK|0.045|F8|
|0610007P14RIK|0.098|D8|
|0610007P14RIK|0.087|E7|
|POLR2K|0.0084|D8|
|ADAM19|0|E7|  

* 3.freq - half processed data with frequencies 
For example : 

| element | frequency_in_document| document | 
|---|---|---|
|fellow|1|1790|
|immediately|1|1790|
|impression|1|1790|
|receive|3|1790| 

## Configuration

The configuration for the LVS project is stored in the `config_xxx.toml` [ xxx is the dataset name] file. This file is written in TOML format and contains the following sections:

### `[data]`

This section contains the configuration for the database. The following keys are supported:

* `file_path`: The input file of the dataset in csv format  [ ex - data/cod/annual-number-of-deaths-by-cause.csv]
* `file_path2`: Optional : Additional input file with summary information [ None , or filename  ex - data/cod/population.csv]
* `dataset`: The dataset name [ex - cod] 
 
file_path = data/cod/annual-number-of-deaths-by-cause.csv
file_path2 = data/cod/population.csv
dataset = cod 

### `[proc]` 
This processing instructions
* `processing_type`= [full/freq/wide_unpivot] - should we calculate frequencies only [freq], or also unpivot the dataset [full]/[wide_unpivot]
* `agg_column`= Year 
* `var_name`= Cause
* `value_name`= Deaths  
* `columns_to_remove` = E5,E6,E8  # columns that we like to remove from the original dataset 

### `[output]`
* `output_path` = results/cod/lvs.csv
* `output_dic` = results/cod/dic.csv 
* `sig_file` = results/cod/signatures.csv
* `graph` = [True/False] should we generate a graphs ? 
* `top` =  N top most dynamic features
* `sig_length` = length of signature (How many elements values in each signature) 
* `short_names` = False  # Do we like to encode long column names from the original dataset , with a shorter codes
### `constants` 
* `ignore_columns` = ['Entity','Code']
* `columns_to_keep` = ['document', 'element', 'frequency_in_document']  

Running full processing : lvs_data_processing.ipynb

| Parameter | Description |
|---|---|
| `file_path` | Path to the main CSV data file. |
| `file_path2` | Path to an optional secondary CSV file with summary information. |
| `dataset` | Name of the dataset. |
| `processing_type` | Specifies the type of processing: `full` for unpivoting and frequency calculation, or `freq` for frequency calculation only. |
| `agg_column` | Column used for aggregation (e.g., "Year"). |
| `var_name` | Name of the variable column (e.g., "Cause"). |
| `value_name` | Name of the value column (e.g., "Deaths"). |
| `output_path` | Path to the output CSV file containing processed data. |
| `output_dic` | Path to the output CSV file containing a dictionary or mapping. |
| `sig_file` | Path to the output CSV file for storing signatures. |
| `graph` | should we generate a graphs ?  |
| `top` | N top most dynamic features. |
| `sig_length` | length of signature (How many elements values in each signature) |
| `short_names`| Do we like to encode long column names from the original dataset , with a shorter codes|
| `columns_to_remove` |columns that we like to remove from the original dataset| 