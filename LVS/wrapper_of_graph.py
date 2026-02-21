import os
import configparser
import pandas as pd
import altair as alt
from LPA import Corpus
import lvs_per_country
import lvs_per_document

dataset='demo'

df_merged = pd.read_csv(f"results/{dataset}/df_merged.csv")
docs = pd.read_csv(f"results/{dataset}/docs.csv")

 

print(df_merged.head())
lvs_per_document.plot_document (df_merged,dataset,docs) 
lvs_per_country.plot_document  (df_merged,dataset,docs) 