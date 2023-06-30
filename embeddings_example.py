import openai
import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken

API_KEY = os.getenv("OPENAI_API_KEY") 
#RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 
RESOURCE_ENDPOINT = "https://oainpusegaistudygroup01.openai.azure.com/"

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2023-03-15-preview"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 

# This encodes the reference data as BPE tokens for GPT models. 
# This is not the vectorized values. But a preprocessing step
def generate_bpe_tokens(df_statements) -> pd.core.frame.DataFrame:
    #tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    df_statements['n_tokens'] = df_statements["metadata"].apply(lambda x: len(tokenizer.encode(x)))
    df_statements = df_statements[df_statements.n_tokens<8192]
    return df_statements


def generate_adav2_embeddings(df_statements: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    # df_statements = generate_bpe_tokens(df_statements)
    df_statements['values'] = df_statements["metadata"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002')) # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    return df_statements

# search through the embeddings for nearest neighbors
def search_docs(df_with_embeddings, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    #assign a similarity value in a new similarities column
    df_with_embeddings["similarities"] = df_with_embeddings["values"].apply(lambda x: cosine_similarity(x, embedding))

    # sort it by descending order to show the items with max similarity at the top
    res = (
        df_with_embeddings.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        print(res)
    return res

def load_data() ->  pd.core.frame.DataFrame:
    df_statements=pd.read_csv(os.path.join(os.getcwd(),'bill_sum_data.csv')) # This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
    pd.options.mode.chained_assignment = None
    return df_statements

if(__name__ == "__main__"):
    # response = requests.get(url, headers={"api-key": API_KEY})
    # print(response.text)
    df_statements = load_data()
    df_statements = generate_adav2_embeddings(df_statements)
    print(df_statements)
    res = search_docs(df_statements, "What do you think about the days of the week?", top_n=5, to_print = False)
    print(res)