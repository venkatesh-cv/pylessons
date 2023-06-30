import openai
import pinecone
from openai_utils import get_embeddings
from embeddings_example import generate_adav2_embeddings, load_data
import pandas as pd
import os

sample_text = "hello world"
index_name = "gen-qa-openai"

def get_embedded_data():
    df_statements = load_data()
    df_statements = generate_adav2_embeddings(df_statements)
    return df_statements;


def init_pinecone_index() -> pinecone.GRPCIndex:
    # initialize connection to pinecone (get API key at app.pinecone.io)
    api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
    # find your environment next to the api key in pinecone console
    env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"
    pinecone.init(api_key=api_key, environment=env)
    embedding_length = len(get_embeddings(sample_text))
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=embedding_length,
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )
    # connect to index
    index = pinecone.GRPCIndex(index_name)
    # view index stats
    # print(index.describe_index_stats())
    return index

def insert_into_vectordb(df_statements, index :pinecone.GRPCIndex):
    index.upsert_from_dataframe(df_statements)

def search(index:pinecone.GRPCIndex,text:str) -> None:
    encoded_query = get_embeddings(text)
    result = index.query(encoded_query,top_k=10, include_metadata=True)
    print(result)

def prime_database(index:pinecone.GRPCIndex) -> None:
    df_statements = get_embedded_data()
    df_statements["id"] = df_statements["metadata"]
    df_statements["metadata"] = df_statements["metadata"].apply(lambda x: {"text":x})
    print(df_statements)
    insert_into_vectordb(df_statements, index)


if __name__ == "__main__":
    index = init_pinecone_index()
    prime_database(index)
    search(index, "tell me about days of the week")