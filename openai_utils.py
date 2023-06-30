import os
import openai
import json
openai.api_type = "azure"
openai.api_base = "https://oainpusegaistudygroup01.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat():    
  # response = openai.Embedding.create(
  #   engine="text-embedding-ada-002",
  #   input = "Hello worl",'
  response = openai.ChatCompletion.create(
    engine ="gpt-35-turbo-0301",
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"Who is Barack Obama"},{"role":"assistant","content":"See this page https://en.wikipedia.org/wiki/Barack_Obama"},{"role":"user","content":"Who is Deepika Padukone"},{"role":"assistant","content":"See this page https://en.wikipedia.org/wiki/Deepika_Padukone"},{"role":"user","content":"who is mike tyson. Can y ou show me his wiki page"}],
    temperature=0.2,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
  print("%s" % response["choices"][0]["message"]["content"])
  # print("Type is %s" % response.openai_id)

def get_embeddings(text:str) -> str:
  response = openai.Embedding.create(
  engine="text-embedding-ada-002",
  input = text,
  temperature=0.2,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
  # print("%s" % response["data"][0]["embedding"])
  # print(response.keys())
  return response["data"][0]["embedding"]


if __name__ == "__main__":
  # print("hello world")
  # chat()
  get_embeddings("world is a funny place")