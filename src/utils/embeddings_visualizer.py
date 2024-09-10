from chromaviz import visualize_collection
from langchain_community.vectorstores import Chroma
from src.utils.functions import  hf_embeddings


db_dir = "src/resources/embeddings/insurance"
vdb = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings)

# print(str(vdb._collection.count()))

visualize_collection(vdb._collection)