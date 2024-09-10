from chromaviz import visualize_collection

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# connection = sqlite3.connect('cache.db', timeout=100)
import os
import sys

from langchain_community.vectorstores import Chroma
from src.utils.functions import  hf_embeddings


db_dir = "src/resources/embeddings/insurance"
vdb = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings)

# print(str(vdb._collection.count()))

visualize_collection(vdb._collection)