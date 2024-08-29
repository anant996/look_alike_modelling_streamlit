import glob
import json
import re
from datetime import datetime

import pandas as pd
import sqlite3
from io import StringIO
# from langchain_openai import ChatOpenAI, OpenAI
from pyspark.sql import SparkSession
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from streamlit_image_select import image_select


def save_chart(query):
    q_s = ' If any charts or graphs or plots were created save them locally.'
    query += ' . ' + q_s
    return query


def check_plot_query(query):
    keywords = ['chart', 'charts', 'graph', 'graphs', 'plot', 'plt', 'plots']
    for keyword in keywords:
        if keyword in query.lower():
            return True
    return False


def run_query(agent, query_):
    if check_plot_query(query_.lower()):
        query_ = save_chart(query_)
    output = agent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    return response, intermediate_steps


__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# connection = sqlite3.connect('cache.db', timeout=100)
import os
import sys
import streamlit as st

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

import streamlit as st
from langchain_community.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings, run_query
from PIL import Image  # Import the Image class from the PIL module

st.set_page_config(page_title='Similance')
title_container = st.container()
col1, mid1, mid2, col2 = st.columns([0.25, 0.15, 0.4, 0.2])
image = Image.open('src/resources/ui_components/dataverze_logo.png')
with title_container:
    with mid1:
        st.image(image, width=100)
    with mid2:
        title = "Similance"
        st.title(title)

with st.container(border=True):
    st.markdown("""  
    ## :blue[CPG Retailers Campaign effectiveness ]  
    **Goal:** To make our Marketing Campaign more effective

    **Story:** 
    As a CPG / Retail company we run email campaigns to promote our latest offerings. Looking back at our previous efforts, we saw that some customers bought the product after receiving our emails. Now, as we plan our next campaign, we want to be more targeted in who we reach out to. 
    Sorting through our extensive list of customers, we want to identify patterns among those who made purchases before. We are not just looking at obvious things like age or location but can use this smart solution that can recognize similarities without explicit instructions. It scans through our data and pinpoints potential matches automatically. 
    Once we identify these similar customers, we craft personalized emails tailored to their interests and past behaviors, hoping to entice them to buy again. It's like inviting friends to join in on something they've enjoyed before. 
    By leveraging this approach, we aim to make our email campaign more effective, reaching those who are most likely to be interested in what we have to offer and driving sales in the process. 
    """)

rows_to_convert_superstore = 'Year_Birth,Education,Marital_Status,Income,Kidhome,Teenhome,Recency,MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,Complain,MonthsCustomer'.split(
    ",")

if 'superstore_output_df' not in st.session_state:
    st.session_state.superstore_output_df = None

if 'supported_file_formats' not in st.session_state:
    st.session_state.supported_file_formats = ["txt", "json", "csv"]

if 'vdb_superstore' not in st.session_state:
    # If not defined, define it
    db_dir = "src/resources/embeddings/superstore"
    # client = chromadb.PersistentClient(path=db_dir)
    vdb_superstore = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                            collection_metadata={"hnsw:space": "cosine"})
    st.session_state.vdb_superstore = vdb_superstore


def save_superstore_output():
    st.session_state.superstore_output_df.toPandas().to_csv("src/resources/data/superstore_output.csv", index=False)


def read_superstore_output():
    return pd.read_csv("src/resources/data/superstore_output.csv", header=0)


def get_superstore_retrieved_df(retriever, val_df, spark):
    input_rows = val_df.rdd.map(lambda x: x.row_as_text).collect()
    relevant_rows = []

    for i in range(0, len(input_rows)):
        for relevant_row in retriever.get_relevant_documents(input_rows[i]):
            relevant_rows.append(
                relevant_row.page_content + f"; Id: {relevant_row.metadata['Id']}")

    converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
    generated_df = spark.createDataFrame(converted_rows).distinct()
    # return input_df.join(generated_df, how="inner", on=["infogroup_id", "mapped_contact_id_cont"])
    return generated_df


def generate_look_alike_superstore(uploaded_file, k):
    if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
        with st.spinner('Uploading...'):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            csv_file = StringIO(stringio.read())
            pandas_df = pd.read_csv(csv_file, header=0)[rows_to_convert_superstore]
            st.markdown("""Uploaded Data""")
            st.write(pandas_df)
            spark = SparkSession.builder.appName("example").getOrCreate()
            input_df = spark.createDataFrame(pandas_df)
    else:
        raise Exception("File format {uploaded_file.name.split('.')[-1]} not supported")

    test_df = get_row_as_text(input_df, rows_to_convert_superstore)

    retriever = st.session_state.vdb_superstore.as_retriever(search_kwargs={"k": int(k)})
    generated_df = get_superstore_retrieved_df(retriever, test_df, spark)
    generated_df.show()
    return generated_df


def display_input_data():
    st.markdown("""**Superstore Data**""")
    superstore_data = pd.read_csv("src/resources/data/superstore_master.csv")
    st.write(superstore_data)
    st.markdown("""---""")


def superstore_generate_form():
    display_input_data()
    st.markdown("""**Input Data**""")
    with st.form('fileform'):
        uploaded_file = st.file_uploader("Upload customer data", type=st.session_state.supported_file_formats)
        k = st.number_input('Number of rows required:', placeholder='Enter number odf rows to fetch per query:',
                            value=20)
        submitted = st.form_submit_button('Generate', disabled=(k == ""))
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
                    try:
                        with st.spinner('Generating...'):
                            try:
                                generated_df = generate_look_alike_superstore(uploaded_file, k)
                            except:
                                generated_df = read_superstore_output()
                        st.write("Generated look-alike audiences.")
                        st.session_state.superstore_output_df = generated_df
                        st.write(st.session_state.superstore_output_df.drop("Response"))
                        save_superstore_output()
                    except AttributeError as e:
                        # Handling the AttributeError
                        st.write("Please submit the uploaded file.")
                        st.write(e)
                        # You can choose to perform alternative actions here if needed
                    except Exception as e:
                        # Handling any other exceptions
                        st.write(f"An unexpected error occurred: {e}")
                        raise e
                else:
                    st.write(f"Supported file types are {', '.join(st.session_state.supported_file_formats)}")
            else:
                st.write("Please select a file to upload first!")


superstore_generate_form()


def delete_images(image_paths):
    deleted_count = 0
    not_found_count = 0

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)
            deleted_count += 1
            # print(f"Deleted: {path}")
        else:
            not_found_count += 1
            # print(f"Not found: {path}")

    # print(f"\nDeleted {deleted_count} image(s).")
    # print(f"{not_found_count} image(s) not found.")


# def superstore_query_form():
#     if st.session_state.superstore_output_df is not None:
#         with st.form('queryform'):
#             input_df = read_superstore_output()
#             query = st.text_input(label="Query generated results", placeholder="Write your query here")  # add value
#             agent = create_pandas_dataframe_agent(OpenAI(temperature=0), input_df,
#                                                   return_intermediate_steps=True, save_charts=True, verbose=True)
#             query_submitted = st.form_submit_button('Submit query')
#             if query_submitted:
#                 with st.spinner("Generating ..."):
#                     response = "Agent stopped"
#                     while ("Agent stopped" in response):
#                         response, intermediate_steps = run_query(agent, query)
#                     st.write(response)
#                     # step = re.search(r'Action Input:\s*(.*)', str(intermediate_steps[-1])).group(1).strip()
#                     # st.write(f"Step:\n {step}")
#                     imgs_png = glob.glob('*.png')
#                     imgs_jpg = glob.glob('*.jpg')
#                     imgs_jpeeg = glob.glob('*.jpeg')
#                     imgs_ = imgs_png + imgs_jpg + imgs_jpeeg
#                     if len(imgs_) > 0:
#                         for img in imgs_:
#                             st.image(img)
#                     delete_images(imgs_)


# superstore_query_form()

# How many total complain have we received?
# Plot income distribution
# How many married individuals are there?
# Who was a customer for maximum months? whats that in years?
# whats the maximum numer of years an individual has been a customer?

st.markdown("---")  
st.markdown(
    "<footer style='text-align: center; color: gray;'>"
    "Powered by Dataverze | © 2024"
    "</footer>",
    unsafe_allow_html=True
)