import pandas as pd
import sqlite3
from io import StringIO

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
connection = sqlite3.connect('cache.db', timeout=100)
import os
import sys
from chromaviz import visualize_collection
import streamlit as st
from langchain_community.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings
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
    ## :blue[Customer Acquisition]  
    **Goal:** Identify customers who might avail our cross selling offers in Theatre
    
    **Story:** 
    As a bank, we possess data on card usage among our customers. Let's consider that we have 500,000 customers who utilized our card for purchasing movie tickets within the last 30 days. Furthermore, we have collected information on instances where some of these customers also made purchases for additional cinema services such as food or beverages. Additionally, as a bank, we are aware of the number of individuals who have already booked movie tickets for the upcoming week, which stands at approximately 15,000. Among these bookings, there are 1,000 cases where customers have already pre-booked additional services.  
    To enhance customer experience and engagement, the bank aims to offer a special service to those customers who have only purchased movie tickets without any additional services. By analyzing the historical data of the 500,000 customers or potentially a larger dataset, we can identify patterns that indicate which customers are more likely to purchase cinema services based on similarities with those who have already made such purchases for the same day.
    """)
rows_to_convert_movie = 'Age,FrequentWatcher,AnnualIncomeClass,ServicesOpted,AccountSyncedToSocialMedia,BookedFoodOrNot'.split(
    ",")

if 'movie_output_df' not in st.session_state:
    st.session_state.movie_output_df = None

if 'supported_file_formats' not in st.session_state:
    st.session_state.supported_file_formats = ["txt", "json", "csv"]

if 'vdb_movie' not in st.session_state:
    # If not defined, define it
    db_dir = "src/resources/embeddings/movie"
    # client = chromadb.PersistentClient(path=db_dir)
    vdb_movie = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                       collection_metadata={"hnsw:space": "cosine"})
    st.session_state.vdb_movie = vdb_movie


def get_movie_retrieved_df(retriever, val_df, pd):
    input_rows = val_df["row_as_text"].tolist()

    relevant_rows = []

    for i in range(0, len(input_rows)):
        target = []
        for relevant_row in retriever.get_relevant_documents(input_rows[i]):
            target.append(int(relevant_row.metadata['Target']))
        print(target)
        relevant_rows.append(
            input_rows[i] + f"; Target: {max(target)}")

    converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
    generated_df = pd.DataFrame(converted_rows).drop_duplicates()

    # return input_df.join(generated_df, how="inner", on=["infogroup_id", "mapped_contact_id_cont"])
    print(generated_df.head())
    st.write("Generated look-alike audiences.")
    st.write(generated_df)
    return generated_df


def generate_look_alike_movie(uploaded_file, k):
    if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
        with st.spinner('Uploading...'):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            csv_file = StringIO(stringio.read())
            pandas_df = pd.read_csv(csv_file, header=0)[rows_to_convert_movie]
            st.markdown("""Uploaded Data""")
            st.write(pandas_df)
            input_df = pd.DataFrame(pandas_df)
    else:
        raise Exception("File format {uploaded_file.name.split('.')[-1]} not supported")

    test_df = get_row_as_text(input_df, rows_to_convert_movie)

    retriever = st.session_state.vdb_movie.as_retriever(search_kwargs={"k": int(k)})
    generated_df = get_movie_retrieved_df(retriever, test_df, pd).drop(columns=["Target"])
    return generated_df


def movie_generate_form():
    succeeded = False
    st.markdown("""**Movie Data**""")
    movie_data = pd.read_csv("src/resources/data/movie_master.csv")
    st.write(movie_data)
    st.markdown("""---""")
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
                            generated_df = generate_look_alike_movie(uploaded_file, k)
                        st.session_state.movie_output_df = generated_df
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


movie_generate_form()
if st.button("Visualize Embeddings"):
    # db_dir = "src/resources/embeddings/insurance"
    # vdb = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings)
    visualize_collection(st.session_state.vdb_movie._collection)

st.markdown("---")  
st.markdown(
    "<footer style='text-align: center; color: gray;'>"
    "Powered by Dataverze | © 2024"
    "</footer>",
    unsafe_allow_html=True
)