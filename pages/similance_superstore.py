import pandas as pd
import sqlite3
from io import StringIO

import logging
import traceback


__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# connection = sqlite3.connect('cache.db', timeout=100)
import os
import sys
import streamlit as st

# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

import streamlit as st
from langchain_community.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings
from PIL import Image  # Import the Image class from the PIL module

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try: 
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
        st.session_state.superstore_output_df.to_csv("src/resources/data/superstore_output.csv", index=False)


    def read_superstore_output():
        return pd.read_csv("src/resources/data/superstore_output.csv", header=0)


    def get_superstore_retrieved_df(retriever, val_df, pd):
        try:
            input_rows = val_df["row_as_text"].tolist()
            relevant_rows = []

            for i in range(0, len(input_rows)):
                for relevant_row in retriever.get_relevant_documents(input_rows[i]):
                    relevant_rows.append(
                        relevant_row.page_content + f"; Id: {relevant_row.metadata['Id']}")

            converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
            generated_df = pd.DataFrame(converted_rows).drop_duplicates()
            return generated_df
        except Exception as e:
            logger.error(f"Error in get_superstore_retrieved_df: {str(e)}")
            raise e


    def generate_look_alike_superstore(uploaded_file, k):
        try:
            if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
                with st.spinner('Uploading...'):
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    csv_file = StringIO(stringio.read())
                    pandas_df = pd.read_csv(csv_file, header=0)[rows_to_convert_superstore]
                    st.markdown("""Uploaded Data""")
                    st.write(pandas_df)
                    input_df = pd.DataFrame(pandas_df)
            else:
                raise ValueError(f"File format {uploaded_file.name.split('.')[-1]} not supported")

            test_df = get_row_as_text(input_df, rows_to_convert_superstore)

            retriever = st.session_state.vdb_superstore.as_retriever(search_kwargs={"k": int(k)})
            generated_df = get_superstore_retrieved_df(retriever, test_df, pd)
            logger.info(f"Generated look-alike data with {len(generated_df)} rows")
            return generated_df
        except Exception as e:
            logger.error(f"Error in generate_look_alike_superstore: {str(e)}")
            raise e


    def display_input_data():
        try:
            st.markdown("""**Superstore Data**""")
            superstore_data = pd.read_csv("src/resources/data/superstore_master.csv")
            st.write(superstore_data)
            st.markdown("""---""")
            logger.info("Superstore input data displayed successfully")
        except FileNotFoundError as e:
            logger.error(f"Error in display_input_data: Superstore master file not found. {str(e)}")
            st.error("Unable to load Superstore data. Please check if the file exists.")
        except pd.errors.EmptyDataError as e:
            logger.error(f"Error in display_input_data: Superstore master file is empty. {str(e)}")
            st.error("The Superstore data file is empty. Please check the file contents.")
        except Exception as e:
            logger.error(f"Unexpected error in display_input_data: {str(e)}")
            st.error("An unexpected error occurred while displaying Superstore data. Please try again later.")


    def superstore_generate_form():
        try:
            display_input_data()
            st.markdown("""**Input Data**""")
            with st.form('fileform'):
                uploaded_file = st.file_uploader("Upload customer data", type=st.session_state.supported_file_formats)
                k = st.number_input('Number of rows required:', placeholder='Enter number of rows to fetch per query:',
                                    value=20)
                submitted = st.form_submit_button('Generate', disabled=(k == ""))
                if submitted:
                    if uploaded_file is not None:
                        if uploaded_file.name.split(".")[-1] in st.session_state.supported_file_formats:
                            try:
                                with st.spinner('Generating...'):
                                    generated_df = generate_look_alike_superstore(uploaded_file, k)
                                st.write("Generated look-alike audiences.")
                                st.session_state.superstore_output_df = generated_df
                                st.write(st.session_state.superstore_output_df.drop(columns=["Response"]))
                                save_superstore_output()
                            except AttributeError as e:
                                logger.error(f"AttributeError in superstore_generate_form: {str(e)}")
                                st.error("Please submit the uploaded file.")
                            except Exception as e:
                                logger.error(f"Unexpected error in superstore_generate_form: {str(e)}")
                                st.error(f"An unexpected error occurred: {str(e)}")
                        else:
                            st.write(f"Supported file types are {', '.join(st.session_state.supported_file_formats)}")
                    else:
                        st.write("Please select a file to upload first!")
        except Exception as e:
            logger.error(f"Error in superstore_generate_form: {str(e)}")
            st.error("An unexpected error occurred. Please try again later.")


    superstore_generate_form()

    st.markdown("---")  
    st.markdown(
        "<footer style='text-align: center; color: gray;'>"
        "Powered by Dataverze | © 2024"
        "</footer>",
        unsafe_allow_html=True
    )
    logger.info("Similance superstore page loaded successfully")

except Exception as e:
    logger.error(f"An error occurred in the main application: {str(e)}")
    logger.error(traceback.format_exc())
    st.error("An unexpected error occurred. Please try again later or contact support.")