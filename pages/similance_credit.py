import pandas as pd
import sqlite3
from io import StringIO

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# connection = sqlite3.connect('cache.db', timeout=100)
import os
import sys

import streamlit as st
from langchain_community.vectorstores import Chroma
from src.utils.functions import get_row_as_text, hf_embeddings
from PIL import Image  # Import the Image class from the PIL module

import logging
import traceback

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
        ## :blue[Credit Card Renewals]  
        **Goal:** To retain credit card customers which might potentially churn away 
        
        **Story:** 
        The team at a bank is worried because many customers are quitting their credit card services. They want to figure out who might leave next so they can try to keep them happy. 
        Here's how we can help: We have a bunch of information about our customers and their credit card use. We also know who has already left us. By looking at all this data, we can find customers who are a lot like the ones who left before. 
        Then, we give each customer a score based on how much they're like the ones who left. The higher the score, the more likely they might leave, too. 
        Once we know who might leave, we can reach out to them with special offers or extra help to make them happy. It's like giving them a good reason to stick with us. 
        This way, we hope to keep more customers happy and stop them from leaving us. 
        """)

    rows_to_convert_credit = 'Customer_Age,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category,Months_on_book,Total_Relationship_Count,Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio'.split(
        ",")

    if 'credit_output_df' not in st.session_state:
        st.session_state.credit_output_df = None

    if 'supported_file_formats' not in st.session_state:
        st.session_state.supported_file_formats = ["txt", "json", "csv"]

    if 'vdb_credit' not in st.session_state:
        # If not defined, define it
        db_dir = "src/resources/embeddings/credit"
        # client = chromadb.PersistentClient(path=db_dir)
        vdb_credit = Chroma(persist_directory=db_dir, embedding_function=hf_embeddings,
                            collection_metadata={"hnsw:space": "cosine"})
        st.session_state.vdb_credit = vdb_credit


    def get_credit_retrieved_df(retriever, val_df, pd):
        try:
            input_rows = val_df["row_as_text"].tolist()
            relevant_rows = []

            for i in range(0, len(input_rows)):
                for relevant_row in retriever.get_relevant_documents(input_rows[i]):
                    relevant_rows.append(
                        relevant_row.page_content + f"; customer_id: {relevant_row.metadata['customer_id']}")

            converted_rows = [dict(pair.split(": ") for pair in row.split("; ")) for row in relevant_rows]
            generated_df = pd.DataFrame(converted_rows).drop_duplicates()
            logger.info(f"Retrieved {len(generated_df)} unique rows")
            return generated_df
        except Exception as e:
            logger.error(f"Error in get_credit_retrieved_df: {str(e)}")
            logger.debug(traceback.format_exc())
            raise e


    def generate_look_alike_credit(uploaded_file, k):
        try:
            if uploaded_file.name.split(".")[-1] not in st.session_state.supported_file_formats:
                raise ValueError(f"File format {uploaded_file.name.split('.')[-1]} not supported")

            with st.spinner('Uploading...'):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                csv_file = StringIO(stringio.read())
                pandas_df = pd.read_csv(csv_file, header=0)[rows_to_convert_credit]
                st.markdown("""Uploaded Data""")
                st.write(pandas_df)
                input_df = pd.DataFrame(pandas_df)

            test_df = get_row_as_text(input_df, rows_to_convert_credit)

            retriever = st.session_state.vdb_credit.as_retriever(search_kwargs={"k": int(k)})
            generated_df = get_credit_retrieved_df(retriever, test_df, pd)
            logger.info(f"Generated look-alike data with {len(generated_df)} rows")
            return generated_df
        except ValueError as e:
            logger.error(f"ValueError in generate_look_alike_credit: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in generate_look_alike_credit: {str(e)}")
            logger.debug(traceback.format_exc())
            raise e


    def credit_generate_form():
        try:
            st.markdown("""**Credit Card Churn Data**""")
            credit_data = pd.read_csv("src/resources/data/credit_master.csv")
            st.write(credit_data)
            st.markdown("""---""")
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
                                    generated_df = generate_look_alike_credit(uploaded_file, k)
                                    st.write("Generated look-alike audiences.")
                                    st.write(generated_df)
                                st.session_state.credit_output_df = generated_df
                                logger.info("Successfully generated and stored credit output data")
                            except AttributeError as e:
                                logger.error(f"AttributeError in credit_generate_form: {str(e)}")
                                st.error("Please submit the uploaded file.")
                            except Exception as e:
                                logger.error(f"Unexpected error in credit_generate_form: {str(e)}")
                                st.error(f"An unexpected error occurred: {str(e)}")
                        else:
                            logger.warning(f"Unsupported file type: {uploaded_file.name.split('.')[-1]}")
                            st.write(f"Supported file types are {', '.join(st.session_state.supported_file_formats)}")
                    else:
                        logger.warning("No file uploaded")
                        st.write("Please select a file to upload first!")
        except Exception as e:
            logger.error(f"Error in credit_generate_form: {str(e)}")
            st.error("An unexpected error occurred. Please try again later.")


    credit_generate_form()

    logger.info("Similance credit page loaded successfully")
except Exception as e:
    logger.error(f"An error occurred in the main application: {str(e)}")
    logger.error(traceback.format_exc())
    st.error("An unexpected error occurred. Please try again later or contact support.")