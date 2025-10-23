#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 14:44:31 2025

@author: admin
"""

from   dotenv import load_dotenv
import os
from   langchain_community.utilities import SQLDatabase
from   sqlalchemy.pool import QueuePool
from   sqlalchemy import create_engine, text
import logging
from   datetime import datetime
from   langchain.prompts import ChatPromptTemplate
from   langchain_openai import ChatOpenAI
import ast
from   logging_utils import setup_logging, get_logger
load_dotenv("prod.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_URI = "postgresql://postgres:admin@localhost:5432/final"
db = SQLDatabase.from_uri(
    DATABASE_URI,
    engine_args={
        "poolclass": QueuePool,
        "pool_size": 20,
        "max_overflow": 10,
        "pool_recycle": 3600,
        "pool_timeout": 30
    }
)

current_date  = datetime.now().strftime('%Y-%m-%d')
# Setup logging with query ID support
setup_logging("metadata", logging.INFO)
logger = get_logger(__name__)

def return_table_list():
    engine = create_engine(DATABASE_URI)
    with engine.connect() as connection:
        table_list_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
              AND (table_type='BASE TABLE' OR table_type='VIEW');
        """
        table_list_return = connection.execute(text(table_list_query))
        table_list = table_list_return.fetchall()
        table_list_return = str([table[0] for table in table_list])
        return table_list_return

def generate_metadata():
    table_list = return_table_list()
    table_list = ast.literal_eval(table_list)
    engine     = create_engine(DATABASE_URI)
    for table in table_list:
        #if not (("gdp_" in str(table)) or ("gst_" in str(table)) or ("_gdp" in str(table)) or ("macro_" in str(table)) or ("statewise_" in str(table))):
        if ("_tbd" in str(table)) or (not ("imf_" in str(table))):
            continue
        logger.info("Running table " + str(table))
        print(str(table))
        try:
            pull_sample_query = "SELECT * FROM " + str(table) + " ORDER BY RANDOM() LIMIT 40;"
            logger.info("Trying to run: " + pull_sample_query)
            with engine.connect() as connection:
                sample_result = connection.execute(text(pull_sample_query))
                sample_cat = ""
                for row in sample_result:
                    sample_cat += str(row) + "\n"
            prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert metadata assistant. Given a set of sample rows from a table, generate a brief description of the information contained in the table.
                     # Important rules
                     - You should focus on citing examples of different categories or products, states of India, or all-India level data.
                     - Specify at what resolution the data is available (national, state-wise, city-wise, category-wise, etc.)
                     - Specify the time frequency of data (annual, quarterly, monthly, etc.)
                     - Provide examples of entries

                     # Most important
                     - Restrict your output to between 60 and 80 words
                     - Do not include any thinking traces apart from the suggested metadata for the table
                     - Separately, generate two sample queries in natural language which may be asked of this model
                     - Provide your output in json format with the following fields: ["summary" (containing 60-80 word summary), "sample_query_1", "sample_query_2"]
                     """),
                    ("human", "Sample: {sample_cat}"),
                ])
            llm = ChatOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY)
            chain = prompt | llm
            resp = chain.invoke({"sample_cat": sample_cat})
            logger.info(str(resp.content))
        except Exception as e:
            print("Failed for " + str(table) + ": " + str(e))

generate_metadata()