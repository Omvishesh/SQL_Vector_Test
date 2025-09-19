from   utils_common import llm_call, openai_call
from   textwrap import dedent
import re
import pandas as pd
import ast

def classify_query(query):
    system_instruction=dedent("""
                long prompt (removed for easier reading)
""")
    query_class = llm_call(system_instruction, query).strip()
    return query_class

def file_selector_agriculture_and_rural(query):
    system_instruction = dedent(f"""
        long prompt (removed for easier reading)
 """)
    selected_file = openai_call(system_instruction, query).strip()
    return selected_file

def file_selector_enterprise_establishment_surveys(query):
    system_instruction=dedent(f"""
       long prompt (removed for easier reading)
     """)
    selected_file = openai_call(system_instruction, query).strip()
    return selected_file

def file_selector_social_migration_and_households(query):
    system_instruction=dedent(f"""
        long prompt (removed for easier reading)
  """)
    selected_file = openai_call(system_instruction, query).strip()
    return selected_file

def file_selector_CPI(query):
    system_instruction=dedent(f"""
        long prompt (removed for easier reading)
 """)
    selected_file = llm_call(system_instruction, query).strip()
    return selected_file

def file_selector_GDP(query):
    system_instruction=dedent(f"""
        long prompt (removed for easier reading)

""")
    selected_file = openai_call(system_instruction, query).strip()
    return selected_file

def file_selector_IIP(query):
    system_instruction=dedent(f"""
        long prompt (removed for easier reading)
""")
    selected_file = openai_call(system_instruction, query).strip()
    return selected_file

def file_selector_MSME(query):
    system_instruction=dedent(f"""
        long prompt (removed for easier reading)
""")
    selected_file = llm_call(system_instruction, query).strip()
    return selected_file



#6.msme_gdp_by_region_view : Use this table when the question involves GDP contribution of MSMEs grouped by region, subregion, or country, it contains GDP contribution of MSMEs by region/subregion and country within Asia
#7.msme_gdp_by_sector_view : Use this table when the question involves GDP contribution of MSMEs by sector within a region or country.it contains GDP contribution of MSMEs by sector across Asian countries and regions

### Table Selection Examples, **Few-shot for msme_gdp_by_region_view**
#Select file: msme_gdp_by_region_view for queries like those listed below-
#Query1: "What is the MSME GDP in South Asia in 2022?"
#Query2: "Compare MSME GDP across Southeast Asia and Pacific Islands."

### Table Selection Examples, **Few-shot for msme_gdp_by_region_view**
#Select file: msme_gdp_by_region_view for queries like those listed below-
#Query1: "What is the MSME GDP in South Asia in 2022?"
#Query2: "Compare MSME GDP across Southeast Asia and Pacific Islands."


def rephrase_for_table(query, schema, context, table_name):
    instructions = f"""
    long prompt (removed for easier reading)
"""
    rephrased_for_table = openai_call(instructions, query).strip() #openai_call(instructions, query)
    return rephrased_for_table

def identify_generic_columns(schema):
    try:
        #print("Received schema:" )
        #print(schema)
        lines = ast.literal_eval(schema)
        col_list = []
        for line in lines:
            if "Schema" not in line:
                # Use ast.literal_eval to safely parse the tuple
                # Optionally, convert to dict for clarity
                col_dict = {
                    'column_name': line[0],
                    'data_type': line[1],
                    'nullable': line[2],
                    'default': line[3]
                }
                #print("Found tuple: " + str(col_dict))
                if ("id" not in col_dict['column_name']) and \
                   ("year" not in col_dict['column_name']) and \
                   ("month" not in col_dict['column_name']) and \
                   ("data" not in col_dict['column_name']) and \
                   ("date" not in col_dict['column_name']) and \
                   ("released_on" not in col_dict['column_name']) and \
                   ("updated_on" not in col_dict['column_name']) and \
                   (("character varying" in col_dict['data_type']) or ("text" in col_dict['data_type'])):
                       col_list.append(col_dict['column_name'])
                       #col_list.append("WHERE " + col_dict["column_name"] + " = ")
    except:
        col_list = []
    return col_list

def generate_sql_query(query, schema, context, table_name, last_error="N/A"):
    #if "resulted in no data" not in last_error:
        #columns = str(identify_generic_columns(query, schema))
        #query = str(query) + '''\nEnsure you set the following columns to generic values: ''' + columns)
    if last_error == "N/A":
        sql_query = rephrase_for_table(query, schema, context, table_name)      
        return sql_query, query
          
    instructions = dedent(f"""long prompt (removed for easier reading)
 """)
    if last_error != "N/A":
        instructions += dedent(f"""EXTREMELY IMPORTANT: Keep in mind that your last attempt returned the error: {last_error}
        """)
    sql_query = openai_call(instructions, query).strip() #openai_call
    return sql_query, query

def table_citation(selected_file):
    table_list = {
    }

    try:
        citation = table_list[selected_file]
    except:
        citation = "Unknown"
    return citation

def data_description(headers):
    system_instruction=dedent("""long prompt (removed for easier reading)
  """)
    description = openai_call(system_instruction, headers)
    return description

def rationalize_information(result, headers, query):
    if query == "":
        query = "Summarize the provided information, and state that this summary is being provided because the data size was too large to answer the query precisely."
    system_instruction=dedent(f""" 
                              long prompt (removed for easier reading)
""")
    rationalized_info = openai_call(system_instruction, query).strip()
    return rationalized_info

def handle_pandas_response(df, query, orig_query, max_rows, nq):
    df.fillna('', inplace=True) 
    df.to_csv("debug_dataframe.csv")
    headers = ""
    try:
        if len(df) <= 4:
            result = df.to_dict(orient='records')
            headers = data_description(headers + "\nData: " + str(result)).strip()
            return result, headers
        headers = ""
        # Find single-valued columns
        definite_drops = ["id", "data_release_date", "data_updated_date"]
        single_valued_cols = [col for col in df.columns if (col in definite_drops) or ((df[col].nunique(dropna=False) == 1) and (col.lower() != 'year'))]
        
        # Append their values to the caption
        for col in single_valued_cols:
            val = df[col].iloc[0]
            headers += f"{col}: {val} | "
    
        # Drop trailing delimiter if needed
        headers = headers.rstrip(" | ")
    
        # Drop single-valued columns from the dataframe
        df = df.drop(columns=single_valued_cols)
    
        # Define keywords that indicate temporal association
        temporal_keywords = ['year', 'month', 'quarter', 'date', 'day', 'week', 'period', 'time']
        
        # Create a regex pattern from the keywords
        pattern = re.compile('|'.join(temporal_keywords), re.IGNORECASE)
    
        # Find columns with headers matching any of the temporal keywords
        temporal_cols = [col for col in df.columns if pattern.search(col)]
        selected_temporal_cols = {}
    
        for keyword in temporal_keywords:
            # Filter matching columns for this keyword
            matches = [col for col in temporal_cols if keyword in col.lower() and 'data' not in col.lower()]
            
            # Prioritize numeric columns among the matches
            numeric_matches = [col for col in matches if pd.api.types.is_numeric_dtype(df[col])]
            
            if numeric_matches:
                selected_temporal_cols[keyword] = numeric_matches[0]  # Use the first numeric match
            elif matches:
                selected_temporal_cols[keyword] = matches[0]  # Fallback: first non-numeric match
    
        # Get the selected columns
        cols_to_merge = list(selected_temporal_cols.values())
        
        if 'date_stamp' in list(df):
            df['date'] = df['date_stamp'].astype(str)
        else:
            # Merge into a 'date' column
            if len(temporal_cols) == 1 and temporal_cols[0] == "year":
                df['date'] = pd.to_datetime(df['year'].str[:4], format='%Y')
            else:
                df['date'] = df[cols_to_merge].astype(str).agg('-'.join, axis=1)
        # Convert 'date' column to datetime (will produce NaT for unparseable rows)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        has_nat = df['date'].isna().any()
        if has_nat:
            df['date'] = df['year'].str.extract(r'^(\d{4})').astype(int)
            df['date'] = pd.to_datetime(df['date'], format='%Y')
        
        # Sort the DataFrame by the 'date' columnß
        df = df.sort_values(by='date',ascending=False).reset_index(drop=True)
        try:
            check_top_k = openai_call("""long prompt (removed for easier reading)
""", orig_query)
            if check_top_k == "YES":
            #if "top" in orig_query.lower():
                print("Identified as a top-k query, retaining only latest date")
                latest_date = df.loc[0, 'date']
                df = df[df['date'] == latest_date]
        except Exception as e:
            print("Could not assess whether query should keep latest date only: " + str(e))
        df = df.drop(columns=["date"])
        nrows = len(df)
        
        if (nrows <= 12) or (nq == 1):
            result = df.to_dict(orient='records')
            headers = data_description(headers + "\nData: " + str(result)).strip()
            return result, headers
        if 12 < nrows < max_rows:
            # We have more than 12 rows. Need some sort of rationalization.
            # headers += f" -- Data contains {nrows} rows, some rationalization will be needed -- "
            result = df.to_dict(orient='records')
            rationalized_info = rationalize_information(result, headers, orig_query + query).strip()
            headers = data_description(headers).strip()
            return {"summarized_info": rationalized_info}, headers
        # Data size has hit maximum limit. Need some sort of rationalization.
        # headers += f" -- Data contains {nrows} rows, some rationalization will be needed -- "
        result = df.to_dict(orient='records')
        rationalized_info = rationalize_information(result, headers, "Too many rows were pulled, but try to answer the following query from the data provided. Ensure that you mention the date period for which this is valid. **You MUST** use the information from the latest available time period for your summary: " + orig_query + query).strip()
        headers = data_description(headers).strip()
        return {"summarized_info": rationalized_info}, headers
    
    except:
        if "date" in list(df):
            df = df.drop(columns=["date"])
        headers = data_description(headers).strip()
        if len(df) > 100:
            df = df.iloc[:100,:]
        return df.to_dict(orient='records'), headers
