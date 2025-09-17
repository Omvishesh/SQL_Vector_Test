import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from encoder_v8 import emb_text, model
from milvus_utils_v8 import get_milvus_client, get_search_results, get_chunks_by_reference_page_pairs
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from dateutil.relativedelta import relativedelta
from textwrap import dedent
from google import genai
from google.genai import types
from   google.genai.types import Tool, GoogleSearch
from time import strftime, gmtime
import re
from typing import List, Dict
from math import ceil
import ast
import numpy as np
# Load environment variables
load_dotenv()

# API Key and Security
API_KEY = os.getenv("ACQ_API_KEY")
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
current_date = datetime.now().strftime('%Y-%m-%d')

# FastAPI instance
app = FastAPI(title="Deep Insights v8 Server for Top Vector Search Results")

# Milvus Configuration
DATA_INSIGHTS_V7_COLLECTION_NAME = os.getenv("DEEP_INSIGHTS_V8_COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
TOP_N_RESULTS = 5  # Configurable number of search results

print(f'MILVUS_ENDPOINT = {MILVUS_ENDPOINT}')
#cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", device="cpu")

# Milvus client
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

# Logging setup
logging.basicConfig(
    filename="deepinsights-v8-"+current_date+".log",  # Log file name
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# API Key verification dependency
async def verify_api_key(api_key: str = Depends(api_key_header)):
    logging.info(f"Received API Key: {api_key[:4]}****")  # Mask API key for security
    if api_key != os.getenv("ACQ_API_KEY"):
        logging.warning(f"Unauthorized API access attempt with key: {api_key[:4]}****")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Input model
class Question(BaseModel):
    question: str

def clarify_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are tasked with rephrasing the given query to make it easier for a RAG agent to pull the right data.
                Rules to produce rephrased query:
                    1. Do not edit the query as far as possible, only augment with a date range if none is present.
                        a. Remember that the current date is {curdate}.
                        b. If no date range is available from the context, try web search to attach a date.
                        c. If context is not available even from web search, use one year before {curdate} to {curdate}.
                        d. If financial year or FY is mentioned, it is April of the previous year to March of the mentioned year.
                            Example: FY25 -> April 2024 to March 2025
                                     FY23-24 -> April 2023 to March 2024
                                     FY 2025-26 -> April 2025 to March 2026
                    2. If the query contains acronyms, include full form in parentheses.
                        Example: Query -> What is RBI's thought process in May 2025?
                                Rephrased query -> What is RBI (Reserve Bank of India) thought process in May 2025?
                    3. Attach date range if no date range is available.
                        a. Include minimum and maximum date as far as possible (e.g. Feb 2025 to April 2025)
                        b. Always use month names and full years (e.g. April 2025, January 2022)
                    4. Do not attach extraneous information apart from this, or include your own thinking traces. Keep it as close to the original query as possible.
                    5. If the query does not mention "India", mention it explicitly.
                    6. Include synonyms and related quantities in the query.
                        Example: "Top 5 states by GDP" -> "Top 5 states by GDP, GVA, GSVA, GSDP, NSDP"
                        Example: "Contribution of agriculture to Madhya Pradesh economy" -> "Madhya Pradesh economy, Madhya Pradesh Agriculture, Madhya Pradesh GSDP"
                        Example: "Inflation in India" -> "Inflation in India, categories such as food and beverages, clothing, wholesale prices"
                    7. You MUST attach a minimum and maximum date to the output.
                    8. You MUST restrict your output to at most 25 words.

            The original query is given below.
            """),
            tools=[google_search_tool],
            temperature=0.0,
            ),
        contents=query
    )

    return response.text

def answer_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent("""Answer the following query by relying on a web search. Restrict your answer to 50 words, and emphasise results within the date range specified.
            """),
            tools=[google_search_tool],
            temperature=0.0,
            ),
        contents=query
    )

    return response.text


def final_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent("""Consider the following query and a suggested answer. Generate a single re-worded query for RAG retrieval, which emphasises key quantities at the start (e.g. Specific answer to the query) and also attempts to contrast with general responses.
                Example: "Top 5 states by GDP, examples are Maharashtra, Uttar Pradesh, Tamil Nadu" --> "Maharashtra state GSDP, top 5 states by GSDP (not India overall)"
                Exmample: "How has agriculture modernization affected per capita GDP of Madhya Pradesh" --> "Madhya Pradesh State GSDP, agriculture  (not other sectors)."
            """),
            tools=[google_search_tool],
            temperature=0.0,
            ),
        contents=query
    )

    return response.text

def identify_lexical_term(query):
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=dedent("""Extract a MAXIMUM of FOUR (4) key entity /entities from the statement below, which can be used for lexical matching in proposed answers. It/they should be state names, categories of product, macro-economic indicators (GDP, GVA, etc.), or sectors (agriculture, mining, etc.).

                IMPORTANT: They should be terms that are the "answer" in the text provided below.
                For example, "**NSDP:** Karnataka has the highest per capita NSDP." --> ['Karnataka']

                IMPORTANT: Look phrases similar containing the word "of", and use the predicates in your list.
                For example, "Impact of startups" --> ["startups"]
                For example, "Economy of Uttar Pradesh" --> ["Uttar Pradesh"]

                DO NOT return any other thinking trace apart from your answer(s) as a list [term1, term2]
                """),
                temperature=0.0,
                ),
            contents=query
        )
        response = ast.literal_eval(response.text)
        logging.info("Identified key terms: " + str(response))
        return response
    except:
        return []
    return []

def fetch_date(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are a date extractor. Your job is to extract a clear time reference point in time from user queries, if one exists.
            Examples:
            - "CPI report for December 2024" → "December 2024"
            - "What was the inflation rate in June 2023?" → "June 2023"
            - "Give me the latest IIP data" → "today"
            - "Tell me the GDP growth over the last five years" → "today"
            - "What happened in Q3 2022?" → "December 2022"
            - "What was the inflation rate H1 FY25?" → "September 2024"
            Return only the extracted date phrase (like "December 2024" or "today"). If no specific date is found, assume "today".
            IMPORTANT: If the extracted date contains multiple dates, output ONLY the LATEST date mentioned in the query.
            Make sure the output format of the date is "%B %Y"
            IMPORTANT: Do not output a date beyond {curdate}
            """),
            temperature=0.0,
        ),
        contents=query
    )

    query_date = response.text.strip()
    return query_date

def fetch_min_date(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are a date extractor. Your job is to extract a clear time reference point in time from user queries, if one exists.
            Examples:
            - "CPI report for December 2024" → "December 2024"
            - "What was the inflation rate in June 2023?" → "June 2023"
            - "Give me the latest IIP data" → "today"
            - "Tell me the GDP growth over the last five years" → "today"
            - "What happened in Q3 2022?" → "December 2022"
            - "What was the inflation rate H1 FY25?" → "September 2024"
            Return only the extracted date phrase (like "December 2024" or "today"). If no specific date is found, assume "today".
            IMPORTANT: If the extracted date contains multiple dates, output ONLY the EARLIEST date mentioned in the query.
            Make sure the output format of the date is "%B %Y"
            IMPORTANT: Do not output a date beyond {curdate}
            """),
            temperature=0.0,
        ),
        contents=query
    )

    query_date = response.text.strip()
    return query_date


def months_since(date_str, query_date='today'):
    #Date from the query if not use latest
    try:
        date_obj = datetime.strptime(date_str, '%B %Y')
        if query_date == 'today':
            today = datetime.today()
        else:
            try:
                today = datetime.strptime(query_date, '%B %Y')
            except:
                today = datetime.today()  # fallback

        diff = relativedelta(today, date_obj)
        logging.info("Pulled date: " + query_date + ", Doc date: " + date_str + ", Delta: " + str(diff.years * 12 + diff.months))
        return diff.years * 12 + diff.months

    except Exception as e:
        logging.warning(f'Error parsing date: {e}')
        return 999

def update_query_min_date(query_min_date: str, query_date: str, query_duration: int, min_months: int) -> str:
    # Convert input date strings to datetime objects
    query_min_date_dt = datetime.strptime(query_min_date, "%B %Y")
    query_date_dt = datetime.strptime(query_date, "%B %Y")

    # If duration is less than 3 months, update min date to 3 months before query date
    if query_duration < min_months:
        query_min_date_dt = query_date_dt - relativedelta(months=min_months)

    # Return updated min date as string in the same format
    return query_min_date_dt.strftime("%B %Y")

def build_range_around_date(center_date_str, months_before, months_after, field_name="date"):
    """
    Given a center date string like 'March 2024' and two integers (months_before, months_after),
    builds a Milvus filter expression for that range.

    Assumes Milvus stores dates in the format 'Month YYYY' (e.g., 'March 2024').
    """
    if center_date_str == 'today':
        center_date_str = datetime.today().strftime("%B %Y")
    try:
        center_date = datetime.strptime(center_date_str, "%B %Y")
    except:
        center_date_str = datetime.today().strftime("%B %Y")
        center_date = datetime.strptime(center_date_str, "%B %Y")

    # Calculate start and end of the range
    start_date = center_date - relativedelta(months=months_before)
    end_date = center_date + relativedelta(months=months_after)
    
    # Convert dates to string in yyyymmdd format
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # Build the filter expression
    filter_expr = f'{field_name} >= "{start_str}" and {field_name} <= "{end_str}"'
    return {"filter": filter_expr}

    # Generate the list of months
    current = start_date
    months = []
    while current <= end_date:
        months.append(current.strftime("%B %Y"))
        current += relativedelta(months=1)

    # Build the filter expression
    filters = [f'{field_name} == "{m}"' for m in months]
    filter_expr = " or ".join(filters)

    return {"filter": filter_expr}

def generalize_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent("""Consider the query given in the content. Your task is to generalize the query to a small extent. You can do this by:
            1. Removing mentions of specific states of India and replacing it by just "India". For example, "Tamil Nadu" can be replaced by "India".
            2. Removing specific day-month references and retaining only the years. For example, "23 May 2023" can be replaced by just "2023".
            3. Removing names of specific commodities and replacing them by broader classes. For example, "moong dal" can be replaced by "pulses" or "food".
            4. Removing generic phrases and focusing only on the data. For example, "Effect of COVID on steel production" can be replaced by "steel production statistics". Similarly, phrases such as "commentary", "effect", "policy" can be removed from the rephrased query.
            INSTRUCTIONS: Return the rephrased query as a single sentence. Do not hallucinate non-existing information. Stick to a maximum length of 12 words.
            """),
            temperature=0.0,
            ),
        contents=query
    )
    logging.info("Rephrased query: " + response.text)
    return response.text

def suggest_answer(query, excerpts):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Consider the following query: {query}. You are given the following content that contains a potential answer for this query. Write a short paragraph or set of bullet points that summarize the answer to the query.

            Here is some brief context about aspects of the Indian economy:
                1. Agriculture: Contributes about 15-20% to GDP, agriculture employs nearly half of the workforce. Monsoon patterns significantly impact agricultural output and rural demand.
                2. Industry: Manufacturing and construction are crucial for GDP growth and employment. Government initiatives like "Make in India" aim to boost manufacturing.
                3. Services: The services sector, including IT, finance, and telecommunications, contributes over 50% to GDP. IT services, in particular, are a major export and growth driver.
                4. Government Policies: Fiscal policies, such as taxation and public spending, influence economic growth. Monetary policies by the Reserve Bank of India (RBI) manage inflation and interest rates.
                5. Inflation: Influenced by food prices, fuel costs, and global commodity prices. The RBI uses repo rates and cash reserve ratios to control inflation.
                6. Foreign Direct Investment (FDI): FDI inflows boost infrastructure, technology, and job creation. Government policies aim to attract FDI in sectors like manufacturing and services.
                7. Global Economic Conditions: Exports, remittances, and foreign investments are affected by global demand and economic stability.
                8. Demographics: A young and growing workforce can drive economic growth, but requires adequate education and employment opportunities.
                9. Infrastructure: Investments in transportation, energy, and digital infrastructure enhance productivity and economic growth.
                10. Technological Advancements: Innovation and digitalization improve efficiency and competitiveness across sectors.

            IMPORTANT: Include ONLY the information that answers the query. If there is insufficient information to answer the specific query, say exactly the following in your response: "<insufficient-data>". Do not include any other text when there is insufficient information to answer the query.
            **Formatting instructions**
            - Make your answer about 300 to 350 words, IF sufficient data is present.
            - Use each of the excerpts to compose your answer, so long as they are relevant to the query.
            - Do not mention any data that is not available or not present. Stick to a summary of what data is available.

            IMPORTANT: In your summary, focus on specific data values, prices, and percentages if they relate to the original query. Avoid returning purely qualitative observations.
            IMPORTANT: If you are returning a full answer, do not include "<insufficient-data>" in the response.
            IMPORTANT: DO NOT HALLUCINATE ANY INFORMATION THAT IS NOT PRESENT IN THE ATTACHED CONTENTS.
            """),
            temperature=0.0,
            ),
        contents=excerpts
    )
    return response.text


def synthesize_with_gemini(
    question: str,
    unstructured_results: List[Dict]
) -> str:
    """
    Synthesizes a final answer using Gemini-2.0-Flash based on unstructured sources.
    """

    # Format top results
    formatted_sources = ""
    for idx, result in enumerate(unstructured_results, start=1):
        content = result.get("content", "").strip()
        reference = result.get("reference", "").strip()
        url = result.get("url", "").strip()

        formatted_sources += (
            f"### Source {idx}:\n"
            f"- **Reference**: {reference}\n"
            f"- **URL**: {url if url else 'N/A'}\n"
            f"- **Extracted Content**:\n{content}\n\n"
        )

    # System instruction prompt without structured data logic
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Based on the original question: {question}, and the following unstructured text data from various sources, synthesize a comprehensive and coherent answer. Integrate the information smoothly.

        **Formatting Instructions:**
        - Begin the final answer with this header: `## Insights from Ingested Data`
        - **When using any content from the 'Unstructured Text Data', cite the corresponding 'Reference' name and URL, but only if it is actually used.**
        - Indicate citations inline using square brackets like this: [1], [2], etc.
        - At the end of the answer, add a section titled `## References` that lists only the used references, numbered to match the inline citations.
        - Only include references that were cited in the text.
        - If no URL is available, skip the reference entirely.
        - Format the `## References` section exactly like this:

        ## References
        1. [Reference Name 1](https://example.com)
        2. [Reference Name 2](https://example.com)

        - Do not include any references that were not cited in the synthesized answer.
        - Avoid duplicate citations for the same source in the same paragraph — cite once per distinct point.
        - If data is conflicting or ambiguous, acknowledge that transparently in the summary.

            """),
            temperature=0.0,
            ),
        contents=formatted_sources
    )
    return response.text


# Search API Endpoint
@app.post("/search-topN", dependencies=[Depends(verify_api_key)])
async def search_topN_milvus(request: Request, question: Question):
    bin_size   =  2
    top_k      =  6
    min_months =  3

    start_time = time.time()
    request_time = datetime.utcnow().isoformat()

    llm_query = clarify_query(question.question).strip()
    #llm_query = (llm_query + "\n" + answer_query(llm_query).strip())
    #llm_query = final_query(llm_query).strip()
    #llm_query = (question.question).strip()
    suggest_answer = answer_query(llm_query).strip()

    try:
        query_date = fetch_date(llm_query).strip()
        if query_date == 'today':
            query_date = datetime.today().strftime("%B %Y")
        query_min_date = fetch_min_date(llm_query).strip()
        if query_min_date == 'today':
            query_min_date = datetime.today().strftime("%B %Y")
        query_duration = abs(months_since(query_min_date,query_date))
        logging.info(f"Query min date: {query_min_date}, max date: {query_date}, Query duration is {query_duration}")
    except:
        query_date = 'today'
        query_duration = 24
        query_min_date = (datetime.today() - relativedelta(months=query_duration)).strftime("%B %Y")
    if query_duration < min_months:
        query_min_date = update_query_min_date(query_min_date, query_date, query_duration, min_months)
        query_duration = min_months
    min_date = datetime.strptime(query_min_date, "%B %Y")
    max_date = datetime.strptime(query_date, "%B %Y")

    client_ip = request.client.host  # Get client IP address

    logging.info(f"Received request from {client_ip} at {request_time}")
    logging.info(f"Question Asked: {question.question}")
    logging.info(f"LLM Query Generated: {llm_query}")
    logging.info("Reference answer: " + suggest_answer)
    key_terms = identify_lexical_term(suggest_answer)

    if query_duration <= min_months:
        date_range = [(min_date, max_date), (max_date + relativedelta(months=1), max_date + relativedelta(months=min_months))]
    else:
        bin_size = min(4,max(2,query_duration // 12))
        step = max(1, ceil(query_duration // bin_size))
        date_range = []
        cur_date = min_date
        while cur_date < max_date:
            next_chunk = min(cur_date + relativedelta(months=step), max_date)
            date_range.append((cur_date, next_chunk))
            cur_date = next_chunk + relativedelta(months=1)
        date_range.append((max_date - relativedelta(months=2), max_date + relativedelta(months=2)))
        if (step > 3) and (query_duration >= 18):
            date_range.append((max_date + relativedelta(months=3), max_date + relativedelta(months=step)))
    bin_size = len(date_range)

    logging.info(
        f"""Overall date_filter: Min Date: {min_date.strftime("%B %Y")}, Max Date: {max_date.strftime("%B %Y")}, Total Months: {query_duration}"""
    )

    logging.info(f"Date Range tuples: {date_range}")

    try:
        # Track token count
        token_count = len(question.question.split())  # Approximate token count
        logging.info(f"Question Token Count: {token_count}")

        llm_token_count = len(llm_query.split())  # Approximate token count
        logging.info(f"LLM Query Token Count: {llm_token_count}")

        # Start embedding generation
        embed_start = time.time()
        query_vector = emb_text(model, llm_query)#; logging.info(query_vector)
        embed_time = time.time() - embed_start

        logging.info(f"Embedding generation time: {embed_time:.4f} seconds")

        total_search_time = 0.0
        top_results_to_return = []

        used_buckets = []
        used_indices = []

        for start_date, end_date in date_range:
            chunk_label = f"{start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
            logging.info(f"Processing range: {chunk_label}")
            months_before = (months_since(start_date.strftime("%B %Y"), query_date))
            months_after = (months_since(query_date, end_date.strftime("%B %Y")))
            milvus_date_filter = build_range_around_date(
                query_date, months_before, months_after
            )["filter"]
            #logging.info(str(milvus_date_filter))

            # Search in Milvus
            search_start = time.time()
            search_res = get_search_results(
                milvus_client, DATA_INSIGHTS_V7_COLLECTION_NAME, query_vector, ["content", "source", "id", "page", "reference", "date", "url"],
                milvus_date_filter, bin_size
            )
            search_time = time.time() - search_start
            total_search_time += search_time
            logging.info(f"Milvus search execution time: {search_time:.4f} seconds")
            logging.info(f"Document search date filter: {milvus_date_filter}")

            if not search_res or not search_res[0]:
                logging.warning(f"No results found for {chunk_label}")
                continue
                #raise HTTPException(status_code=404, detail="No results found")

            # Retrieve the top results
            top_results = [
                {
                    "id": result["id"],
                    "content": result["entity"]["content"],
                    "distance": result["distance"],
                    "source": result["entity"]["source"],
                    "page": result["entity"]["page"],
                    "reference": result["entity"]["reference"],
                    "date": result["entity"]["date"],
                    "url": result["entity"].get("url", "N/A")  # Default if not present
                }
                for result in search_res[0]
            ]
            n_results = len(top_results)

            # Log Top 15
            logging.info(f"Top {n_results} sources before reranking:")
            for i, item in enumerate(top_results, start=1):
                logging.info(
            #        f"Result - Content: {item['content']}, Page: {item['page']}, "
                    f"Source: {item['source']}, Reference: {item['reference']}, Date: {item['date']}, Distance: {item['distance']:.4f}"
                )

            #  Rerank with CrossEncoder
            #pairs = [(llm_query, str(item["content"]) + "\n\nResult from " + str(item["reference"]) + ", " + str(item['date'])) for item in top_results]
            pairs = [(llm_query + "\n" + suggest_answer, str(item["content"]) + "\n\nResult from " + str(item["reference"]) + ", " + str(item['date'])) for item in top_results]
            scores = cross_encoder.predict(pairs)
            counts = []
            for item in top_results:
                count = sum(term in str(item['content']) for term in key_terms)
                counts.append(count)
            counts = np.array(counts).astype(float)
            counts -= 0.25*len(key_terms)
            scores += counts
            penalty = np.array([10*(item['content'].count("\n")+item['content'].count("|"))/len(item['content']) for item in top_results])
            scores -= penalty
            scores = np.round(1 / (1 + np.exp(-scores)),decimals=3)
            logging.info("Scores: " + str(scores))
            logging.info("Lexical boosts: " + str(counts))
            logging.info("Noise penalty : " + str(penalty))
            try:
                logging.info("Attempting chunk addition")

                # 1. Find items in top_results with score >= 0.5 after normalization
                candidates = [
                    (item, score)
                    for item, score in zip(top_results, scores)
                    if score >= 0.5
                ]

                if candidates:

                    # 2. Generate [reference, page]—also including one page before and after

                    for item, _, in candidates:
                        reference  = item["reference"]
                        page       = int(item["page"])
                        current_id = int(item["id"])
                        #logging.info("Found current id " + str(current_id))
                        # Assuming 'page' is an integer
                        group_content = ""
                        new_search = []
                        for p in [page - 1, page, page + 1]:
                            new_search.append([reference, str(p)])

                        # 3. Retrieve all matching chunks
                        add_result = get_chunks_by_reference_page_pairs(
                            milvus_client,
                            DATA_INSIGHTS_V7_COLLECTION_NAME,
                            new_search
                        )

                        id_list    = [int(item["id"]) for item in add_result]
                        #logging.info("Found id list " + str(id_list))
                        secn_start = [1 if "[SECTION]" in item['content'] else 0 for item in add_result]
                        #logging.info("Sections start at " + str(secn_start))
                        pos        = id_list.index(current_id)
                        #logging.info("Found current section at " + str(pos))

                        before = None
                        for i in range(pos, -1, -1):
                            if secn_start[i] == 1:
                                before = i
                                break
                        if before is None:
                            before = max(0,pos - 1)

                        after = None
                        for i in range(pos+1, len(secn_start)):
                            if secn_start[i] == 1:
                                after = i
                                break
                        if after is None:
                            after = len(secn_start)

                        if [before, after] not in used_buckets:
                            used_buckets.append([before, after])
                            #logging.info("Collating between " + str([before, after]))
                            group_content = ""
                            for i in range(before, after):
                                if (len(group_content) < 10000) or (i <= pos):
                                    group_content += add_result[i]['content']

                            item['content'] = group_content

                    """
                    for item, _, in candidates:
                        reference = item["reference"]
                        page = int(item["page"])
                        # Assuming 'page' is an integer
                        group_content = ""
                        for p in [page - 1, page, page + 1]:
                            new_search = []
                            new_search.append([reference, str(p)])

                            # 3. Retrieve all matching chunks
                            add_result = get_chunks_by_reference_page_pairs(
                                milvus_client,
                                CPI_V6_COLLECTION_NAME,
                                new_search
                            )
                            if not (not add_result or not add_result[0]):
                                if p == page - 1:
                                    group_content += add_result[-1]['content']
                                if p == page + 1:
                                    group_content += add_result[0]['content']
                                if p == page:
                                    for item_inner in add_result:
                                        group_content += item_inner['content'] + "\n"
                        item['content'] = group_content[:10000]
                        """
                    n_results = len(top_results)

                else:
                    logging.info("No qualifying chunks")

            except Exception as e:
                logging.info("Failed with exception: " + str(e))

            # Let's assume each item in top_15 has a "date" field
            deltas   = [(months_since(datetime.strptime(item["date"], "%Y%m").strftime("%B %Y"),query_date)) for item in top_results] # Signed deltas, positive = older and negative = newer than query date
            #deltas   = [(months_since(item["date"],query_date)) for item in top_results] # Signed deltas, positive = older and negative = newer than query date
            if min(deltas) > 0:
                # Date is too recent, we do not have matching documents
                maxdelta = min(deltas)
            else:
                # We have at least one document matching the query date
                if max(deltas) < 0:
                    # Date is too old, we do not have documents that old
                    maxdelta = max(deltas)
                else:
                    maxdelta = 0
            maxdelta += 0.5*query_duration
            mindelta = maxdelta - query_duration
            lookup_delta = [[-2,max(2,query_duration)], [mindelta,maxdelta], [mindelta+6,maxdelta+6]]
            logging.info(str([mindelta,maxdelta,query_duration,min(deltas),max(deltas)]))
            chunks_found = False
            chunk_attempt = 0
            top_internal = []
            top_index   = []
            chunk_index = [xx for xx in range(len(deltas))]
            while ((not chunks_found) and (chunk_attempt < 3) and (len(top_internal) < 3)):
                if chunk_attempt < len(lookup_delta):
                    curtuple = lookup_delta[chunk_attempt]
                else:
                    curtuple = lookup_delta[-1]
                mindelta = curtuple[0]
                maxdelta = curtuple[1]

                chunk_attempt += 1
                logging.info("Deltas being used: " + str([mindelta,maxdelta]))
                #logging.info("Computed deltas")
                #logging.info(deltas)
                date_boosts = [0 if maxdelta >= deltas[xx] >= mindelta else 25 for xx in range(len(deltas))]

                # Add boosted scores to the cross_encoder scores
                #logging.info("Original scores: " + str(scores))
                final_scores = [s - boost for s, boost in zip(scores, date_boosts)]
                #logging.info("Modified scores: " + str(final_scores))

                # Now rerank based on the final boosted score
                reranked = sorted(
                zip(top_results, final_scores, chunk_index),
                key=lambda x: x[1],
                reverse=True
                )

                # Top 5 with cross_score filtering
                content_concat = []
                cross_thresh = 0.5 - 0.1*chunk_attempt
                for item, score, cur_index in reranked[:top_k]:
                    item["cross_score"] = np.round(float(score),decimals=3)
                    if (item["cross_score"] > cross_thresh) and (cur_index not in top_index) and (item["id"] not in used_indices):  # Only include results where cross_score > threshold
                        # Attach the reference URL
                        if item["cross_score"] < 0:
                            item["cross_score"] += 25
                        top_internal.append(item)
                        top_results_to_return.append(item)
                        top_index.append(cur_index)
                        used_indices.append(item["id"])
                        content_concat += item["content"]
                        #best_relevance = max(best_relevance,item["cross_score"])

                if len(top_internal) < 3:
                    logging.warning("Not enough valid results found in current attempt: (" + str(len(top_internal)) + "/"+str(top_k)+"). Relaxing cross score and deltas ..")
                    #mindelta += 6
                    #maxdelta += 6
                else:
                    chunks_found = True
            if len(top_internal) > 0:
                n_this_time = len(top_internal)
                logging.info(f"Appending {n_this_time} snippets for {chunk_label}")
            else:
                n_this_time = 0

        # Check if no valid results with cross_score > 0 were found

        if not top_results_to_return:
            logging.warning("No valid results with cross_score > 0")
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.4f} seconds")
            return {
                "question": question.question,
                "llm_query": llm_query,
                "query_date": query_date,
                "retrieved_results": [{
                    "content": "<insufficient_data>", #"We could not find any relevant content related to your query.",
                    "distance": "N/A",
                    "source": "N/A",
                    "page": "N/A",
                    "reference": "N/A",
                    "date": "N/A",
                    "url": "N/A"
                }],
                "time": total_time,
            }
        else:
            # Log Top 5
            top_results_to_return.sort(key=lambda item: item["cross_score"], reverse=True)
            final_return = []
            char_count   = 0
            for item in top_results_to_return:
                if char_count < 40000:
                    final_return.append(item.copy())
                    char_count += len(item["content"])
            n_final = len(final_return)
            logging.info(f"Top {n_final} results after reranking:")
            for i, res in enumerate(final_return, start=1):
                logging.info(
                    f"{i}. Content: {res['content'][:200]}..., Page: {res['page']}, "
                    f"Source: {res['source']}, Reference: {res['reference']}, Date: {item['date']}, Distance: {res['distance']:.4f}, Cross Score: {res['cross_score']:.4f}"
                )
            total_time = time.time() - start_time
            logging.info(f"Total processing time: {total_time:.4f} seconds")
            return {
                "question": question.question,
                "llm_query": llm_query,
                "query_date": query_date,
                "retrieved_results": final_return,
                "time": total_time,
            }


    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        logging.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=error_message)
