from  handler_sql import total_input_tokens, total_output_tokens
from api_main import markdown_answer


user_query = "List insurers with health premiums above 1000"

suggest_answer = markdown_answer(user_query, sql_responses["responses"], counter, total_input_tokens, total_output_tokens)