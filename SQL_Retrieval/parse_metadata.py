#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 18:46:26 2025

@author: admin
"""

import re
import json

def extract_table_data(log_text):
    table_name_pattern = re.compile(r"Running table (\S+)")
    extracted = []
    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = table_name_pattern.search(line)
        if match:
            table_name = match.group(1)
            # Look for the JSON in subsequent lines
            json_buffer = []
            j = i + 1
            found_json = False
            while j < len(lines):
                next_line = lines[j]
                if ' - __main__ - INFO - {' in next_line:
                    found_json = True
                    # Split to get only the JSON part
                    _, json_part = next_line.split(' - __main__ - INFO - ', 1)
                    json_buffer.append(json_part)
                    j += 1
                    # Continue collecting until we have balanced braces or end
                    brace_count = json_part.count('{') - json_part.count('}')
                    while j < len(lines) and brace_count > 0:
                        json_buffer.append(lines[j])
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        j += 1
                    break
                elif next_line.strip().startswith('{'):
                    found_json = True
                    json_buffer.append(next_line)
                    j += 1
                    break
                j += 1
            if found_json:
                # Join the buffer
                json_str = '\n'.join(json_buffer).strip()
                # Remove any trailing commas before closing braces
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                try:
                    json_obj = json.loads(json_str)
                    if all(key in json_obj for key in ["summary", "sample_query_1", "sample_query_2"]):
                        extracted.append({"table": table_name, "details": json_obj})
                except json.JSONDecodeError as e:
                    print(f"JSON parse error for table {table_name}: {e}")
            # Move i to j
            i = j
        else:
            i += 1
    return extracted

def load_log_to_string(file_path):
    """Load the content of a .log file into a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def format_extracted_data(extracted):
    """Format the extracted data into the specified structure."""
    formatted = []
    for idx, entry in enumerate(extracted, start=1):
        table = entry['table']
        json_data = entry['details']
        summary = json_data['summary']
        query1 = json_data['sample_query_1']
        query2 = json_data['sample_query_2']
        
        formatted.append(f"\n\n{idx}. {table}: {summary}")
        formatted.append(f"Sample queries:\na. {query1}")
        formatted.append(f"b. {query2}")
        
    formatted.append("\n\n[")
    for idx, entry in enumerate(extracted, start=1):
        table = entry['table']
        formatted.append(f"{table}, ")
    formatted.append("]")
    
    return "\n".join(formatted)



log_content = load_log_to_string('metadata-msme.log')
result = extract_table_data(log_content)
print(format_extracted_data(result))