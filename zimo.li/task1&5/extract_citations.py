# Task 1
# question: I handled the termination when 429 (specifically when daily call > 100k), 
# but didn't work on how to resume the next day

import sys
sys.path.append("/home/ubuntu/openalex")
import json
import requests
import time
from tqdm import tqdm
from src.tools.get_info import batch_extract_fields

# starttime = time.time()

input_path = "/home/ubuntu/openalex/workspace/zimo.li/test_input.jsonl"
output_path = "/home/ubuntu/openalex/workspace/zimo.li/test_output.jsonl"
perpage = 200 # 200 is already the max, more efficient than 50, 100, etc
sleep_time = 0.11 # good combo with perpage = 200, utilizing max per day with max per request

def extract_citations(input_path, output_path, perpage, sleep_time):
    with open(input_path, 'r') as file, open(output_path, 'w') as output_file:
        for line in file:
            paper = json.loads(line)
            base_url = paper["cited_by_api_url"]
            # base_url = "https://api.openalex.org/works?filter=cites:W2076003538"
            print("\nbase url:", base_url)
            citations = []
            page = 1

            while True:
                url = f"{base_url}&page={page}&per-page={perpage}"
                try:
                    response = requests.get(url)
                    data = response.json()
                    results = data["results"]
                    results = batch_extract_fields(results)
                    citations.extend(results)
                    if len(results) < perpage: #last page
                            break  
                    page += 1
                    time.sleep(sleep_time)

                except Exception as e: #should only be 429 caused by >100k visit per day
                    print(f"Error fetching url {url}: {e}") 
                    return

            print("Total citations fetched:", len(citations))

            paper["citations"] = citations
            output_file.write(json.dumps(paper) + "\n")


    print("\nCitations extraction completed.")
    # endtime = time.time()
    # print("Total time taken:", endtime - starttime)

extract_citations(input_path, output_path, perpage, sleep_time)