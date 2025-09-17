import json
from collections import Counter
input_path = "/home/ubuntu/raw_data/works/updated_date=2025-02-09/part_001.jsonl"
# input_path = "/home/ubuntu/openalex/workspace/zimo.li/test_input.jsonl"
output_path = "/home/ubuntu/openalex/workspace/zimo.li/test_output.jsonl"

data = []

with open(input_path, 'r') as f:
    for line in f:
        data.append(json.loads(line).get("ids", {}))

key_counter = Counter()

for d in data:
    key_counter.update(d.keys())

print(key_counter)