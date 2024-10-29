import os
import re
import json
from collections import defaultdict

def extract_references(directory):
    references = defaultdict(list)
    
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            industry = filename.replace(".md", "")
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                
                # Extract industry description
                industry_desc = re.search(r'# Industry Description\n\n(.*?)\n\n', content, re.DOTALL)
                if industry_desc:
                    references[industry].append({
                        "type": "Industry Description",
                        "text": industry_desc.group(1).strip(),
                        "page": find_page_number(content, industry_desc.start())
                    })
                
                # Extract Table 1
                table1 = re.search(r'## Table 1\. Sustainability Disclosure Topics & Metrics\n\n\|.*?\n\|.*?\n(.*?)\n\n', content, re.DOTALL)
                if table1:
                    rows = table1.group(1).strip().split('\n')
                    for row in rows:
                        cells = [cell.strip() for cell in row.split('|')[1:-1]]
                        references[industry].append({
                            "type": "Metric",
                            "topic": cells[0],
                            "metric": cells[1],
                            "category": cells[2],
                            "unit": cells[3],
                            "code": cells[4],
                            "page": find_page_number(content, table1.start())
                        })
                
                # Extract Table 2
                table2 = re.search(r'## Table 2\. Activity Metrics\n\n\|.*?\n\|.*?\n(.*?)\n\n', content, re.DOTALL)
                if table2:
                    rows = table2.group(1).strip().split('\n')
                    for row in rows:
                        cells = [cell.strip() for cell in row.split('|')[1:-1]]
                        references[industry].append({
                            "type": "Activity Metric",
                            "metric": cells[0],
                            "category": cells[1],
                            "unit": cells[2],
                            "code": cells[3],
                            "page": find_page_number(content, table2.start())
                        })
                
                # Extract metric details
                metrics = re.findall(r'# (.*?)\n\n(.*?)\n\n', content, re.DOTALL)
                for metric_name, metric_content in metrics:
                    references[industry].append({
                        "type": "Metric Detail",
                        "metric": metric_name,
                        "text": metric_content.strip(),
                        "page": find_page_number(content, content.index(metric_name))
                    })
    
    return references

def find_page_number(content, position):
    pages = content.split('#### Page')
    for i, page in enumerate(pages):
        if len(page) > position:
            return i
    return len(pages)

# Usage
directory = "path/to/markdown/files"
references = extract_references(directory)

# Save to JSON file
with open('industry_references.json', 'w') as json_file:
    json.dump(references, json_file, indent=2)

print("References have been saved to industry_references.json")