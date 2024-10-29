import os
import re
import json

def extract_industry_description(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        match = re.search(r'# Industry Description\n\n(.*?)(?=\n#|\Z)', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

def create_industry_description_dict(root_folder):
    industry_descriptions = {}
    
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            markdown_path = os.path.join(folder_path, "full_content.md")
            if os.path.exists(markdown_path):
                description = extract_industry_description(markdown_path)
                if description:
                    industry_descriptions[folder_name] = description
    
    return industry_descriptions

root_folder = "markdown_output"
result = create_industry_description_dict(root_folder)

# Save the dictionary to a JSON file
output_file = "industry_descriptions.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)

print(f"Industry descriptions have been saved to {output_file}")