import json
import os
import glob

def format_for_llama3(qa_item):
    system_prompt = "You are a knowledgeable assistant specialized in sustainability reporting standards. Provide accurate and concise answers based on the given information."
    
    question = qa_item['question']
    options = f"A: {qa_item['optionA']}\nB: {qa_item['optionB']}\nC: {qa_item['optionC']}\nD: {qa_item['optionD']}\nE: {qa_item['optionE']}"
    correct_answer = qa_item['answer']
    reference = "\n".join(qa_item['reference_text'])
    
    user_message = f"Question: {question}\n\nOptions:\n{options}\n\nPlease provide the correct answer and explain why it's correct based on the following reference:\n{reference}"
    
    assistant_message = f"The correct answer is option {correct_answer}. Here's the explanation:\n\n"
    if correct_answer == 'A':
        assistant_message += f"{qa_item['optionA']} is the correct answer because it aligns with the information provided in the reference text. {reference}"
    elif correct_answer == 'B':
        assistant_message += f"{qa_item['optionB']} is the correct answer because it matches the details given in the reference text. {reference}"
    elif correct_answer == 'C':
        assistant_message += f"{qa_item['optionC']} is the correct answer as it accurately reflects the information stated in the reference text. {reference}"
    elif correct_answer == 'D':
        assistant_message += f"{qa_item['optionD']} is the correct answer as it corresponds to the information provided in the reference text. {reference}"
    elif correct_answer == 'E':
        assistant_message += f"{qa_item['optionE']} is the correct answer based on the details given in the reference text. {reference}"
    
    formatted_data = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_message}<|eot_id|>"""
    
    return formatted_data

def process_files(input_directory, output_file):
    all_formatted_data = []
    
    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_directory, '*_qa.json'))
    
    for file_path in json_files:
        with open(file_path, 'r') as file:
            qa_data = json.load(file)
            
            for qa_item in qa_data:
                formatted_item = format_for_llama3(qa_item)
                all_formatted_data.append({"text": formatted_item})
    
    # Write all formatted data to the output file
    with open(output_file, 'w') as outfile:
        for item in all_formatted_data:
            json.dump(item, outfile)
            outfile.write('\n')

# Usage
input_directory = 'generate_qa_traceable/mcq_local_output_traceable'
output_file = 'finetuning/llama3_finetuning_data.jsonl'

process_files(input_directory, output_file)
print(f"Formatted data has been written to {output_file}")