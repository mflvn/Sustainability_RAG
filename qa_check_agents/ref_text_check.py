import json
from typing import List, Dict
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator
import os
# from uptrain import EvalLLM, Evals

# Create LLM
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)

def load_json_file(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def load_markdown(industry: str) -> str:
    with open(f"/homes/ml6823/fyp/Thesis/markdowns/{industry}/full_content.md", 'r') as f:
        return f.read()

def evaluate_reference_text(reference_text: str, markdown_content: str):
    faithfulness_eval_result = faithfulness_evaluator.evaluate(
        query="Evaluate the faithfulness of the given text.",
        response=reference_text,
        contexts=[markdown_content]
    )

    return {
        "faithfulness": {
            "passing": faithfulness_eval_result.passing,
            "score": faithfulness_eval_result.score,
        }
    }

def evaluate_entry(entry: Dict):
    industry = entry['industry']
    question = entry['question']
    reference_text = ' ;\n'.join(entry['reference_text'])
    markdown_content = load_markdown(industry)
    
    result = evaluate_reference_text(reference_text, markdown_content)
     
    return {
        'industry': industry,
        'question': question,
        'reference_text_evaluation': result
    }

# Load JSON file
entries = load_json_file('/homes/ml6823/fyp/Thesis/generate_qa_traceable/mcq_local_output_traceable/b1-apparel-accessories-and-footwear_qa.json')

# Evaluate each entry
results = [evaluate_entry(entry) for entry in entries]

# Save results to JSON file
with open('/homes/ml6823/fyp/Thesis/qa_check_agents/reference_text_faithfulness_check.json', 'w') as f:
    json.dump(results, f, indent=2)