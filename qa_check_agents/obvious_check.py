import json
from typing import List, Dict, Any
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_questions(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def load_context(industry: str) -> str:
    with open(f"/homes/ml6823/fyp/Thesis/markdowns/{industry}/full_content.md", 'r') as f:
        return f.read()

obvious_check_schema = {
  "name": "obvious_check_schema",
  "description": "Checks that the multiple choice has a single best option",
  "type": "object",
  "properties": {
      "llm_answer": {
        "type": "string",
        "description": "Capital letter of the answer option chosen"
      }
    },
    "required": ["llm_answer"]
  }

def verify_obvious(question_data: Dict[str, Any]) -> Dict[str, Any]:

    prompt = f"""
        Give the letter of the correct answer option.

        Question: {question_data['question']}

        Option A: {question_data['optionA']}
        Option B: {question_data['optionB']}
        Option C: {question_data['optionC']}
        Option D: {question_data['optionD']}
        Option E: {question_data['optionE']}
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant that analyzes multiple-choice questions."},{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": {"name": "obvious_check_schema", "parameters": obvious_check_schema}}],
        tool_choice={"type": "function", "function": {"name": "obvious_check_schema"}},
        temperature=0
    )

    # llm_output = json.loads(response.choices[0].message.content)
    # return llm_output
    try:
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = tool_call.function.arguments
        return {
            "question": question_data["question"],
            "correct_answer": question_data["answer"],
            "llm_answer": arguments["llm_answer"],
            "is_obvious": arguments["llm_answer"][0].lower == question_data["answer"][0].lower,
        }
    except (IndexError, AttributeError) as e:
        print(f"Error parsing LLM response: {e}")
        return None

if __name__ == "__main__":
  questions = load_questions('/homes/ml6823/fyp/Thesis/generate_qa_traceable/mcq_local_output_traceable/b2-appliance-manufacturing_qa.json')

  results = [verify_obvious(q) for q in questions]

  with open('/homes/ml6823/fyp/Thesis/qa_check_agents/obvious_check.json', 'w', encoding="utf-8") as f:
      json.dump(results, f, indent=2)