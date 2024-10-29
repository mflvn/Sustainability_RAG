import json
from typing import Any, Dict
import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def load_context(industry: str) -> str:
    with open(f"./markdowns/{industry}/full_content.md", "r") as f:
        return f.read()


sba_check_schema = {
    "name": "sba_check_schema",
    "description": "Checks that the multiple choice has a single best option",
    "type": "object",
    "properties": {
        "correct_answers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of the capital letter of the correct answer option(s), if any",
        }
    },
    "required": ["correct_answers"],
}


def verify_one_and_only_one_correct_answer(question_data, context):

    prompt = f"""
        Full context:
        {context}
        Based on the following question and the given context, determine if there is only one correct answer option, by identifying all correct answer options based on the reference text and full content provided. 
        Example: If the correct answer is A and B, the correct_answers should be ["A", "B"].
        If there is no correct answer, the correct_answers should be an empty list [].

        Question: {question_data['question']}

        Option A: {question_data['optionA']}
        Option B: {question_data['optionB']}
        Option C: {question_data['optionC']}
        Option D: {question_data['optionD']}
        Option E: {question_data['optionE']}
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": "You are an expert question designer. Given a question and full context, you check if it has one and only one correct option.",
            },
            {"role": "user", "content": prompt},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "sba_check_schema",
                    "parameters": sba_check_schema,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "sba_check_schema"}},
        temperature=0,
    )

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        return {
            "one_and_only_one_correct_answer": len(arguments["correct_answers"]) == 1,
            "correct_answers": arguments["correct_answers"]
        }
    except (IndexError, AttributeError) as e:
        print(f"Error parsing LLM response: {e}")
        return None
