import json
from typing import Any, Dict, Union
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def load_context(industry: str) -> str:
    with open(f"./markdowns/{industry}/full_content.md", "r") as f:
        return f.read()


metrics_schema = {
    "name": "metrics_schema",
    "description": "Conducts binary checks for a question",
    "type": "object",
    "properties": {
        "negative_question": {
            "type": "integer",
            "description": "1 if the question is a 'negative' question, 0 otherwise",
        },
        "multihop_check": {
            "type": "integer",
            "description": "1 if the question is 'multihop', 0 otherwise",
        },
    },
    "required": [
        "negative_question",
        "multihop_check",
    ],
}


def verify_binary_checks(
    question_data: Dict[str, Any], context: str
) -> Union[Dict[str, int], None]:
    if "industries" in question_data:
        industries = question_data["industries"]
    else:
        industries = question_data.get("industry", [])

    is_multiple_choice = all(f"option{chr(65+i)}" in question_data for i in range(5))

    if is_multiple_choice:
        prompt = f"""
        Here is a multiple-choice question:
        Question: {question_data['question']}

        Option A: {question_data['optionA']}
        Option B: {question_data['optionB']}
        Option C: {question_data['optionC']}
        Option D: {question_data['optionD']}
        Option E: {question_data['optionE']}

        Evaluate the question based on the following metrics:

        "Negative Question": 1 if the question is phrased negatively, such as "Which of the following is NOT true?"; 0 otherwise.
        "Multihop Check": 1 if the question requires multiple steps to answer; 0 otherwise.

        Score all metrics as 1 or 0. Be harsh. If not clear, mark it as 0. Give the response in the given schema.
        """
    else:
        prompt = f"""
        Here is a free-text question:
        Question: {question_data['question']}

        Evaluate the question based on the following metrics:

        "Negative Question": 1 if the question is phrased negatively, such as "What is NOT a correct statement?"; 0 otherwise.
        "Multihop Check": 1 if the question requires multiple steps or pieces of information to answer; 0 otherwise.

        Score all metrics as 1 or 0. Be harsh. If not clear, mark it as 0. Give the response in the given schema.
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": "You are an expert question evaluator. Given a question and full context, you assess the given metrics.",
            },
            {"role": "user", "content": prompt},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "metrics_schema",
                    "parameters": metrics_schema,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "metrics_schema"}},
        temperature=0,
    )

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        return {
            "negative_question": arguments["negative_question"],
            "multihop_check": arguments["multihop_check"],
        }
    except (IndexError, AttributeError) as e:
        print(f"Error parsing LLM response: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example multiple-choice question
    mc_question = {
        "question": "Which of the following is NOT a renewable energy source?",
        "optionA": "Solar",
        "optionB": "Wind",
        "optionC": "Natural Gas",
        "optionD": "Hydroelectric",
        "optionE": "Geothermal",
        "industries": ["renewable_energy"],
    }

    # Example free-text question
    ft_question = {
        "question": "Explain the process of photosynthesis and its importance in the carbon cycle.",
        "industries": ["biology", "environmental_science"],
    }

    context = "Sample context for the questions."

    print("Multiple-choice question results:")
    print(verify_binary_checks(mc_question, context))

    print("\nFree-text question results:")
    print(verify_binary_checks(ft_question, context))
