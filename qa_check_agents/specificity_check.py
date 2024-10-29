import json
from typing import Any, Dict, Union
import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def load_context(industry: str) -> str:
    with open(f"./markdowns/{industry}/full_content.md") as f:
        return f.read()


specificity_check_schema = {
    "name": "specificity_check_schema",
    "description": "Scores the specificity of the question-answer pair on a score of 1 to 10",
    "type": "object",
    "properties": {
        "specificity": {
            "type": "number",
            "description": "Specificity score between 1 and 10 with 1 being the least specific(most vague), and 10 being the most specific",
        }
    },
    "required": ["specificity"],
}


def verify_specificity(
    question_data: Dict[str, Any], full_context: str
) -> Union[Dict[str, float], None]:
    is_multiple_choice = all(f"option{chr(65+i)}" in question_data for i in range(5))

    prompt = f"""
    Full content:
    {full_context}

    Evaluate the specificity of the following question based on the provided context and compared to the highly specific question examples provided.
    Score the specificity on a scale of 1 to 10, where 1 is extremely broad or general and 10 is very specific. 
    {"Consider both the question itself and the answer options given." if is_multiple_choice else "Consider the question and the expected level of detail in the answer."}
    If a question requires a very specific answer directly from a specific sentence/part of the document, it is considered more specific. If a question can be answered in multiple ways or is broad, it is considered less specific.

    To help you, here are some example questions below:

    Highest specificity questions that score 10:
    "What is the unit of measure for the 'Percentage of raw materials third-party certified to an environmental and/or social sustainability standard, by standard' metric in the Apparel, Accessories & Footwear industry (as listed in the relevant table)?"

    High specificity questions that score 8:
    "What topics are covered in the 'Raw Materials Sourcing' section of the Apparel, Accessories & Footwear document, and what are the key takeaways for a company writing its sustainability report in this industry?"

    Medium specificity questions that score 6:
    "A company in the Household & Personal Products industry is facing water scarcity issues in multiple manufacturing locations. What is the most comprehensive approach to address this challenge in their sustainability report?"

    Low specificity questions that score 3:
    "How might the increasing focus on energy efficiency certifications in the appliance industry influence future regulatory trends and consumer behavior?"

    Lowest specificity questions that score 1:
    "What broader implications does the industry's focus on energy management have for environmental sustainability?"

    Now the question to be evaluated:
    Question: {question_data['question']}
    """

    if is_multiple_choice:
        prompt += f"""
        Answer: {question_data['option' + question_data['answer']]}
        """
    else:
        prompt += question_data["answer"]

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an expert question designer."},
            {"role": "user", "content": prompt},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "specificity_check_schema",
                    "parameters": specificity_check_schema,
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "specificity_check_schema"},
        },
        temperature=0,
    )

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = tool_call.function.arguments
        result = json.loads(arguments)
        return {
            "specificity_score": result["specificity"],
        }
    except (IndexError, AttributeError, json.JSONDecodeError) as e:
        print(f"Error parsing LLM response: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example multiple-choice question
    mc_question = {
        "question": "What is the unit of measure for the 'Percentage of raw materials third-party certified to an environmental and/or social sustainability standard, by standard' metric in the Apparel, Accessories & Footwear industry?",
        "optionA": "Number",
        "optionB": "Weight",
        "optionC": "Percentage (%) by weight",
        "optionD": "n/a",
        "optionE": "Volume",
    }

    # Example free-text question
    ft_question = {
        "question": "Describe the key topics covered in the 'Raw Materials Sourcing' section of the Apparel, Accessories & Footwear document, and explain the main takeaways for a company writing its sustainability report in this industry."
    }

    context = "Sample context for the questions."

    print("Multiple-choice question results:")
    print(verify_specificity(mc_question, context))

    print("\nFree-text question results:")
    print(verify_specificity(ft_question, context))
