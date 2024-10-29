import json
from types import SimpleNamespace
from typing import Any, Dict, List
import os

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


class MetricsSchema(BaseModel):
    relevancy_score: List[int]
    relevancy_industries_name: List[str]
    faithfulness_score: int

    @property
    def industry_metrics(self):
        return {
            industry: SimpleNamespace(relevancy_score=score)
            for industry, score in zip(
                self.relevancy_industries_name, self.relevancy_score
            )
        }


def load_context(industry: str) -> str:
    with open(f"./markdowns/{industry}/full_content.md") as f:
        return f.read()


def verify_question_quality(question_data: Dict[str, Any],) -> MetricsSchema:
    industries = question_data.get("industries", [question_data.get("industry")])
    industries = [industries] if isinstance(industries, str) else industries

    industry_contexts = {industry: load_context(industry) for industry in industries}
    full_context = "\n\n".join(industry_contexts.values())

    is_multiple_choice = all(f"option{chr(65+i)}" in question_data for i in range(5))

    prompt = f"""
    Critically evaluate the following question based on the provided context for the relevant {'industry' if len(industries) == 1 else 'industries'}. Be extremely rigorous and unforgiving in your assessment.

    Relevancy: Assess how precisely the question aligns with {'the' if len(industries) == 1 else 'each'} industry's specific context, challenges, and goals. A highly relevant question should directly address key aspects, metrics, or challenges unique to the industry. Be very strict - even slight deviations from industry-specific concerns should result in lower scores. ALSO: if the question does not cover all the industries required, it must be considered irrelevant and given a score of 1.

    Faithfulness: Measure how accurately the question {'and its options' if is_multiple_choice else ''} {'are' if is_multiple_choice else 'is'} grounded in the provided context{'s' if len(industries) > 1 else ''}. A faithful question must be directly answerable from the given information without any need for external knowledge or inference. Be extremely critical - even minor discrepancies or omissions should significantly impact the score.

    Score both metrics on a scale of 1 to 10, where:
    1-2: Completely irrelevant or unfaithful
    3-4: Major flaws in relevancy or faithfulness
    5-6: Moderate issues, but still lacking
    7-8: Generally good, with minor issues
    9-10: Excellent, near-perfect alignment

    Example Context:
    "The Apparel, Accessories & Footwear industry faces significant sustainability challenges, particularly in raw materials sourcing. Key metrics include:
    1. Percentage of raw materials third-party certified to environmental and/or social sustainability standards.
    2. Priority raw materials: Description of environmental and social risks and/or hazards associated with priority raw materials used for products.
    3. Environmental impacts in the supply chain: Percentage of (1) Tier 1 supplier facilities and (2) supplier facilities beyond Tier 1 that have completed the Sustainable Apparel Coalition's Higg Facility Environmental Module (Higg FEM) assessment or an equivalent environmental data assessment."

    Examples:
    Good Question (Relevancy: 9, Faithfulness: 10):
    "What percentage of raw materials in the Apparel, Accessories & Footwear industry should be third-party certified to environmental or social sustainability standards, according to the context?"
    This question directly addresses a specific metric mentioned in the industry context and can be answered solely based on the provided information.

    Very Bad Question (Relevancy: 2, Faithfulness: 1):  
    "What is the average salary of a fashion designer in New York City?"
    This question is neither relevant to the industry's sustainability metrics nor answerable from the given context.

    Full context for {'the' if len(industries) == 1 else 'each'} industry:
    {full_context}
 
    Question to evaluate: {question_data['question']} 

    {"".join([f"Option {chr(65+i)}: {question_data[f'option{chr(65+i)}']}" for i in range(5)]) if is_multiple_choice else "This is a free-text question. Consider the level of detail and specificity required to provide a comprehensive answer based solely on the given context."}

    Provide a relevancy score for {'the' if len(industries) == 1 else 'each'} industry and an overall faithfulness score. Be extremely critical and justify your scores.
    Your response MUST include 'industry_metrics' and 'faithfulness_score'. Be very critical in your assessment.
    """  # noqa: E501

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": f"You are an expert question evaluator. Given a question{'and options' if is_multiple_choice else ''} and full context for {'an' if len(industries) == 1 else 'multiple'} {'industry' if len(industries) == 1 else 'industries'}, you assess its relevancy {'for the industry' if len(industries) == 1 else 'for each industry'}, overall faithfulness. Your response MUST include 'industry_metrics', 'faithfulness_score'.",  # noqa: E501
            },
            {"role": "user", "content": prompt},
        ],
        response_format=MetricsSchema,
        temperature=0,
    )

    try:
        response_structure = response.choices[0].message.parsed

        # Create a MetricsSchema object
        metrics = MetricsSchema(
            relevancy_score=response_structure.relevancy_score,
            relevancy_industries_name=response_structure.relevancy_industries_name,
            faithfulness_score=response_structure.faithfulness_score,
        )

        return metrics
    except (IndexError, AttributeError, json.JSONDecodeError) as e:
        print(f"Error parsing LLM response: {e}")
        # Return a default MetricsSchema if parsing fails
        return MetricsSchema(
            relevancy_score=[1],
            relevancy_industries_name=["Unknown"],
            faithfulness_score=1,
        )


if __name__ == "__main__":
    # Example multiple-choice question
    mc_question = {
        "question": "What is the unit of measure for the 'Percentage of raw materials third-party certified to an environmental and or social sustainability standard, by standard' metric in the Apparel, Accessories & Footwear industry?",
        "optionA": "Number",
        "optionB": "Weight",
        "optionC": "Percentage (%) by weight",
        "optionD": "n/a",
        "optionE": "Volume",
        "industry": "b1-apparel-accessories-and-footwear",
    }

    # Example free-text question
    ft_question = {
        "question": "Describe the key topics covered in the 'Raw Materials Sourcing' section of the Apparel, Accessories & Footwear document, and explain the main takeaways for a company writing its sustainability report in this industry.",
        "industry": "b1-apparel-accessories-and-footwear",
    }

    # Run evaluations
    mc_result = verify_question_quality(mc_question)
    ft_result = verify_question_quality(ft_question)

    print("Multiple-choice question results:")
    print(f"Industry Metrics: {mc_result.industry_metrics}")
    print(f"Faithfulness Score: {mc_result.faithfulness_score}")

    print("\nFree-text question results:")
    print(f"Industry Metrics: {ft_result.industry_metrics}")
    print(f"Faithfulness Score: {ft_result.faithfulness_score}")
