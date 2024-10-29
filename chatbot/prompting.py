import json
import os
from typing import Dict, List, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

# Load industry descriptions
with open("./generate_qa/industry_descriptions.json") as f:
    INDUSTRY_DESCRIPTIONS = json.load(f)

INDUSTRY_DESCRIPTIONS_STRING = ",\n".join(
    [f"{k}: {v}" for k, v in INDUSTRY_DESCRIPTIONS.items()]
)


class IndustryClassification(BaseModel):
    industries: List[str] = Field(
        default_factory=list,
        description="List of relevant industries. Should be empty if no industries are directly related.",
    )


class MCQAnswer(BaseModel):
    answer: str = Field(
        ...,
        description="The correct option for the question, a single character from A to D.",
    )


class IndustryClassificationRetriever:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = f"""
        You are a sustainability reporting expert specializing in corporate sustainability reports using IFRS standards. Your task is to identify industries directly related to a given question, referring to these industry descriptions:

        {INDUSTRY_DESCRIPTIONS_STRING}

        Guidelines:
        1. Only return industries that are DIRECTLY and PRIMARILY relevant to the company's core business activities mentioned in the question.
        3. Be extremely precise: do not include industries that are only tangentially related or those that the company might interact with but are not part of its primary operations.
        4. Consider the context of corporate sustainability reporting when making your decision.
        5. If multiple industries are relevant, limit your selection to the 1-3 most applicable ones.
        6. Avoid including industries that might be part of the supply chain or waste management unless they are explicitly stated as a core part of the company's operations.
        GIVE at least 1 undustry and at most 3 industries. return the full code like b1-apparel-accessories-and-footwear.
        """

    def retrieve(self, query: str,) -> Tuple[str, List[str]]:
        # Step 1: Identify relevant industries
        industries = self._identify_industries(query)

        # Step 2: Extract relevant parts from each industry's markdown
        relevant_parts = self._extract_relevant_parts(query, industries)
        # print(relevant_parts)
        # Step 3: Combine relevant parts
        combined_content = "\n\n".join(
            [
                f"start of {industry} content:\n{content.strip()}end of {industry}****"
                for industry, content in relevant_parts.items()
            ]
        )

        # Step 4: Answer the question (MCQ only)
        answer = self._answer_mcq(query, combined_content)

        return answer.answer, industries, combined_content

    def _identify_industries(self, query: str) -> List[str]:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Identify the industry or industries DIRECTLY and PRIMARILY related to the company's core business activities in this question. Return an empty list if none are specifically relevant:\n{query}",
                },
            ],
            response_format=IndustryClassification,
        )

        return completion.choices[0].message.parsed.industries

    def _extract_relevant_parts(
        self, query: str, industries: List[str]
    ) -> Dict[str, str]:
        relevant_parts = {}
        for industry in industries:
            markdown_content = self.load_industry_markdown(industry)

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert in extracting all the relevant information from sustainability reports. Given a question and industry-specific content, your task is to extract all relevant parts that directly address the question. Do not answer the question. The goal is to use this information, given in this instance from {industry} to collect one by one the information in different industries.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nIndustry content:\n{markdown_content}\n\nExtract all  relevant parts that could partly help address the question from the content of the industry {industry}. I will ask you for the other industries later.",
                    },
                ],
                temperature=0.1,
            )

            relevant_parts[industry] = completion.choices[0].message.content
        return relevant_parts

    def _answer_mcq(self, query: str, combined_content: str) -> MCQAnswer:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in sustainability reporting. Your task is to answer the given multiple-choice question based on the relevant information provided from various industry sources. Respond with only the letter (A, B, C, or D) corresponding to the correct answer.",
                },
                {
                    "role": "user",
                    "content": f"\nRelevant information:\n{combined_content}\n\nQuestion: {query}. Please answer the question with only the letter corresponding to the correct answer:",
                },
            ],
            response_format=MCQAnswer,
        )

        return completion.choices[0].message.parsed

    @staticmethod
    def load_industry_markdown(industry: str) -> str:
        file_path = f"./markdowns/{industry}/full_content.md"
        if os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        else:
            return f"Markdown file for {industry} not found."
