import json
from typing import List, Dict
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def load_questions(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def load_context(industry: str) -> str:
    with open(f"./markdowns/{industry}/full_content.md", 'r') as f:
        return f.read()

def deepeval_evaluate_question(question_data: Dict):
    # Load the full context
    full_context = load_context(question_data['industry'])
    
    # Extract reference text and answer
    reference_text = question_data['reference_text']
    answer = question_data[f"option{question_data['answer'][0]}"]
    
    # Construct the query
    query = f"{question_data['question']}\n"
    query += f"A. {question_data['optionA']}\n"
    query += f"B. {question_data['optionB']}\n"
    query += f"C. {question_data['optionC']}\n"
    query += f"D. {question_data['optionD']}\n"

    # Initialize metrics
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
        verbose_mode=True
    )
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
        verbose_mode=True
    )

    # Check 1: Reference text against full context
    test_case_1 = LLMTestCase(
        input=query,
        actual_output=reference_text,
        retrieval_context=[full_context]
    )
    
    faithfulness_metric.measure(test_case_1)
    relevancy_metric.measure(test_case_1)

    check1_results = {
        'faithfulness': {
            'score': faithfulness_metric.score
        },
        'relevancy': {
            'score': relevancy_metric.score
        }
    }

    # Check 2: Answer against reference text
    test_case_2 = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=[reference_text]
    )
    
    faithfulness_metric.measure(test_case_2)
    relevancy_metric.measure(test_case_2)

    check2_results = {
        'faithfulness': {
            'score': faithfulness_metric.score
        },
        'relevancy': {
            'score': relevancy_metric.score
        }
    }

    return {
        'industry': question_data['industry'],
        'qa_type': question_data['qa_type'],
        'question': query,
        'reference_text': reference_text,
        'answer': answer,
        'check1_results': check1_results,
        'check2_results': check2_results
    }