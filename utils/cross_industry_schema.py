cross_industry_qa_schema = {
    "name": "cross_industry_qa_schema",
    "description": "Generate multiple choice question-answer pairs comparing multiple industries",
    "input_schema": {
        "type": "object",
        "properties": {
            "qa_pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "industries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of industries involved in the question, using the b1-... format"
                        },
                        "qa_type": {
                            "type": "string", 
                            "description": "The type of cross-industry question (e.g., 'Cross-Industry Comparison', 'Metric Comparison', 'Trend Identification', 'Policy Impact Assessment')"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "The difficulty level of the question"
                        },
                        "question": {"type": "string", "description": "The question"},
                        "optionA": {"type": "string", "description": "Option A"},
                        "optionB": {"type": "string", "description": "Option B"},
                        "optionC": {"type": "string", "description": "Option C"},
                        "optionD": {"type": "string", "description": "Option D"},
                        "optionE": {"type": "string", "description": "Option E"},
                        "answer": {
                            "type": "string", 
                            "description": "The correct answer option with both the option capital letter and the actual answer"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "A brief explanation of why the answer is correct and how it relates to the industries involved"
                        },
                        "references": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "industry": {"type": "string", "description": "The industry code (b1-...)"},
                                    "volume": {"type": "string", "description": "The volume name (e.g., 'Volume B1—Apparel, Accessories & Footwear')"},
                                    "pages": {"type": "string", "description": "Relevant page number(s)"}
                                },
                                "required": ["industry", "volume", "pages"]
                            },
                            "description": "List of references for each industry involved in the question"
                        },
                        "themes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of sustainability themes addressed in the question (e.g., 'emissions', 'resource management', 'social responsibility')"
                        }
                    },
                    "required": [
                        "industries", "qa_type", "difficulty", "question", 
                        "optionA", "optionB", "optionC", "optionD", "optionE", 
                        "answer", "explanation", "references", "themes"
                    ]
                }
            }
        },
        "required": ["qa_pairs"]
    }
}