import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


def calculate_text_similarity_metrics(reference: str, candidate: str) -> dict:
    """
    Calculate BLEU and ROUGE scores for the given reference and candidate texts.

    Args:
    reference (str): The reference text
    candidate (str): The candidate text to be evaluated

    Returns:
    dict: A dictionary containing BLEU score and ROUGE scores
    """
    # Download necessary NLTK data
    nltk.download("punkt", quiet=True)
    if isinstance(reference, str) and isinstance(candidate, str):
        # Tokenize the texts
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())

        # Calculate BLEU score
        bleu_score = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method3,
        )

        # Calculate ROUGE scores
        rouge = Rouge()
        rouge_scores = rouge.get_scores(candidate, reference)[0]

        # Combine the scores into a single dictionary
        results = {
            "BLEU": bleu_score,
            "ROUGE": rouge_scores["rouge-l"]["f"],
        }

        return results
    else:
        return {
            "BLEU": 0,
            "ROUGE": 0,
        }


# Example usage
if __name__ == "__main__":
    reference = "The quick brown fox jumps over the lazy dog."
    candidate = "The fast brown fox leaps over the sleepy dog."

    results = calculate_text_similarity_metrics(reference, candidate)
    print("BLEU score:", results["bleu_score"])
    print("ROUGE-1 F1 score:", results["rouge_1_f"])
    print("ROUGE-2 F1 score:", results["rouge_2_f"])
    print("ROUGE-L F1 score:", results["rouge_l_f"])
