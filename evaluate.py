from rouge_score import rouge_scorer

def evaluate_summary(generated, reference):
    """
    Computes ROUGE-1 and ROUGE-L scores between a generated summary and a reference summary.

    :param generated: the summary produced by the model
    :param reference: the ground truth summary
    :return: dictionary with f-measure scores for ROUGE-1 and ROUGE-L
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }

# Example usage
if __name__ == "__main__":
    reference_summary = "Cells produce energy primarily in their mitochondria."
    generated_summary = "The mitochondria is the powerhouse of the cell."

    results = evaluate_summary(generated_summary, reference_summary)
    print("ROUGE-1:", round(results['ROUGE-1'], 4))
    print("ROUGE-L:", round(results['ROUGE-L'], 4))