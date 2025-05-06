#  Evaluation Report — Extractive Summarization (Week 5)

This report includes a quantitative (ROUGE) and qualitative evaluation of automatically generated summaries for three educational videos.

---

##  Example 1: Why Do We Dream?

### ROUGE scores
- **ROUGE-1** F1: 0.0690
- **ROUGE-2** F1: 0.0000 (wasn't performed)
- **ROUGE-L** F1: 0.0690

### Observations
-  The generated summary is quite generic and misses key elements from the original video (e.g., references to Freud and other theories).
-  A ROUGE-2 score of 0 indicates almost no overlap in word pairs.
-  Potential improvement: raise the TF-IDF relevance threshold or include more than one representative sentence.

---

##  Example 2: What is DNA?

### ROUGE scores
- **ROUGE-1** F1: 0.6667
- **ROUGE-2** F1: 0.3226
- **ROUGE-L** F1: 0.5455

### Observations
-  This summary is very close in content and structure to the original description.
-  Strong use of technical terms and clear sentence construction.
-  Great example of the effectiveness of the TF-IDF sentence scoring method.

---

##  Example 3: The History of the Universe

### ROUGE scores
- **ROUGE-1** F1: 0.3571
- **ROUGE-2** F1: 0.2308
- **ROUGE-L** F1: 0.3571

### Observations
-  The summary captures the main events (Big Bang, galaxy formation) but lacks the broader context.
-  Missing emphasis on the scale of time and cosmic evolution mentioned in the reference.
-  Could be improved by adding more historically rich or timeline-focused sentences.

---

##  Final Notes

- ROUGE metrics are useful for measuring similarity but cannot replace qualitative human judgment.
- The “What is DNA?” summary shows that even simple TF-IDF based extractive methods can produce high-quality results.
- For more complex topics, a supervised approach or semantic embeddings (e.g., BERT) could yield better summaries.

---

** Next step**: Refactor and modularize the `summarize()` function in preparation for **Week 6**.
