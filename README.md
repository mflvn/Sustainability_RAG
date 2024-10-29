# IC_Thesis

## Prerequisites

- Python 3.x
- Install packages from `requirements.txt`.
- Input your own API keys as per the .env.example file

## Steps

1. **Remove Repeated PDF Pages**:

   ```bash
   python pdf_parse/remove_pdf_intro.py
   python pdf_parse/remove_pdf_intro_standards.py
   ```

2. **Convert to Images**:

   ```bash
   python pdf_parse/pdf_to_images.py
   python pdf_parse/pdf_to_images_standards.py
   ```

3. **Parse Images**:

   ```bash
   python pdf_parse/parser.py
   ```

4. **Generate Embeddings**:

   ```bash
   python pdf_parse/embed.py
   ```

5. **Generate Questions**:
   - Free-text:
     - Naive:

     ```bash
     python qa_experiments/naive_free_local.py
     python qa_experiments/naive_free_cross.py
     ```

     - Prompt-based:

     ```bash
     python qa_experiments/prompt_free_local.py
     python qa_experiments/prompt_free_cross.py
     ```

     - Few-shot:

     ```bash
     python qa_experiments/fewshot_free_local.py
     python qa_experiments/fewshot_free_cross.py
     ```



   - MCQ (Multiple Choice Questions):

        - Naive:

        ```bash
        python qa_experiments/naive_mcq_local.py
        python qa_experiments/naive_mcq_cross.py
        ```

        - Prompt-based:

        ```bash
        python qa_experiments/prompt_mcq_local.py
        python qa_experiments/prompt_mcq_cross.py
        ```

       - Few-shot:

        ```bash
        python qa_experiments/fewshot_mcq_local.py
        python qa_experiments/fewshot_mcq_cross.py
        ```

The final set of questions can be found in `final_questions/all_qas.json`

6. **Evaluate Questions**:

   ```bash
   python qa_check_agents/master_check.py
   ```

   change the input files  in the script to compare the different question generation techniques

7. **Evaluate RAG Performance**:
   a. Standard RAG techniques:

   ```bash
   python evaluation/rag_eval.py
   ```

   change the chosen parameters in the script to test different experiments, like model size, top k, retrievers, chunking namespace, rag technique. The list of each are defined in the utils/model.py.
   
    b. RAG pipeline:

   ```bash
   python evaluation/rag_pipeline.py
   ```

   c. LLM pipeline:

   ```bash
   python evaluation/llm_pipeline.py
   ```


## Extra

- **Fine-tune Model**:
  - Use `finetuning/preprocess.py` to generate the JSON:

   ```bash
   python finetuning/preprocess.py
   ```

  - Use Together AI CLI to fine-tune:

   ```bash
   together fine-tuning create --training-file <FILE-ID> -m <MODEL>
   ```

- **Test classifiers**:
  - BERT:

   ```bash
   python chatbot/bert/bert.py
   ```

  - ML based:

   ```bash
   python chatbot/ML/ml_class.py
   ```

  - LLM based:

   ```bash
   python chatbot/classifier_evaluate.py
    ```
