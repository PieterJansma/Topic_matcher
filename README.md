# Topic_matcher

# GECKO Variable Mapping Experiments

This repository contains experiments for **mapping variable names** to **topics** also exploring existing **Gecko Ontology**, exploring how different text enrichment and embedding strategies affect performance.

We experiment with:
- raw labeled data,
- LLM-enriched definitions and synonyms,
- GECKO ontology true-matches,
- different embedding models (general vs. biomedical),
- and longer enriched definitions.

---

## Data Files

### 1. `Labeled_variables.xlsx`
- **Description**: Manually labeled dataset of variables with their parent categories.  
- **Columns**:  
  - `name`: variable name (e.g., *Pregnancy number*).  
  - `parent`: manually assigned parent category.  
  - `definition`: optional short definition (often missing or short).  

**Example:**
| name              | parent              | definition |
|-------------------|---------------------|------------|
| Pregnancy number  | Identifiers         | Pregnancy number      |
| Food groups (M)   | 	Nutrition | Vegetables without potatoes  during pregnancy,Fruits during pregnancy. etcc|
| Suprailiac skinfold   | Suprailiac skinfold | NaN |

---

### 2 LLM Enrichment

To improve coverage of missing or too-short definitions, we used **LLMs to auto-generate enriched definitions**.

- **Models used**
  - [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) for one-sentence definitions.
  - Same model also tested with prompts for **longer multi-sentence definitions**.

- **Prompts**
  - **Short definition:**  
    *"Write a concise, factual, single-sentence definition for the term between <term> tags. Use neutral scientific language."*
  - **Longer definition (experiment):**  
    *"Write a clear, factual 2–4 sentence definition for the term between <term> tags. Include what it measures or why it matters."*

- **Process**
  - The LLM generates a candidate definition.  
  - We check **cosine similarity** between the variable name and definition.  
  - If similarity ≥ threshold, the definition is accepted and stored as `definition_enriched`.  
  - `definition_source` records whether the final value is `original` or `llm_generated`.

- **Synonyms**
  - We also tried prompts such as:  
    *"Provide 3–6 synonyms or closely related terms for the term between <term> tags. Format: comma-separated list."*
  - **Result:** inconsistent outputs → synonyms are logged for audit but not yet used in evaluation.  
  - Future work: improve synonym generation and filtering.

**Example (`Labeled_variables_with_llm_enrichment.csv`):**

| name                | parent         | definition_enriched                                            | definition_source |
|---------------------|----------------|----------------------------------------------------------------|------------------|
| Pregnancy number    | Identifiers    | Pregnancy number is an information content entity that is the outcome of a dubbing process and is used to refer to one instance of entity shared by a group of people to refer to that individual entity               | llm_generated    |
| Food groups (M)     | Nutrition      | Nutritional history is a lifestyle history that is about the diet and nutrition of an individual            | llm_generated    |
| Suprailiac skinfold | Skinfold | Suprailiac skinfold is a morphological measurement of the skin fold.     | llm_generated    |


### 3. `parent_gecko_candidates_top5.xlsx`

This file contains the **top-5 candidate GECKO ontology classes** for each of the 89 parent categories.  
Candidates were generated using our **best-performing embedding model** (explained in the Results section).

- **Process**
  - For each parent, the model produced 5 candidate GECKO classes ranked by cosine similarity.  
  - A **human annotator** reviewed the candidates and selected the most appropriate match.  
  - In some cases, none of the 5 options were a perfect fit — the annotator then chose the *closest* class.  
  - Because of the difficulty of the task, not all matches are 100% correct.  
  - The goal was primarily to **enrich the dataset with additional ontology context** (labels + definitions), rather than to create a flawless gold standard.

- **Columns**
  - `parent` – the parent category from our dataset.  
  - `parent_definition` – enriched definition for the parent (LLM-generated when available).  
  - `gecko_label` – candidate GECKO ontology label.  
  - `gecko_definition` – definition from the GECKO ontology.  
  - `sim` – cosine similarity score between parent text and GECKO text.  
  - `auto_suggest` – automatically flagged if similarity ≥ threshold.  
  - `approved` – manual selection (`TRUE` if chosen as the best match).

**Example:**

| parent                | gecko_label     | gecko_definition                                     | sim  | approved |
|-----------------------|-----------------|------------------------------------------------------|------|----------|
| Pregnancy history     | Gravidity       | The number of times a female has been pregnant.      | 0.87 | TRUE     |
| Diet during pregnancy | Food intake     | The consumption of food and drink by an organism.    | 0.82 | TRUE     |


---

## 🧪 Experimental Strategies

We evaluated progressively richer strategies for text representation:

1. **Baseline (No enrichment)**  
   - Input = `name` only, or `name + original definition` (when available).  
   - Model: `all-MiniLM-L6-v2` baseline.  

2. **LLM Enrichment**  
   - Input = `name + definition_enriched` (LLM-generated).  
   - Synonyms included where available.  
   - Model: general semantic search models like `multi-qa-mpnet-base-dot-v1`.  

3. **LLM + GECKO Mapping**  
   - Add GECKO label/definition for parents (based on manual `TRUE` matches).  
   - Training embeddings use GECKO; query embeddings may exclude it (to simulate real-world usage).  

4. **Extended Definitions (Longer LLM enrichment)**  
   - LLM prompted to generate longer, more detailed definitions.  
   - Idea: give embedding models more semantic signal.  

---

## 🔬 Models Evaluated

We compared multiple embedding models:

### General-purpose Language Models
- `thenlper/gte-large`
- `multi-qa-mpnet-base-dot-v1`
- `sentence-t5-base`

### Biomedical / Domain Models
- `allenai/specter`
- `biobert-base-cased-v1.1`
- (optional future: `microsoft/BiomedNLP-PubMedBERT`)

---

## 📊 Results

### LOOCV Accuracy (Top-1 / Top-3)

| Strategy                    | Model                     | Top-1  | Top-3  | Notes |
|------------------------------|---------------------------|--------|--------|-------|
| Baseline (no enrichment)     | all-MiniLM-L6-v2          | 0.xxx  | 0.xxx  | Short input only |
| LLM Enrichment               | multi-qa-mpnet-base-dot-v1| 0.720  | 0.861  | Big boost with enriched defs |
| LLM Enrichment               | sentence-t5-base          | 0.718  | 0.857  | Similar to mpnet |
| LLM + GECKO (query=with GECKO) | thenlper/gte-large      | 0.774  | 0.896  | Best so far |
| LLM + GECKO (query=no GECKO) | thenlper/gte-large        | 0.769  | 0.886  | Robust even w/out GECKO |

👉 **Takeaway:**  
- Enrichment consistently improves accuracy.  
- GECKO adds further boost, but results hold up even when GECKO isn’t available at query time.  

---

## 📉 Error Analysis

- **Confusions** often occur between semantically close parents (e.g., *Pregnancy history* vs. *Pregnancy outcomes*).  
- **Synonyms help** disambiguate similar categories.  
- **GECKO helps** by aligning with ontology structure.  

Future work:  
- per-parent accuracy tables,  
- confusion matrix visualization.  

---

## ⚙️ Reproducing the Results

1. **Enrich definitions & synonyms**
   ```bash
   python enrich_llm_defs.py
