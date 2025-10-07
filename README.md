# Topic Matching – Local LLM Toolkit (Qwen + llama.cpp)

Two terminal tools to help you **clean up variable definitions** and **predict taxonomy parents** — fully **offline** on your machine with a **quantized** instruction model (e.g., **Qwen3-4B-Instruct Q4_K_M**) served via `llama.cpp`.

---

## Table of Contents

* [What it does](#what-it-does)
* [Why Top-1 *and* Top-3](#why-top1-and-top3)
* [Quickstart](#quickstart)
* [Run the model locally (llama.cpp)](#run-the-model-locally-llamacpp)
* [Scripts & usage](#scripts--usage)

  * [A) Create compact definitions](#a-create-compact-definitions)
  * [B) Predict parents, explain, and evaluate](#b-predict-parents-explain-and-evaluate)
* [Input/Output formats](#inputoutput-formats)
* [How it works (under the hood)](#how-it-works-under-the-hood)
* [Configuration knobs](#configuration-knobs)
* [Troubleshooting](#troubleshooting)
* [Results & analysis templates](#results--analysis-templates)

---

## What it does

1. **Compacts messy definitions**
   Reads `Labeled_variables.xlsx` with columns `name`, `parent`, `definition` and writes
   `Labeled_variables.with_llm_short_definition.xlsx` by adding **`llm_short_definition`**.

* Produces **exactly one** concise sentence by default.
* If the original is long, it allows **up to two short sentences** to keep signal.

2. **Predicts the best parent** for each variable
   Uses **`name` + `llm_short_definition`** (falls back to `definition` or just `name`).

* Returns **Top-1** and **Top-3** suggestions.
* Can include a **self-score (0–100)** and a **two-sentence rationale** per suggestion.
* Supports a quick **`test N`** command to measure **Top-1 / Top-3 accuracy**.

Everything runs **locally** via an **OpenAI-compatible** HTTP endpoint (no data leaves your machine).

---

## Why Top-1 and Top-3?

* **Top-1** is what you’d auto-apply if you accept the model’s first choice.
* **Top-3** is practical for **triage/review**: if the gold label appears in the top three, a human can quickly click the right one. This reduces correction time and gives a better sense of usefulness in noisy taxonomies.

---

## Quickstart

```bash
# 1) Create and activate a virtual environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install pandas openpyxl requests

# 3) Start your local model server (see next section)

# 4) Generate short definitions (interactive; writes a new Excel)
python interactive_compact_defs_all_v2.py

# 5) Predict parents / evaluate (interactive; reads the new Excel)
python predict_parent_interactive.py
```

---

## Run the model locally (llama.cpp)

You’ll run a **quantized** Qwen Instruct model locally and expose it via an OpenAI-style API.

### Quick CLI sanity check

Verify the model loads:

```bash
# User's example (quantized Qwen 3 4B Instruct)
./llama-cli -m ~/Downloads/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

### Start the HTTP server

Run `llama-server` so the Python tools can POST to `http://127.0.0.1:8080`:

```bash
./llama-server \
  -m ~/Downloads/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --port 8080 --host 127.0.0.1
```

> The scripts try both `/v1/chat/completions` and `/chat/completions` automatically.
> If you use another OpenAI-compatible host (e.g., LM Studio), just keep the same base URL.

---

## Scripts & usage

### A) Create compact definitions

**File:** `interactive_compact_defs_all_v2.py`
**Reads:** `Labeled_variables.xlsx`
**Writes:** `Labeled_variables.with_llm_short_definition.xlsx` (adds `llm_short_definition` next to `definition`)

**Run**

```bash
python interactive_compact_defs_all_v2.py
```

**Interactive commands**

* Type a row number (1-based) or comma-separated (e.g., `1, 2, 16`) to preview a compact definition for those rows.
* Type `all` to generate **`llm_short_definition`** for every row and save a new Excel.
* Type `r` to reload the file, `q` to quit.

**Behavior**

* One concise sentence (~10–22 words).
* If the original `definition` is long (>300 chars), the model may produce **two** short sentences.
* Extremely long inputs are **soft-truncated** in the prompt to avoid server `400` errors.

---

### B) Predict parents, explain, and evaluate

**File:** `predict_parent_interactive.py`
**Reads:** `Labeled_variables.with_llm_short_definition.xlsx`

**Run**

```bash
python predict_parent_interactive.py
```

**Interactive commands**

* `predict N` or `predict 1,2,16` — show **Top-1** + **Top-3** predictions.

  * If `llm_short_definition` is missing for that row, the script **generates** one on the fly for context (not saved).
* `test N [seed S]` — sample *N* random rows and compute **Top-1 / Top-3** accuracy.

  * Uses a fixed seed by default for reproducibility (change with `seed S`).
* `options` — print the full, numbered list of unique parent labels the model sees.
* `r` — reload the Excel, `q` — quit.

**Sample output (abridged)**

```
Row: 1
name                 : Pregnancy number
parent (original)    : Identifiers
definition (original):
Pregnancy number
------------------------------------------------------------
llm_short_definition (from sheet or generated now):
Number of pregnancies recorded for the subject.
------------------------------------------------------------
[using llm_short_definition for prediction]

PREDICTED parent (top-1): Birth, pregnancy and reproductive health history
mode                    : json   similarity≈1.00
confidence              : 72/100 (model self-score)
------------------------------------------------------------
TOP-3 suggestions:
1. Birth, pregnancy and reproductive health history  [score: 72/100 | sim≈1.00]
    why: Tracks parity for obstetric context. Strong semantic overlap with reproductive history.
2. Identifiers                                      [score: 18/100 | sim≈0.84]
    why: Sometimes stored as a code-like field. Less semantic fit than reproductive history.
3. Demographics                                     [score: 10/100 | sim≈0.81]
    why: Could be reported with person attributes. Not primarily a demographic measure.
```

> **Notes**
>
> * *similarity* is a string-match quality vs. your exact option list (not a probability).
> * *confidence* is the model’s **self-score** (0–100), normalized to sum ~100 across Top-3. Treat as heuristic.

---

## Input/Output formats

### Input (required)

`Labeled_variables.xlsx` with columns (case-insensitive):

* `name` *(string)*
* `parent` *(string; your current taxonomy label; used as “gold” in testing)*
* `definition` *(string; may be long)*

### Output (after step A)

`Labeled_variables.with_llm_short_definition.xlsx` adds:

* `llm_short_definition` *(string; the compact definition)*

The predictor reads this file by default.

---

## How it works (under the hood)

* **Definition generation**:
  Uses a strict instruction style (no lists, no marketing language) to produce one compact sentence; for long inputs, up to two short sentences. Long enumerations are truncated in prompt to avoid HTTP 400.

* **Parent classification**:
  Builds a prompt with **all unique `parent` options** from your data. Prefers `llm_short_definition` as context; falls back gracefully to `definition` or `name`.

* **Top-3 JSON with rationale (preferred)**:
  Asks the model to return **strict JSON**:

  ```json
  { "choices": [
    {"label": "<exact option>", "score": 0..100, "why": "<exactly two short sentences>"},
    ...
  ]}
  ```

  If JSON parsing fails, the script falls back to “labels” or “numbered indices”.

* **Accuracy**:

  * **Top-1**: first suggestion equals the gold `parent`.
  * **Top-3**: gold appears among the first 3 suggestions.

---

## Configuration knobs

Open the script and adjust:

* `BASE_URL` — default `http://127.0.0.1:8080`
* `MODEL` — e.g., `"gwen-instruct"` (or whatever your server exposes)
* `PROMPT_DEF_CHAR_LIMIT` — max definition chars passed to the model (default `1200`)
* `LONG_DEF_CHAR_THRESHOLD` — above this, allow **two** short sentences (default `300`)
* `MAX_TOK_OUT` — output token cap; increase if JSON gets cut off (default `256`)
* `TEST_SEED` — seed for `test N` (default `42`)

---

## Troubleshooting

* **`400 Bad Request` on chat/completions**
  Prompt too large. The scripts **truncate** long definitions; if this still happens, reduce `PROMPT_DEF_CHAR_LIMIT`.

* **LibreSSL / urllib3 warning on macOS**
  It’s just a warning. If you want to hide it:

  * Pin urllib3 `<2`: `pip install "urllib3<2"`; or
  * Use a newer Python (e.g., 3.12 via Homebrew) and recreate the venv.

* **Server route differences**
  Tools try `/v1/chat/completions` and `/chat/completions`. If you use a different API (e.g., Ollama’s `/api/chat`), adapt `post_chat` accordingly.

* **Model outputs indices (e.g., `68`)**
  That’s **numbered mode**. The script maps indices back to labels; type `options` to see the numbering.

---

## Results & analysis templates

### 🔢 Accuracy summary (paste from `test N`)

| Sample Size | Evaluated | Skipped (no gold) | Failures | Top-1 Accuracy | Top-3 Accuracy | Seed |
| ----------: | --------: | ----------------: | -------: | -------------: | -------------: | ---: |
|          100 |        97 |                 3 |        0 |         50.52% |         75.26% |   42 |

### Examples of when model is right

| Row | Name                                             | Gold Parent                    | Top-1 Pred                        | Score | Why (two sentences)                                                                                                                                                                                                          |
| --: | ------------------------------------------------ | ------------------------------ | --------------------------------- | ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  56 | Other and unspecified non-pharmacological inter… | Non-ph…                        | Non-pharmacological interventions |    95 | The name directly refers to interventions that do not use medication. This category best groups behavioral and supportive strategies, making it the most specific and relevant parent category for this entry.               |
| 484 | ndvi100                                          | Natural spaces                 | Natural spaces                    |    87 | The definition specifies ‘natural’ areas such as parks, forests, and green spaces. These locations are prototypical contexts for exposure studies and align with ecological and environmental contexts within natural areas. |
|  64 | Any eating disorder                              | Maternal psychiatric disorders | Maternal psychiatric disorders    |    86 | Eating disorders are classified as psychiatric conditions. When occurring in mothers, they are studied within maternal mental health across the lifespan, which falls under maternal psychiatric disorders.                  |
| 421 | main_land_use_agricultural                       | Built environment/neighbour…   | Built environment/neighbourhood   |    85 | The definition involves neighbourhood form and land use factors. These are classic built-environment attributes that shape how land use patterns influence residential environments.                                         |
| 269 | Nutrients (M)                                    | Nutrition                      | Nutrition                         |    68 | The definition explicitly mentions key aspects of dietary intake and nutrient quality. That scope directly aligns with the core focus of nutrition as a domain.                                                              |


### Examples of where model is wrong.

| Row | Name                                  | Gold Parent                                | Top-1 Pred                        | In Top-3? | Score | Why (two sentences)                                                                                                                                                                                                   |
| --: | ------------------------------------- | ------------------------------------------ | --------------------------------- | :-------: | ----: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 230 | Sibling position                      | Birth outcomes                             | Family and household structure    |     ✅     |    86 | Sibling position directly reflects ordinal birth order within the family unit. This is inherently a structural aspect of family dynamics and birth order within a household.                                          |
| 285 | Use of assistive devices              | Perception of health, quality…             | Non-pharmacological interventions |     ❌     |    85 | The use of assistive devices is part of supportive care and rehabilitation strategies. These are typically grouped under non-pharmacological strategies for maintaining daily function.                               |
| 408 | Death of child                        | Vital status                               | Death                             |     ✅     |    68 | The name ‘Death of child’ directly aligns with mortality outcomes. Death is the appropriate high-level category for mortality indicators in vital status classification.                                              |
| 487 | Traffic                               | Other physical environment characteristics | Traffic                           |     ❌     |    44 | The name and definition directly describe road traffic and related exposure. Traffic is a canonical subtopic of the physical environment; however, it may sit under a more specific subcategory in the gold taxonomy. |
| 132 | Preeclampsia/gestational hypertension | Pregnancy, delivery and birth              | Diseases                          |     ❌     |    44 | Preeclampsia is a pregnancy-specific disease entity with established diagnostic criteria. The model maps to the broader ‘Diseases’ category rather than the narrower pregnancy context used in the gold label.        |

---



