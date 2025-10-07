# interactive_compact_defs_all_v2.py
# Python 3.9 compatible
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)  # hide LibreSSL warning

import os
import re
import sys
import requests
import pandas as pd

# ==== CONFIG ====
EXCEL_PATH = "Labeled_variables.xlsx"   # same folder as this script
BASE_URL   = "http://127.0.0.1:8080"    # your OpenAI-compatible endpoint
MODEL      = "gwen-instruct"            # model name on your server
API_KEY    = None                       # e.g., "sk-..." if required
OUT_PATH   = None                       # None -> auto: "<input>.with_llm_short_definition.xlsx"
TEMPERATURE = 0.2
TIMEOUT     = 90
MAX_TOK_OUT = 128                       # some servers require max_tokens; safe default

TARGET_COL = "llm_short_definition"
DEF_COL    = "definition"

# thresholds for long definitions
LONG_DEF_CHAR_THRESHOLD = 300           # switch to "up to 2 sentences" beyond this
PROMPT_DEF_CHAR_LIMIT   = 1200          # truncate definition text in the prompt to this size

SYSTEM_ONE_SENTENCE = (
    "You refine definitions. Produce EXACTLY ONE concise, useful definition sentence in English.\n"
    "Rules: 10–22 words; only essentials; no lists/examples; no marketing/filler; "
    "avoid repeating the term if redundant; use Parent as context if given; "
    "output ONLY the sentence (no quotes or extra text)."
)

SYSTEM_TWO_SENTENCES_MAX = (
    "You refine definitions. Produce a compact English definition in AT MOST TWO short sentences "
    "(prefer one if possible).\n"
    "Rules: only essentials; no lists/examples; no marketing/filler; "
    "avoid repeating the term if redundant; use Parent as context if given; "
    "output ONLY the sentence(s), no quotes or extra text."
)

USER_TMPL = """Data:
- Name: {name}
- Parent: {parent}
- Current definition (may be truncated): {definition}

Task: produce a compact English definition for 'Name' in the context of 'Parent', following the system rules above.
"""

def build_urls(base):
    """Try both /v1/chat/completions and /chat/completions for compatibility."""
    base = base.rstrip("/")
    return [f"{base}/v1/chat/completions", f"{base}/chat/completions"]

def post_chat(url, model, system, prompt, api_key=None, temperature=0.2, timeout=60, max_tokens=MAX_TOK_OUT):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        # bubble up with server message for debugging
        raise requests.HTTPError(f"{r.status_code} {r.reason} — {r.text}", response=r)
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def openai_compatible_chat(base, model, system, prompt, api_key=None, temperature=0.2, timeout=60):
    errors = []
    for url in build_urls(base):
        try:
            return post_chat(url, model, system, prompt, api_key, temperature, timeout)
        except requests.HTTPError as e:
            errors.append(str(e))
            continue
    raise RuntimeError("Both /v1/chat/completions and /chat/completions failed:\n - " + "\n - ".join(errors))

def load_df(path):
    df = pd.read_excel(path)
    # normalize headers to lowercase and trim
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["name", "parent", "definition"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {path}. Found: {list(df.columns)}")
    return df

def sanitize_definition(def_text):
    """Trim super-long definitions to keep prompts within server limits."""
    if not def_text:
        return "", 0, False
    txt = str(def_text).strip()
    original_len = len(txt)

    # Compact whitespace
    txt = re.sub(r"\s+", " ", txt)

    # Soft shortening of very repetitive metadata patterns (optional & safe)
    # Collapse long sequences like 'Exact age, height_###' to a marker
    txt = re.sub(r"(Exact age,\s*height_\d+)(,\s*height_\d+){10,}", r"\1, ... [many items omitted]", txt)

    # Truncate if still too long
    truncated = False
    if len(txt) > PROMPT_DEF_CHAR_LIMIT:
        cut = txt.rfind(" ", 0, PROMPT_DEF_CHAR_LIMIT)
        if cut < 0:
            cut = PROMPT_DEF_CHAR_LIMIT
        txt = txt[:cut] + f" [...] [definition truncated from {original_len} chars]"
        truncated = True

    return txt, original_len, truncated

def choose_system(original_len):
    if original_len > LONG_DEF_CHAR_THRESHOLD:
        return SYSTEM_TWO_SENTENCES_MAX
    return SYSTEM_ONE_SENTENCE

def make_prompt(row):
    name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
    parent = "" if pd.isna(row["parent"]) else str(row["parent"]).strip()
    definition = "" if pd.isna(row[DEF_COL]) else str(row[DEF_COL]).strip()

    def_for_prompt, orig_len, truncated = sanitize_definition(definition)
    system = choose_system(orig_len)

    prompt = USER_TMPL.format(
        name=name or "(empty)",
        parent=parent or "(none)",
        definition=def_for_prompt or "(none)",
    )
    return system, prompt, definition  # return original definition for printing

def generate_one(row):
    system, prompt, original_def = make_prompt(row)
    compact = openai_compatible_chat(
        BASE_URL, MODEL, system, prompt,
        api_key=API_KEY, temperature=TEMPERATURE, timeout=TIMEOUT
    )
    return compact

def show_row(df, idx1):
    """idx1 is 1-based index shown to user; convert to 0-based for df."""
    i = idx1 - 1
    if i < 0 or i >= len(df):
        print(f"[!] Row {idx1} is out of range (valid 1..{len(df)}).")
        return

    row = df.iloc[i]
    name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
    parent = "" if pd.isna(row["parent"]) else str(row["parent"]).strip()
    original_def = "" if pd.isna(row[DEF_COL]) else str(row[DEF_COL]).strip()

    print("=" * 80)
    print(f"Row: {idx1}")
    print(f"name       : {name}")
    print(f"parent     : {parent}")
    print("definition :")
    print(original_def if original_def else "(empty)")
    print("-" * 80)

    try:
        compact = generate_one(row)
    except Exception as e:
        print(f"[Request failed] {e}")
        return

    print("COMPACT ENGLISH DEFINITION (1–2 sentences depending on input length)")
    print(compact)
    print("=" * 80)

def insert_next_to_definition(df, new_series, new_col=TARGET_COL, def_col=DEF_COL):
    """Insert/position new_col immediately after def_col; if new_col exists, update it."""
    if new_col not in df.columns:
        df[new_col] = ""
    df[new_col] = new_series

    cols = list(df.columns)
    if def_col in cols and new_col in cols:
        cols.remove(new_col)
        def_idx = cols.index(def_col)
        cols.insert(def_idx + 1, new_col)
        df = df[cols]
    return df

def process_all(df):
    print(f"Processing ALL {len(df)} rows → writing '{TARGET_COL}' next to '{DEF_COL}'...")
    outputs = []
    failures = 0
    for i, row in df.iterrows():
        try:
            out = generate_one(row)
        except Exception as e:
            failures += 1
            out = ""  # leave empty on failure
        outputs.append(out)
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"  ... {i+1}/{len(df)} done")

    series = pd.Series(outputs, index=df.index, name=TARGET_COL)
    df_out = insert_next_to_definition(df.copy(), series, TARGET_COL, DEF_COL)

    # Decide output path
    if OUT_PATH:
        out_path = OUT_PATH
    else:
        base, ext = os.path.splitext(EXCEL_PATH)
        out_path = f"{base}.with_{TARGET_COL}.xlsx"

    df_out.to_excel(out_path, index=False)
    print(f"Saved to: {out_path}")
    if failures:
        print(f"Note: {failures} rows failed and were left empty in '{TARGET_COL}'.")
    return out_path

def parse_indices(s):
    """Accept single number or comma-separated list; return list of 1-based ints in input order."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        if not re.fullmatch(r"\d+", p):
            print(f"[!] Ignoring invalid token: {p}")
            continue
        out.append(int(p))
    return out

def main():
    df = load_df(EXCEL_PATH)
    n = len(df)
    print(f"Loaded {EXCEL_PATH} with {n} rows.")
    print("Columns:", ", ".join(df.columns))
    print("Type a row number (1-based), multiple like '1, 2, 16', or:")
    print("  - 'all' to create/update column llm_short_definition next to 'definition' and save a new Excel")
    print("  - 'r' to reload the Excel")
    print("  - 'q' to quit\n")

    while True:
        try:
            s = input("Row(s) [1..{}] > ".format(n)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if s in ("q", "quit", "exit"):
            print("Bye!")
            break
        if s in ("r", "reload"):
            df = load_df(EXCEL_PATH)
            n = len(df)
            print(f"Reloaded. Now {n} rows.")
            continue
        if s == "all":
            try:
                process_all(df)
            except Exception as e:
                print(f"[ALL failed] {e}")
            continue
        if not s:
            continue

        idx_list = parse_indices(s)
        if not idx_list:
            continue
        for idx1 in idx_list:
            show_row(df, idx1)

if __name__ == "__main__":
    main()
