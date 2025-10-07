# predict_parent_interactive.py
# Python 3.9 compatible
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)  # hide LibreSSL warning

import re
import sys
import os
import json
import requests
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime

# ==== CONFIG ====
EXCEL_PATH = "Labeled_variables.with_llm_short_definition.xlsx"   # same folder as this script
BASE_URL   = "http://127.0.0.1:8080"    # your OpenAI-compatible endpoint
MODEL      = "gwen-instruct"            # model name on your server
API_KEY    = None                       # e.g., "sk-..." if required

TEMPERATURE = 0.0                       # classification: keep it deterministic/concise
TIMEOUT     = 90
MAX_TOK_OUT = 256                       # room for JSON + rationales

# Logging (error analysis)
PRED_LOG_PATH = "predictions_log.xlsx"

# Prompts: truncation & generation
PROMPT_DEF_CHAR_LIMIT = 1200            # truncate very long defs in prompts
LONG_DEF_CHAR_THRESHOLD = 300           # >300 chars → allow up to 2 sentences when generating
TEST_SEED = 42                           # seed for 'test N' sampling (set to None for non-reproducible)

SYSTEM_ONE_SENTENCE = (
    "You refine definitions. Produce EXACTLY ONE concise, useful definition sentence in English.\n"
    "Rules: 10–22 words; only essentials; no lists/examples; no marketing/filler; "
    "avoid repeating the term if redundant; use Parent as context if given; "
    "output ONLY the sentence, no quotes or extra text."
)
SYSTEM_TWO_SENTENCES_MAX = (
    "You refine definitions. Produce a compact English definition in AT MOST TWO short sentences "
    "(prefer one if possible).\n"
    "Rules: only essentials; no lists/examples; no marketing/filler; "
    "avoid repeating the term if redundant; use Parent as context if given; "
    "output ONLY the sentence(s), no quotes or extra text."
)
USER_TMPL_DEF = """Data:
- Name: {name}
- Parent: {parent}
- Current definition (may be truncated): {definition}

Task: produce a compact English definition for 'Name' in the context of 'Parent', following the system rules above.
"""

# ===== Helpers =====
def normalize_header(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def load_df(path):
    df = pd.read_excel(path)
    df = normalize_header(df)
    for col in ["name", "parent"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {path}. Found: {list(df.columns)}")
    if "llm_short_definition" not in df.columns and "definition" not in df.columns:
        print("[!] Neither 'llm_short_definition' nor 'definition' found; predictions will use only 'name'.")
    return df

def _sanitize_for_prompt(def_text):
    if not def_text:
        return "", 0, False
    txt = str(def_text).strip()
    original_len = len(txt)
    txt = re.sub(r"\s+", " ", txt)
    # collapse repetitive patterns like 'Exact age, height_###'
    txt = re.sub(r"(Exact age,\s*height_\d+)(,\s*height_\d+){10,}", r"\1, ... [many items omitted]", txt)
    truncated = False
    if len(txt) > PROMPT_DEF_CHAR_LIMIT:
        cut = txt.rfind(" ", 0, PROMPT_DEF_CHAR_LIMIT)
        if cut < 0:
            cut = PROMPT_DEF_CHAR_LIMIT
        txt = txt[:cut] + f" [...] [definition truncated from {original_len} chars]"
        truncated = True
    return txt, original_len, truncated

def _choose_system_for_length(original_len):
    return SYSTEM_TWO_SENTENCES_MAX if original_len > LONG_DEF_CHAR_THRESHOLD else SYSTEM_ONE_SENTENCE

def build_urls(base):
    base = base.rstrip("/")
    return [f"{base}/v1/chat/completions", f"{base}/chat/completions"]

def post_chat(url, model, messages, api_key=None, temperature=0.0, max_tokens=MAX_TOK_OUT, timeout=TIMEOUT):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code} {r.reason} — {r.text}", response=r)
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def call_chat(messages, temperature=TEMPERATURE):
    errors = []
    for url in build_urls(BASE_URL):
        try:
            return post_chat(url, MODEL, messages, api_key=API_KEY, temperature=temperature)
        except requests.HTTPError as e:
            errors.append(str(e))
            continue
    raise RuntimeError("Both /v1/chat/completions and /chat/completions failed:\n - " + "\n - ".join(errors))

# --- Generate compact definition on-the-fly (if llm_short_definition missing) ---
def generate_compact_definition(name, parent, original_def):
    def_for_prompt, orig_len, _ = _sanitize_for_prompt(original_def or "")
    system = _choose_system_for_length(orig_len)
    prompt = USER_TMPL_DEF.format(
        name=name or "(empty)",
        parent=parent or "(none)",
        definition=def_for_prompt or "(none)",
    )
    return call_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ], temperature=0.2)

def list_parent_options(df):
    opts = [str(v).strip() for v in df["parent"].dropna().astype(str).unique()]
    seen = set()
    unique_opts = []
    for o in sorted(opts, key=lambda x: x.lower()):
        key = re.sub(r"\s+", " ", o.lower())
        if key and key not in seen:
            seen.add(key)
            unique_opts.append(o)
    return unique_opts

def options_block(options, numbered=False):
    if numbered:
        return "\n".join(f"{i+1}) {opt}" for i, opt in enumerate(options))
    return "\n".join(f"- {opt}" for opt in options)

def map_to_option(label, options):
    """Map model output text to one of the options using exact and fuzzy matching."""
    if not label:
        return None, 0.0
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    label_n = norm(label)
    opt_norms = [norm(o) for o in options]
    if label_n in opt_norms:
        i = opt_norms.index(label_n)
        return options[i], 1.0
    best_i, best_sc = -1, 0.0
    for i, o in enumerate(options):
        sc = SequenceMatcher(None, label_n, norm(o)).ratio()
        if sc > best_sc:
            best_i, best_sc = i, sc
    if best_sc >= 0.80:
        return options[best_i], best_sc
    return None, best_sc

# --- TOP-K with SCORES + RATIONALES (preferred) ---
def system_prompt_topk_json():
    return (
        "You are a classifier. From the provided OPTIONS, choose the TOP 3 Parent labels that best match the Name and Definition. "
        "Return STRICT, VALID JSON with this schema:\n"
        "{ \"choices\": [ {\"label\": \"<exact label from options>\", \"score\": <integer 0..100>, \"why\": \"<exactly two short sentences>\"}, ... ] }\n"
        "Rules: (1) Labels must be copied EXACTLY from the options; (2) Provide exactly 3 items; "
        "(3) Scores should be plausible and sum roughly to 100; (4) 'why' must be EXACTLY TWO sentences, concise and evidence-based; "
        "(5) Do NOT include any text before or after the JSON."
    )

def user_prompt_topk_json(name, deftext, options_text):
    return (
        "Data:\n"
        f"- Name: {name}\n"
        f"- Definition: {deftext if deftext else '(none)'}\n\n"
        "OPTIONS (select from these only):\n"
        f"{options_text}\n\n"
        "Task: Return the JSON object as specified."
    )

def _extract_json(text):
    """Attempt to extract a JSON object from the model output (robust to fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text

def _normalize_scores(choices):
    total = 0.0
    for ch in choices:
        try:
            total += float(ch.get("score", 0))
        except Exception:
            pass
    if total <= 0:
        n = len(choices)
        for ch in choices:
            ch["score"] = round(100.0 / n, 2)
        return choices
    for ch in choices:
        try:
            ch["score"] = round(float(ch.get("score", 0)) * 100.0 / total, 2)
        except Exception:
            ch["score"] = 0.0
    return choices

def predict_parent_topk_json(row, options, override_deftext=None, k=3):
    name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
    if override_deftext is not None:
        deftext = override_deftext.strip()
    elif "llm_short_definition" in row.index and pd.notna(row["llm_short_definition"]):
        deftext = str(row["llm_short_definition"]).strip()
    elif "definition" in row.index and pd.notna(row["definition"]):
        deftext = str(row["definition"]).strip()
    else:
        deftext = ""

    msgs = [
        {"role": "system", "content": system_prompt_topk_json()},
        {"role": "user", "content": user_prompt_topk_json(name, deftext, options_block(options, numbered=False))},
    ]
    out = call_chat(msgs, temperature=0.0)
    jtxt = _extract_json(out)
    data = json.loads(jtxt)
    choices = data.get("choices", [])[:k]

    mapped = []
    for ch in choices:
        lbl_raw = str(ch.get("label", "")).strip()
        why = str(ch.get("why", "")).strip()
        score = ch.get("score", 0)
        label, sim = map_to_option(lbl_raw, options)
        if not label:
            continue
        mapped.append((label, f"{lbl_raw}", "json", sim, float(score), why))
    if not mapped:
        raise ValueError("No valid labels parsed from JSON.")

    tmp = [{"label": m[0], "score": m[4], "why": m[5], "raw": m[1], "sim": m[3], "mode": m[2]} for m in mapped]
    tmp = _normalize_scores(tmp)
    normed = []
    for t in tmp:
        normed.append((t["label"], t["raw"], "json", t["sim"], t["score"], t["why"]))
    return normed[:k]

# --- Fallback TOP-K (labels or numbered) if JSON fails ---
def system_prompt_topk(numbered=False):
    if numbered:
        return (
            "You are a classifier. From the NUMBERED options, choose the TOP 3 indices "
            "that best match the Name and Definition.\n"
            "Return ONLY three numbers separated by commas (e.g., 7, 12, 4). No extra text."
        )
    else:
        return (
            "You are a classifier. From the options, choose the TOP 3 labels that best match "
            "the Name and Definition.\n"
            "Return EXACTLY three labels from the options, in order, one per line. No extra text."
        )

def user_prompt_topk(name, deftext, options_text, numbered=False):
    return (
        "Data:\n"
        f"- Name: {name}\n"
        f"- Definition: {deftext if deftext else '(none)'}\n\n"
        "Options:\n"
        f"{options_text}\n\n"
        + ("Task: Reply with the THREE NUMBERS only, separated by commas.\n"
           if numbered else
           "Task: Reply with THREE LABELS exactly as written, one per line.\n")
    )

def parse_topk_labels(text, k=3):
    parts = [p.strip() for p in re.split(r"[\n,;|]+", text) if p.strip()]
    out, seen = [], set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        out.append(p)
        seen.add(key)
        if len(out) >= k:
            break
    return out

def parse_topk_numbers(text, k=3):
    nums = re.findall(r"\d+", text)
    uniq, seen = [], set()
    for n in nums:
        if n in seen:
            continue
        uniq.append(int(n))
        seen.add(n)
        if len(uniq) >= k:
            break
    return uniq

def predict_parent_topk_for_row(row, options, override_deftext=None, k=3):
    # Preferred JSON path
    try:
        return predict_parent_topk_json(row, options, override_deftext=override_deftext, k=k)
    except Exception:
        pass

    # Fallback: LABELS then NUMBERED
    name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
    if override_deftext is not None:
        deftext = override_deftext.strip()
    elif "llm_short_definition" in row.index and pd.notna(row["llm_short_definition"]):
        deftext = str(row["llm_short_definition"]).strip()
    elif "definition" in row.index and pd.notna(row["definition"]):
        deftext = str(row["definition"]).strip()
    else:
        deftext = ""

    results = []
    # labels
    messages = [
        {"role": "system", "content": system_prompt_topk(numbered=False)},
        {"role": "user", "content": user_prompt_topk(name, deftext, options_block(options, numbered=False), numbered=False)},
    ]
    try:
        out = call_chat(messages, temperature=0.0)
        candidates = parse_topk_labels(out, k=k)
        for cand in candidates:
            mapped, sc = map_to_option(cand, options)
            if mapped and mapped not in [m for m, *_ in results]:
                results.append((mapped, cand, "labels", sc, None, ""))
            if len(results) >= k:
                break
    except Exception:
        pass

    # numbered
    if len(results) < k:
        messages = [
            {"role": "system", "content": system_prompt_topk(numbered=True)},
            {"role": "user", "content": user_prompt_topk(name, deftext, options_block(options, numbered=True), numbered=True)},
        ]
        try:
            out2 = call_chat(messages, temperature=0.0)
            nums = parse_topk_numbers(out2, k=k)
            for n in nums:
                idx = n - 1
                if 0 <= idx < len(options):
                    mapped = options[idx]
                    if mapped not in [m for m, *_ in results]:
                        results.append((mapped, str(n), "numbered", 1.0, None, ""))
                if len(results) >= k:
                    break
        except Exception:
            pass

    return results[:k]

def predict_parent_for_row(row, options, override_deftext=None):
    topk = predict_parent_topk_for_row(row, options, override_deftext=override_deftext, k=1)
    if not topk:
        return None, "", "none", 0.0
    label, raw, mode, sim, score, why = topk[0]
    return label, raw, mode, sim

# --- Small utils ---
def _labels_equal(a, b):
    norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
    return norm(a) == norm(b)

def _prepare_used_text(row, generate_if_missing=True):
    """Return text used for prediction (prefer llm_short_definition; optionally generate)."""
    name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
    parent = "" if pd.isna(row["parent"]) else str(row["parent"]).strip()
    long_def = ""
    if "definition" in row.index and pd.notna(row["definition"]):
        long_def = str(row["definition"]).strip()
    short_def = ""
    if "llm_short_definition" in row.index and pd.notna(row["llm_short_definition"]):
        short_def = str(row["llm_short_definition"]).strip()
    if short_def:
        return short_def
    if generate_if_missing and long_def:
        try:
            gen = generate_compact_definition(name, parent, long_def)
            if gen:
                return gen
        except Exception:
            pass
    return long_def

def _new_run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# --- Logging to Excel ---
def _topk_to_record_dict(idx1, row, used_source, used_text, topk, parent_true, run_id, cmd):
    """Build one flat dict for error analysis Excel."""
    # unpack with safe defaults
    def _get(t, j):
        return t[j] if j < len(t) else None

    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "cmd": cmd,
        "row_index": idx1,
        "name": ("" if pd.isna(row["name"]) else str(row["name"]).strip()),
        "parent_gold": ("" if pd.isna(parent_true) else str(parent_true).strip()),
        "definition": ("" if "definition" not in row.index or pd.isna(row["definition"]) else str(row["definition"]).strip()),
        "llm_short_definition": ("" if "llm_short_definition" not in row.index or pd.isna(row["llm_short_definition"]) else str(row["llm_short_definition"]).strip()),
        "used_source": used_source,
    }

    gold = rec["parent_gold"]

    # Top-1 / 2 / 3
    for k, tup in enumerate(topk, start=1):
        label = tup[0] if len(tup) > 0 else None
        raw   = tup[1] if len(tup) > 1 else None
        mode  = tup[2] if len(tup) > 2 else None
        sim   = tup[3] if len(tup) > 3 else None
        score = tup[4] if len(tup) > 4 else None
        why   = tup[5] if len(tup) > 5 else None

        rec[f"pred_top{k}_label"]      = label
        rec[f"pred_top{k}_score"]      = score
        rec[f"pred_top{k}_similarity"] = sim
        rec[f"pred_top{k}_mode"]       = mode
        rec[f"pred_top{k}_raw"]        = raw
        rec[f"pred_top{k}_why"]        = why

    # Correctness flags
    top1_label = rec.get("pred_top1_label") or ""
    in_top3 = any(_labels_equal(gold, (rec.get(f"pred_top{k}_label") or "")) for k in (1, 2, 3))
    rec["pred_top1_correct"] = _labels_equal(gold, top1_label)
    rec["in_top3"] = in_top3

    return rec

def _append_records_to_excel(records, out_path=PRED_LOG_PATH):
    if not records:
        return
    df_new = pd.DataFrame(records)
    if os.path.exists(out_path):
        try:
            df_old = pd.read_excel(out_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            # if read fails, just write new
            df_all = df_new
    else:
        df_all = df_new
    df_all.to_excel(out_path, index=False)
    print(f"[saved] Appended {len(records)} row(s) to {out_path}")

# --- CLI actions ---
def show_prediction(df, idx1, options):
    i = idx1 - 1
    if i < 0 or i >= len(df):
        print(f"[!] Row {idx1} is out of range (valid 1..{len(df)}).")
        return

    row = df.iloc[i]
    name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
    parent_true = "" if pd.isna(row["parent"]) else str(row["parent"]).strip()

    long_def = str(row["definition"]).strip() if "definition" in df.columns and pd.notna(row["definition"]) else ""
    short_def = str(row["llm_short_definition"]).strip() if "llm_short_definition" in df.columns and pd.notna(row["llm_short_definition"]) else ""

    used_text = _prepare_used_text(row, generate_if_missing=True)
    used_source = "llm_short_definition" if short_def else ("generated" if used_text and used_text != long_def else ("definition" if long_def else "name-only"))

    print("=" * 100)
    print(f"Row: {idx1}")
    print(f"name                 : {name}")
    print(f"parent (original)    : {parent_true if parent_true else '(none)'}")
    print("definition (original):")
    print(long_def if long_def else "(empty)")
    print("-" * 100)
    print("llm_short_definition (from sheet or generated now):")
    print(short_def if short_def else (used_text if used_source == "generated" else "(empty)"))
    print("-" * 100)
    print(f"[using {used_source} for prediction]\n")

    try:
        topk = predict_parent_topk_for_row(row, options, override_deftext=used_text, k=3)
    except Exception as e:
        print(f"[Prediction failed] {e}")
        print("=" * 100)
        return

    if not topk:
        print("No prediction returned.")
        print("=" * 100)
        return

    best = topk[0]
    label1, raw1, mode1, sim1, score1, why1 = best
    print(f"PREDICTED parent (top-1): {label1}")
    print(f"model raw output        : {raw1}")
    print(f"mode                    : {mode1}   similarity≈{sim1:.2f}")
    if score1 is not None:
        print(f"confidence              : {score1:.0f}/100 (model self-score)")
    print("-" * 100)
    print("TOP-3 suggestions:")
    for rank, (label, raw, mode, sc, score, why) in enumerate(topk, start=1):
        score_str = f"{score:.0f}/100" if score is not None else "n/a"
        why_str = why if why else "(no rationale available in fallback mode)"
        print(f"{rank}. {label}    [score: {score_str} | sim≈{sc:.2f} | mode: {mode} | raw: {raw}]")
        print(f"    why: {why_str}")
    print("=" * 100)
    print("Note: scores are model self-assessments (not calibrated probabilities).\n")

    # --- write to Excel log ---
    rec = _topk_to_record_dict(
        idx1=idx1,
        row=row,
        used_source=used_source,
        used_text=used_text,
        topk=topk,
        parent_true=parent_true,
        run_id=_new_run_id(),
        cmd="predict",
    )
    _append_records_to_excel([rec], out_path=PRED_LOG_PATH)

def run_test(df, n, options, seed=TEST_SEED):
    if n <= 0:
        print("[!] N must be > 0")
        return
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=seed) if seed is not None else df.sample(n=n)
    idxs = list(sample.index)
    run_id = _new_run_id()

    total = 0
    skipped = 0
    top1_hits = 0
    top3_hits = 0
    failures = 0
    records = []

    for j, i in enumerate(idxs, start=1):
        row = df.iloc[i]
        gold = str(row["parent"]).strip() if pd.notna(row["parent"]) else ""
        if not gold:
            skipped += 1
            continue

        used_text = _prepare_used_text(row, generate_if_missing=False)  # faster; skip generation in batch
        try:
            topk = predict_parent_topk_for_row(row, options, override_deftext=used_text, k=3)
        except Exception:
            failures += 1
            continue

        if not topk:
            failures += 1
            continue

        total += 1
        pred1 = topk[0][0]
        if _labels_equal(pred1, gold):
            top1_hits += 1
        if any(_labels_equal(t[0], gold) for t in topk):
            top3_hits += 1

        rec = _topk_to_record_dict(
            idx1=(i + 1),
            row=row,
            used_source=("llm_short_definition" if ("llm_short_definition" in row.index and pd.notna(row["llm_short_definition"])) else ("definition" if ("definition" in row.index and pd.notna(row["definition"])) else "name-only")),
            used_text=used_text,
            topk=topk,
            parent_true=gold,
            run_id=run_id,
            cmd=f"test({n}, seed={seed})",
        )
        records.append(rec)

        if (j % 10 == 0) or (j == len(idxs)):
            print(f"  progress: {j}/{len(idxs)} processed")

    if total == 0:
        print("[Test] No evaluable rows (all missing parent?)")
        return

    top1_acc = 100.0 * top1_hits / total
    top3_acc = 100.0 * top3_hits / total

    print("\n================= TEST SUMMARY =================")
    print(f"Sample size requested : {n} (seed={seed})")
    print(f"Evaluated (non-empty) : {total}")
    print(f"Skipped (no parent)   : {skipped}")
    print(f"Failures (errors)     : {failures}")
    print("-----------------------------------------------")
    print(f"Top-1 accuracy        : {top1_hits}/{total} = {top1_acc:.2f}%")
    print(f"Top-3 accuracy        : {top3_hits}/{total} = {top3_acc:.2f}%")
    print("================================================\n")

    # --- write all test predictions to Excel log ---
    _append_records_to_excel(records, out_path=PRED_LOG_PATH)

def parse_predict_cmd(s):
    m = re.match(r"^predict\s+(.+)$", s.strip().lower())
    if not m:
        return None
    rest = m.group(1)
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    out = []
    for p in parts:
        if not re.fullmatch(r"\d+", p):
            print(f"[!] Ignoring invalid index: {p}")
            continue
        out.append(int(p))
    return out

def parse_test_cmd(s):
    # supports: "test 50"  (optional: "test 50 seed 123")
    m = re.match(r"^test\s+(\d+)(?:\s+seed\s+(\d+))?$", s.strip().lower())
    if not m:
        return None, None
    n = int(m.group(1))
    seed = int(m.group(2)) if m.group(2) is not None else TEST_SEED
    return n, seed

def main():
    df = load_df(EXCEL_PATH)
    n = len(df)
    options = list_parent_options(df)
    print(f"Loaded {EXCEL_PATH} with {n} rows.")
    print(f"Found {len(options)} unique parent options.")
    print("Commands:")
    print("  predict N              → predict parent for row N (1-based index) and log to Excel")
    print("  predict N,M,K          → predict multiple and log to Excel")
    print("  test N [seed S]        → sample N random rows, report Top-1/Top-3, and log details to Excel")
    print("  options                → list all parent options")
    print("  r                      → reload Excel (and options)")
    print("  q                      → quit\n")

    while True:
        try:
            s = input("cmd > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not s:
            continue
        low = s.lower()
        if low in ("q", "quit", "exit"):
            print("Bye!")
            break
        if low in ("r", "reload"):
            df = load_df(EXCEL_PATH)
            n = len(df)
            options = list_parent_options(df)
            print(f"Reloaded. {n} rows. {len(options)} parent options.")
            continue
        if low == "options":
            print("\nPARENT OPTIONS:")
            for i, opt in enumerate(options, 1):
                print(f"{i:>3}. {opt}")
            print("")
            continue

        # test command
        ntest, seed = parse_test_cmd(s)
        if ntest is not None:
            run_test(df, ntest, options, seed=seed)
            continue

        # predict command
        idxs = parse_predict_cmd(s)
        if idxs is None:
            print("Unknown command. Try: predict 151   or: predict 1,2,16   or: test 50   or: options")
            continue
        for idx1 in idxs:
            show_prediction(df, idx1, options)

if __name__ == "__main__":
    main()
