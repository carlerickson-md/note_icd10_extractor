"""
Clinical Note Diagnostic Pipeline: Abridge Assessment & Plan to ICD-10 Extractor.
"""

from pathlib import Path
import re
import pandas as pd
from rapidfuzz import process, fuzz


# =============================================================================
# 1. CONFIGURATION & CONSTANTS (NEUROLOGY FOCUS)
# =============================================================================
DEFAULT_TARGET_CHAPTERS = ("G", "I", "F", "Z")
KNOWN_ETIOLOGIES = {
    "vascular", "alzheimer", "lewy", "frontotemporal", "pick", "parkinson"
}


# =============================================================================
# 2. I/O & REGISTRY LOADERS
# =============================================================================
def select_note_file(notes_dir: Path) -> Path:
    """Scans the notes directory and prompts the user to select one note file."""
    if not notes_dir.exists():
        raise FileNotFoundError(f"Notes directory '{notes_dir}' does not exist.")

    note_files = sorted(list(notes_dir.glob("*.txt")))
    if not note_files:
        raise FileNotFoundError(f"No .txt files found in '{notes_dir}'.")

    print("\n--- Available Note Files ---")
    for idx, file_path in enumerate(note_files, start=1):
        print(f"[{idx}] {file_path.name}")

    while True:
        choice = input(f"\nSelect a file number (1-{len(note_files)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(note_files):
            return note_files[int(choice) - 1]
        print(f"Invalid input. Enter a number between 1 and {len(note_files)}.")


def load_clinical_note(file_path: Path, provider: str) -> pd.DataFrame:
    """Reads a clinical text file into a single-row DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_note = f.read()

    return pd.DataFrame([{
        "note_id": file_path.stem,
        "provider": provider,
        "clinical_text": raw_note
    }])


def load_reference_data(data_dir: Path, target_chapters: tuple = DEFAULT_TARGET_CHAPTERS):
    """Loads and preprocesses hedge phrases, acronym maps, and ICD-10 registries."""
    # 1. Hedge phrases
    hedge_path = data_dir / "hedge_phrases.csv"
    hedge_patterns = []
    if hedge_path.exists():
        df_hedges = pd.read_csv(hedge_path, dtype=str)
        df_hedges.columns = df_hedges.columns.str.strip()
        raw_hedges = df_hedges["hedge_phrase"].dropna().str.strip().tolist()
        sorted_hedges = sorted(raw_hedges, key=len, reverse=True)
        hedge_patterns = [rf"\b{re.escape(p)}\b" for p in sorted_hedges]
    else:
        print(f"Warning: '{hedge_path.name}' not found. Skipping external hedge stripping.")

    # 2. Acronym map
    acronym_path = data_dir / "acronym_map.csv"
    acronym_dict = {}
    if acronym_path.exists():
        df_acronyms = pd.read_csv(acronym_path, dtype=str)
        df_acronyms.columns = df_acronyms.columns.str.strip()
        acronym_dict = dict(zip(
            df_acronyms["acronym"].str.strip().str.lower(),
            df_acronyms["expansion"].str.strip().str.lower()
        ))
    else:
        print(f"Warning: '{acronym_path.name}' not found. Skipping acronym expansion.")

    # 3. ICD-10 Codes
    icd_path = data_dir / "icd10_codes.csv"
    if not icd_path.exists():
        raise FileNotFoundError(f"ICD registry '{icd_path.name}' not found in '{data_dir}'.")

    df_icd = pd.read_csv(icd_path, dtype={"icd10_code": str, "description": str})
    df_icd.columns = df_icd.columns.str.strip()
    df_icd["icd10_code"] = df_icd["icd10_code"].astype(str).str.strip().str.replace('"', '')
    df_icd["description"] = df_icd["description"].astype(str).str.strip().str.replace('"', '')

    # Filter by target chapters
    df_icd_filtered = df_icd[df_icd["icd10_code"].str.startswith(target_chapters)].copy()
    df_icd_filtered.reset_index(drop=True, inplace=True)

    return hedge_patterns, acronym_dict, df_icd_filtered


# =============================================================================
# 3. CLINICAL EXTRACTION & NORMALIZATION
# =============================================================================
def extract_assessment_and_plan(text: str) -> str:
    """Extracts text between 'Assessment & Plan' and four or more underscores."""
    pattern = r"Assessment & Plan\s*\n(.*?)(?=\n_{4,})"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_candidate_diagnoses(
    ap_text: str, 
    hedge_patterns: list[str] = None, 
    acronym_dict: dict[str, str] = None
) -> list[dict]:
    """Parses problem blocks into raw display terms and cleaned matching terms."""
    if not ap_text:
        return []

    blocks = re.split(r"\n\s*\n", ap_text.strip())
    candidates = []

    for block in blocks:
        first_line = block.split("\n")[0].strip()

        if first_line.startswith(("-", "•", "*", "Follow Up")):
            continue

        parts = re.split(r"[:—]", first_line, maxsplit=1)
        raw_term = parts[0].strip()

        # 1. Strip billing tags
        clean = re.sub(r"[\(\[\{]\s*(?:CMS/HCC|CMS|HCC)\s*[\)\]\}]", "", raw_term, flags=re.IGNORECASE)

        # 2. Strip external hedge words
        if hedge_patterns:
            for pattern in hedge_patterns:
                clean = re.sub(pattern, " ", clean, flags=re.IGNORECASE)

        # 3. Strip punctuation
        clean = re.sub(r"[^\w\s]", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        # 4. Expand acronyms / synonyms (longest first)
        if acronym_dict:
            sorted_terms = sorted(acronym_dict.keys(), key=len, reverse=True)
            for phrase in sorted_terms:
                pattern = rf"\b{re.escape(phrase)}\b"
                clean = re.sub(pattern, acronym_dict[phrase], clean, flags=re.IGNORECASE)
            clean = re.sub(r"\s+", " ", clean).strip()

        # 5. Deduplicate redundant words preserving order
        tokens = clean.split()
        seen = set()
        deduped = [w for w in tokens if not (w.lower() in seen or seen.add(w.lower()))]
        clean = " ".join(deduped)

        if clean and len(clean.split()) <= 10:
            candidates.append({"raw": raw_term, "clean": clean})

    return candidates


# =============================================================================
# 4. TERMINOLOGY MATCHER
# =============================================================================
def match_candidates_to_icd(
    candidates: list[dict], 
    registry_df: pd.DataFrame, 
    threshold: float = 60.0
) -> pd.DataFrame:
    """Matches candidate diagnoses using a 2-stage token-subset and fuzzy ranker."""
    matches = []

    for cand_dict in candidates:
        raw_term = cand_dict["raw"]
        clean_term = cand_dict["clean"]
        cand_tokens = set(clean_term.lower().split())

        def has_all_tokens(desc: str) -> bool:
            desc_tokens = set(re.findall(r"\w+", str(desc).lower()))
            return cand_tokens.issubset(desc_tokens)

        filtered_subset = registry_df[registry_df["description"].apply(has_all_tokens)].copy()

        if not filtered_subset.empty:
            has_explicit_etiology = any(e in cand_tokens for e in KNOWN_ETIOLOGIES)

            if not has_explicit_etiology:
                unspecified_subset = filtered_subset[
                    filtered_subset["description"].str.contains(r"\bunspecified\b", case=False, regex=True)
                ]
                if not unspecified_subset.empty:
                    filtered_subset = unspecified_subset

            candidate_pool = filtered_subset["description"].tolist()
            best_match = process.extractOne(
                clean_term, candidate_pool, scorer=fuzz.token_sort_ratio
            )
            matched_desc, score, pool_idx = best_match
            matched_code = filtered_subset.iloc[pool_idx]["icd10_code"]

            matches.append({
                "candidate_term": raw_term,
                "icd10_code": matched_code,
                "icd10_description": matched_desc,
                "confidence_score": 100.0,
            })
        else:
            all_pool = registry_df["description"].tolist()
            best_match = process.extractOne(
                clean_term, all_pool, scorer=fuzz.token_set_ratio
            )

            if best_match and best_match[1] >= threshold:
                matched_desc, score, idx = best_match
                matched_code = registry_df.iloc[idx]["icd10_code"]
                matches.append({
                    "candidate_term": raw_term,
                    "icd10_code": matched_code,
                    "icd10_description": matched_desc,
                    "confidence_score": round(score, 2),
                })
            else:
                matches.append({
                    "candidate_term": raw_term,
                    "icd10_code": "NO_MATCH",
                    "icd10_description": "None",
                    "confidence_score": 0.0,
                })

    return pd.DataFrame(matches)


# =============================================================================
# 5. PIPELINE ORCHESTRATION & DISPLAY
# =============================================================================
def main():
    """Main execution entry point."""
    project_root = Path(__file__).resolve().parent.parent
    notes_dir = project_root / "notes"
    data_dir = project_root / "data"

    # 1. User Interaction & Note Loading
    selected_file = select_note_file(notes_dir)
    print(f"\nYou selected: {selected_file.name}")
    provider = input("Enter the provider name: ").strip()

    df_notes = load_clinical_note(selected_file, provider)

    # 2. Extract Assessment & Plan
    df_notes["assessment_plan"] = df_notes["clinical_text"].apply(extract_assessment_and_plan)

    print("\n" + "=" * 60)
    print(f"Note ID  : {df_notes.loc[0, 'note_id']}")
    print(f"Provider : {df_notes.loc[0, 'provider']}")
    print("=" * 60)
    print("EXTRACTED ASSESSMENT & PLAN:")
    print("-" * 60)
    print(df_notes.loc[0, "assessment_plan"])
    print("=" * 60)

    # 3. Load Registries & Support Tables
    hedge_patterns, acronym_dict, df_icd_filtered = load_reference_data(data_dir)

    # 4. Extract & Normalize Diagnosis Candidates
    extracted_candidates = extract_candidate_diagnoses(
        df_notes.loc[0, "assessment_plan"],
        hedge_patterns=hedge_patterns,
        acronym_dict=acronym_dict
    )

    print("\nEXTRACTED CANDIDATE PHRASES:")
    for c in extracted_candidates:
        print(f" - Raw: '{c['raw']}' | Cleaned: '{c['clean']}'")

    # 5. Match Candidates to ICD-10
    df_matches = match_candidates_to_icd(extracted_candidates, df_icd_filtered)

    # 6. Format and Display Clean Output
    print("\n" + "=" * 60)
    print("MATCHED ICD-10 ENTITIES:")
    print("=" * 60)

    # Left-align formatted output
    formatters = {
        col: f"{{:<{max(len(col), df_matches[col].astype(str).str.len().max())}}}".format
        for col in df_matches.columns
    }
    print(df_matches.to_string(index=False, justify="left", formatters=formatters))
    print("=" * 60)


if __name__ == "__main__":
    main()