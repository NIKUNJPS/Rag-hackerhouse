"""
Pulls ai4bharat/MSMARCO-XI (Hindi config) and normalizes it into a flat list of
{doc_id, title, section, text} records the chunkers can consume.

Why not `datasets.load_dataset(..., streaming=True)`: this repo ships one parquet
file per language via a custom loading script, and the script's default config
tries to materialize a combined multi-language table with a single ~9.7GB row
group -- that blows past reasonable memory/time even in streaming mode (see the
dataset's own README/loading script). Downloading one language's parquet file
directly via huggingface_hub and reading it with pyarrow sidesteps that entirely
and is what's actually fast here.

Each source row is one query with several candidate passages (MS MARCO passage
ranking format) and an `is_selected` flag marking which candidate(s) actually
answer the query. We flatten every candidate into its own document -- this keeps
the "hard negative" passages in the index too, which is what makes the chunking
hit-rate eval in backend/eval/chunking_eval.py meaningful instead of trivial.

Run standalone: python -m backend.ingestion.load_dataset
"""

from backend.config import settings

# ai4bharat/MSMARCO-XI ships one parquet per language, named <lang3>train.parquet
# / <lang3>val.parquet under train/ and validation/. Map ISO-639-1 -> that prefix.
LANG_PREFIX = {
    "hi": "hin", "bn": "ben", "gu": "guj", "kn": "kan", "ml": "mal",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san",
    "ta": "tam", "te": "tel", "ur": "urd", "as": "asm",
}


def _parquet_filename(config: str, split: str) -> str:
    prefix = LANG_PREFIX.get(config, config)
    suffix = "train" if split == "train" else "val"
    folder = "train" if split == "train" else "validation"
    return f"{folder}/{prefix}{suffix}.parquet"


def load_msmarco_xi(sample_size: int | None = None, config: str | None = None,
                     split: str = "validation") -> list[dict]:
    """
    sample_size caps the number of *source query rows* consumed (each row yields
    several passage records, typically ~8-10) -- not the final record count.
    `validation` split is used by default: it's ~98k queries in a single ~460MB
    file, small enough to download and hold in memory directly, vs. the
    multi-GB `train` split.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    sample_size = sample_size if sample_size is not None else settings.DATASET_SAMPLE_SIZE
    config = config or settings.DATASET_CONFIG
    filename = _parquet_filename(config, split)
    path = hf_hub_download(repo_id=settings.DATASET_NAME, filename=filename, repo_type="dataset")

    table = pq.read_table(path)
    if sample_size:
        table = table.slice(0, sample_size)
    df = table.to_pandas()

    records = []
    for _, row in df.iterrows():
        query_id = row.get("query_id")
        query = (row.get("query") or "").strip()
        query_type = row.get("query_type") or ""
        if not query:
            continue

        passages = row.get("passages")
        if passages is None:
            continue
        raw_translated = passages.get("Translated_passages")
        raw_selected = passages.get("is_selected")
        translated = list(raw_translated) if raw_translated is not None else []
        selected = list(raw_selected) if raw_selected is not None else []

        for i, text in enumerate(translated):
            text = (text or "").strip()
            if not text:
                continue
            is_selected = bool(i < len(selected) and selected[i] == 1)
            records.append({
                "doc_id": f"{query_id}-{i}",
                "title": query,
                "section": query_type,
                "text": text,
                "is_selected": is_selected,
                "query_id": str(query_id),
            })

    return records


if __name__ == "__main__":
    import json

    recs = load_msmarco_xi()
    print(f"loaded {len(recs)} passage records from {settings.DATASET_SAMPLE_SIZE} query rows "
          f"({sum(1 for r in recs if r['is_selected'])} marked relevant)")
    with open("backend/data/raw_records.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
