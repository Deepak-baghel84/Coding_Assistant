




import json
import csv
from pathlib import Path
from collections import Counter

import tiktoken


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INPUT_FILE = "chunks.json"
OUTPUT_FILE = "chunk_quality_report.csv"

MAX_CHARS = 1500
MAX_TOKENS = 500

encoding = tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------
# 2. Load chunks
# --------------------------------------------------

def load_chunks():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


# --------------------------------------------------
# 3. Calculate chunk statistics
# --------------------------------------------------

def analyze_chunk(chunk):

    code = chunk.get("code", "")

    char_count = len(code)

    token_count = len(
        encoding.encode(code)
    )

    preview = code[:150].replace("\n", " ")

    return {
        "chunk_id": chunk.get("chunk_id"),
        "name": chunk.get("name"),
        "chunk_type": chunk.get("chunk_type"),
        "file_path": chunk.get("file_path"),
        "parent_id": chunk.get("parent_id"),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "character_count": char_count,
        "token_count": token_count,
        "is_embedding_chunk": chunk.get(
            "is_embedding_chunk", False
        ),
        "is_empty": not code.strip(),
        "is_oversized": (
            char_count > MAX_CHARS
            or token_count > MAX_TOKENS
        ),
        "code_preview": preview
    }


# --------------------------------------------------
# 4. Validate chunks
# --------------------------------------------------

def validate_chunks(chunks):

    ids = [
        chunk.get("chunk_id")
        for chunk in chunks
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count in Counter(ids).items()
        if count > 1
    ]

    code_counts = Counter(
        chunk.get("code", "")
        for chunk in chunks
        if chunk.get("code", "").strip()
    )

    duplicate_code_count = sum(
        count - 1
        for count in code_counts.values()
        if count > 1
    )

    missing_parents = []

    existing_ids = set(ids)

    for chunk in chunks:

        parent_id = chunk.get("parent_id")

        if (
            parent_id is not None
            and parent_id not in existing_ids
        ):
            missing_parents.append(
                chunk.get("chunk_id")
            )

    return {
        "duplicate_ids": duplicate_ids,
        "duplicate_code_count": duplicate_code_count,
        "missing_parents": missing_parents
    }


# --------------------------------------------------
# 5. Display report
# --------------------------------------------------

def display_report(reports, validation):

    print("\n========== CHUNK QUALITY REPORT ==========\n")

    for report in reports:

        print(f"Chunk ID      : {report['chunk_id']}")
        print(f"Name          : {report['name']}")
        print(f"Type          : {report['chunk_type']}")
        print(f"Parent ID     : {report['parent_id']}")
        print(f"Characters    : {report['character_count']}")
        print(f"Estimated tokens: {report['token_count']}")
        print(f"Embedding     : {report['is_embedding_chunk']}")
        print(f"Oversized     : {report['is_oversized']}")
        print(f"Empty         : {report['is_empty']}")
        print(f"Preview       : {report['code_preview']}")
        print("-" * 65)

    # Overall statistics
    total = len(reports)

    embedding_chunks = [
        r for r in reports
        if r["is_embedding_chunk"]
    ]

    empty_chunks = [
        r for r in reports
        if r["is_empty"]
    ]

    oversized_chunks = [
        r for r in reports
        if r["is_oversized"]
    ]

    avg_chars = (
        sum(r["character_count"] for r in reports) / total
        if total else 0
    )

    avg_tokens = (
        sum(r["token_count"] for r in reports) / total
        if total else 0
    )

    print("\n========== SUMMARY ==========")

    print("Total chunks:", total)
    print("Embedding-ready chunks:", len(embedding_chunks))
    print("Empty chunks:", len(empty_chunks))
    print("Oversized chunks:", len(oversized_chunks))
    print("Duplicate chunk IDs:", len(validation["duplicate_ids"]))
    print("Duplicate code occurrences:", validation["duplicate_code_count"])
    print("Missing parent references:", len(validation["missing_parents"]))
    print("Average characters:", round(avg_chars, 2))
    print("Average estimated tokens:", round(avg_tokens, 2))


# --------------------------------------------------
# 6. Export CSV report
# --------------------------------------------------

def export_report(reports):

    if not reports:
        print("No chunks available for export.")
        return

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=reports[0].keys()
        )

        writer.writeheader()
        writer.writerows(reports)

    print(f"\nReport saved to: {OUTPUT_FILE}")


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():

    if not Path(INPUT_FILE).exists():
        print("chunks.json not found.")
        return

    chunks = load_chunks()

    reports = [
        analyze_chunk(chunk)
        for chunk in chunks
    ]

    validation = validate_chunks(chunks)

    display_report(reports, validation)

    export_report(reports)


if __name__ == "__main__":
    main()
