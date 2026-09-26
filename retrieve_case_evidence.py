import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parent
SCENARIOS = ROOT / "syn_data/product_a_scenarios_v1"
DB_PATH = ROOT / "artifacts/chroma_product_a"
OUTPUT = ROOT / "artifacts/retrieval/case_07_evidence.json"

STAGES = ("observation", "investigation", "action", "retest")


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    # 1. Load the incoming failure separately from historical evidence.
    query_path = SCENARIOS / "case_07/observation.json"
    query = read_json(query_path)

    if (
        query["role"] != "query_only"
        or query.get("index_as_history") is not False
    ):
        raise ValueError("Case 07 must remain query-only.")

    query_test = query["test_result"]["test_number"]

    # 2. Build a DUT -> historical case mapping from the local JSON files.
    historical_cases = {}

    for folder in sorted(SCENARIOS.glob("case_*")):
        observation_path = folder / "observation.json"
        observation = read_json(observation_path)

        # This excludes Case 07.
        if observation["role"] != "historical":
            continue

        dut_id = observation["dut_id"]

        if dut_id == query["dut_id"]:
            raise ValueError("The incoming DUT appears in historical cases.")

        if dut_id in historical_cases:
            raise ValueError(f"Duplicate historical DUT: {dut_id}")

        records = {"observation": observation}

        for stage in STAGES[1:]:
            record = read_json(folder / f"{stage}.json")

            if (
                record["dut_id"] != dut_id
                or record["case_id"] != observation["case_id"]
                or record["observation_id"] != observation["record_id"]
                or record["stage"] != stage
                or record["role"] != "historical"
            ):
                raise ValueError(f"Invalid case link: {folder.name}/{stage}")

            records[stage] = record

        historical_cases[dut_id] = {
            "folder": folder,
            "records": records,
        }

    # First prototype rule: compare cases with the same failed test.
    eligible = {
        dut_id: case
        for dut_id, case in historical_cases.items()
        if case["records"]["observation"]["original_chunk"]["failed_test"]
        == query_test
    }

    print("Query DUT:", query["dut_id"])
    print("Failed test:", query_test)
    print("Historical cases available:", len(historical_cases))
    print("Same-test candidates:", len(eligible))

    if not eligible:
        print("No eligible historical cases. No evidence bundle created.")
        return

    # 3. Open the existing Chroma collection.
    if not DB_PATH.exists():
        raise FileNotFoundError("Chroma database folder was not found.")

    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_collection(
        name="product_a_observations",
        embedding_function=None,
    )

    # Check that the candidate observations are present and up to date.
    stored = collection.get(
        ids=list(eligible),
        include=["documents", "metadatas"],
    )

    stored_documents = dict(zip(stored["ids"], stored["documents"]))
    stored_metadata = dict(zip(stored["ids"], stored["metadatas"]))

    for dut_id, case in eligible.items():
        expected = case["records"]["observation"]["retrieval_text"]

        if stored_documents.get(dut_id) != expected:
            raise ValueError(
                f"Missing or outdated Chroma observation: {dut_id}"
            )

        if (stored_metadata.get(dut_id) or {}).get("dut_id") != dut_id:
            raise ValueError(f"Missing DUT metadata in Chroma: {dut_id}")

    # 4. Embed only the incoming observation and rank candidates.
    embedder = DefaultEmbeddingFunction()
    query_vector = embedder([query["retrieval_text"]])[0]

    results = collection.query(
        query_embeddings=[
            [float(value) for value in query_vector]
        ],
        where={"dut_id": {"$in": list(eligible)}},
        n_results=min(3, len(eligible)),
        include=["documents", "metadatas", "distances"],
    )

    # 5. Assemble the current observation and linked historical evidence.
    bundle = {
        "schema_version": "1.0",
        "purpose": "Evidence input for later LLM integration",
        "current_incident": {
            "source_file": relative(query_path),
            "record": query,
        },
        "retrieval": {
            "collection": collection.name,
            "embedding_model": "all-MiniLM-L6-v2",
            "method": "observation_vector_search_then_case_lookup",
            "candidate_scope": (
                "Historical scenario DUTs with the same failed test"
            ),
            "failed_test": query_test,
            "candidate_count": len(eligible),
            "distance_is_cause_probability": False,
        },
        "historical_matches": [],
        "answer_constraints": [
            "Historical findings belong only to their historical DUT.",
            "Do not claim historical checks were performed on Case 07.",
            "Similarity supports possible causes, not confirmation.",
            "Preserve synthetic and review-status labels.",
            "Cite the supporting record IDs and source files.",
            "If evidence is insufficient, state what check is needed.",
        ],
    }

    for rank, dut_id in enumerate(results["ids"][0], start=1):
        case = eligible[dut_id]
        records = case["records"]
        observation = records["observation"]
        original = observation["original_chunk"]

        # A concise observation record avoids duplicating the raw chunk.
        observation_evidence = {
            "record_id": observation["record_id"],
            "case_id": observation["case_id"],
            "dut_id": dut_id,
            "stage": "observation",
            "synthetic": observation["synthetic"],
            "retrieval_text": observation["retrieval_text"],
            "raw_log_source": {
                "file": original["source_file"],
                "lines": original["source_lines"],
                "sha256": original["source_sha256"],
            },
        }

        distance = float(results["distances"][0][rank - 1])

        match = {
            "rank": rank,
            "case_id": observation["case_id"],
            "dut_id": dut_id,
            "vector_distance": distance,
            "evidence": [],
        }

        for stage in STAGES:
            record = (
                observation_evidence
                if stage == "observation"
                else records[stage]
            )

            match["evidence"].append({
                "citation_id": record["record_id"],
                "source_file": relative(case["folder"] / f"{stage}.json"),
                "record": record,
            })

        bundle["historical_matches"].append(match)

        print(f"\nMatch {rank}: {observation['case_id']}")
        print("  DUT:", dut_id)
        print("  Distance:", round(distance, 4))
        print("  Loaded: observation, investigation, action, retest")

    if not bundle["historical_matches"]:
        raise RuntimeError("Chroma returned no historical matches.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(bundle, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\nEvidence bundle saved:", relative(OUTPUT))
    print("Chroma records:", collection.count())
    print("No records inserted or updated. No LLM called.")


if __name__ == "__main__":
    main()
