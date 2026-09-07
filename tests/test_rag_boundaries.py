"""Scope-before-budget and schema-drift regressions; no model downloads."""

import duckdb
import yaml

from engine import retrieval, vector_index
from scripts import run_retrieval_eval


def test_vector_scopes_before_top_k(monkeypatch):
    class Index:
        def count(self):
            return 24

        def query(self, *, query_texts, n_results):
            assert n_results == 24
            names = [f"private_{i}" for i in range(22)] + ["visible_a", "visible_b"]
            return {"ids": [names], "metadatas": [[{}] * 24],
                    "distances": [[i / 24 for i in range(24)]]}

    monkeypatch.setattr(retrieval, "build_index", lambda con: Index())
    hits = retrieval.retrieve("revenue", k=2, allowed_tables={"visible_a", "visible_b"})
    assert [hit.table for hit in hits] == ["visible_a", "visible_b"]


def test_keyword_scopes_before_top_k(monkeypatch):
    corpus = [{"id": name, "document": doc,
               "metadata": {"domain": "demo", "description": ""}}
              for name, doc in [("private", "revenue sales"), ("visible", "revenue")]]
    monkeypatch.setattr(retrieval, "build_corpus", lambda con: corpus)
    hits = retrieval.retrieve_keyword("revenue sales", k=1, allowed_tables={"visible"})
    assert [hit.table for hit in hits] == ["visible"]


def test_hybrid_passes_scope_to_both_rankings(monkeypatch):
    calls = []

    def ranked(question, *, k, con, allowed_tables):
        calls.append(allowed_tables)
        return [retrieval.RetrievedTable("visible", "demo", "", .9)]

    monkeypatch.setattr(retrieval, "retrieve", ranked)
    monkeypatch.setattr(retrieval, "retrieve_keyword", ranked)
    assert retrieval.retrieve_hybrid("revenue", allowed_tables={"visible"})[0].table == "visible"
    assert calls == [{"visible"}, {"visible"}]


def test_empty_scope_does_not_build_or_query_an_index(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty scope must not retrieve")

    monkeypatch.setattr(retrieval, "build_index", unexpected)
    monkeypatch.setattr(retrieval, "build_corpus", unexpected)
    for picker in (retrieval.retrieve, retrieval.retrieve_keyword, retrieval.retrieve_hybrid):
        assert picker("revenue", allowed_tables=set()) == []


def test_index_rebuilds_when_columns_change(monkeypatch):
    builds = []
    monkeypatch.setattr(retrieval, "_collection", None)
    monkeypatch.setattr(retrieval, "_collection_source", None)

    def build(name, corpus, fingerprint):
        value = {"fingerprint": fingerprint, "corpus": corpus}
        builds.append(value)
        return value

    monkeypatch.setattr(vector_index.LocalVectorIndex, "build", build)
    with duckdb.connect() as con:
        con.execute("CREATE TABLE healthcare_fact_claims (claim_id INT)")
        first = retrieval.build_index(con)
        assert retrieval.build_index(con) is first
        con.execute("ALTER TABLE healthcare_fact_claims ADD COLUMN fresh_measure DOUBLE")
        second = retrieval.build_index(con)
        assert second is not first
        assert first["fingerprint"] != second["fingerprint"]
        record = next(row for row in second["corpus"] if row["id"] == "healthcare_fact_claims")
        assert "fresh_measure" in record["document"]
        assert len(builds) == 2


def test_schema_revision_ignores_row_values_but_tracks_descriptions(monkeypatch):
    with duckdb.connect() as con:
        con.execute("CREATE TABLE healthcare_fact_claims (claim_id INT)")
        before = retrieval.corpus_revision(con)
        con.execute("INSERT INTO healthcare_fact_claims VALUES (999)")
        assert retrieval.corpus_revision(con) == before
        monkeypatch.setattr(retrieval, "DOMAINS", {**retrieval.DOMAINS, "healthcare": "updated"})
        assert retrieval.corpus_revision(con) != before


def test_paraphrase_labels_are_separate_and_reference_known_contracts():
    root = run_retrieval_eval.ROOT
    references = {row["id"]: row for row in yaml.safe_load(
        (root / "evals" / "golden_questions.yaml").read_text(encoding="utf-8"))}
    cases = yaml.safe_load((root / "evals" / "retrieval_paraphrases.yaml")
                           .read_text(encoding="utf-8"))
    assert len(cases) == len({row["id"] for row in cases}) == 22
    assert len({references[row["reference"]]["domain"] for row in cases}) == 11
    for case in cases:
        assert case["question"] != references[case["reference"]]["question"]


def test_retrieval_quality_gate_fails_on_missing_required_tables(monkeypatch, tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("- id: probe\n  question: probe\n  sql: SELECT * FROM required_table\n",
                    encoding="utf-8")
    monkeypatch.setattr(run_retrieval_eval, "GOLDEN", path)
    monkeypatch.setattr(run_retrieval_eval, "build_warehouse", lambda: duckdb.connect())
    monkeypatch.setattr(run_retrieval_eval, "table_names", lambda con: ["required_table"])
    monkeypatch.setattr(run_retrieval_eval, "schema_catalog", lambda con: "catalog")
    monkeypatch.setattr(retrieval, "build_index", lambda con: None)
    monkeypatch.setattr(retrieval, "schema_catalog_for", lambda *args, **kwargs: "catalog")
    for name in ("retrieve", "retrieve_keyword", "retrieve_hybrid"):
        monkeypatch.setattr(retrieval, name, lambda *args, **kwargs: [])
    assert run_retrieval_eval.main(["--min-coverage", "1"]) == 1
