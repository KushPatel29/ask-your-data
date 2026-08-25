"""Schema retrieval stays measured, bounded, and safe for follow-up turns."""

from data_manifest import MANIFEST
from engine import retrieval


def test_corpus_matches_manifest_and_includes_real_columns(con):
    corpus = retrieval.build_corpus(con)
    assert len(corpus) == len(MANIFEST)
    assert len({row["id"] for row in corpus}) == len(corpus)

    claims = next(row for row in corpus if row["id"] == "healthcare_fact_claims")
    assert "claim_id" in claims["document"]
    assert claims["metadata"]["domain"] == "healthcare"


def test_retrieve_converts_cosine_distance_and_clamps_k(monkeypatch):
    class FakeCollection:
        def count(self):
            return 2

        def query(self, *, query_texts, n_results):
            assert query_texts == ["claim denial rate"]
            assert n_results == 2
            return {
                "ids": [["healthcare_fact_claims", "healthcare_dim_denial_reason"]],
                "metadatas": [[
                    {"domain": "healthcare", "description": "claims"},
                    {"domain": "healthcare", "description": "denial reasons"},
                ]],
                "distances": [[0.1, 0.25]],
            }

    monkeypatch.setattr(retrieval, "build_index", lambda con: FakeCollection())
    hits = retrieval.retrieve("claim denial rate", k=99, con=object())

    assert [hit.table for hit in hits] == [
        "healthcare_fact_claims",
        "healthcare_dim_denial_reason",
    ]
    assert [hit.score for hit in hits] == [0.9, 0.75]


def test_empty_question_does_not_build_an_index(monkeypatch):
    def fail(_con):
        raise AssertionError("an empty question must not initialize Chroma")

    monkeypatch.setattr(retrieval, "build_index", fail)
    assert retrieval.retrieve("   ", con=object()) == []


def test_follow_up_tables_are_forced_into_the_schema(monkeypatch, con):
    monkeypatch.setattr(
        retrieval,
        "retrieve",
        lambda _question, *, k, con: [
            retrieval.RetrievedTable(
                table="healthcare_fact_claims",
                domain="healthcare",
                description="claim transactions",
                score=0.9,
            )
        ],
    )

    catalog = retrieval.schema_catalog_for(
        "and by payer?",
        con,
        include_tables=("healthcare_dim_payer",),
    )

    assert "healthcare_fact_claims" in catalog
    assert "healthcare_dim_payer" in catalog
    assert "hr_fact_employees" not in catalog
