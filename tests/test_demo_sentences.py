"""The worked examples answer in sentences, and the number is never typed.

`evals/golden_questions.yaml` now carries an `answer` beside every reference
SQL. That sentence is hand-written, which is unusual for this repository — so
the tests here are about the one property that makes it safe: the VALUE in the
sentence is always the value the query just returned, never a literal somebody
typed next to it.

If prose and query could disagree, this file would be the only place in the app
where a wrong number could survive review, because it is the part a reader
quotes and the part nobody re-runs.
"""

from __future__ import annotations

import pytest
import yaml

from engine import demo_mode


@pytest.fixture(scope="module")
def cases():
    return demo_mode.load_golden_questions()


def test_every_golden_question_carries_a_sentence(cases):
    assert len(cases) == 39
    for case in cases:
        assert case.get("answer"), case["id"]


def test_every_sentence_substitutes_the_value_rather_than_stating_it(cases):
    """The placeholder is the whole safety property. A template that spelled
    the number out would let the prose drift away from the query silently."""
    for case in cases:
        assert "{value}" in case["answer"], case["id"]


def test_no_sentence_hard_codes_its_own_expected_value(cases):
    """The stronger version of the rule above, and the one that catches a
    careless edit: the expected value must not appear as a literal anywhere in
    the sentence, because then it would read correctly even after the data
    moved underneath it."""
    for case in cases:
        expected = str(case["expect"])
        template = case["answer"].replace("{value}", "")
        assert expected not in template, (
            f"{case['id']} spells its own answer out: {case['answer']!r}")


def test_a_sentence_without_a_placeholder_is_rejected_at_load(tmp_path):
    bad = tmp_path / "golden.yaml"
    bad.write_text(yaml.safe_dump([{
        "id": "x", "domain": "d", "question": "q?",
        "sql": "SELECT 1", "expect": 1, "answer": "The answer is 1.",
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        demo_mode.load_golden_questions(bad)


def test_an_unformattable_sentence_is_rejected_at_load(tmp_path):
    bad = tmp_path / "golden.yaml"
    bad.write_text(yaml.safe_dump([{
        "id": "x", "domain": "d", "question": "q?",
        "sql": "SELECT 1", "expect": 1, "answer": "{value} and {mystery}",
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="unformattable"):
        demo_mode.load_golden_questions(bad)


def test_a_missing_answer_is_rejected_at_load(tmp_path):
    bad = tmp_path / "golden.yaml"
    bad.write_text(yaml.safe_dump([{
        "id": "x", "domain": "d", "question": "q?", "sql": "SELECT 1", "expect": 1,
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="answer"):
        demo_mode.load_golden_questions(bad)


def test_the_live_sentence_carries_the_live_number(con, cases):
    """Executed against the real warehouse: every sentence contains the value
    the reference SQL actually returned, and reads as a sentence rather than a
    cell."""
    for case in cases:
        result = demo_mode.answer(con, case)
        assert result.ok, case["id"]
        sentence = result.sentence
        assert result.headline in sentence, case["id"]
        assert sentence != result.headline, f"{case['id']} is still a bare value"
        assert sentence.endswith((".", "%", "!")), case["id"]
        assert len(sentence.split()) >= 4, f"{case['id']}: {sentence!r}"


def test_the_sentence_moves_when_the_data_moves(con, cases):
    """The point of substituting rather than stating. A result that no longer
    matches the contract still narrates its OWN number, and the badge beside it
    goes false — the prose can never be the thing that is stale."""
    case = next(c for c in cases if c["id"] == "active_employees")
    answered = demo_mode.answer(con, case)
    assert answered.matches_contract
    assert "1,483" in answered.sentence

    drifted = demo_mode.DemoAnswer(
        question=case["question"], domain=case["domain"], sql=case["sql"],
        result=answered.result, expected=999_999, template=case["answer"],
    )
    assert not drifted.matches_contract
    assert "1,483" in drifted.sentence, "the sentence reports the query, not the contract"


def test_a_failed_query_narrates_nothing(cases):
    """No sentence is better than a fluent one wrapped around a missing value."""
    from engine.query import QueryResult

    case = cases[0]
    broken = demo_mode.DemoAnswer(
        question=case["question"], domain=case["domain"], sql=case["sql"],
        result=QueryResult(sql=case["sql"], error="boom"),
        expected=case["expect"], template=case["answer"],
    )
    assert broken.sentence == "—"


def test_a_missing_template_degrades_to_the_value(cases):
    """Belt and braces behind the loader: an entry that reached the renderer
    without a template still shows a true number."""
    from engine.query import QueryResult

    case = cases[0]
    bare = demo_mode.DemoAnswer(
        question=case["question"], domain=case["domain"], sql=case["sql"],
        result=QueryResult(sql=case["sql"], columns=["n"], rows=[(42,)], row_count=1),
        expected=42, template="",
    )
    assert bare.sentence == "42"
