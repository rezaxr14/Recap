from recap.ask.models import Citation
from recap.ask.service import _Source, verify_citations


def _source(
    *, source_type: str = "memory", source_id: str = "mem_one", text: str = "A cursor repair."
) -> _Source:
    return _Source(
        type=source_type,
        id=source_id,
        label="test source",
        text=text,
        quote=text[:20],
        score=1.0,
    )


def test_citation_must_reference_a_supplied_source() -> None:
    valid, verification = verify_citations(
        [Citation(type="memory", id="mem_missing", quote="A cursor repair.")], [_source()]
    )
    assert valid == []
    assert verification.checked == 1
    assert verification.invalid == 1
    assert verification.rejected[0]["reason"] == "source_not_in_context"


def test_quote_must_be_a_whitespace_normalized_substring() -> None:
    source = _source(text="A cursor repair.\nRestart from page one.")
    valid, verification = verify_citations(
        [
            Citation(type="memory", id="mem_one", quote="A cursor   repair."),
            Citation(type="memory", id="mem_one", quote="rewritten from memory"),
        ],
        [source],
    )
    assert [citation.quote for citation in valid] == ["A cursor   repair."]
    assert verification.valid == 1
    assert verification.invalid == 1
    assert verification.rejected[0]["reason"] == "quote_not_in_source"


def test_quote_verification_is_case_sensitive() -> None:
    valid, verification = verify_citations(
        [Citation(type="memory", id="mem_one", quote="a cursor repair.")],
        [_source()],
    )
    assert valid == []
    assert verification.invalid == 1
