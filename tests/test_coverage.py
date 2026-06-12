"""Tests for entity_coverage (D6)."""

from langchain_core.documents import Document

from docquery.config import Settings
from docquery._coverage import entity_coverage


def _settings_with_store(vector_store):
    s = Settings()
    s.vs = vector_store  # _build_chroma reuses an existing .vs
    return s


def test_coverage_counts_distinct_per_type(vector_store):
    vector_store.add_documents([
        Document(page_content="a", metadata={"page": 1, "entity_instruction": "ADD;ADC"}),
        Document(page_content="b", metadata={"page": 2, "entity_instruction": "ADD"}),  # dup ADD
        Document(page_content="c", metadata={"page": 3, "entity_register": "R0;R1"}),
    ])
    cov = entity_coverage(_settings_with_store(vector_store))
    assert cov["instruction"]["count"] == 2   # ADD, ADC (ADD counted once)
    assert cov["register"]["count"] == 2


def test_coverage_by_section(vector_store):
    vector_store.add_documents([
        Document(page_content="a", metadata={"page": 1, "entity_instruction": "ADD",
                 "section": "Data Processing"}),
        Document(page_content="b", metadata={"page": 9, "entity_instruction": "LDR",
                 "section": "Memory"}),
    ])
    cov = entity_coverage(_settings_with_store(vector_store))
    assert cov["instruction"]["by_section"] == {"Data Processing": 1, "Memory": 1}


def test_coverage_empty_store_returns_empty_dict(vector_store):
    assert entity_coverage(_settings_with_store(vector_store)) == {}
