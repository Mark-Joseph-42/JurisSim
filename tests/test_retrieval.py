import pytest
import os
from src.vector_db import VectorDB

@pytest.fixture(scope="module")
def db():
    # Set up in-memory DB for testing
    os.environ["QDRANT_URL"] = ":memory:"
    vdb = VectorDB(collection_name="test_laws")
    vdb.index_all_mock_data("mock_data")
    return vdb

def test_carbon_retrieval(db):
    query = "subsidiary company carbon emissions splitting"
    results = db.search(query, top_k=3)
    texts = [r['text'] for r in results]
    assert any("carbon emissions per facility" in t.lower() for t in texts)
    assert any("[Corporate Carbon Emissions Control Act]" in t for t in texts)

def test_tax_retrieval(db):
    query = "loyalty points as payment avoiding tax"
    results = db.search(query, top_k=3)
    texts = [r['text'] for r in results]
    assert any("currency" in t.lower() for t in texts)
    assert any("[Digital Invoicing and Taxation Act]" in t for t in texts)

def test_privacy_retrieval(db):
    query = "routing data through foreign proxy"
    results = db.search(query, top_k=3)
    texts = [r['text'] for r in results]
    assert any("domestic controller" in t.lower() for t in texts)
    assert any("[Digital Privacy and Data Protection Act]" in t for t in texts)

def test_age_retrieval(db):
    query = "user registration age requirement"
    results = db.search(query, top_k=3)
    texts = [r['text'] for r in results]
    assert any("18 years of age" in t.lower() for t in texts)
    assert any("[Unified User Registration Act]" in t for t in texts)
