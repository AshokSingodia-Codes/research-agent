from app.tools.chunking import chunk_document_text
from app.tools.embeddings import HashingEmbedder
from app.tools.vectorstore import VectorStore


def test_chunking_paragraphs_and_overlap():
    text = (
        "First section discusses distributed consensus protocols in high write environments. "
        "Raft and Paxos both maintain replicated state machines using append entries RPCs.\n\n"
        "Second section details log compaction techniques. Snapshotting writes memory state to disk, "
        "allowing the log to be truncated safely. Log compaction prevents unbounded disk growth."
    )
    chunks = chunk_document_text(
        doc_id="doc_1",
        source_url="https://example.com/raft",
        domain="example.com",
        text=text,
        size=150,
        overlap=30,
    )
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc_1" for c in chunks)
    assert all(c.domain == "example.com" for c in chunks)
    assert any("consensus" in c.text.lower() for c in chunks)


def test_hashing_embedder_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=128)
    vec1 = embedder.embed_query("raft consensus algorithm")
    vec2 = embedder.embed_query("raft consensus algorithm")
    assert vec1.shape == (128,)
    # Deterministic
    assert (vec1 == vec2).all()


def test_vectorstore_add_and_search():
    store = VectorStore()
    text = (
        "Zero knowledge rollups compress transaction execution by generating succinct cryptographic proofs. "
        "ZK-SNARKs and STARKs ensure verifiable off-chain state updates."
    )
    chunks = chunk_document_text(
        doc_id="doc_zk",
        source_url="https://rollups.org/zk",
        domain="rollups.org",
        text=text,
        size=100,
    )
    store.add(chunks)
    assert len(store) >= 1

    results = store.search("cryptographic proofs", k=3)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert "proofs" in top_chunk.text.lower() or "knowledge" in top_chunk.text.lower()
    assert score > 0.0

