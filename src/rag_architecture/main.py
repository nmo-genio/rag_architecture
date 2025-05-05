"""Entry point for running the ingestion pipeline on a given PDF.

This script loads a PDF file, cleans the text, tags metadata, chunks the content,
and stores the resulting vector embeddings in MongoDB Atlas.
"""

import sys

from ingestion import load_and_clean, tag_metadata, chunk_docs, store_embeddings


def main() -> None:
    """Run the document ingestion pipeline.

    Loads the PDF file, cleans and processes its content, chunks the data,
    and stores vector embeddings in the MongoDB Atlas vector search index.
    """
    FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample_files/mongodb.pdf"
    LOG_PREVIEW_CHARS = 120

    cleaned = load_and_clean(FILE_PATH)
    print(f"🧹  {len(cleaned)} pages after cleaning")

    tagged = tag_metadata(cleaned)
    chunks = chunk_docs(tagged)
    print(f"📄  Produced {len(chunks)} chunks")

    sample = chunks[0]
    print(f"[Chunks] total={len(chunks)}")
    print(f"[Chunk 0 preview] {sample.page_content[:LOG_PREVIEW_CHARS]!r}")
    print(f"[Chunk 0 metadata] {sample.metadata}")

    store_embeddings(chunks)


if __name__ == "__main__":
    main()