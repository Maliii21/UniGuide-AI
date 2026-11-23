if __package__:
    from .chunk_documents import CHUNK_OVERLAP, CHUNK_SIZE
    from .generate_embeddings import generate_embeddings, save_knowledge_base
    from .rebuild_chunks import CHUNKS_PATH, atomic_write, build_chunks, chunk_json, chunk_settings, load_model
else:
    from chunk_documents import CHUNK_OVERLAP, CHUNK_SIZE
    from generate_embeddings import generate_embeddings, save_knowledge_base
    from rebuild_chunks import CHUNKS_PATH, atomic_write, build_chunks, chunk_json, chunk_settings, load_model


def run(*, chunks_only=False, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    try:
        if not 0 <= overlap < chunk_size:
            raise ValueError("Chunk size must be positive and overlap must be smaller.")
        model = load_model()
        chunks, processed, skipped = build_chunks(model, chunk_size=chunk_size, overlap=overlap)
        if chunks_only:
            atomic_write(CHUNKS_PATH, chunk_json(chunks))
            embedding_summary = "embeddings not generated"
        else:
            vectors = generate_embeddings(chunks, model)
            save_knowledge_base(chunks, vectors)
            embedding_summary = f"{len(vectors)} embedding(s), {vectors.shape[1]} dimensions"
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Unable to rebuild the knowledge base: {error}")
        return 1

    print(f"Processed {processed} document(s), skipped {skipped}, "
          f"created {len(chunks)} chunk(s), {embedding_summary}.")
    return 0


def main():
    args = chunk_settings("Rebuild UniGuide AI's document index.")
    return run(chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    raise SystemExit(main())
