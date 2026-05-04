def text2chunk(text: str, chunk_size: int, overlap: int = 0):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks