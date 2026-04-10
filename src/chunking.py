from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text:
            return []
        
        # Split by . , ! , ? or .\n as defined in the rules
        raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_group))
        
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            # No more separators, hard split at chunk_size
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]
        
        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]
        
        # Split by the current separator
        if sep == "":
            # Character split
            parts = list(current_text)
        else:
            parts = current_text.split(sep)
            
        final_chunks: list[str] = []
        current_chunk = ""
        
        for part in parts:
            # If the current_chunk + new part + separator is within size
            # We need to account for the separator we removed during split
            tmp_chunk = current_chunk + (sep if current_chunk else "") + part
            
            if len(tmp_chunk) <= self.chunk_size:
                current_chunk = tmp_chunk
            else:
                # current_chunk is full, save it
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                # Now decide what to do with 'part'
                if len(part) <= self.chunk_size:
                    current_chunk = part
                else:
                    # 'part' itself is too large, recurse on it with more separators
                    sub_chunks = self._split(part, next_seps)
                    # The last sub_chunk might be merged with the next parts if they fit
                    final_chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                    
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
        
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        
        results = {}
        for name, chunker in chunkers.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0
            results[name] = {
                "count": count,
                "avg_length": avg_len,
                "chunks": chunks
            }
        return results
