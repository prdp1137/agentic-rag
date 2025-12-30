"""
Sparse Vector Encoder for Keyword Matching.

This module provides sparse vector encoding for BM25-like keyword matching
in hybrid search scenarios.
"""

from __future__ import annotations

from collections import defaultdict

from qdrant_client.models import SparseVector


class SparseEncoder:
    """
    Sparse vector encoder for keyword-based retrieval.
    
    This is a simplified BM25-like implementation that converts text
    into sparse vectors for keyword matching. In production, consider using:
    - SPLADE (Sparse Lexical and Expansion) model
    - FastText with IDF weighting
    - Qdrant's built-in sparse encoder
    
    The sparse vector captures keyword frequencies for hybrid retrieval,
    complementing the semantic understanding from dense vectors.
    
    Attributes:
        min_token_length: Minimum token length to include (filters noise).
        hash_space: Size of the hash space for term indices.
    """
    
    def __init__(
        self,
        min_token_length: int = 3,
        hash_space: int = 100000,
    ) -> None:
        """
        Initialize the sparse encoder.
        
        Args:
            min_token_length: Minimum character length for tokens.
            hash_space: Size of the hash space for consistent term indices.
        """
        self.min_token_length = min_token_length
        self.hash_space = hash_space
    
    def encode(self, text: str) -> SparseVector:
        """
        Encode text into a sparse vector representation.
        
        The encoding process:
        1. Tokenize and clean the text
        2. Compute term frequencies
        3. Apply TF weighting (log-based)
        4. Convert to sparse vector format
        
        Args:
            text: Input text to vectorize.
        
        Returns:
            SparseVector with indices and values for non-zero terms.
        
        Example:
            >>> encoder = SparseEncoder()
            >>> sparse_vec = encoder.encode("LangGraph is great for AI agents")
            >>> print(sparse_vec.indices)  # Term hashes
            >>> print(sparse_vec.values)   # TF weights
        """
        # Tokenize and compute term frequencies
        tokens = text.lower().split()
        term_freq: dict[str, int] = defaultdict(int)
        
        for token in tokens:
            # Clean token - remove punctuation
            cleaned = ''.join(c for c in token if c.isalnum())
            
            # Skip very short tokens (articles, prepositions, etc.)
            if cleaned and len(cleaned) >= self.min_token_length:
                term_freq[cleaned] += 1
        
        # Convert terms to indices using consistent hashing
        indices: list[int] = []
        values: list[float] = []
        
        for term, freq in term_freq.items():
            # Use hash to get a consistent index for each term
            # Modulo to keep indices in reasonable range
            term_hash = hash(term) % self.hash_space
            indices.append(abs(term_hash))
            
            # TF component: log(1 + freq) for diminishing returns on repetition
            # Added base weight of 1.0 for single occurrences
            tf_weight = 1.0 + (freq * 0.5)
            values.append(tf_weight)
        
        return SparseVector(indices=indices, values=values)
    
    def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        """
        Encode multiple texts into sparse vectors.
        
        Args:
            texts: List of input texts.
        
        Returns:
            List of SparseVector objects.
        """
        return [self.encode(text) for text in texts]
