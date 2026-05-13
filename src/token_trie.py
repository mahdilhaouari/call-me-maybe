# src/token_trie.py
from typing import Dict, List, Set


class TokenTrie:
    """
    A prefix trie built from token-ID sequences.
    Used to ensure the model only generates valid function name tokens.
    """

    def __init__(self, sequences: List[List[int]]) -> None:
        self.root: Dict = {}
        for seq in sequences:
            node = self.root
            for tok in seq:
                node = node.setdefault(tok, {})
            node["$"] = True  # mark end of a valid sequence

    def get_allowed_next_tokens(self, prefix: List[int]) -> Set[int]:
        """Return the set of tokens that can follow the given prefix."""
        node = self.root
        for tok in prefix:
            if tok not in node:
                return set()
            node = node[tok]
        return {t for t in node if t != "$"}

    def is_complete_prefix(self, prefix: List[int]) -> bool:
        """Return True if the prefix matches one of the inserted sequences."""
        node = self.root
        for tok in prefix:
            if tok not in node:
                return False
            node = node[tok]
        return "$" in node