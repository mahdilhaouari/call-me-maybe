# src/token_trie.py
from typing import Dict, List, Set


class TokenTrie:
    """Prefix trie over token-ID sequences for constraining function name generation."""

    def __init__(self, sequences: List[List[int]]) -> None:
        self.root: Dict = {}
        for seq in sequences:
            node = self.root
            for tok in seq:
                node = node.setdefault(tok, {})
            node["$"] = True  # marks a complete sequence

    def get_allowed_next_tokens(self, prefix: List[int]) -> Set[int]:
        node = self.root
        for tok in prefix:
            if tok not in node:
                return set()
            node = node[tok]
        return {t for t in node if t != "$"}

    def is_complete_prefix(self, prefix: List[int]) -> bool:
        node = self.root
        for tok in prefix:
            if tok not in node:
                return False
            node = node[tok]
        return "$" in node