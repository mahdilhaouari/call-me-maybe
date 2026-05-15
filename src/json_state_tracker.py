# src/json_state_tracker.py


class JSONStateTracker:
    """
    A simple state machine that reads a JSON object one character at a time
    and tracks where we are in the structure.

    This tells the constrained decoder which characters are legal to generate
    at each step, so the output is always valid JSON.

    States:
        START       - haven't seen anything yet
        IN_KEY      - reading a key name inside quotes
        AFTER_KEY   - key done, waiting for ':'
        AFTER_COLON - saw ':', waiting for a value
        IN_STRING   - reading a string value
        IN_NUMBER   - reading a numeric value
        AFTER_VALUE - value done, waiting for ',' or '}'
        AFTER_COMMA - saw ',', waiting for the next key
        END         - the top-level object is closed
    """

    def __init__(self) -> None:
        self.state = "START"
        self.depth = 0          # how many { we've opened without closing
        self.current_key = ""   # the key name being built right now
        self.escape_mode = False
        self.in_quotes = False

    def reset(self) -> None:
        self.__init__()

    def update(self, char: str) -> str:
        """Feed one character and advance the state."""

        # after a backslash, the next character is always literal
        if self.escape_mode:
            self.escape_mode = False
            return self.state

        s = self.state

        if s == "START":
            if char == "{":
                self.state = "IN_KEY"
                self.depth = 1
                self.current_key = ""
            else:
                raise ValueError(f"JSON must start with '{{', got '{char}'")

        elif s == "IN_KEY":
            if char == '"' and not self.in_quotes:
                self.in_quotes = True
            elif char == '"':
                self.in_quotes = False
                self.state = "AFTER_KEY"
            elif char == "\\":
                self.escape_mode = True
            elif self.in_quotes:
                self.current_key += char
            elif char == "}":
                # empty object like {}
                self.depth -= 1
                self.state = "END" if self.depth == 0 else "AFTER_VALUE"
            else:
                raise ValueError(f"Expected '\"', got '{char}'")

        elif s == "AFTER_KEY":
            if char == ":":
                self.state = "AFTER_COLON"
            else:
                raise ValueError(f"Expected ':', got '{char}'")

        elif s == "AFTER_COLON":
            if char == '"':
                self.state = "IN_STRING"
                self.in_quotes = True
            elif char == "{":
                self.depth += 1
                self.state = "IN_KEY"
                self.current_key = ""   # reset for the nested object
            elif char == "[":
                self.state = "AFTER_VALUE"
            elif char in "-0123456789":
                self.state = "IN_NUMBER"
            elif char in "tfn":         # true / false / null
                self.state = "AFTER_VALUE"
            else:
                raise ValueError(f"Invalid value start '{char}'")

        elif s == "IN_NUMBER":
            if char in "0123456789.eE+-":
                pass                    # still reading the number
            elif char == ",":
                self.state = "AFTER_COMMA"
                self.current_key = ""
            elif char == "}":
                self.depth -= 1
                self.state = "END" if self.depth == 0 else "AFTER_VALUE"
            else:
                raise ValueError(f"Unexpected '{char}' inside number")

        elif s == "IN_STRING":
            if char == '"':
                self.in_quotes = False
                self.state = "AFTER_VALUE"
            elif char == "\\":
                self.escape_mode = True
            # any other character is just string content

        elif s == "AFTER_VALUE":
            if char == ",":
                self.state = "AFTER_COMMA"
                self.current_key = ""
            elif char == "}":
                self.depth -= 1
                self.state = "END" if self.depth == 0 else "AFTER_VALUE"
            else:
                raise ValueError(f"Expected ',' or '}}', got '{char}'")

        elif s == "AFTER_COMMA":
            if char == '"':
                self.state = "IN_KEY"
                self.current_key = ""
                self.in_quotes = True
            else:
                raise ValueError(f"Expected '\"' after comma, got '{char}'")

        elif s == "END":
            raise ValueError("JSON is already complete")

        return self.state

    def get_valid_next_chars(self) -> set:
        """Return all characters that are legal in the current state."""
        if self.escape_mode:
            return set('\\/bfnrtu"')

        s = self.state

        if s == "START":
            return {"{"}

        if s == "IN_KEY":
            if not self.in_quotes:
                return {'"', "}"}   # '}' covers empty objects
            return {chr(i) for i in range(32, 127)}

        if s == "AFTER_KEY":
            return {":"}

        if s == "AFTER_COLON":
            digits = {str(d) for d in range(10)}
            return {'"', "{", "[", "-", "t", "f", "n"} | digits

        if s == "IN_NUMBER":
            digits = {str(d) for d in range(10)}
            return digits | {".", "e", "E", "+", "-", ",", "}"}

        if s == "IN_STRING":
            return {chr(i) for i in range(32, 127)}

        if s == "AFTER_VALUE":
            return {",", "}"} if self.depth > 0 else {"}"}

        if s == "AFTER_COMMA":
            return {'"'}

        return set()

    def is_complete(self) -> bool:
        return self.state == "END"

    def get_current_state(self) -> str:
        return self.state

    def get_depth(self) -> int:
        return self.depth