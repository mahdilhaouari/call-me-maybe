# src/json_state_tracker.py


class JSONStateTracker:
    """
    Tracks position inside a JSON object character by character.
    Used by the constrained decoder to know which characters are
    legal at every generation step.
    """

    def __init__(self) -> None:
        self.state       = "START"
        self.depth       = 0
        self.current_key = ""
        self.escape_mode = False
        self.in_quotes   = False

    def reset(self) -> None:
        self.__init__()

    def update(self, char: str) -> str:
        if self.escape_mode:
            self.escape_mode = False
            return self.state

        s = self.state

        if s == "START":
            if char == '{':
                self.state, self.depth, self.current_key = "IN_KEY", 1, ""
            else:
                raise ValueError(f"Expected '{{', got '{char}'")

        elif s == "IN_KEY":
            if char == '"' and not self.in_quotes:
                self.in_quotes = True
            elif char == '"':
                self.in_quotes, self.state = False, "AFTER_KEY"
            elif char == '\\':
                self.escape_mode = True
            elif self.in_quotes:
                self.current_key += char
            elif char == '}':
                self.depth -= 1
                self.state = "END" if self.depth == 0 else "AFTER_VALUE"
            else:
                raise ValueError(f"Expected '\"', got '{char}'")

        elif s == "AFTER_KEY":
            if char == ':':
                self.state = "AFTER_COLON"
            else:
                raise ValueError(f"Expected ':', got '{char}'")

        elif s == "AFTER_COLON":
            if char == '"':
                self.state, self.in_quotes = "IN_STRING", True
            elif char == '{':
                self.depth += 1
                self.state, self.current_key = "IN_KEY", ""
            elif char == '[':
                self.state = "AFTER_VALUE"
            elif char in '-0123456789':
                self.state = "IN_NUMBER"
            elif char in 'tfn':
                self.state = "AFTER_VALUE"
            else:
                raise ValueError(f"Invalid value start '{char}'")

        elif s == "IN_NUMBER":
            if char in '0123456789.eE+-':
                pass
            elif char == ',':
                self.state, self.current_key = "AFTER_COMMA", ""
            elif char == '}':
                self.depth -= 1
                self.state = "END" if self.depth == 0 else "AFTER_VALUE"
            else:
                raise ValueError(f"Unexpected '{char}' in number")

        elif s == "IN_STRING":
            if char == '"':
                self.in_quotes, self.state = False, "AFTER_VALUE"
            elif char == '\\':
                self.escape_mode = True

        elif s == "AFTER_VALUE":
            if char == ',':
                self.state, self.current_key = "AFTER_COMMA", ""
            elif char == '}':
                self.depth -= 1
                self.state = "END" if self.depth == 0 else "AFTER_VALUE"
            else:
                raise ValueError(f"Expected ',' or '}}', got '{char}'")

        elif s == "AFTER_COMMA":
            if char == '"':
                self.state, self.current_key, self.in_quotes = "IN_KEY", "", True
            else:
                raise ValueError(f"Expected '\"', got '{char}'")

        elif s == "END":
            raise ValueError("JSON already complete")

        return self.state

    def get_valid_next_chars(self) -> set:
        if self.escape_mode:
            return set('\\/bfnrtu"')
        s = self.state
        if s == "START":        return {'{'}
        if s == "IN_KEY":       return ({'"', '}'} if not self.in_quotes
                                        else {chr(i) for i in range(32, 127)})
        if s == "AFTER_KEY":    return {':'}
        if s == "AFTER_COLON":  return {'"', '{', '[', '-', 't', 'f', 'n'} | {str(d) for d in range(10)}
        if s == "IN_NUMBER":    return {str(d) for d in range(10)} | {'.', 'e', 'E', '+', '-', ',', '}'}
        if s == "IN_STRING":    return {chr(i) for i in range(32, 127)}
        if s == "AFTER_VALUE":  return {',', '}'} if self.depth > 0 else {'}'}
        if s == "AFTER_COMMA":  return {'"'}
        return set()

    def is_complete(self) -> bool:     return self.state == "END"
    def get_current_state(self) -> str: return self.state
    def get_depth(self) -> int:         return self.depth