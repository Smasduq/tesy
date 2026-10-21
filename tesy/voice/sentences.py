"""Split a reply into sentences so speech can start before the reply ends."""

from __future__ import annotations

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "approx",
    "no",
    "fig",
    "vol",
    "inc",
    "ltd",
}
CLOSERS = "\"'’”)]}"


def _is_abbreviation(text: str, period_index: int) -> bool:
    """Is the dot at ``period_index`` the end of an abbreviation?"""
    start = period_index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    word = text[start:period_index].lower().rstrip(".")
    if word in ABBREVIATIONS:
        return True
    # A single letter, as in "J. R. R. Tolkien".
    return len(word) == 1 and word.isalpha()


def _split(text: str) -> tuple[list[str], str]:
    """Return the complete sentences in ``text`` and the unfinished remainder."""
    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char not in ".!?":
            index += 1
            continue

        if char == ".":
            if index > 0 and text[index - 1] == ".":
                # part of an ellipsis
                index += 1
                continue
            if _is_abbreviation(text, index):
                index += 1
                continue

        end = index + 1
        while end < length and text[end] in CLOSERS:
            end += 1

        if end < length and not text[end].isspace():
            index += 1
            continue

        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        index = end

    return sentences, text[start:]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences, remainder = _split(text)
    if remainder.strip():
        sentences.append(remainder.strip())
    return sentences


class SentenceBuffer:
    """Turns a stream of text into complete sentences as they arrive."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add text and return any sentences that are now complete."""
        self._buffer += text
        sentences, self._buffer = _split(self._buffer)
        return sentences

    def flush(self) -> list[str]:
        """Return whatever is left over."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return [remainder] if remainder else []
