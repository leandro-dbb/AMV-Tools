"""XML escaping shared by the FCPXML and FCP 7 writers.

``xml.sax.saxutils.escape`` only handles ``& < >``: enough for text nodes,
not for values written inside a double-quoted attribute.  XML 1.0 also
forbids most C0 control characters (even escaped), so a clip name pasted
with a stray ``\\x07`` would make Final Cut/Resolve/Premiere reject the file.
"""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

# XML 1.0 Char production: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD]
# | [#x10000-#x10FFFF].  Lone surrogates can't come out of valid UTF-8 text,
# but strip them too so encoding never fails.
_FORBIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")

_ATTR_ENTITIES = {'"': "&quot;", "'": "&apos;", "\t": "&#9;", "\n": "&#10;", "\r": "&#13;"}


def xml_clean(value: object) -> str:
    """``str(value)`` without the characters XML 1.0 forbids."""
    return _FORBIDDEN.sub("", "" if value is None else str(value))


def xml_text(value: object) -> str:
    """Escape for a text node."""
    return escape(xml_clean(value))


def xml_attr(value: object) -> str:
    """Escape for a quoted attribute value (``& < > " '`` and whitespace
    that attribute-value normalisation would otherwise turn into spaces)."""
    return escape(xml_clean(value), _ATTR_ENTITIES)
