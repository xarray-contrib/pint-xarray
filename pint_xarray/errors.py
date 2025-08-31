from collections.abc import Hashable
from typing import Any


class PintExceptionGroup(ExceptionGroup):
    pass


def _add_note(e: Exception, note: str) -> Exception:
    e.add_note(note)

    return e


def format_error_message(mapping: dict[Hashable, Any], op: str) -> ExceptionGroup:
    messages = {
        "attach": "Cannot attach units",
        "parse": "Cannot parse units",
        "convert": "Cannot convert variables",
        "convert_indexers": "Cannot convert indexers",
    }
    message = messages.get(op)
    if message is None:  # pragma: no cover
        raise ValueError("invalid op")

    match op:
        case "attach":
            errors = [
                _add_note(e, f"cannot attach units to variable {key!r}: {unit}")
                for key, (unit, e) in mapping.items()
            ]
        case "parse":
            errors = [
                _add_note(e, f"invalid units for variable {key!r}: {unit} ({type})")
                for key, (unit, type, e) in mapping.items()
            ]
        case "convert":
            errors = [
                _add_note(e, f"incompatible units for variable {key!r}")
                for key, e in mapping.items()
            ]
        case "convert_indexers":
            errors = [
                _add_note(e, f"incompatible units for indexer for {key!r}")
                for key, e in mapping.items()
            ]

    return PintExceptionGroup(message, errors)
