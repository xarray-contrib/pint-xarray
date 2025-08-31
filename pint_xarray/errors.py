from collections.abc import Hashable
from typing import Any


class PintExceptionGroup(ExceptionGroup):
    pass


def _add_note(e: Exception, note: str) -> Exception:
    e.add_note(note)

    return e


def create_exception_group(mapping: dict[Hashable, Any], op: str) -> ExceptionGroup:
    match op:
        case "attach":
            message = "Cannot attach units"
            errors = [
                _add_note(e, f"cannot attach units to variable {key!r}: {unit}")
                for key, (unit, e) in mapping.items()
            ]
        case "parse":
            message = "Cannot parse units"
            errors = [
                _add_note(e, f"invalid units for variable {key!r}: {unit} ({type})")
                for key, (unit, type, e) in mapping.items()
            ]
        case "convert":
            message = "Cannot convert variables"
            errors = [
                _add_note(e, f"incompatible units for variable {key!r}")
                for key, e in mapping.items()
            ]
        case "convert_indexers":
            message = "Cannot convert indexers"
            errors = [
                _add_note(e, f"incompatible units for indexer for {key!r}")
                for key, e in mapping.items()
            ]
        case _:  # pragma: no cover
            raise ValueError("invalid op")

    return PintExceptionGroup(message, errors)
