def format_error_message(mapping, op):
    sep = "\n    " if len(mapping) == 1 else "\n -- "
    if op == "attach":
        message = "Cannot attach units:"
        message = sep.join(
            [message]
            + [
                f"cannot attach units to variable {key!r}: {unit} (reason: {str(e)})"
                for key, (unit, e) in mapping.items()
            ]
        )
    elif op == "parse":
        message = "Cannot parse units:"
        message = sep.join(
            [message]
            + [
                f"invalid units for variable {key!r}: {unit} ({type}) (reason: {str(e)})"
                for key, (unit, type, e) in mapping.items()
            ]
        )
    elif op == "convert":
        message = "Cannot convert variables:"
        message = sep.join(
            [message]
            + [
                f"incompatible units for variable {key!r}: {error}"
                for key, error in mapping.items()
            ]
        )
    elif op == "convert_indexers":
        message = "Cannot convert indexers:"
        message = sep.join(
            [message]
            + [
                f"incompatible units for indexer for {key!r}: {error}"
                for key, error in mapping.items()
            ]
        )
    else:
        raise ValueError("invalid op")

    return message
