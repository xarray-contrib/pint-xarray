# Copyright 2014-2024, xarray developers
# Copyright 2025, Callan Gray

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
