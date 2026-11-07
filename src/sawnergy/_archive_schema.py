from __future__ import annotations

from collections.abc import Mapping


def validate_archive_schema(
    attributes: Mapping[str, object], *, schema_name: str, schema_version: int,
) -> None:
    """Reject missing, malformed, or incompatible archive markers with ValueError."""
    marker_types = {"producer_version": str, "schema_name": str, "schema_version": int}
    missing = [key for key in marker_types if key not in attributes]
    if missing:
        raise ValueError(
            f"Missing required archive schema metadata: {', '.join(missing)}. "
            "Unmarked legacy and partially marked archives are unsupported."
        )
    for key, expected_type in marker_types.items():
        value = attributes[key]
        if not isinstance(value, expected_type) or (key == "schema_version" and isinstance(value, bool)): # bool is int subtype, reject
            raise ValueError(
                f"Malformed archive marker {key!r}: expected {expected_type.__name__}, got {value!r}."
            )
    if attributes["schema_name"] != schema_name or attributes["schema_version"] != schema_version:
        raise ValueError(
            f"Incompatible archive schema: expected schema_name={schema_name!r}, "
            f"schema_version={schema_version!r}; encountered schema_name={attributes['schema_name']!r}, "
            f"schema_version={attributes['schema_version']!r}."
        )
