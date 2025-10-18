from typing import Any, Mapping, MutableMapping


def merge_metadata(*metadatas: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    """
    Combine metadata mappings, where later entries override earlier keys.

    Args:
        *metadatas: Optional mapping objects to merge.

    Returns:
        A dictionary containing the merged key/value pairs.
    """
    merged: dict[str, Any] = {}
    for meta in metadatas:
        if meta is None:
            continue
        if isinstance(meta, dict):
            merged.update(meta)
        else:
            merged.update(dict(meta))
    return merged
