import datetime
import json
import logging
from functools import lru_cache
from typing import TypedDict

from settings import PATCHES_FILE

logger = logging.getLogger(__name__)


class PatchEntry(TypedDict):
    """Single patch metadata row."""

    id: int
    name: str
    date: str


@lru_cache(maxsize=1)
def load_patch_entries() -> list[PatchEntry]:
    with PATCHES_FILE.open(encoding="utf-8") as patch_file:
        loaded = json.load(patch_file)

    entries: list[PatchEntry] = []
    for item in loaded:
        entry = PatchEntry(
            id=int(item["id"]),
            name=str(item["name"]),
            date=str(item["date"]),
        )
        entries.append(entry)

    entries.sort(key=lambda entry: _to_unix_timestamp(entry["date"]))
    if not entries:
        msg = "patch.json is empty."
        raise RuntimeError(msg)
    return entries


def _to_unix_timestamp(iso_datetime: str) -> int:
    normalized = iso_datetime.replace("Z", "+00:00")
    parsed = datetime.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return int(parsed.timestamp())


def get_latest_patch_id() -> int:
    entries = load_patch_entries()
    return entries[-1]["id"]


def get_patch_vocab_size() -> int:
    entries = load_patch_entries()
    return max(entry["id"] for entry in entries) + 1


def resolve_patch_id(start_time: int) -> int:
    entries = load_patch_entries()
    first_patch = entries[0]
    first_start_time = _to_unix_timestamp(first_patch["date"])

    if start_time < first_start_time:
        logger.warning(
            "Match start_time %s predates earliest known patch (%s). "
            "Using earliest patch_id=%s.",
            start_time,
            first_patch["name"],
            first_patch["id"],
        )
        return first_patch["id"]

    latest_patch = entries[-1]
    latest_start_time = _to_unix_timestamp(latest_patch["date"])
    if start_time > latest_start_time:
        logger.warning(
            "Match start_time %s is newer than latest known patch (%s). "
            "Using latest patch_id=%s.",
            start_time,
            latest_patch["name"],
            latest_patch["id"],
        )
        return latest_patch["id"]

    resolved_patch_id = first_patch["id"]
    for patch in entries:
        patch_start_time = _to_unix_timestamp(patch["date"])
        if patch_start_time <= start_time:
            resolved_patch_id = patch["id"]
        else:
            break

    return resolved_patch_id
