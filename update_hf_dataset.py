#!/usr/bin/env python3
"""
Update BohanJia/CompBench on HuggingFace with new local metadata.

The HF dataset stores images (input_image, edited_image, mask) that are NOT
available locally. This script downloads the existing dataset, updates only
the text fields (task, instruction, caption) from local JSONL/JSON files,
then re-uploads.

Usage:
    pip install datasets huggingface_hub pyarrow pillow
    python update_hf_dataset.py --token hf_XXXX [--dry-run] [--inspect]

HF dataset structure (confirmed via --inspect):
  train split tasks    : remove, add, replace, implicit_reasoning,
                         location, action, view,
                         multi_object_add, multi_object_remove
  multi_turn split tasks: multi_turn_add, multi_turn_remove

Key encoding details:
  - implicit_reasoning image_path : underscore format as-is  e.g. "02b72b8d_2_00016.png"
  - act_loc_view image_path       : plain filename            e.g. "00000.png"
  - multi_object caption          : pipe-separated            e.g. "a yellow fish|a white fish"
  - multi_turn image_path         : prefixed by turn+task     e.g. "turn1_add/02b72b8d_00002_t1.png"
  - multi_turn instruction        : plain string (one turn)
  - multi_turn caption (remove)   : not present in local files → preserved from HF
"""

import json
import argparse
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"
HF_REPO = "BohanJia/CompBench"


# ─── File loaders ────────────────────────────────────────────────────────────

def load_json_array(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─── Build local lookup tables ───────────────────────────────────────────────
# All lookups use (task, image_path) as key to handle tasks that share image paths.
# Value is a dict of fields to update; omitted fields are preserved from HF.

def build_train_lookup() -> dict[tuple[str, str], dict]:
    """
    Returns {(task, image_path): {instruction, caption}} for all train tasks.

    Train tasks and their source files:
      remove / add / replace   → tasks/{task}/metadata.json
                                  fields: image_path, instruction, caption
      implicit_reasoning       → tasks/implicit_reasoning/implicit_info.jsonl
                                  fields: image_name (= image_path as-is), instruction, caption
      location / action / view → tasks/act_loc_view/{subtask}/instructions.jsonl
                                  fields: image_name (= image_path as-is), instruction
      multi_object_add         → tasks/multi_turn_editing/multi_object_add.jsonl
                                  fields: image_name, add_ins, caption1, caption2
                                  caption encoded as "caption1|caption2"
      multi_object_remove      → tasks/multi_turn_editing/multi_object_remove.jsonl
                                  fields: image_name, remove_ins, caption1, caption2
                                  caption encoded as "caption1|caption2"
    """
    lookup: dict[tuple[str, str], dict] = {}
    counts: dict[str, int] = {}

    # remove / add / replace — JSON array, image_path field matches HF directly
    for task in ("remove", "add", "replace"):
        records = load_json_array(TASKS_DIR / task / "metadata.json")
        for r in records:
            lookup[(task, r["image_path"])] = {
                "instruction": r["instruction"],
                "caption": r.get("caption", ""),
            }
        counts[task] = len(records)

    # implicit_reasoning — image_name is the image_path as-is (no conversion)
    # Use implicit_info.jsonl (has caption); implicit_data.jsonl has no caption — skip.
    records = load_jsonl(TASKS_DIR / "implicit_reasoning" / "implicit_info.jsonl")
    for r in records:
        lookup[("implicit_reasoning", r["image_name"])] = {
            "instruction": r["instruction"],
            "caption": r.get("caption", ""),
        }
    counts["implicit_reasoning"] = len(records)

    # act_loc_view subtasks — image_name is a plain filename, also used as-is
    for subtask in ("location", "action", "view"):
        records = load_jsonl(TASKS_DIR / "act_loc_view" / subtask / "instructions.jsonl")
        for r in records:
            lookup[(subtask, r["image_name"])] = {
                "instruction": r["instruction"],
                "caption": r.get("caption", ""),
            }
        counts[subtask] = len(records)

    # multi_object_add — in train split, caption is pipe-separated
    records = load_jsonl(TASKS_DIR / "multi_turn_editing" / "multi_object_add.jsonl")
    for r in records:
        lookup[("multi_object_add", r["image_name"])] = {
            "instruction": r["add_ins"],
            "caption": f"{r.get('caption1', '')}|{r.get('caption2', '')}",
        }
    counts["multi_object_add"] = len(records)

    # multi_object_remove — in train split, caption is pipe-separated
    records = load_jsonl(TASKS_DIR / "multi_turn_editing" / "multi_object_remove.jsonl")
    for r in records:
        lookup[("multi_object_remove", r["image_name"])] = {
            "instruction": r["remove_ins"],
            "caption": f"{r.get('caption1', '')}|{r.get('caption2', '')}",
        }
    counts["multi_object_remove"] = len(records)

    print("  Local train records per task:")
    for task, n in counts.items():
        print(f"    {task}: {n}")
    print(f"  Total unique (task, image_path) pairs: {len(lookup)}")
    return lookup


def build_multi_turn_lookup() -> dict[tuple[str, str], dict]:
    """
    Returns {(task, image_path): {instruction}} for multi_turn tasks.
    Caption is NOT updated for multi_turn_remove (not present in local files).
    Caption mapping for multi_turn_add is also complex — preserved from HF for safety.

    Each local row generates TWO HF rows (one per turn):
      multi_turn_add:
        turn1_add/{image_name}  → instruction=turn1_add_ins
        turn2_add/{image_name}  → instruction=turn2_add_ins
      multi_turn_remove:
        turn1_remove/{image_name} → instruction=turn1_remove_ins
        turn2_remove/{image_name} → instruction=turn2_remove_ins
    """
    lookup: dict[tuple[str, str], dict] = {}

    records = load_jsonl(TASKS_DIR / "multi_turn_editing" / "multi_turn_add.jsonl")
    for r in records:
        name = r["image_name"]
        lookup[("multi_turn_add", f"turn1_add/{name}")] = {
            "instruction": r["turn1_add_ins"],
        }
        lookup[("multi_turn_add", f"turn2_add/{name}")] = {
            "instruction": r["turn2_add_ins"],
        }
    print(f"    multi_turn_add: {len(records)} local rows → {len(records)*2} HF rows")

    records = load_jsonl(TASKS_DIR / "multi_turn_editing" / "multi_turn_remove.jsonl")
    for r in records:
        name = r["image_name"]
        lookup[("multi_turn_remove", f"turn1_remove/{name}")] = {
            "instruction": r["turn1_remove_ins"],
        }
        lookup[("multi_turn_remove", f"turn2_remove/{name}")] = {
            "instruction": r["turn2_remove_ins"],
        }
    print(f"    multi_turn_remove: {len(records)} local rows → {len(records)*2} HF rows")

    print(f"  Total unique (task, image_path) pairs: {len(lookup)}")
    return lookup


# ─── Row updater ─────────────────────────────────────────────────────────────

class RowUpdater:
    """
    datasets.map-compatible callable that updates text fields in-place
    while leaving image data and unspecified fields untouched.

    Lookup key is always (task, image_path). Only fields present in the
    lookup entry are updated — missing fields (e.g. caption for multi_turn_remove)
    are preserved from the existing HF row.
    """

    def __init__(self, lookup: dict[tuple[str, str], dict]):
        self.lookup = lookup
        self.updated = 0
        self.unchanged = 0
        self.not_in_local = 0

    def __call__(self, row: dict) -> dict:
        key = (row.get("task", ""), row["image_path"])
        meta = self.lookup.get(key)

        if meta is None:
            self.not_in_local += 1
            return row

        changed = any(row.get(f) != v for f, v in meta.items())
        if changed:
            self.updated += 1
            return {**row, **meta}
        else:
            self.unchanged += 1
            return row

    def report(self) -> str:
        total = self.updated + self.unchanged + self.not_in_local
        return (f"updated={self.updated}, unchanged={self.unchanged}, "
                f"not_in_local(kept as-is)={self.not_in_local}, total={total}")


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def inspect_dataset(dataset):
    """Print one sample row per task type and save to inspect_result.json."""
    output_path = Path(__file__).parent / "inspect_result.json"
    result = {}

    for split_name, split in dataset.items():
        print(f"\n── Split: {split_name} ({len(split)} rows) ──")
        print(f"   Columns: {split.column_names}")

        task_samples: dict[str, dict] = {}
        for row in split:
            t = row.get("task", "")
            if t not in task_samples:
                task_samples[t] = row
            if len(task_samples) >= 20:
                break

        result[split_name] = {
            "num_rows": len(split),
            "columns": split.column_names,
            "task_samples": {},
        }

        for task in sorted(task_samples):
            s = task_samples[task]
            entry = {
                "image_path": s["image_path"],
                "instruction": str(s["instruction"]),
                "caption": str(s.get("caption", "")),
                "mask": "present" if s.get("mask") else None,
            }
            result[split_name]["task_samples"][task] = entry
            print(f"\n  [{task}]")
            print(f"    image_path : {entry['image_path']}")
            print(f"    instruction: {entry['instruction'][:120]!r}")
            print(f"    caption    : {entry['caption'][:80]!r}")
            print(f"    mask       : {entry['mask']}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nInspect results saved to {output_path}")


def report_coverage(train_lookup, multi_turn_lookup, dataset):
    """Report local entries not found in HF (need images to be added separately)."""
    hf_train_keys  = {(r["task"], r["image_path"]) for r in dataset["train"]}
    hf_multi_keys  = {(r["task"], r["image_path"]) for r in dataset["multi_turn"]}

    new_train = [k for k in train_lookup      if k not in hf_train_keys]
    new_multi = [k for k in multi_turn_lookup if k not in hf_multi_keys]

    def show(label, items, lookup):
        if items:
            print(f"\n  {label}: {len(items)} entries in local but not in HF")
            for k in items[:5]:
                print(f"    {k[0]}: {k[1]}")
            if len(items) > 5:
                print(f"    ... and {len(items)-5} more")
        else:
            print(f"\n  {label}: all local entries matched in HF")

    show("train (new)", new_train, train_lookup)
    show("multi_turn (new)", new_multi, multi_turn_lookup)

    if new_train or new_multi:
        print("\n  NOTE: new entries require image files (input_image, edited_image, mask)")
        print("        to be added. This script only updates existing records.")

    return new_train, new_multi


# ─── Load or reuse dataset ───────────────────────────────────────────────────

def load_or_download(cache_dir: Path):
    from datasets import load_dataset, load_from_disk

    saved_dir = cache_dir / "dataset"
    if saved_dir.exists() and any(saved_dir.iterdir()):
        print(f"Found existing dataset at {saved_dir}, loading from disk...")
        return load_from_disk(str(saved_dir))

    cache_dir.mkdir(exist_ok=True)
    print(f"Downloading {HF_REPO} from HuggingFace (this may take a while — ~3.8 GB)...")
    dataset = load_dataset(HF_REPO, cache_dir=str(cache_dir))
    print(f"Saving to {saved_dir} for future runs...")
    dataset.save_to_disk(str(saved_dir))
    return dataset


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Update CompBench HF dataset metadata")
    parser.add_argument("--token", help="HuggingFace write token (or set HF_TOKEN env var)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print one sample row per task and save inspect_result.json, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes but do not push to HuggingFace")
    args = parser.parse_args()

    if args.token:
        from huggingface_hub import login
        login(token=args.token)

    cache_dir = Path(__file__).parent / "hf_cache"
    dataset = load_or_download(cache_dir)

    for name, split in dataset.items():
        print(f"  {name}: {len(split)} rows")

    if args.inspect:
        inspect_dataset(dataset)
        return

    # ── Build local lookups ──
    from datasets import DatasetDict

    print("\nBuilding train lookup from local files...")
    train_lookup = build_train_lookup()

    print("\nBuilding multi_turn lookup from local files...")
    multi_turn_lookup = build_multi_turn_lookup()

    # ── Coverage report ──
    print("\nCoverage check (local vs HF):")
    report_coverage(train_lookup, multi_turn_lookup, dataset)

    # ── Update train split ──
    print("\nUpdating train split...")
    train_updater = RowUpdater(train_lookup)
    updated_train = dataset["train"].map(train_updater, desc="train")
    print(f"  {train_updater.report()}")

    # ── Update multi_turn split ──
    print("\nUpdating multi_turn split...")
    multi_updater = RowUpdater(multi_turn_lookup)
    updated_multi = dataset["multi_turn"].map(multi_updater, desc="multi_turn")
    print(f"  {multi_updater.report()}")

    # ── Summary ──
    print("\n── Update Summary ──")
    print(f"  train     : {train_updater.report()}")
    print(f"  multi_turn: {multi_updater.report()}")

    if args.dry_run:
        print("\nDry run — not pushing to HuggingFace.")
        return

    # ── Push ──
    print(f"\nPushing updated dataset to {HF_REPO}...")
    DatasetDict({"train": updated_train, "multi_turn": updated_multi}).push_to_hub(HF_REPO)
    print("Done!")


if __name__ == "__main__":
    main()
