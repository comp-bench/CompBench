#!/usr/bin/env python3
"""
Download BohanJia/CompBench from HuggingFace and restructure into the local
directory layout expected by eval_all.py.

Usage:
    pip install datasets huggingface_hub pillow tqdm
    python download_from_hf.py [--output_dir PATH] [--split all|train|multi_turn] [--overwrite]

Directory layout produced (mirrors eval_all.py expectations):

  tasks/
    remove/
      input_image/{hash}/{idx}/{frame}.png
      edited_image/{hash}/{idx}/{frame}.png
      mask/{hash}/{idx}/{frame}.png
      metadata.json   # [{image_path, instruction, caption}, ...]
    add/    (same)
    replace/ (same)
    implicit_reasoning/
      ori_images/{image_name}
      edited_images/{image_name}
      masks/{image_name}
      implicit_info.jsonl   # {image_name, instruction, caption}
      implicit_data.jsonl   # {image_name, instruction}
    act_loc_view/
      action/
        ori_images/{image_name}
        edited_images/{image_name}
        masks/{image_name}  (empty for these tasks)
        instructions.jsonl  # {image_name, instruction}
      location/  (same)
      view/      (same)
    multi_turn_editing/
      turn1_add/{name}  (ori_images, edited_images, masks subdirs)
      turn2_add/        (same)
      turn1_remove/     (same)
      turn2_remove/     (same)
      multi_object_add/ (same)
      multi_object_remove/ (same)
      multi_turn_add.jsonl
      multi_turn_remove.jsonl
      multi_object_add.jsonl
      multi_object_remove.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

HF_REPO = "BohanJia/CompBench"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_image(pil_img, dest: Path, overwrite: bool) -> bool:
    """Save a PIL image to dest (PNG). Returns True if written, False if skipped."""
    if pil_img is None:
        return False
    if dest.exists() and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(dest, format="PNG")
    return True


def append_jsonl(path: Path, obj: dict):
    """Append one JSON object as a line to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset_cached(cache_dir: Path):
    """Load from disk if already cached, otherwise download and save."""
    from datasets import load_dataset, load_from_disk

    saved_dir = cache_dir / "dataset"
    if saved_dir.exists() and any(saved_dir.iterdir()):
        print(f"Loading dataset from cache: {saved_dir}")
        return load_from_disk(str(saved_dir))

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_REPO} from HuggingFace (this may take a while — ~3.8 GB)...")
    dataset = load_dataset(HF_REPO, cache_dir=str(cache_dir))
    print(f"Saving to {saved_dir} for future use...")
    dataset.save_to_disk(str(saved_dir))
    return dataset


# ─── Train split processors ───────────────────────────────────────────────────

def process_local_editing(rows: list[dict], output_dir: Path, task: str, overwrite: bool):
    """
    Handle tasks: add, remove, replace.
    image_path format: {hash}/{idx}/{frame}.png  e.g. "001ca3cb/0/00003.png"
    Output:
      tasks/{task}/input_image/{hash}/{idx}/{frame}.png
      tasks/{task}/edited_image/{hash}/{idx}/{frame}.png
      tasks/{task}/mask/{hash}/{idx}/{frame}.png
      tasks/{task}/metadata.json
    """
    task_dir = output_dir / task
    meta_path = task_dir / "metadata.json"

    # Load existing metadata to support resuming
    existing_metadata: list[dict] = []
    existing_paths: set[str] = set()
    if meta_path.exists() and not overwrite:
        try:
            with open(meta_path, encoding="utf-8") as f:
                existing_metadata = json.load(f)
            existing_paths = {e["image_path"] for e in existing_metadata}
        except Exception as e:
            print(f"  Warning: could not read existing {meta_path}: {e}")

    new_metadata: list[dict] = list(existing_metadata)
    skipped = written = 0

    for row in tqdm(rows, desc=f"  {task}", unit="img", leave=False):
        image_path: str = row["image_path"]

        if image_path in existing_paths and not overwrite:
            skipped += 1
            continue

        # Save images
        save_image(row.get("input_image"), task_dir / "input_image" / image_path, overwrite)
        save_image(row.get("edited_image"), task_dir / "edited_image" / image_path, overwrite)
        if row.get("mask") is not None:
            save_image(row["mask"], task_dir / "mask" / image_path, overwrite)

        # Accumulate metadata
        if image_path not in existing_paths:
            new_metadata.append({
                "image_path": image_path,
                "instruction": row.get("instruction", ""),
                "caption": row.get("caption", ""),
            })
            existing_paths.add(image_path)
        elif overwrite:
            # Update in-place
            for entry in new_metadata:
                if entry["image_path"] == image_path:
                    entry["instruction"] = row.get("instruction", "")
                    entry["caption"] = row.get("caption", "")
                    break

        written += 1

    # Write metadata.json (full rewrite when done)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)

    print(f"  {task}: {written} written, {skipped} skipped, {len(new_metadata)} total in metadata.json")


def process_implicit_reasoning(rows: list[dict], output_dir: Path, overwrite: bool):
    """
    image_path is already in underscore format: e.g. "02b72b8d_2_00016.png"
    Output:
      tasks/implicit_reasoning/ori_images/{image_name}
      tasks/implicit_reasoning/edited_images/{image_name}
      tasks/implicit_reasoning/masks/{image_name}
      tasks/implicit_reasoning/implicit_info.jsonl   (has caption)
      tasks/implicit_reasoning/implicit_data.jsonl   (no caption)
    """
    task_dir = output_dir / "implicit_reasoning"
    info_path = task_dir / "implicit_info.jsonl"
    data_path = task_dir / "implicit_data.jsonl"

    # Load already-written names
    existing: set[str] = set()
    if info_path.exists() and not overwrite:
        try:
            with open(info_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.add(json.loads(line).get("image_name", ""))
        except Exception as e:
            print(f"  Warning: could not read existing {info_path}: {e}")

    if overwrite:
        # Truncate existing JSONL files
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text("", encoding="utf-8")
        data_path.write_text("", encoding="utf-8")

    skipped = written = 0
    for row in tqdm(rows, desc="  implicit_reasoning", unit="img", leave=False):
        image_name: str = row["image_path"]  # already the correct filename

        if image_name in existing and not overwrite:
            skipped += 1
            continue

        save_image(row.get("input_image"), task_dir / "ori_images" / image_name, overwrite)
        save_image(row.get("edited_image"), task_dir / "edited_images" / image_name, overwrite)
        if row.get("mask") is not None:
            save_image(row["mask"], task_dir / "masks" / image_name, overwrite)

        if image_name not in existing or overwrite:
            record_info = {
                "image_name": image_name,
                "instruction": row.get("instruction", ""),
                "caption": row.get("caption", ""),
            }
            record_data = {
                "image_name": image_name,
                "instruction": row.get("instruction", ""),
            }
            append_jsonl(info_path, record_info)
            append_jsonl(data_path, record_data)
            existing.add(image_name)

        written += 1

    print(f"  implicit_reasoning: {written} written, {skipped} skipped")


def process_act_loc_view(rows: list[dict], output_dir: Path, subtask: str, overwrite: bool):
    """
    Handle tasks: action, location, view.
    image_path is a plain filename: e.g. "00000.png"
    Output:
      tasks/act_loc_view/{subtask}/ori_images/{image_name}
      tasks/act_loc_view/{subtask}/edited_images/{image_name}
      tasks/act_loc_view/{subtask}/masks/{image_name}  (likely no mask)
      tasks/act_loc_view/{subtask}/instructions.jsonl
    """
    task_dir = output_dir / "act_loc_view" / subtask
    inst_path = task_dir / "instructions.jsonl"

    existing: set[str] = set()
    if inst_path.exists() and not overwrite:
        try:
            with open(inst_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.add(json.loads(line).get("image_name", ""))
        except Exception as e:
            print(f"  Warning: could not read existing {inst_path}: {e}")

    if overwrite:
        task_dir.mkdir(parents=True, exist_ok=True)
        inst_path.write_text("", encoding="utf-8")

    skipped = written = 0
    for row in tqdm(rows, desc=f"  {subtask}", unit="img", leave=False):
        image_name: str = row["image_path"]

        if image_name in existing and not overwrite:
            skipped += 1
            continue

        save_image(row.get("input_image"), task_dir / "ori_images" / image_name, overwrite)
        save_image(row.get("edited_image"), task_dir / "edited_images" / image_name, overwrite)
        if row.get("mask") is not None:
            save_image(row["mask"], task_dir / "masks" / image_name, overwrite)

        if image_name not in existing or overwrite:
            append_jsonl(inst_path, {
                "image_name": image_name,
                "instruction": row.get("instruction", ""),
            })
            existing.add(image_name)

        written += 1

    print(f"  {subtask}: {written} written, {skipped} skipped")


def process_multi_object(rows: list[dict], output_dir: Path, task: str, overwrite: bool):
    """
    Handle tasks: multi_object_add, multi_object_remove.
    image_path format: "02b72b8d_00002.png" (plain, no slash)
    caption is pipe-separated: "caption1|caption2"
    Output:
      tasks/multi_turn_editing/{task}/ori_images/{image_name}
      tasks/multi_turn_editing/{task}/edited_images/{image_name}
      tasks/multi_turn_editing/{task}/masks/{image_name}
      tasks/multi_turn_editing/{task}.jsonl
    """
    mt_dir = output_dir / "multi_turn_editing"
    task_dir = mt_dir / task
    jsonl_path = mt_dir / f"{task}.jsonl"

    existing: set[str] = set()
    if jsonl_path.exists() and not overwrite:
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.add(json.loads(line).get("image_name", ""))
        except Exception as e:
            print(f"  Warning: could not read existing {jsonl_path}: {e}")

    if overwrite:
        mt_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

    # Determine instruction and caption field names based on task
    ins_key = "add_ins" if task == "multi_object_add" else "remove_ins"

    skipped = written = 0
    for row in tqdm(rows, desc=f"  {task}", unit="img", leave=False):
        image_name: str = row["image_path"]

        if image_name in existing and not overwrite:
            skipped += 1
            continue

        save_image(row.get("input_image"), task_dir / "ori_images" / image_name, overwrite)
        save_image(row.get("edited_image"), task_dir / "edited_images" / image_name, overwrite)
        if row.get("mask") is not None:
            save_image(row["mask"], task_dir / "masks" / image_name, overwrite)

        if image_name not in existing or overwrite:
            # Decode pipe-separated caption
            raw_caption = row.get("caption", "")
            parts = raw_caption.split("|") if raw_caption else ["", ""]
            caption1 = parts[0] if len(parts) > 0 else ""
            caption2 = parts[1] if len(parts) > 1 else ""

            append_jsonl(jsonl_path, {
                "image_name": image_name,
                ins_key: row.get("instruction", ""),
                "caption1": caption1,
                "caption2": caption2,
            })
            existing.add(image_name)

        written += 1

    print(f"  {task}: {written} written, {skipped} skipped")


# ─── Multi-turn split processors ─────────────────────────────────────────────

def process_multi_turn(rows_by_task: dict[str, list[dict]], output_dir: Path, overwrite: bool):
    """
    Handles multi_turn_add and multi_turn_remove from the multi_turn split.

    Each HF row has image_path like "turn1_add/{name}" or "turn2_add/{name}".
    We need to group by base name and reconstruct per-turn JSONL files.

    Output:
      tasks/multi_turn_editing/turn1_add/ori_images/{name}
      tasks/multi_turn_editing/turn1_add/edited_images/{name}
      tasks/multi_turn_editing/turn1_add/masks/{name}
      tasks/multi_turn_editing/turn2_add/  (same)
      tasks/multi_turn_editing/turn1_remove/ (same)
      tasks/multi_turn_editing/turn2_remove/ (same)
      tasks/multi_turn_editing/multi_turn_add.jsonl
      tasks/multi_turn_editing/multi_turn_remove.jsonl
    """
    mt_dir = output_dir / "multi_turn_editing"
    mt_dir.mkdir(parents=True, exist_ok=True)

    # ── multi_turn_add ──────────────────────────────────────────────────────
    add_rows = rows_by_task.get("multi_turn_add", [])
    if add_rows:
        add_jsonl_path = mt_dir / "multi_turn_add.jsonl"

        existing_add: set[str] = set()
        if add_jsonl_path.exists() and not overwrite:
            try:
                with open(add_jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_add.add(json.loads(line).get("image_name", ""))
            except Exception as e:
                print(f"  Warning: could not read existing {add_jsonl_path}: {e}")

        if overwrite:
            add_jsonl_path.write_text("", encoding="utf-8")

        # Group rows by prefix: turn1_add vs turn2_add
        # image_path = "turn1_add/{name}" → strip prefix to get base name
        turns_add: dict[str, dict] = defaultdict(dict)  # base_name -> {turn1: row, turn2: row}
        for row in add_rows:
            img_path: str = row["image_path"]
            if img_path.startswith("turn1_add/"):
                name = img_path[len("turn1_add/"):]
                turns_add[name]["turn1"] = row
            elif img_path.startswith("turn2_add/"):
                name = img_path[len("turn2_add/"):]
                turns_add[name]["turn2"] = row

        skipped = written = 0
        for name, turn_data in tqdm(turns_add.items(), desc="  multi_turn_add", unit="pair", leave=False):
            if name in existing_add and not overwrite:
                skipped += 1
                continue

            # Save images for each turn
            for prefix, turn_key in [("turn1_add", "turn1"), ("turn2_add", "turn2")]:
                row = turn_data.get(turn_key)
                if row is None:
                    continue
                turn_dir = mt_dir / prefix
                save_image(row.get("input_image"), turn_dir / "ori_images" / name, overwrite)
                save_image(row.get("edited_image"), turn_dir / "edited_images" / name, overwrite)
                if row.get("mask") is not None:
                    save_image(row["mask"], turn_dir / "masks" / name, overwrite)

            if name not in existing_add or overwrite:
                t1 = turn_data.get("turn1", {})
                t2 = turn_data.get("turn2", {})
                append_jsonl(add_jsonl_path, {
                    "image_name": name,
                    "turn1_add_ins": t1.get("instruction", ""),
                    "turn2_add_ins": t2.get("instruction", ""),
                    "turn1_caption": t1.get("caption", ""),
                    "turn2_caption": t2.get("caption", ""),
                })
                existing_add.add(name)

            written += 1

        print(f"  multi_turn_add: {written} pairs written, {skipped} skipped")

    # ── multi_turn_remove ───────────────────────────────────────────────────
    remove_rows = rows_by_task.get("multi_turn_remove", [])
    if remove_rows:
        remove_jsonl_path = mt_dir / "multi_turn_remove.jsonl"

        existing_remove: set[str] = set()
        if remove_jsonl_path.exists() and not overwrite:
            try:
                with open(remove_jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_remove.add(json.loads(line).get("image_name", ""))
            except Exception as e:
                print(f"  Warning: could not read existing {remove_jsonl_path}: {e}")

        if overwrite:
            remove_jsonl_path.write_text("", encoding="utf-8")

        turns_remove: dict[str, dict] = defaultdict(dict)
        for row in remove_rows:
            img_path: str = row["image_path"]
            if img_path.startswith("turn1_remove/"):
                name = img_path[len("turn1_remove/"):]
                turns_remove[name]["turn1"] = row
            elif img_path.startswith("turn2_remove/"):
                name = img_path[len("turn2_remove/"):]
                turns_remove[name]["turn2"] = row

        skipped = written = 0
        for name, turn_data in tqdm(turns_remove.items(), desc="  multi_turn_remove", unit="pair", leave=False):
            if name in existing_remove and not overwrite:
                skipped += 1
                continue

            for prefix, turn_key in [("turn1_remove", "turn1"), ("turn2_remove", "turn2")]:
                row = turn_data.get(turn_key)
                if row is None:
                    continue
                turn_dir = mt_dir / prefix
                save_image(row.get("input_image"), turn_dir / "ori_images" / name, overwrite)
                save_image(row.get("edited_image"), turn_dir / "edited_images" / name, overwrite)
                if row.get("mask") is not None:
                    save_image(row["mask"], turn_dir / "masks" / name, overwrite)

            if name not in existing_remove or overwrite:
                t1 = turn_data.get("turn1", {})
                t2 = turn_data.get("turn2", {})
                append_jsonl(remove_jsonl_path, {
                    "image_name": name,
                    "turn1_remove_ins": t1.get("instruction", ""),
                    "turn2_remove_ins": t2.get("instruction", ""),
                })
                existing_remove.add(name)

            written += 1

        print(f"  multi_turn_remove: {written} pairs written, {skipped} skipped")


# ─── Train split dispatcher ───────────────────────────────────────────────────

def process_train_split(dataset, output_dir: Path, overwrite: bool):
    """Route each task in the train split to the appropriate processor."""
    print("\nProcessing train split...")

    # Group rows by task
    tasks_map: dict[str, list[dict]] = defaultdict(list)
    print("  Grouping rows by task...")
    for row in tqdm(dataset["train"], desc="  indexing train", unit="row", leave=False):
        tasks_map[row["task"]].append(row)

    for task_name, count in sorted((t, len(r)) for t, r in tasks_map.items()):
        print(f"  Found task '{task_name}': {count} rows")

    # Local editing
    for task in ("add", "remove", "replace"):
        if task in tasks_map:
            process_local_editing(tasks_map[task], output_dir, task, overwrite)
        else:
            print(f"  Warning: task '{task}' not found in train split")

    # Implicit reasoning
    if "implicit_reasoning" in tasks_map:
        process_implicit_reasoning(tasks_map["implicit_reasoning"], output_dir, overwrite)
    else:
        print("  Warning: 'implicit_reasoning' not found in train split")

    # Action / location / view
    for subtask in ("action", "location", "view"):
        if subtask in tasks_map:
            process_act_loc_view(tasks_map[subtask], output_dir, subtask, overwrite)
        else:
            print(f"  Warning: task '{subtask}' not found in train split")

    # Multi-object (still in train split)
    for task in ("multi_object_add", "multi_object_remove"):
        if task in tasks_map:
            process_multi_object(tasks_map[task], output_dir, task, overwrite)
        else:
            print(f"  Warning: task '{task}' not found in train split")


# ─── Multi-turn split dispatcher ─────────────────────────────────────────────

def process_multi_turn_split(dataset, output_dir: Path, overwrite: bool):
    """Route each task in the multi_turn split to the appropriate processor."""
    print("\nProcessing multi_turn split...")

    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    print("  Grouping rows by task...")
    for row in tqdm(dataset["multi_turn"], desc="  indexing multi_turn", unit="row", leave=False):
        rows_by_task[row["task"]].append(row)

    for task_name, count in sorted((t, len(r)) for t, r in rows_by_task.items()):
        print(f"  Found task '{task_name}': {count} rows")

    process_multi_turn(rows_by_task, output_dir, overwrite)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download BohanJia/CompBench from HuggingFace and restructure into local layout."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "tasks",
        help="Root output directory (default: ./tasks next to this script)",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "multi_turn"],
        default="all",
        help="Which split(s) to download: train, multi_turn, or all (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip existing files for resumability)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    print(f"Output directory : {output_dir}")
    print(f"Split            : {args.split}")
    print(f"Overwrite        : {args.overwrite}")

    # Load (or download + cache) the dataset
    cache_dir = Path(__file__).parent / "hf_cache"
    try:
        dataset = load_dataset_cached(cache_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    available_splits = list(dataset.keys())
    print(f"\nAvailable splits : {available_splits}")
    for split_name in available_splits:
        print(f"  {split_name}: {len(dataset[split_name])} rows")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch
    if args.split in ("all", "train"):
        if "train" in dataset:
            process_train_split(dataset, output_dir, args.overwrite)
        else:
            print("Warning: 'train' split not found in dataset.")

    if args.split in ("all", "multi_turn"):
        if "multi_turn" in dataset:
            process_multi_turn_split(dataset, output_dir, args.overwrite)
        else:
            print("Warning: 'multi_turn' split not found in dataset.")

    print("\nDone. Local directory structure is ready for eval_all.py.")


if __name__ == "__main__":
    main()
