import base64
import imghdr
import json
import mimetypes
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


DATASET_ROOT = Path(__file__).parent / "dataset"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _pick_split(dataset_dict: Any, preferred: Iterable[str]) -> Any:
    if not hasattr(dataset_dict, "keys"):
        return dataset_dict
    for name in preferred:
        if name in dataset_dict:
            return dataset_dict[name]
    first_key = next(iter(dataset_dict.keys()))
    return dataset_dict[first_key]


def _image_to_data_uri(image_value: Any) -> str:
    if not image_value:
        return ""
    if isinstance(image_value, str):
        if image_value.startswith("data:"):
            return image_value
        image_path = Path(image_value)
        if not image_path.exists():
            return ""
        data = image_path.read_bytes()
        mime_type = mimetypes.guess_type(str(image_path))[0]
        if not mime_type:
            fmt = imghdr.what(None, data)
            if fmt:
                mime_type = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
        if not mime_type:
            mime_type = "image/png"
        return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
    if isinstance(image_value, dict):
        data = image_value.get("bytes")
        path = image_value.get("path")
        if data is None and path:
            try:
                data = Path(path).read_bytes()
            except OSError:
                data = None
        if data is None:
            return ""
        mime_type = mimetypes.guess_type(path)[0] if path else None
        if not mime_type:
            fmt = imghdr.what(None, data)
            if fmt:
                mime_type = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
        if not mime_type:
            mime_type = "image/png"
        return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
    try:
        from PIL import Image  # type: ignore[import-not-found]

        if isinstance(image_value, Image.Image):
            buffer = BytesIO()
            fmt = image_value.format or "PNG"
            image_value.save(buffer, format=fmt)
            data = buffer.getvalue()
            mime_type = Image.MIME.get(fmt, f"image/{fmt.lower()}")
            return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
    except Exception:
        pass
    return ""


def _xor_decrypt(data: bytes, key: str) -> bytes:
    key_bytes = key.encode("utf-8")
    if not key_bytes:
        return data
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


def _decrypt_xbench_field(value: Any, key: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if not key:
        return value
    try:
        decoded = base64.b64decode(value)
    except Exception:
        return value
    try:
        return _xor_decrypt(decoded, key).decode("utf-8", errors="replace")
    except Exception:
        return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, data: Any) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _prepare_hle_dataset(dataset_path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    dataset_dict = load_dataset("cais/hle")
    dataset = _pick_split(dataset_dict, ["test", "validation", "val", "dev", "train"])
    items: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        question = example.get("question") or example.get("prompt") or ""
        image_uri = _image_to_data_uri(example.get("image"))
        raw_subject = example.get("raw_subject") or example.get("subject") or ""
        category = example.get("category") or raw_subject or "Other"
        item = {
            "id": str(example.get("id") or example.get("question_id") or example.get("qid") or idx),
            "question": question,
            "image": image_uri,
            "answer": example.get("answer") or example.get("target") or example.get("final_answer") or "",
            "answer_type": example.get("answer_type") or example.get("type") or "",
            "author_name": example.get("author_name") or "",
            "rationale": example.get("rationale") or example.get("explanation") or "",
            "raw_subject": raw_subject,
            "category": category,
            "canary": example.get("canary") or "",
            "image_preview_exists": bool(image_uri),
            "rationale_image_exists": bool(example.get("rationale_image") or example.get("rationale_image_exists")),
        }
        items.append(item)
    category_allocation: Dict[str, int] = {}
    for item in items:
        category_allocation[item["category"]] = category_allocation.get(item["category"], 0) + 1
    payload = {
        "metadata": {
            "total_returned": len(items),
            "category_allocation": category_allocation,
        },
        "data": items,
    }
    _write_json(dataset_path, payload)


def _prepare_xbench_dataset(dataset_name: str, dataset_path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    dataset_dict = load_dataset(dataset_name)
    dataset = _pick_split(dataset_dict, ["train", "test", "validation", "val", "dev"])
    items: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        key = example.get("canary") or ""
        prompt = _decrypt_xbench_field(example.get("prompt"), key)
        answer = _decrypt_xbench_field(example.get("answer"), key)
        metadata: Dict[str, Any] = {}
        for k, v in example.items():
            if k in {"prompt", "answer", "canary", "task_id", "id"}:
                continue
            if k == "metadata" and isinstance(v, dict):
                metadata.update(_json_safe(v))
            else:
                metadata[k] = _json_safe(v)
        item = {
            "task_id": example.get("id") or example.get("task_id") or (idx + 1),
            "task_question": prompt,
            "ground_truth": answer,
            "metadata": metadata,
        }
        items.append(item)
    _write_json(dataset_path, items)


def _prepare_deepsearchqa_dataset(dataset_path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    dataset_dict = load_dataset("google/deepsearchqa")
    dataset = _pick_split(dataset_dict, ["train", "test", "validation", "val", "dev"])
    items: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        item = {
            "example_id": str(example.get("example_id") or example.get("id") or idx),
            "problem": example.get("problem") or example.get("question") or example.get("prompt") or "",
            "problem_category": example.get("problem_category") or example.get("category") or example.get("subject") or "",
            "answer": example.get("answer") or example.get("target") or "",
            "answer_type": example.get("answer_type") or example.get("type") or "",
        }
        items.append(item)
    _write_json(dataset_path, items)


def _prepare_finsearchcomp_dataset(dataset_path: Path) -> None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    dataset_dict = load_dataset("ByteSeedXpert/FinSearchComp")
    dataset = _pick_split(dataset_dict, ["train", "test", "validation", "val", "dev"])
    items: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        prompt_id = example.get("prompt_id") or example.get("id") or str(idx)
        task_type = example.get("task_type") or example.get("type")
        if not task_type and isinstance(prompt_id, str) and prompt_id.startswith("(") and ")" in prompt_id:
            task_type = prompt_id.split(")")[0].lstrip("(")
        if task_type not in {"T2", "T3"}:
            continue
        item = {
            "prompt_id": prompt_id,
            "prompt": example.get("prompt") or example.get("question") or "",
            "response_reference": example.get("response_reference") or example.get("answer") or "",
            "judge_prompt_template": example.get("judge_prompt_template") or "",
            "judge_system_prompt": example.get("judge_system_prompt") or "",
            "label": example.get("label") or "",
        }
        items.append(item)
    _write_json(dataset_path, items)


def load_hle_dataset(batch_size: int) -> Iterator[dict]:
    dataset_path = DATASET_ROOT / "HLE" / "hle.json"
    if not dataset_path.exists():
        _prepare_hle_dataset(dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data["data"]
    items = []
    for item in data:
        if (question := item.get("question", None)) is None:
            continue
        if image := item.get("image", None):
            images_dir = Path(__file__).parent / "dataset" / "HLE" / "images"
            os.makedirs(images_dir, exist_ok=True)
            matching_files = list(images_dir.glob(f"{item['id']}.*"))
            if matching_files:
                image_path = matching_files[0] 
                question = f"{question} [Image path: {image_path}]"
                items.append((item["id"], question))
                continue
            match = re.match(r"data:(.*);base64,(.*)", image, re.DOTALL)
            if not match:
                print(f"Parse error: {item['id']}")
            mime_type = match.group(1).strip()
            base64_string = match.group(2).strip()
            binary_data = base64.b64decode(base64_string)
            image_path = f"dataset/HLE/images/{item['id']}.{mime_type.split('/')[-1]}"
            image_path = Path(__file__).parent / image_path
            with open(image_path, "wb") as f:
                f.write(binary_data)
            question = f"{question} [Image path: {image_path}]"
        items.append((item["id"], question))

    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        data_items = [{"task_id": task_id, "query": question} for task_id, question in batch_items]
        yield {"data_items": data_items}


def load_xbench_dataset(batch_size: int, type: str = "deepsearch") -> Iterator[dict]:
    dataset_paths = []
    if type == "deepsearch":
        dataset_paths.append(DATASET_ROOT / "XBENCH" / "DeepSearch-2510.json")
    elif type == "scienceqa":
        dataset_paths.append(DATASET_ROOT / "XBENCH" / "ScienceQA.json")
    elif type == "all":
        dataset_paths.append(DATASET_ROOT / "XBENCH" / "DeepSearch-2510.json")
        dataset_paths.append(DATASET_ROOT / "XBENCH" / "ScienceQA.json")
    else:
        raise ValueError(f"Unknown XBENCH dataset type: {type}")
    data = []
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            dataset_name = "xbench/DeepSearch-2510" if "DeepSearch-2510" in dataset_path.name else "xbench/ScienceQA"
            _prepare_xbench_dataset(dataset_name, dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
        data.extend(temp_data)
    for i in range(0, len(data), batch_size):
        batch_items = data[i : i + batch_size]
        data_items = [{"task_id": item["task_id"], "query": item["task_question"]} for item in batch_items]
        yield {"data_items": data_items}


def load_deepsearchqa_dataset(batch_size: int) -> Iterator[dict]:
    dataset_path = DATASET_ROOT / "DEEPSEARCHQA" / "DSQA-full.json"
    if not dataset_path.exists():
        _prepare_deepsearchqa_dataset(dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i in range(0, len(data), batch_size):
        batch_items = data[i : i + batch_size]
        data_items = [{"task_id": item["example_id"], "query": item["problem"]} for item in batch_items]
        yield {"data_items": data_items}


def load_finsearchcomp_dataset(batch_size: int) -> Iterator[dict]:
    dataset_path = DATASET_ROOT / "FinSearchComp" / "t2_t3_questions.json"
    if not dataset_path.exists():
        _prepare_finsearchcomp_dataset(dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i in range(0, len(data), batch_size):
        batch_items = data[i : i + batch_size]
        data_items = [{"task_id": item["prompt_id"], "query": item["prompt"]} for item in batch_items]
        yield {"data_items": data_items}


def load_dataset(dataset: str, batch_size: int) -> Iterator[dict]:
    """
    Load dataset based on the dataset name.

    Args:
        dataset: Dataset name ('GAIA-valid', 'GAIA-test', 'HLE', etc.)
        batch_size: Number of queries per batch

    Returns:
        Iterator that yields dictionaries with 'data_items' key, where each data_item
        contains 'task_id' and 'query' fields
    """
    if dataset == "HLE":
        return load_hle_dataset(batch_size)
    elif dataset == "XBENCH-deepsearch":
        return load_xbench_dataset(batch_size, type="deepsearch")
    elif dataset == "XBENCH-scienceqa":
        return load_xbench_dataset(batch_size, type="scienceqa")
    elif dataset == "XBENCH-all":
        return load_xbench_dataset(batch_size, type="all")
    elif dataset == "DEEPSEARCHQA":
        return load_deepsearchqa_dataset(batch_size)
    elif dataset == "FINSEARCHCOMP":
        return load_finsearchcomp_dataset(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
