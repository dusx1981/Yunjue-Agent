#!/usr/bin/env python3
"""Unified evaluation entrypoint for DeepSearchQA, XBench, HLE, and others."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import collections
import concurrent.futures
import copy
import json
import logging
import math
import os
import random
import re
import textwrap
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import httpx
import yaml
from openai import OpenAI
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MessageType:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: str
    content: str


class BenchmarkType(Enum):
    HLE = "hle"
    DSQA = "dsqa"
    FINSEARCHCOMP = "finsearchcomp"


@dataclass
class Prediction:
    id: str
    question: str
    prediction: Optional[str]
    raw_prediction_json: Optional[Dict[str, Any]] = None
    run_success: Optional[bool] = None


@dataclass
class GroundTruth:
    id: str
    question: str
    answer: str
    answer_type: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OpenAIChatModel:
    """Thin wrapper around the official OpenAI client with a LangChain-like interface."""

    def __init__(self, client: OpenAI, model: str, **invoke_params: Any) -> None:
        self._client = client
        self._model = model
        self._invoke_params = invoke_params

    def invoke(
        self, messages: Iterable[Any], response_format: Optional[Dict[str, str]] = None
    ) -> SimpleNamespace:
        formatted_messages: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, Message):
                formatted_messages.append({"role": message.role, "content": message.content})
            elif isinstance(message, dict):
                formatted_messages.append(message)
            else:
                content = str(message.content if hasattr(message, "content") else message)
                role = "user"
                if hasattr(message, "role"):
                    role = message.role
                formatted_messages.append({"role": role, "content": content})

        call_params = dict(self._invoke_params)
        if response_format:
            call_params["response_format"] = response_format

        response = self._client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            **call_params,
        )

        if not response.choices:
            raise RuntimeError("OpenAI response contained no choices")

        choice = response.choices[0]
        message_content = getattr(choice.message, "content", "") or ""
        return SimpleNamespace(content=message_content)


def load_predictions(path: Path, benchmark_type: BenchmarkType) -> List[Prediction]:
    predictions: List[Prediction] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                pred = payload.get("prediction")
                if isinstance(pred, str):
                    try:
                        pred_dict = json.loads(pred)
                        prediction = pred_dict.get("final_answer")
                    except json.JSONDecodeError:
                        match = re.search(r'"final_answer"\s*:\s*"([^"]*)"', pred)
                        prediction = match.group(1) if match else pred
                else:
                    prediction = pred.get("final_answer") if pred else None

                if benchmark_type == BenchmarkType.HLE:
                    pred_id = payload.get("question_id") or str(payload.get("question_index", line_no))
                elif benchmark_type == BenchmarkType.DSQA:
                    pred_id = (
                        payload.get("example_id")
                        or payload.get("question_id")
                        or payload.get("question_index")
                    )
                    if pred_id is None:
                        pred_id = line_no - 1
                    pred_id = str(pred_id)
                elif benchmark_type == BenchmarkType.FINSEARCHCOMP:
                    pred_id = payload.get("question_index") or payload.get("question_id") or str(line_no)
                else:
                    pred_id = str(line_no)

                predictions.append(
                    Prediction(
                        id=pred_id,
                        question=str(payload.get("question") or payload.get("problem", "")),
                        prediction=prediction,
                        raw_prediction_json=payload.get("raw_prediction_json"),
                        run_success=payload.get("run_success"),
                    )
                )
            except Exception as exc:
                print(f"Warning: Failed to parse prediction on line {line_no}: {exc}")

    return predictions


def load_ground_truth(path: Path, benchmark_type: BenchmarkType) -> Tuple[Dict[str, GroundTruth], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if benchmark_type == BenchmarkType.HLE:
        items = items["data"]
    ground_truth_map: Dict[str, GroundTruth] = {}
    order: List[str] = []

    for idx, item in enumerate(items):
        if benchmark_type == BenchmarkType.HLE:
            gt_id = item.get("id") or str(idx)
            question = item.get("question")
            answer = item.get("answer")
            answer_type = item.get("answer_type")
            category = None
        elif benchmark_type == BenchmarkType.DSQA:
            gt_id = str(item.get("example_id", idx))
            question = item.get("problem")
            answer = item.get("answer")
            answer_type = item.get("answer_type", "Single Answer")
            category = item.get("problem_category")
        elif benchmark_type == BenchmarkType.FINSEARCHCOMP:
            gt_id = item.get("prompt_id") or str(idx)
            question = item.get("prompt")
            answer = item.get("response_reference")
            answer_type = None
            category = item.get("label")
        else:
            continue

        if not isinstance(question, str) or not isinstance(answer, str):
            continue

        ground_truth_map[gt_id] = GroundTruth(
            id=gt_id,
            question=question,
            answer=answer,
            answer_type=answer_type,
            category=category,
            metadata=item,
        )
        order.append(gt_id)

    return ground_truth_map, order


def normalise_answer(value: Optional[str]) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def normalise_set_answer(value: Optional[str]) -> set:
    if value is None:
        return set()
    elements = [elem.strip().lower() for elem in value.split(",")]
    return set(elem for elem in elements if elem)


def build_eval_model(
    config_path: Path, model_override: Optional[str] = None
) -> Tuple[OpenAIChatModel, Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    eval_conf = conf.get("EVAL_MODEL")
    if not isinstance(eval_conf, dict) or not eval_conf:
        raise ValueError("EVAL_MODEL configuration is missing or invalid in conf.yaml")

    llm_conf = dict(eval_conf)
    llm_conf.pop("token_limit", None)
    verify_ssl = llm_conf.pop("verify_ssl", True)

    api_key = llm_conf.pop("api_key", None)
    model_name = llm_conf.pop("model", None)
    if model_override:
        model_name = model_override
    base_url = llm_conf.pop("base_url", None)
    max_retries = llm_conf.pop("max_retries", 3)

    if not api_key:
        raise ValueError("EVAL_MODEL.api_key is required for OpenAI client configuration")
    if not model_name:
        raise ValueError("EVAL_MODEL.model is required for OpenAI client configuration")

    http_client = None
    if not verify_ssl:
        http_client = httpx.Client(verify=False)

    client_kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": max_retries}
    if base_url:
        client_kwargs["base_url"] = base_url
    if http_client:
        client_kwargs["http_client"] = http_client

    client = OpenAI(**client_kwargs)
    model = OpenAIChatModel(client, model_name, **llm_conf)

    if http_client:
        atexit.register(http_client.close)

    if model_override:
        eval_conf = dict(eval_conf)
        eval_conf["model"] = model_name
    return model, eval_conf


def judge_answer_finsearchcomp(
    llm: OpenAIChatModel,
    question: str,
    ground_truth: str,
    prediction: str,
    judge_system_prompt: str,
    judge_prompt_template: str,
) -> Dict[str, Any]:
    user_prompt = judge_prompt_template.format(
        prompt=question, response_reference=ground_truth, response=prediction or "<empty>"
    )
    messages = [Message(MessageType.SYSTEM, judge_system_prompt), Message(MessageType.USER, user_prompt)]
    response = llm.invoke(messages, response_format={"type": "json_object"})
    text = response.content
    if isinstance(text, list):
        text = "".join(str(chunk) for chunk in text)

    text = text.strip()
    try:
        json_match = re.search(r'\{\s*"answer_score"\s*:\s*(\d+)\s*\}', text)
        if json_match:
            score = int(json_match.group(1))
            explanation_match = re.search(r"【评分依据】[：:](.*?)(?=【JSON】|$)", text, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else text
            return {"correct": score == 1, "explanation": explanation, "answer_score": score}
    except Exception as exc:
        print(f"Warning: Failed to parse FinSearchComp response: {exc}")

    print("Warning: Failed to parse FinSearchComp response")
    print(f"Response text: {text[:500]}...")
    return {"correct": False, "explanation": "Failed to parse LLM response", "answer_score": 0}


def judge_answer(
    llm: OpenAIChatModel,
    question: str,
    ground_truth: str,
    prediction: str,
    answer_type: Optional[str] = None,
) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert judge for benchmark answers. "
        "You must respond with valid JSON only, using this exact schema:\n"
        "{\n"
        '  "correct": <boolean>,\n'
        '  "explanation": "<string>"\n'
        "}\n"
        "Do not include any text outside the JSON object."
    )

    answer_type_instruction = ""
    if answer_type == "Set Answer":
        answer_type_instruction = (
            "\n\nNote: This is a 'Set Answer' question. The prediction should contain all elements "
            "from the ground truth (order doesn't matter), possibly separated by commas. "
            "The prediction is correct if it contains all required elements, even if formatted differently."
        )

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Ground truth answer:\n"
        f"{ground_truth}\n\n"
        "Model prediction:\n"
        f"{prediction or '<empty>'}\n\n"
    )

    if answer_type:
        user_prompt += f"Answer type: {answer_type}{answer_type_instruction}\n\n"

    user_prompt += "Decide if the model prediction satisfies the ground truth. Respond with valid JSON matching the schema."

    messages = [Message(MessageType.SYSTEM, system_prompt), Message(MessageType.USER, user_prompt)]
    response = llm.invoke(messages, response_format={"type": "json_object"})
    text = response.content
    if isinstance(text, list):
        text = "".join(str(chunk) for chunk in text)

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        print(f"Warning: Failed to parse LLM response as JSON: {exc}")
        print(f"Response text: {text[:200]}...")
        return {"correct": False, "explanation": "Failed to parse LLM response"}


def evaluate_predictions(
    llm: OpenAIChatModel,
    predictions: Iterable[Prediction],
    ground_truth_map: Dict[str, GroundTruth],
    benchmark_type: Optional[BenchmarkType] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    results: List[Dict[str, Any]] = []
    evaluated = 0
    correct = 0
    to_judge = []

    for pred in predictions:
        truth = ground_truth_map.get(pred.id)
        if not truth:
            results.append(
                {
                    "id": pred.id,
                    "question": pred.question,
                    "prediction": pred.prediction,
                    "status": "missing_ground_truth",
                }
            )
            continue

        predicted_value = pred.prediction or ""

        if benchmark_type != BenchmarkType.FINSEARCHCOMP:
            auto_matched = False
            if truth.answer_type == "Set Answer":
                pred_set = normalise_set_answer(predicted_value)
                truth_set = normalise_set_answer(truth.answer)
                if pred_set == truth_set:
                    auto_matched = True
            else:
                if normalise_answer(predicted_value) == normalise_answer(truth.answer):
                    auto_matched = True

            if auto_matched:
                results.append(
                    {
                        "id": pred.id,
                        "question": truth.question,
                        "category": truth.category,
                        "answer_type": truth.answer_type,
                        "ground_truth": truth.answer,
                        "prediction": predicted_value,
                        "status": "auto_match",
                        "correct": True,
                        "explanation": "Prediction matches ground truth after normalisation.",
                    }
                )
                evaluated += 1
                correct += 1
                continue

        to_judge.append((pred, truth, predicted_value))

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        if benchmark_type == BenchmarkType.FINSEARCHCOMP:
            futures_to_data = {
                executor.submit(
                    judge_answer_finsearchcomp,
                    llm,
                    truth.question,
                    truth.answer,
                    predicted_value,
                    truth.metadata.get("judge_system_prompt", ""),
                    truth.metadata.get("judge_prompt_template", ""),
                ): (pred, truth, predicted_value)
                for pred, truth, predicted_value in to_judge
            }
        else:
            futures_to_data = {
                executor.submit(
                    judge_answer, llm, truth.question, truth.answer, predicted_value, truth.answer_type
                ): (pred, truth, predicted_value)
                for pred, truth, predicted_value in to_judge
            }
        for future in tqdm(
            concurrent.futures.as_completed(futures_to_data), total=len(to_judge), desc="Evaluating"
        ):
            pred, truth, predicted_value = futures_to_data[future]
            try:
                judgement = future.result()
                is_correct = bool(judgement.get("correct"))
                explanation = judgement.get("explanation", "")
            except Exception as exc:
                print(f"Warning: Failed to judge prediction for {pred.id}: {exc}")
                is_correct = False
                explanation = f"Evaluation error: {exc}"
                judgement = {}

            results.append(
                {
                    "id": pred.id,
                    "question": truth.question,
                    "category": truth.category,
                    "answer_type": truth.answer_type,
                    "ground_truth": truth.answer,
                    "prediction": predicted_value,
                    "status": "llm_judged",
                    "correct": is_correct,
                    "explanation": explanation,
                    "raw_judgement": judgement,
                }
            )
            evaluated += 1
            if is_correct:
                correct += 1

    return results, evaluated, correct


def sort_results_by_ground_truth(
    results: List[Dict[str, Any]], ground_truth_order: List[str]
) -> List[Dict[str, Any]]:
    order_map = {gt_id: idx for idx, gt_id in enumerate(ground_truth_order)}
    indexed_results = list(enumerate(results))

    def sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        original_idx, result = item
        gt_idx = order_map.get(result.get("id"))
        if gt_idx is None:
            gt_idx = len(order_map) + original_idx
        return gt_idx, original_idx

    indexed_results.sort(key=sort_key)
    return [result for _, result in indexed_results]


def compute_category_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    category_stats: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if "correct" not in result:
            continue
        category = result.get("category") or "Unknown"
        if category not in category_stats:
            category_stats[category] = {"total": 0, "correct": 0, "accuracy": 0.0}

        category_stats[category]["total"] += 1
        if result["correct"]:
            category_stats[category]["correct"] += 1

    for category in category_stats:
        total = category_stats[category]["total"]
        correct = category_stats[category]["correct"]
        category_stats[category]["accuracy"] = (correct / total) if total > 0 else 0.0

    return category_stats


def _load_eval_conf(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    eval_conf = conf.get("EVAL_MODEL")
    if not isinstance(eval_conf, dict) or not eval_conf:
        raise ValueError("EVAL_MODEL configuration is missing or invalid in conf.yaml")
    return dict(eval_conf)


def _resolve_eval_model(benchmark: str, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    if benchmark in {"deepsearchqa", "dsqa"}:
        return "gemini-2.5-flash"
    if benchmark == "xbench":
        return "gemini-2.0-flash"
    if benchmark == "hle":
        return "o3-mini-2025-01-31"
    return None


def _require_path(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


###################################################
# DeepSearchQA evaluator (LLM-graded)
###################################################

DEEPSEARCH_QA_PROMPT = textwrap.dedent(
    """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.
"""
)

GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


@dataclass
class ItemRating:
    original_index: int | None = None
    example_id: str = ""
    query: str = ""
    response: str = ""
    category_type: str | None = None
    expected_correct_answer: str | None = None

    answer_correctness_explanation: str | None = None
    expected_correct_answer_list: list[str] | None = None
    response_wrong_answers_list: list[str] | None = None
    grader_ratings_list: list[bool] | None = None

    empty_model_response: bool = False
    empty_auto_rater_response: bool = False
    invalid_auto_rater_response: bool = False
    rating_response: str = ""
    rating_prompt: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectRating:
    num_total_ratings: int = 0
    num_empty_model_response: int = 0
    num_invalid_auto_rater_response: int = 0
    num_empty_auto_rater_response: int = 0
    num_valid_ratings: int = 0
    num_answer_correctness_evaluated: int = 0

    pct_w_ci_all_answers_correct: str = ""
    pct_w_ci_fully_incorrect_items: str = ""
    pct_w_ci_correct_with_excessive_answers: str = ""

    pct_empty_model_response: float = 0.0
    pct_invalid_auto_rater_response: float = 0.0
    pct_empty_auto_rater_response: float = 0.0

    precision: str = ""
    recall: str = ""
    f1_score: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json_response(ori_json_response: str) -> Any:
    try:
        json_str = ori_json_response.strip()
        start_marker = "```json"
        start_idx = json_str.find(start_marker)
        if start_idx != -1:
            json_str = json_str[start_idx + len(start_marker) :].strip()
            end_marker = "```"
            end_idx = json_str.rfind(end_marker)
            if end_idx != -1:
                json_str = json_str[:end_idx].strip()
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.info("JSON decode error: %s for response: %s", exc, ori_json_response)
        return None


def _get_answer_correctness_details(json_response: Any) -> dict[str, bool] | None:
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
        if isinstance(details, dict):
            all_keys_are_strings = all(isinstance(key, str) for key in details.keys())
            all_values_are_booleans = all(isinstance(value, bool) for value in details.values())
            if all_keys_are_strings and all_values_are_booleans:
                return details
        logger.warning("Invalid format for Answer Correctness Details: %s", details)
        return None
    except (KeyError, TypeError) as exc:
        logger.info("Missing correctness details: %s", exc)
        return None


def _get_excessive_answers(json_response: Any) -> list[str] | None:
    try:
        excessive_answers = json_response["Answer Correctness"]["Excessive Answers"]
        if isinstance(excessive_answers, list):
            all_items_are_strings = all(isinstance(item, str) for item in excessive_answers)
            if all_items_are_strings:
                return excessive_answers
        logger.warning("Invalid format for Excessive Answers: %s", excessive_answers)
        return None
    except KeyError as exc:
        logger.info("Missing excessive answers: %s", exc)
        return []


def _reduce_llm_response_to_item_rating(
    item_rating: ItemRating,
    grader_llm_response_text: str,
    grader_llm_prompt_text: str,
) -> ItemRating:
    item_rating.rating_prompt = grader_llm_prompt_text
    item_rating.rating_response = grader_llm_response_text

    if not item_rating.response:
        item_rating.empty_model_response = True
        item_rating.error_message = "AI response was empty."
        return item_rating

    if not grader_llm_response_text:
        item_rating.empty_auto_rater_response = True
        item_rating.error_message = "Auto-rater response was empty."
        return item_rating

    parsed_json_response = _parse_json_response(grader_llm_response_text)
    if not parsed_json_response:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid JSON response from auto-rater."
        return item_rating

    try:
        answer_correctness_node = parsed_json_response.get("Answer Correctness")
        if not isinstance(answer_correctness_node, dict):
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Missing or malformed 'Answer Correctness' node."
            return item_rating

        grader_explanation = answer_correctness_node.get("Explanation")
        if not isinstance(grader_explanation, str):
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Missing or malformed 'Explanation' in Answer Correctness."
            return item_rating
        item_rating.answer_correctness_explanation = grader_explanation

        details = _get_answer_correctness_details(parsed_json_response)
        if details is None:
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Invalid 'Correctness Details' in Answer Correctness."
            return item_rating
        item_rating.expected_correct_answer_list = list(details.keys())
        item_rating.grader_ratings_list = list(details.values())

        excessive_answers = _get_excessive_answers(parsed_json_response)
        if excessive_answers is None:
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Invalid 'Excessive Answers' in Answer Correctness."
            return item_rating
        if excessive_answers:
            item_rating.response_wrong_answers_list = excessive_answers

    except (KeyError, TypeError, ValueError) as exc:
        logger.exception("Error processing parsed JSON: %s", exc)
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = f"Error during JSON processing: {exc}"
        return item_rating

    return item_rating


def _get_grader_model_input_for_row(row: pd.Series) -> str:
    prompt = str(row["problem"]).strip()
    response = str(row["response"]).strip()
    prompt_type = str(row["answer_type"]).strip()
    answer = str(row["answer"]).strip()

    template = DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE
    return template.format(prompt=prompt, prompt_type=prompt_type, answer=answer, response=response)


def _calculate_ci_str(count: int, total: int, z: float = 1.96) -> str:
    if total == 0:
        return f"N/A ({count}/{total})"
    if count < 0:
        logger.warning("CI calculation: count %d is less than 0.", count)
        count = 0
    if count > total:
        logger.warning("CI calculation: count %d is greater than total %d.", count, total)
        count = total

    p = count / total
    p_percent = p * 100.0
    variance = p * (1.0 - p)
    margin_of_error = z * math.sqrt(variance / total)
    moe_percent = margin_of_error * 100.0
    result_str = f"{round(p_percent, 2):.2f} ± {round(moe_percent, 2):.2f} ({count}/{total})"
    if total <= 5:
        result_str += " (CI not robust for n<=5)"
    return result_str


def _calculate_metric(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, float]:
    precision_val = 0.0
    if (true_positives + false_positives) > 0:
        precision_val = true_positives / (true_positives + false_positives)

    recall_val = 0.0
    if (true_positives + false_negatives) > 0:
        recall_val = true_positives / (true_positives + false_negatives)

    f1_score_val = 0.0
    if (precision_val + recall_val) > 0:
        f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

    return {"precision": precision_val, "recall": recall_val, "f1_score": f1_score_val}


def _aggregate_metrics_and_format_strings(per_item_metrics: dict[str, list[float]]) -> dict[str, str]:
    return {
        "precision": f"{np.mean(per_item_metrics['precision']):.2%}",
        "recall": f"{np.mean(per_item_metrics['recall']):.2%}",
        "f1_score": f"{np.mean(per_item_metrics['f1_score']):.2%}",
    }


def _aggregate_ratings(item_ratings: list[ItemRating]) -> ProjectRating:
    assert item_ratings, "No item ratings to aggregate."
    total_items = len(item_ratings)
    project_rating = ProjectRating(num_total_ratings=total_items)

    num_answer_correctness_evaluated = 0
    num_answer_correctness_all_correct = 0
    num_fully_incorrect_items = 0
    num_items_correct_with_excessive_answers = 0

    category_stats = collections.defaultdict(lambda: {"evaluated": 0, "all_correct": 0})
    per_item_metrics = {"precision": [], "recall": [], "f1_score": [], "accuracy": []}

    for item_rating in item_ratings:
        if item_rating.invalid_auto_rater_response:
            project_rating.num_invalid_auto_rater_response += 1
            continue
        if item_rating.empty_auto_rater_response:
            project_rating.num_empty_auto_rater_response += 1
            continue
        if item_rating.empty_model_response:
            project_rating.num_empty_model_response += 1
            continue

        project_rating.num_valid_ratings += 1
        current_category = item_rating.category_type if item_rating.category_type else "Unknown"

        if item_rating.grader_ratings_list is not None:
            num_answer_correctness_evaluated += 1
            category_stats[current_category]["evaluated"] += 1
            ratings = item_rating.grader_ratings_list
            num_correct = sum(1 for r in ratings if r)

            true_positives = num_correct
            false_negatives = len(ratings) - num_correct
            has_expected_answers = bool(ratings)

            all_expected_answers_correct = False
            if has_expected_answers:
                all_expected_answers_correct = num_correct == len(ratings)
                if num_correct == 0:
                    num_fully_incorrect_items += 1

            excessive_answers = item_rating.response_wrong_answers_list
            has_excessive_answers = bool(excessive_answers)
            false_positives = len(excessive_answers) if has_excessive_answers else 0
            if has_excessive_answers and (all_expected_answers_correct or not has_expected_answers):
                num_items_correct_with_excessive_answers += 1

            is_all_correct = (all_expected_answers_correct or not has_expected_answers) and not has_excessive_answers
            if is_all_correct:
                num_answer_correctness_all_correct += 1
                category_stats[current_category]["all_correct"] += 1

            per_item_metric = _calculate_metric(true_positives, false_positives, false_negatives)
            for key, value in per_item_metric.items():
                per_item_metrics[key].append(value)

    if total_items > 0:
        project_rating.pct_empty_model_response = round(project_rating.num_empty_model_response * 100.0 / total_items, 2)
        project_rating.pct_invalid_auto_rater_response = round(
            project_rating.num_invalid_auto_rater_response * 100.0 / total_items, 2
        )
        project_rating.pct_empty_auto_rater_response = round(
            project_rating.num_empty_auto_rater_response * 100.0 / total_items, 2
        )

    if num_answer_correctness_evaluated > 0:
        project_rating.num_answer_correctness_evaluated = num_answer_correctness_evaluated
        project_rating.pct_w_ci_all_answers_correct = _calculate_ci_str(
            num_answer_correctness_all_correct, num_answer_correctness_evaluated
        )
        project_rating.pct_w_ci_fully_incorrect_items = _calculate_ci_str(
            num_fully_incorrect_items, num_answer_correctness_evaluated
        )
        project_rating.pct_w_ci_correct_with_excessive_answers = _calculate_ci_str(
            num_items_correct_with_excessive_answers, num_answer_correctness_evaluated
        )
        aggregated_metrics = _aggregate_metrics_and_format_strings(per_item_metrics)
        project_rating.precision = aggregated_metrics["precision"]
        project_rating.recall = aggregated_metrics["recall"]
        project_rating.f1_score = aggregated_metrics["f1_score"]

    return project_rating


def _load_dsqa_ground_truth(dataset_path: Path) -> pd.DataFrame:
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(items)
    required_cols = {"example_id", "problem", "answer", "answer_type", "problem_category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth missing columns: {missing}")
    df["example_id"] = df["example_id"].astype(str)
    return df


def _parse_prediction_payload(payload: dict[str, Any]) -> tuple[str, str]:
    pred_raw = payload.get("prediction")
    response = ""
    if isinstance(pred_raw, str):
        try:
            pred_dict = json.loads(pred_raw)
            response = pred_dict.get("final_answer", "")
        except json.JSONDecodeError:
            response = pred_raw or ""
    elif isinstance(pred_raw, dict):
        response = pred_raw.get("final_answer", "") or ""
    else:
        response = str(pred_raw or "")

    pred_id = payload.get("example_id") or payload.get("question_id") or payload.get("question_index")
    if pred_id is None:
        raise ValueError("Prediction entry missing example/question id.")
    return str(pred_id), str(response).strip()


def _load_predictions_df(predictions_path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                pred_id, response = _parse_prediction_payload(payload)
            except Exception as exc:
                logger.warning("Skipping malformed prediction line %s: %s", line_no, exc)
                continue
            records.append(
                {
                    "question_index": pred_id,
                    "response": response,
                    "problem": payload.get("question") or payload.get("problem", ""),
                }
            )
    if not records:
        raise ValueError("No predictions loaded.")
    df = pd.DataFrame(records)
    df["question_index"] = df["question_index"].astype(str)
    return df


def _build_dsqa_input_df(dataset_path: Path, predictions_path: Path) -> pd.DataFrame:
    gt_df = _load_dsqa_ground_truth(dataset_path)
    pred_df = _load_predictions_df(predictions_path)
    merged = gt_df.merge(
        pred_df,
        left_on="example_id",
        right_on="question_index",
        how="left",
        suffixes=("", "_pred"),
    )
    if merged["response"].isna().any():
        missing = merged[merged["response"].isna()][["example_id"]]
        logger.warning("Missing predictions for %d items; responses set to empty.", len(missing))
        merged["response"] = merged["response"].fillna("")
    return merged


def _run_dsqa(
    predictions_path: Path,
    dataset_path: Path,
    config_path: Path,
    output_path: Optional[Path],
    max_workers: int,
    model_override: Optional[str],
) -> None:
    input_df = _build_dsqa_input_df(dataset_path, predictions_path)
    if input_df.empty:
        raise ValueError("Input DataFrame is empty. Cannot perform rating.")

    def process_single_row(index_row_tuple: Tuple[int, pd.Series]) -> ItemRating:
        original_idx, row = index_row_tuple
        item_rating = ItemRating(
            original_index=original_idx,
            example_id=str(row.get("example_id", "")).strip(),
            query=str(row.get("problem", "")).strip(),
            response=str(row.get("response", "")).strip(),
            category_type=str(row.get("problem_category", "")).strip(),
            expected_correct_answer=str(row.get("answer", "")).strip(),
        )

        llm, _ = build_eval_model(config_path, model_override=model_override)
        rating_prompt_text = _get_grader_model_input_for_row(row)
        grader_llm_response_str = None
        for i in range(3):
            try:
                grader_llm_response_str = llm.invoke(
                    messages=[{"role": "user", "content": rating_prompt_text}]
                ).content
                break
            except Exception as exc:
                logger.error("LLM call failed (attempt %s/3) for idx %s: %s", i + 1, original_idx, exc)
                if i == 2:
                    item_rating.error_message = f"LLM call failed after 3 attempts: {exc}"
                    return item_rating
                time.sleep(1 + (2 ** (i + random.random())))

        if grader_llm_response_str is None:
            item_rating.error_message = "LLM response was None after retries."
            return item_rating

        return _reduce_llm_response_to_item_rating(item_rating, grader_llm_response_str, rating_prompt_text)

    tasks = list(input_df.iterrows())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(process_single_row, tasks)
        all_item_ratings = list(tqdm(results_iterator, total=len(tasks)))

    project_rating_result = _aggregate_ratings(all_item_ratings)
    output_path = output_path or (predictions_path.parent / "evaluation.json")
    output = {"summary": project_rating_result.to_dict(), "items": [it.to_dict() for it in all_item_ratings]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation to {output_path}")


###################################################
# XBench evaluator
###################################################

QUESTION_PROMPT = (
    "你是一个通用人工智能助手。我将向你提出一个学术问题, 请尽可能简洁地给出解题思路, 并用以下模版作为回答的结尾:\n\n"
    "最终答案:[你的答案]\n\n"
    "不要在最终答案周围添加任何多余的符号, 不要使用换行（在同一行中完成回答）"
)
OBJECTIVE_PROMPT = "对于本题, 你的答案必须是尽可能简洁的数值, 短语, 或者数学表达式; 如果答案有多个, 使用逗号将它们隔开。"
CHOICE_PROMPT = "对于本题, 选出所有符合的选项, 少选、多选或错选都不得分; 如果选项有多个, 连续列出所有选项, 不要使用逗号或空格分隔。"

LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


@dataclass
class QuestionResult:
    question_id: str
    question: str
    question_type: str
    correct_answer: str
    prediction: str
    scores: List[int]
    extracted_answers: List[str]
    llm_responses: List[str]
    grader_explanations: List[str]
    costs: List[float]
    times: List[float]
    length_cutoffs: List[bool]
    safety_cutoffs: List[bool]
    api_errors: List[bool]
    average_score: float
    best_score: int
    majority_vote_answer: Optional[str]
    majority_vote_score: int
    average_cost: float
    average_time: float
    judge_responses: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    total_questions: int
    total_repeats: int
    average_score: float
    best_score_rate: float
    majority_vote_score: float
    total_cost: float
    total_time: float
    average_time_per_question: float


def _parse_match_result(match: Optional[re.Match]) -> str:
    if match is None:
        return ""
    match_text = match.group(0)
    try:
        target = match_text.split(":")[1].strip()
        return target
    except Exception:
        return match_text


def _majority_vote(answers: List[str]) -> Optional[str]:
    if not answers:
        return None
    count = collections.Counter(answers)
    max_votes = max(count.values())
    candidates = [answer for answer, votes in count.items() if votes == max_votes]
    return random.choice(candidates)


def _grade_question(llm, question_text: str, correct_answer: str, llm_response: Optional[str]) -> Tuple[int, str, str, str]:
    if llm_response is None or not llm_response.strip():
        return 0, "", "Response was empty", ""

    simple_match = re.search(r"最终答案:*(.*)", llm_response)
    simple_match_result = _parse_match_result(simple_match)
    if simple_match_result and simple_match_result == correct_answer:
        return 1, simple_match_result, "答案完全正确, 无需调用LLM Judge", ""

    judge_prompt = LLM_JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=llm_response,
    )

    judge_response = ""
    for i in range(3):
        try:
            judge_response = llm.invoke(messages=[{"role": "user", "content": judge_prompt}]).content
            break
        except Exception as exc:
            logger.error("Judge LLM call failed (attempt %s/3): %s", i + 1, exc)
            if i == 2:
                return 0, "", f"Judge Response error: {exc}", ""
            time.sleep(1 + (2 ** (i + random.random())))

    if not judge_response:
        return 0, "", "Judge Response error: empty response", ""

    extract_match = re.search(r"最终答案:*(.*)", judge_response)
    extract_match_result = _parse_match_result(extract_match)
    correct_match = re.search(r"结论:*.(正确|错误)", judge_response)
    correct_match_result = _parse_match_result(correct_match)
    explain_match = re.search(r"解释:*(.*)", judge_response)
    explain_match_result = _parse_match_result(explain_match)

    score = 1 if (correct_match_result == "正确") else 0
    return score, extract_match_result or "", explain_match_result or "", judge_response


def _load_xbench_ground_truth(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ground_truth = {}
    for item in data:
        question_id = str(item.get("task_id") or item.get("id", ""))
        question_text = item.get("task_question") or item.get("prompt", "")
        answer = item.get("ground_truth") or item.get("answer", "")
        question_type = item.get("type", "问答题")
        if "metadata" in item and isinstance(item["metadata"], dict):
            question_type = item["metadata"].get("type", question_type)
        ground_truth[question_id] = {"question": question_text, "answer": answer, "type": question_type}
    return ground_truth


def _load_xbench_predictions(predictions_path: Path) -> Dict[str, str]:
    predictions = {}
    with predictions_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question_id = str(data.get("question_index", ""))
                prediction = data.get("prediction", "")
                if isinstance(prediction, dict):
                    prediction = prediction.get("final_answer", "")
                elif isinstance(prediction, str):
                    try:
                        pred_dict = json.loads(prediction)
                        prediction = pred_dict.get("final_answer", prediction)
                    except json.JSONDecodeError:
                        pass
                predictions[question_id] = str(prediction)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse line %s: %s", line_num, exc)
                continue
    return predictions


def _eval_single_question_worker(
    args: Tuple[str, Dict[str, Any], str, int, Path, Optional[str]]
) -> QuestionResult:
    question_id, gt_item, prediction, n_repeats, config_path, model_override = args
    llm, _ = build_eval_model(config_path, model_override=model_override)
    try:
        scores = []
        extracted_answers = []
        llm_responses = []
        grader_explanations = []
        costs = []
        times = []
        length_cutoffs = []
        safety_cutoffs = []
        api_errors = []
        judge_responses = []

        for _ in range(n_repeats):
            start_time = time.time()
            llm_response = prediction
            response_time = time.time() - start_time
            api_cost = 0.0

            score, extracted_answer, grader_explanation, judge_response = _grade_question(
                llm, gt_item["question"], gt_item["answer"], llm_response
            )

            llm_responses.append(llm_response)
            extracted_answers.append(extracted_answer)
            scores.append(score)
            grader_explanations.append(grader_explanation)
            costs.append(api_cost)
            times.append(response_time)
            length_cutoffs.append(False)
            safety_cutoffs.append(False)
            api_errors.append(False)
            judge_responses.append(judge_response)

        average_score = float(np.mean(scores))
        best_score = int(np.max(scores))
        majority_vote_answer = _majority_vote(extracted_answers)
        majority_vote_score = 1 if (majority_vote_answer == gt_item["answer"]) else 0
        average_cost = float(np.mean(costs))
        average_time = float(np.mean(times))

        return QuestionResult(
            question_id=question_id,
            question=gt_item["question"],
            question_type=gt_item["type"],
            correct_answer=gt_item["answer"],
            prediction=prediction,
            scores=scores,
            extracted_answers=extracted_answers,
            llm_responses=llm_responses,
            grader_explanations=grader_explanations,
            costs=costs,
            times=times,
            length_cutoffs=length_cutoffs,
            safety_cutoffs=safety_cutoffs,
            api_errors=api_errors,
            average_score=average_score,
            best_score=best_score,
            majority_vote_answer=majority_vote_answer,
            majority_vote_score=majority_vote_score,
            average_cost=average_cost,
            average_time=average_time,
            judge_responses=judge_responses,
        )
    except Exception as exc:
        logger.error("Error evaluating question %s: %s", question_id, exc, exc_info=True)
        return QuestionResult(
            question_id=question_id,
            question=gt_item["question"],
            question_type=gt_item["type"],
            correct_answer=gt_item["answer"],
            prediction=prediction,
            scores=[0],
            extracted_answers=[""],
            llm_responses=[prediction],
            grader_explanations=[f"Error during evaluation: {exc}"],
            costs=[0.0],
            times=[0.0],
            length_cutoffs=[False],
            safety_cutoffs=[False],
            api_errors=[True],
            average_score=0.0,
            best_score=0,
            majority_vote_answer=None,
            majority_vote_score=0,
            average_cost=0.0,
            average_time=0.0,
            judge_responses=[],
        )


def _run_xbench(
    predictions_path: Path,
    dataset_path: Path,
    config_path: Path,
    output_dir: Optional[Path],
    n_repeats: int,
    max_workers: int,
    model_override: Optional[str],
) -> None:
    if output_dir is None:
        output_dir = predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Validating evaluation model config from %s", config_path)
    _, eval_conf = build_eval_model(config_path, model_override=model_override)
    logger.info("Using model: %s", eval_conf.get("model", "unknown"))
    logger.info("Concurrent workers: %s", max_workers)

    ground_truth = _load_xbench_ground_truth(dataset_path)
    predictions = _load_xbench_predictions(predictions_path)
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    logger.info(
        "Evaluating %d questions (Ground truth: %d, Predictions: %d)",
        len(common_ids),
        len(ground_truth),
        len(predictions),
    )
    if not common_ids:
        logger.warning("No common question IDs found between ground truth and predictions!")
        return

    tasks = [
        (question_id, ground_truth[question_id], predictions[question_id], n_repeats, config_path, model_override)
        for question_id in sorted(common_ids)
    ]
    results: List[QuestionResult] = []
    total_cost = 0.0
    total_time = 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qid = {executor.submit(_eval_single_question_worker, task): task[0] for task in tasks}
        with tqdm(total=len(tasks), desc="Evaluating questions") as pbar:
            for future in concurrent.futures.as_completed(future_to_qid):
                question_id = future_to_qid[future]
                try:
                    result = future.result()
                    results.append(result)
                    total_cost += result.average_cost
                    total_time += result.average_time
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "avg_score": f"{np.mean([r.average_score for r in results]):.2%}",
                            "current_score": result.average_score,
                        }
                    )
                except Exception as exc:
                    logger.error("Error processing result for question %s: %s", question_id, exc)
                    pbar.update(1)

    results.sort(key=lambda x: str(x.question_id))
    total_questions = len(results)
    average_score = float(np.mean([r.average_score for r in results])) if results else 0.0
    best_score_rate = float(np.mean([r.best_score for r in results])) if results else 0.0
    majority_vote_score = float(np.mean([r.majority_vote_score for r in results])) if results else 0.0
    average_time_per_question = total_time / total_questions if total_questions > 0 else 0.0

    summary = EvaluationSummary(
        total_questions=total_questions,
        total_repeats=n_repeats,
        average_score=average_score,
        best_score_rate=best_score_rate,
        majority_vote_score=majority_vote_score,
        total_cost=total_cost,
        total_time=total_time,
        average_time_per_question=average_time_per_question,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"xbench_results_{timestamp}.json"
    summary_file = output_dir / f"xbench_summary_{timestamp}.json"
    combined_output = output_dir / "xbench_evaluation.json"

    results_file.write_text(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    summary_file.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    combined_output.write_text(
        json.dumps({"summary": asdict(summary), "results": [r.to_dict() for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n✓ Evaluation complete!")
    print(f"  Results: {results_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Combined: {combined_output}")


###################################################
# HLE evaluator
###################################################

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: str
    confidence: int
    strict: bool


async def _extract_answer(
    client: AsyncOpenAI,
    model: str,
    question: str,
    correct_answer: str,
    response: str,
) -> Optional[Dict[str, Any]]:
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    try:
        response = await client.beta.chat.completions.parse(
            model=model,
            max_completion_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        content = response.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence,
        }
    except Exception as exc:
        logger.error("HLE judge error: %s", exc)
        return None


async def _add_judge_response(
    client: AsyncOpenAI,
    model: str,
    question: Dict[str, Any],
    predictions: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    unique_id = question["id"]
    prediction = copy.deepcopy(predictions[unique_id])
    question_text = question["question"]
    correct_answer = question["answer"]

    if "judge_response" in prediction:
        return unique_id, prediction

    response = prediction["response"]
    content = await _extract_answer(client, model, question_text, correct_answer, response)
    if content is not None:
        prediction["judge_response"] = content
        return unique_id, prediction
    return None, None


async def _judge_all_responses(
    client: AsyncOpenAI,
    model: str,
    questions: List[Dict[str, Any]],
    predictions: Dict[str, Any],
    num_workers: int,
) -> List[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
    semaphore = asyncio.Semaphore(num_workers)

    async def bound_func(question: Dict[str, Any]):
        async with semaphore:
            return await _add_judge_response(client, model, question, predictions)

    tasks = [bound_func(q) for q in questions]
    results = await tqdm_asyncio.gather(*tasks)
    return results


def _hle_load_predictions(filepath: str) -> Dict[str, Any]:
    predictions: Dict[str, Any] = {}
    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    question_id = item.get("question_index") or item.get("id")
                    prediction_data = item.get("prediction", "")
                    if isinstance(prediction_data, str):
                        try:
                            parsed_pred = json.loads(prediction_data)
                            response = (
                                f"Final Answer: {parsed_pred.get('final_answer', 'N/A')}\n\n"
                                f"Reasoning: {parsed_pred.get('reasoning_summary', 'N/A')}"
                            )
                        except json.JSONDecodeError:
                            response = prediction_data
                    else:
                        response = str(prediction_data)
                    predictions[str(question_id)] = {"response": response}
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    return predictions


def _calib_err(confidence, correct, p="2", beta=100):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)
        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in {"infty", "infinity", "max"}:
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")
    if p == "2":
        cerr = np.sqrt(cerr)
    return cerr


def _dump_hle_metrics(predictions: Dict[str, Any], n: int, questions_dict: Optional[Dict[str, str]] = None):
    correct = []
    confidence = []
    detailed_results = []

    for k, v in predictions.items():
        if "judge_response" in v:
            judge_response = v["judge_response"]
            is_correct = "yes" in judge_response["correct"]
            correct.append(is_correct)
            confidence.append(judge_response["confidence"])
            detailed_result = {
                "question_id": k,
                "correct": is_correct,
                "confidence": judge_response["confidence"],
                "correct_answer": judge_response.get("correct_answer", "N/A"),
                "model_answer": judge_response.get("model_answer", "N/A"),
                "reasoning": judge_response.get("reasoning", "N/A"),
            }
            if questions_dict and k in questions_dict:
                detailed_result["question_text"] = questions_dict[k]
            detailed_results.append(detailed_result)
        else:
            print(f"Missing judge response for {k}, you should rerun the judge")

    correct_arr = np.array(correct)
    confidence_arr = np.array(confidence) / 100
    available_predictions = len(correct_arr)
    if available_predictions != n:
        print(f"Available predictions: {available_predictions} | Total questions: {n}")

    accuracy = round(100 * sum(correct_arr) / n, 2) if n else 0.0
    confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2) if n else 0.0
    calibration_error = 100 * round(_calib_err(confidence_arr, correct_arr, p="2", beta=100), 2) if n else 0.0

    print("*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | n = {n}")
    print(f"Calibration Error: {calibration_error}")

    return {
        "accuracy": accuracy,
        "confidence_interval": confidence_half_width,
        "total_questions": n,
        "available_predictions": available_predictions,
        "calibration_error": calibration_error,
    }, detailed_results


def _run_hle(
    predictions_path: Path,
    dataset_name: str,
    config_path: Path,
    output_dir: Optional[Path],
    num_workers: int,
    model_override: Optional[str],
) -> None:
    eval_conf = _load_eval_conf(config_path)
    model_name = model_override or eval_conf.get("model")
    if not model_name:
        raise ValueError("EVAL_MODEL.model is required for HLE evaluation")
    api_key = eval_conf.get("api_key")
    if not api_key:
        raise ValueError("EVAL_MODEL.api_key is required for HLE evaluation")
    base_url = eval_conf.get("base_url")
    max_retries = eval_conf.get("max_retries", 3)
    timeout = eval_conf.get("timeout", 180)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout)

    output_dir = output_dir or predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_basename = os.path.basename(predictions_path)
    output_filepath = output_dir / f"judged_{input_basename}".replace(".jsonl", ".json")

    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            all_questions = payload["data"]
        elif isinstance(payload, list):
            all_questions = payload
        else:
            raise ValueError(
                f"Unsupported HLE dataset JSON format at {dataset_path}. Expected {{'data': [...]}} or a list."
            )
        # Normalise fields we rely on.
        questions = [
            {
                "id": str(q.get("id") or q.get("question_id") or q.get("qid")),
                "question": q.get("question") or q.get("prompt") or "",
                "answer": q.get("answer") or q.get("target") or q.get("final_answer") or "",
            }
            for q in all_questions
            if (q.get("question") or q.get("prompt")) is not None
        ]
        total_questions = len(questions)
        questions_dict = {q["id"]: q["question"] for q in questions if q.get("id") and q.get("question")}
    else:
        dataset = load_dataset(dataset_name, split="test").to_dict()
        questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        total_questions = len(questions)
        questions_dict = {q["id"]: q["question"] for q in questions if "question" in q}
    predictions = _hle_load_predictions(str(predictions_path))

    if output_filepath.exists():
        judged_predictions = json.loads(output_filepath.read_text(encoding="utf-8"))
    else:
        judged_predictions = {}

    questions = [q for q in questions if q["id"] in predictions and q["id"] not in judged_predictions]

    results = asyncio.run(_judge_all_responses(client, model_name, questions, predictions, num_workers))
    for unique_id, prediction in results:
        if unique_id is not None:
            judged_predictions[unique_id] = prediction

    output_filepath.write_text(json.dumps(judged_predictions, indent=4), encoding="utf-8")

    # Expand question_id -> question_text mapping when using HF datasets.
    if not dataset_path.exists():
        all_questions_hf = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        for q in all_questions_hf:
            if q["id"] not in questions_dict and "question" in q:
                questions_dict[q["id"]] = q["question"]

    metrics, detailed_results = _dump_hle_metrics(
        judged_predictions, n=total_questions, questions_dict=questions_dict
    )

    eval_output_filepath = str(output_filepath).replace("judged_", "evaluation_").replace(".json", "_metrics.json")
    with open(eval_output_filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    detailed_output_filepath = str(output_filepath).replace("judged_", "evaluation_").replace(".json", "_detailed.json")
    with open(detailed_output_filepath, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    print(f"\nEvaluation metrics saved to: {eval_output_filepath}")
    print(f"Detailed evaluation results saved to: {detailed_output_filepath}")


###################################################
# Generic evaluator (FinSearchComp)
###################################################


def _run_generic(
    benchmark: BenchmarkType,
    predictions_path: Path,
    dataset_path: Path,
    config_path: Path,
    output_path: Optional[Path],
    model_override: Optional[str],
) -> None:
    predictions = load_predictions(predictions_path, benchmark)
    ground_truth_map, ground_truth_order = load_ground_truth(dataset_path, benchmark)
    llm, eval_conf = build_eval_model(config_path, model_override=model_override)
    results, evaluated, correct = evaluate_predictions(llm, predictions, ground_truth_map, benchmark)
    results = sort_results_by_ground_truth(results, ground_truth_order)
    total_predictions = len(predictions)
    accuracy = (correct / evaluated) if evaluated else 0.0
    category_stats = compute_category_stats(results)
    summary = {
        "benchmark_type": benchmark.value,
        "total_predictions": total_predictions,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "category_stats": category_stats,
        "model": eval_conf.get("model"),
        "results": results,
    }
    output_path = output_path or (predictions_path.parent / "evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")


###################################################
# CLI
###################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation runner")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["deepsearchqa", "dsqa", "xbench", "hle", "finsearchcomp"],
        help="Benchmark type",
    )
    parser.add_argument("--predictions", type=str, required=True, help="Predictions path (JSONL/JSON)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path or HF dataset name for HLE")
    parser.add_argument("--config", type=Path, default=Path("conf.yaml"), help="Config file for eval model")
    parser.add_argument("--eval-model", type=str, default=None, help="Override eval model name")
    parser.add_argument("--output", type=Path, default=None, help="Output path for DSQA/generic evaluation")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir for XBench/HLE evaluation")
    parser.add_argument("--max-workers", type=int, default=30, help="Max worker threads")
    parser.add_argument("--n-repeats", type=int, default=1, help="XBench repeat count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = args.benchmark.lower()
    model_override = _resolve_eval_model(benchmark, args.eval_model)

    predictions_path = _require_path(args.predictions, "Predictions")
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if benchmark in {"deepsearchqa", "dsqa"}:
        dataset_path = _require_path(args.dataset, "Dataset")
        _run_dsqa(
            predictions_path=predictions_path,
            dataset_path=dataset_path,
            config_path=args.config,
            output_path=args.output,
            max_workers=args.max_workers,
            model_override=model_override,
        )
        return

    if benchmark == "xbench":
        dataset_path = _require_path(args.dataset, "Dataset")
        _run_xbench(
            predictions_path=predictions_path,
            dataset_path=dataset_path,
            config_path=args.config,
            output_dir=args.output_dir,
            n_repeats=args.n_repeats,
            max_workers=args.max_workers,
            model_override=model_override,
        )
        return

    if benchmark == "hle":
        _run_hle(
            predictions_path=predictions_path,
            dataset_name=args.dataset,
            config_path=args.config,
            output_dir=args.output_dir,
            num_workers=args.max_workers,
            model_override=model_override,
        )
        return

    dataset_path = _require_path(args.dataset, "Dataset")
    _run_generic(
        benchmark=BenchmarkType(benchmark),
        predictions_path=predictions_path,
        dataset_path=dataset_path,
        config_path=args.config,
        output_path=args.output,
        model_override=model_override,
    )


if __name__ == "__main__":
    main()
