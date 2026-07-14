# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Answer parsing and image-path resolution for Toto+VLM evaluation.

Trimmed to the two functions this pipeline actually uses (parse_response,
get_image_paths) — both are self-contained, no cross-references to the rest
of the original file.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List


def parse_response(prediction: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model response to extract answer and reasoning.

    Tries multiple strategies to extract structured information from model
    responses, from most specific to most general:
    1. JSON format with "answer" and "reasoning" fields
    2. "Answer:" prefix format
    3. Simple sentence format (answer before first period)
    4. Raw prediction as fallback

    Args:
        prediction (str): Raw model prediction string

    Returns:
        Tuple[Optional[str], Optional[str]]: (answer, reasoning) where answer
        is the extracted answer and reasoning is the extracted reasoning. Both
        can be None if parsing fails.

    Examples:
        >>> parse_response('{"answer": "A", "reasoning": "Because X"}')
        ('A', 'Because X')

        >>> parse_response('Answer: B\\nThe data shows a clear trend.')
        ('B', 'The data shows a clear trend.')

        >>> parse_response('C. This is the correct choice.')
        ('C', ' This is the correct choice.')

        >>> parse_response('Invalid response')
        ('Invalid response', None)
    """
    if not prediction or not isinstance(prediction, str):
        return None, None

    prediction = prediction.strip()
    if not prediction:
        return None, None

    # Strategy 1: Try to extract JSON from response
    try:
        if "{" in prediction and "}" in prediction:
            start = prediction.find("{")
            end = prediction.rfind("}") + 1
            json_str = prediction[start:end]
            prediction_json = json.loads(json_str)
            answer = prediction_json.get("answer")
            reasoning = prediction_json.get("reasoning")
            return answer, reasoning
        elif "{" in prediction:
            prediction = prediction + "}"
            start = prediction.find("{")
            end = prediction.rfind("}") + 1
            json_str = prediction[start:end]
            prediction_json = json.loads(json_str)
            answer = prediction_json.get("answer")
            reasoning = prediction_json.get("reasoning")
            return answer, reasoning
    except json.JSONDecodeError:
        # Continue to other strategies if JSON parsing fails
        pass

    # Strategy 2: Look for "Answer:" prefix format
    answer_prefix = "Answer:"
    if answer_prefix in prediction:
        start = prediction.find(answer_prefix) + len(answer_prefix)
        end = prediction.find("\n", start)
        if end == -1:  # No newline found, take rest of string
            answer = prediction[start:].strip()
            reasoning = None
        else:
            answer = prediction[start:end].strip()
            reasoning = prediction[end + 1 :].strip()
            reasoning = reasoning if reasoning else None
        return answer, reasoning

    # Strategy 3: Simple sentence format (answer before first period)
    if "." in prediction:
        end = prediction.find(".")
        answer = prediction[:end].strip()
        reasoning = prediction[end + 1 :].strip()
        reasoning = reasoning if reasoning else None
        return answer, reasoning

    # Strategy 4: Return raw prediction as answer with no reasoning
    return prediction, None


def get_image_paths(
    query_groups: str,
    image_dir: str = "arfbench-images-v2",
    task_category: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Find all image files for given query groups.

    Args:
        query_groups (str): Comma-separated query group identifiers
        image_dir (str): Directory containing image files
        task_category (str, optional): Task category (for special handling
            of Anomaly Indicator/Correlation)

    Returns:
        Optional[List[str]]: List of image paths if all found,
        None if any missing
    """
    image_dir = Path(image_dir)
    image_paths = []
    groups = [g.strip() for g in query_groups.split(",")]

    # Special handling for Anomaly Indicator and Anomaly Correlation
    # These use both a combined image AND separate images:
    # {query_group1}-{query_group2}.png + individual images
    if task_category and (
        "Indicator" in task_category or "Correlation" in task_category
    ):
        if len(groups) == 2:
            # First, try to find the combined image
            combined_filename = f"{groups[0]}-{groups[1]}.png"
            combined_path = image_dir / combined_filename
            combined_found = False

            if combined_path.exists():
                image_paths.append(str(combined_path))
                combined_found = True

            # Then find the individual images for each group
            # Look for exact matches first (e.g., "36504_0.png"), then fallback to prefix matches
            # but exclude paired images (files containing a dash)
            for group in groups:
                found = False
                # First try exact match: {group}.png
                exact_match = image_dir / f"{group}.png"
                if exact_match.exists():
                    image_paths.append(str(exact_match))
                    found = True
                else:
                    # Fallback to prefix match, but exclude paired images (containing dash)
                    for file in image_dir.glob(f"{group}*"):
                        file_name = file.name
                        # Skip paired images (contain dash) and the combined image we already added
                        if "-" not in file_name and str(file) not in image_paths:
                            image_paths.append(str(file))
                            found = True
                            break

                if not found:
                    return None

            # Return the combined list if we found the combined image
            if combined_found:
                return image_paths
            else:
                # If no combined image found, fallback to default behavior
                image_paths = []
        else:
            # Fallback to original behavior if not exactly 2 groups
            pass

    # Default behavior: find separate images for each query group
    # Prefer exact matches (e.g., "36504_0.png") and skip paired images (containing dash)
    for group in groups:
        found = False
        # First try exact match: {group}.png
        exact_match = image_dir / f"{group}.png"
        if exact_match.exists():
            image_paths.append(str(exact_match))
            found = True
        else:
            # Fallback to prefix match, but exclude paired images (containing dash)
            for file in image_dir.glob(f"{group}*"):
                file_name = file.name
                # Skip paired images (contain dash)
                if "-" not in file_name:
                    image_paths.append(str(file))
                    found = True
                    break
        if not found:
            return None

    return image_paths
