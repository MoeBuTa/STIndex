# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List


def is_yes_no_question(answer_labels: List[str]) -> bool:
    for answer in answer_labels:
        # TODO: is there any "True", "False" in answer labels? should we convert they to "Yes"/"No" if "Yes"/"No" not in answer labels?
        if not answer.lower() in ["yes", "no"]:
            return False
    return True


def infer_question_type(answer_labels: List[str]) -> str:
    if is_yes_no_question(answer_labels):
        return "yes_no"

    return "undefined"


def infer_nq_question_type(answer_labels: List[str], yes_no_answer: int) -> str:
    if yes_no_answer == 1:
        return "yes_no"

    return "undefined"
