# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List


SOURCE_TYPES_TO_DOWNLOAD: List[str] = ["wikipedia", "wikidata"]
FILE_TYPES_TO_DOWNLOAD: List[str] = ["pdf", "html"]


DATASET_TO_SPLIT_LIST: Dict[str, List[str]] = {
    "nq": ["train", "validation"],
    "triviaqa": ["train", "validation"],
    "hotpotqa": ["train", "dev"],
    "two_wiki": ["train", "dev"],
    "popqa": ["test"],
    "webqa": ["train", "test"],
    "musique": ["train", "dev"],
    "mirage": ["train"],  # MIRAGE medical benchmark (no separate dev/test splits in source)
    "medcorp": ["train"],  # MedCorp corpus: Textbooks + StatPearls
}


def check_dataset_split(dataset: str, split: str) -> None:
    assert dataset in DATASET_TO_SPLIT_LIST.keys(), f"Dataset {dataset} not found in predefined `DATASET_TO_SPLIT_LIST`"
    assert split in DATASET_TO_SPLIT_LIST[dataset], (
        f"Dataset {dataset} do not have split {split} in `DATASET_TO_SPLIT_LIST`"
    )
    return
