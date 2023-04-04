#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from pathlib import Path

from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

from icefall import setup_logger

def preprocess_aishell():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    # Note: By default, we preprocess all sub-parts.
    # You can delete those that you don't need.
    # For instance, if you don't want to use the L subpart, just remove
    # the line below containing "L"
    dataset_parts = (
        "train",
        "test",
        "dev",
    )

    logging.info("Loading manifest (may take 10 minutes)")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        suffix="jsonl.gz",
        prefix="aishell2",
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"aishell2_cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        # Create long-recording cut manifests.
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        # Run data augmentation that needs to be done in the
        # time domain.
        if partition not in ["test", "dev"]:
            logging.info(
                f"Speed perturb for {partition} with factors 0.9 and 1.1 "
                "(Perturbing may take 8 minutes and saving may take 20 minutes)"
            )
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    setup_logger(log_filename="./log-preprocess-aishell")

    preprocess_aishell()
    logging.info("Done")


if __name__ == "__main__":
    main()
