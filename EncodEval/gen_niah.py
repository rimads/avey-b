#!/usr/bin/env python3
"""
Needle-in-Haystack extractive-QA – CONTROLLABLE DIFFICULTY VERSION
Specify exact distribution of distractor counts to predict final benchmark scores.
"""

import argparse
import json
import os
import random
import string
from typing import Dict, List

import numpy as np
from datasets import Dataset, DatasetInfo, Features, Sequence, Value
from tqdm.auto import tqdm


class NeedleHaystackConfig:
    def __init__(
        self,
        needle_types: List[str] = None,
        context_lengths: List[int] = None,
        num_train_samples: int = 1000,
        num_test_samples: int = 1000,
        output_dir: str = "needle_haystack_datasets",
        random_seed: int = 42,
        train_distractor_dist: str = "0:1.0",
        test_distractor_dist: str = "0:1.0",
    ):
        self.needle_types = needle_types or ["word", "number"]
        self.context_lengths = context_lengths or [2048, 4096, 8192]
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.output_dir = output_dir
        self.random_seed = random_seed

        # Parse distribution strings
        self.train_distractor_dist = self._parse_dist(train_distractor_dist)
        self.test_distractor_dist = self._parse_dist(test_distractor_dist)

        # Validate distributions sum to 1.0
        self._validate_dist(self.train_distractor_dist, "train")
        self._validate_dist(self.test_distractor_dist, "test")

    def _parse_dist(self, dist_str: str) -> Dict[int, float]:
        """Parse '0:0.5,1:0.3,2:0.2' into {0: 0.5, 1: 0.3, 2: 0.2}"""
        result = {}
        for pair in dist_str.split(","):
            if ":" not in pair:
                raise ValueError(
                    f"Invalid distribution format: {dist_str}. Expected 'k:v' pairs separated by commas."
                )
            k, v = pair.split(":")
            result[int(k)] = float(v)
        return result

    def _validate_dist(self, dist: Dict[int, float], split_name: str):
        """Ensure distribution probabilities sum to 1.0"""
        total = sum(dist.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"{split_name}_distractor_dist probabilities sum to {total}, not 1.0"
            )

        # Validate keys are non-negative integers
        for k in dist.keys():
            if k < 0:
                raise ValueError(f"Number of distractors must be non-negative, got {k}")


class NeedleHaystackGenerator:
    def __init__(self, config: NeedleHaystackConfig):
        self.cfg = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # English vocabulary for haystack
        self.vocab = [
            "the",
            "and",
            "or",
            "but",
            "because",
            "however",
            "therefore",
            "nevertheless",
            "apple",
            "banana",
            "computer",
            "phone",
            "desk",
            "chair",
            "table",
            "window",
            "door",
            "book",
            "paper",
            "pen",
            "city",
            "street",
            "house",
            "garden",
            "tree",
            "flower",
            "car",
            "train",
            "plane",
            "ship",
            "mountain",
            "river",
            "ocean",
            "forest",
            "run",
            "walk",
            "jump",
            "read",
            "write",
            "think",
            "process",
            "create",
            "build",
            "design",
            "analyze",
            "develop",
            "implement",
            "test",
            "validate",
            "optimize",
            "execute",
            "perform",
            "operate",
            "manage",
            "organize",
            "plan",
            "schedule",
            "monitor",
            "report",
            "large",
            "small",
            "big",
            "little",
            "high",
            "low",
            "fast",
            "slow",
            "new",
            "old",
            "good",
            "bad",
            "simple",
            "complex",
            "easy",
            "difficult",
            "important",
            "necessary",
            "useful",
            "effective",
            "efficient",
            "reliable",
            "accurate",
            "precise",
            "quickly",
            "slowly",
            "carefully",
            "easily",
            "effectively",
            "efficiently",
            "reliably",
            "accurately",
            "precisely",
            "usually",
            "often",
            "sometimes",
            "rarely",
            "always",
            "never",
            "furthermore",
            "moreover",
            "additionally",
            "consequently",
            "subsequently",
            "meanwhile",
            "otherwise",
            "instead",
            "although",
            "though",
            "while",
            "whereas",
            "despite",
            "since",
            "until",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "within",
            "throughout",
            "across",
            "along",
            "behind",
            "beyond",
            "under",
            "over",
            "against",
            "toward",
            "forward",
            "backward",
        ]

    def _generate_alien_string(self) -> str:
        """Generate truly random, un-memorizable string."""
        length = random.randint(8, 12)
        chars = string.ascii_lowercase + string.digits
        return "".join(random.choices(chars, k=length))

    def _generate_number(self) -> str:
        return str(random.randint(1_000_000, 9_999_999))

    def _haystack(self, n: int) -> List[str]:
        return [random.choice(self.vocab) for _ in range(n)]

    def _ordinal_word(self, n: int) -> str:
        """Convert number to ordinal word."""
        ordinals = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
            11: "eleventh",
            12: "twelfth",
            13: "thirteenth",
            14: "fourteenth",
            15: "fifteenth",
        }
        return ordinals.get(n, f"{n}th")

    def _build_one_sample(
        self,
        ctx_len: int,
        needle_type: str,
        position_ratios: List[float],
        num_distractors: int,
    ) -> Dict:
        """
        Build a sample with a specific number of distractors.

        Args:
            ctx_len: Context length in tokens
            needle_type: "word" or "number"
            position_ratios: Position for each needle (target + distractors)
            num_distractors: Number of distractor needles (not counting target)
        """
        # Generate all needles: 1 target + N distractors
        if needle_type == "number":
            all_needles = [self._generate_number() for _ in range(num_distractors + 1)]
        else:
            all_needles = [
                self._generate_alien_string() for _ in range(num_distractors + 1)
            ]

        # Randomly select which ordinal is the target
        target_ordinal = random.randint(1, num_distractors + 1)
        target_needle = all_needles[target_ordinal - 1]

        # Calculate token budget
        base_haystack_len = ctx_len - len(all_needles)  # subtract all needles
        tokens = self._haystack(base_haystack_len)

        # Insert needles at their designated positions
        needles_with_positions = []
        for i, needle in enumerate(all_needles):
            pos_ratio = position_ratios[i]
            insert_pos = int(pos_ratio * len(tokens))
            insert_pos = max(0, min(insert_pos, len(tokens)))
            needles_with_positions.append((insert_pos, needle))

        # Sort by insert position (descending) to avoid index shifting
        needles_with_positions.sort(key=lambda x: x[0], reverse=True)

        for insert_pos, needle in needles_with_positions:
            tokens.insert(insert_pos, needle)

        context = " ".join(tokens)

        # Find answer_start for target needle (unique string, so context.find() works)
        answer_start = context.find(target_needle)
        assert answer_start != -1, "Target needle not found in context"

        # Verify
        extracted = context[answer_start : answer_start + len(target_needle)]
        assert extracted == target_needle, (
            f"Verification failed: expected '{target_needle}', got '{extracted}'"
        )

        return {
            "id": f"{needle_type}_{ctx_len}_{random.randint(0, 1_000_000):06d}",
            "title": f"haystack_{needle_type}_{ctx_len}",
            "context": context,
            "question": f"What is the {self._ordinal_word(target_ordinal)} special {needle_type} in this passage?",
            "answers": {"text": [target_needle], "answer_start": [answer_start]},
        }

    def _sample_num_distractors(self, distribution: Dict[int, float]) -> int:
        """Sample number of distractors based on distribution."""
        choices = list(distribution.keys())
        probabilities = list(distribution.values())
        return np.random.choice(choices, p=probabilities)

    def create_dataset(
        self, ctx_len: int, needle_type: str, distribution: Dict[int, float], total_num
    ) -> Dataset:
        """Generate dataset with controlled distractor distribution."""
        samples = []

        for i in tqdm(
            range(total_num),
            desc=f"Generating {needle_type} | {ctx_len} tokens",
        ):
            # Sample number of distractors for this example
            num_distractors = self._sample_num_distractors(distribution)

            # Generate position ratios for all needles (target + distractors)
            total_needles = num_distractors + 1
            position_ratios = np.random.rand(total_needles)

            # Ensure minimum gap between needles (spread them out)
            min_gap = 0.05  # 5% minimum gap
            sorted_pos = np.sort(position_ratios)
            for j in range(1, len(sorted_pos)):
                sorted_pos[j] = max(sorted_pos[j], sorted_pos[j - 1] + min_gap)
            # Renormalize if needed
            if sorted_pos[-1] > 1.0:
                sorted_pos = sorted_pos / sorted_pos[-1]
            # Shuffle to avoid ordered positions biasing the task
            np.random.shuffle(sorted_pos)
            position_ratios = sorted_pos

            # Build sample
            sample = self._build_one_sample(
                ctx_len, needle_type, position_ratios, num_distractors
            )
            samples.append(sample)

        features = Features(
            {
                "id": Value("string"),
                "title": Value("string"),
                "context": Value("string"),
                "question": Value("string"),
                "answers": {
                    "text": Sequence(Value("string")),
                    "answer_start": Sequence(Value("int32")),
                },
            }
        )

        ds = Dataset.from_list(samples, info=DatasetInfo(features=features))
        return ds

    def run(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        print("=" * 80)
        print("Needle-in-Haystack Dataset Generator (CONTROLLABLE DIFFICULTY)")
        print("=" * 80)
        print(f"Needle types: {self.cfg.needle_types}")
        print(f"Context lengths: {self.cfg.context_lengths}")
        print(f"Train samples per config: {self.cfg.num_train_samples}")
        print(f"Test samples per config: {self.cfg.num_test_samples}")
        print(f"Train distractor distribution: {self.cfg.train_distractor_dist}")
        print(f"Test distractor distribution: {self.cfg.test_distractor_dist}")
        print("-" * 80)

        # Calculate expected scores
        def expected_score(dist: Dict[int, float]) -> float:
            """Calculate expected score given distractor distribution."""
            # For 0 distractors: 100% accuracy (1/(0+1) = 1)
            # For 1 distractor: 50% accuracy (1/(1+1) = 0.5)
            # For N distractors: 1/(N+1) accuracy
            return sum(prob * (1 / (count + 1)) for count, prob in dist.items())

        train_expected = expected_score(self.cfg.train_distractor_dist)
        test_expected = expected_score(self.cfg.test_distractor_dist)
        print(f"Expected train score: {train_expected:.1%}")
        print(f"Expected test score: {test_expected:.1%}")
        print("-" * 80)

        for needle_type in self.cfg.needle_types:
            for ctx_len in self.cfg.context_lengths:
                print(f"\n[{needle_type.upper()} | {ctx_len} tokens]")

                # Generate train and test sets
                train_ds = self.create_dataset(
                    ctx_len,
                    needle_type,
                    self.cfg.train_distractor_dist,
                    self.cfg.num_train_samples,
                )
                test_ds = self.create_dataset(
                    ctx_len,
                    needle_type,
                    self.cfg.test_distractor_dist,
                    self.cfg.num_test_samples,
                )

                # Combine into DatasetDict
                from datasets import DatasetDict

                dataset = DatasetDict({"train": train_ds, "test": test_ds})

                # Save in load_dataset format
                folder = os.path.join(
                    self.cfg.output_dir, f"{needle_type}_{ctx_len}_controllable"
                )
                os.makedirs(folder, exist_ok=True)

                # Save splits
                for split_name, split_data in [
                    ("train", dataset["train"]),
                    ("test", dataset["test"]),
                ]:
                    file_path = os.path.join(folder, f"{split_name}.jsonl")
                    with open(file_path, "w", encoding="utf-8") as f:
                        for example in split_data:
                            f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    # Log distribution stats
                    from collections import Counter

                    distractor_counts = [
                        len(ex["answers"]["text"]) - 1 for ex in split_data
                    ]  # -1 to get distractors
                    count_dist = Counter(distractor_counts)
                    print(
                        f"  ✓ {split_name}.jsonl ({len(split_data)} samples) – {dict(count_dist)}"
                    )

                # Save dataset_infos.json
                info = {
                    "description": f"Controllable-difficulty needle-in-haystack QA – {needle_type} needle, {ctx_len} tokens",
                    "features": dataset["train"].features,
                    "splits": [
                        {"name": "train", "num_examples": len(dataset["train"])},
                        {"name": "test", "num_examples": len(dataset["test"])},
                    ],
                    "distractor_distributions": {
                        "train": self.cfg.train_distractor_dist,
                        "test": self.cfg.test_distractor_dist,
                    },
                    "expected_scores": {
                        "train": train_expected,
                        "test": test_expected,
                    },
                }
                with open(os.path.join(folder, "dataset_infos.json"), "w") as f:
                    json.dump(info, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("✓ All datasets generated successfully!")
        print(
            "You can now control expected benchmark scores via distractor distributions."
        )
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate controllable-difficulty needle-in-haystack datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--needle-types",
        nargs="+",
        choices=["word", "number"],
        default=["word", "number"],
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[1000, 2000, 3800, 4000, 8000, 16000, 32000],
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=10000,
        help="Samples per configuration per split",
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=1000,
        help="Samples per configuration per split",
    )
    parser.add_argument(
        "--train-distractor-dist",
        default="0:0.2,1:0.8",
        help='Distribution of distractor counts in train set, e.g., "0:0.5,1:0.3,2:0.2"',
    )
    parser.add_argument(
        "--test-distractor-dist",
        default="0:0.4,1:0.6",
        help='Distribution of distractor counts in test set, e.g., "0:0.2,1:0.3,2:0.3,5:0.2"',
    )
    parser.add_argument("--output-dir", default="niah")
    parser.add_argument("--random-seed", type=int, default=11)
    args = parser.parse_args()

    config = NeedleHaystackConfig(
        needle_types=args.needle_types,
        context_lengths=args.context_lengths,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        train_distractor_dist=args.train_distractor_dist,
        test_distractor_dist=args.test_distractor_dist,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )

    generator = NeedleHaystackGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
