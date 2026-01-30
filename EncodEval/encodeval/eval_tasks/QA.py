from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification, Trainer

from .abstract_eval import AbstractEval


class QuestionAnsweringEval(AbstractEval):
    def train(self) -> None:
        print("Tokenizing training dataset")
        train_dataset = self.dataset["train"].map(
            self.tokenize,
            batched=True,
            load_from_cache_file=False,
            remove_columns=self.dataset["train"].column_names,
        )

        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"].map(
                self.tokenize,
                batched=True,
                load_from_cache_file=False,
                remove_columns=self.dataset["validation"].column_names,
            )
        else:
            val_dataset = None

        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        print("==== Training Arguments ====")
        print(self.tr_args)
        print("=============================")

        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=self.callbacks,
            args=self.tr_args,
        )

        print("Training model")
        trainer.train()

        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, float]:
        print("Evaluating on validation dataset")
        return self.evaluate("validation")

    def test(self) -> Dict[str, float]:
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split: str) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        self.model.eval()

        print(f"Tokenizing {split} dataset")
        eval_dataset = self.dataset[split]
        eval_dataset = eval_dataset.map(
            self.tokenize,
            batched=True,
            load_from_cache_file=False,
            remove_columns=eval_dataset.column_names,
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

        predictions, labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                logits = output.logits.cpu()
                preds = (
                    logits[0].argmax(2)
                    if isinstance(logits, tuple)
                    else logits.argmax(2)
                )
                predictions += preds.tolist()
                labels += batch["labels"].cpu().tolist()

        sanitized_predictions_labels = self.sanitize_predictions_labels(
            predictions, labels
        )
        f1s = self.compute_f1s(**sanitized_predictions_labels)
        return {
            # **sanitized_predictions_labels,
            **f1s,
        }

    def tokenize(self, examples: Dict) -> Dict:
        inputs = self.tokenizer(
            [f"Question: {question}\nContext: " for question in examples["question"]],
            [context + self.tokenizer.eos_token for context in examples["context"]],
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]

            if len(answer["text"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        labels = []
        for input_ids, start_pos, end_pos in zip(
            inputs["input_ids"], start_positions, end_positions
        ):
            if start_pos == 0 and end_pos == 0:
                labels.append([0] * len(input_ids))
            else:
                labels.append(
                    [0] * start_pos
                    + [1] * (end_pos - start_pos + 1)
                    + [0] * (len(input_ids) - end_pos - 1)
                )
            inputs["labels"] = labels

        return inputs

    def sanitize_predictions_labels(self, predictions, labels):
        sanitized_predictions, sanitized_labels = [], []

        for preds, labs in zip(predictions, labels):
            sanitized_predictions.append(
                [i for i, pred in enumerate(preds) if pred == 1]
            )
            sanitized_labels.append([i for i, lab in enumerate(labs) if lab == 1])

        return {
            "pred": sanitized_predictions,
            "true": sanitized_labels,
        }

    def compute_f1s(self, pred, true):
        pred = [set(p) if len(p) > 0 else {0} for p in pred]
        true = [set(t) if len(t) > 0 else {0} for t in true]
        precisions_i = [len(p & t) / len(p) for p, t in zip(pred, true)]
        recalls_i = [len(p & t) / len(t) for p, t in zip(pred, true)]
        f1s_i = [
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
            for precision, recall in zip(precisions_i, recalls_i)
        ]
        f1_avg = np.mean(f1s_i)
        return {"score": f1_avg, "scores_i": f1s_i}
