from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer

from .abstract_eval import AbstractEval


class SequenceClassificationEval(AbstractEval):
    def train(self) -> None:
        print("Tokenizing training dataset")
        train_dataset = self.dataset["train"].map(
            self.tokenize, batched=True, load_from_cache_file=False
        )
        train_dataset = train_dataset.remove_columns(
            [
                f
                for f in train_dataset.features
                if f not in ["input_ids", "attention_mask", "label"]
            ]
        )

        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"].map(
                self.tokenize, batched=True, load_from_cache_file=False
            )
            val_dataset = val_dataset.remove_columns(
                [
                    f
                    for f in val_dataset.features
                    if f not in ["input_ids", "attention_mask", "label"]
                ]
            )
        else:
            val_dataset = None

        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

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

    def validate(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        print("Evaluating on validation dataset")
        return self.evaluate("validation")

    def test(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split: str) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        self.model.eval()

        print(f"Tokenizing {split} dataset")
        eval_dataset = self.dataset[split].map(
            self.tokenize, batched=True, load_from_cache_file=False
        )
        eval_dataset = eval_dataset.remove_columns(
            [
                f
                for f in eval_dataset.features
                if f not in ["input_ids", "attention_mask", "label"]
            ]
        )

        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                predictions += output.logits.cpu().tolist()

        if self.model.num_labels == 1:
            predictions = [pred[0] for pred in predictions]
        else:
            predictions = [np.argmax(pred).item() for pred in predictions]

        accuracies = self.compute_accuracies(predictions, eval_dataset["label"])

        return {
            # "pred": predictions,
            # "true": eval_dataset["label"],
            **accuracies,
        }

    def tokenize(self, examples: Dict) -> Dict:
        return self.tokenizer(
            [text + self.tokenizer.eos_token for text in examples["text"]],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

    def compute_accuracies(self, pred, true):
        accuracies_i = [(p == t) * 1 for p, t in zip(pred, true)]
        accuracy_avg = np.mean(accuracies_i)
        return {"score": accuracy_avg, "scores_i": accuracies_i}
