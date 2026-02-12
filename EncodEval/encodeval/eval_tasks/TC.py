from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification, Trainer

from .abstract_eval import AbstractEval


class TokenClassificationEval(AbstractEval):
    def train(self) -> None:
        print("Tokenizing training dataset")
        train_dataset = self.dataset["train"].map(self.tokenize, batched=True, load_from_cache_file=False)
        train_dataset = train_dataset.remove_columns(
            [f for f in train_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
        )

        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"]
            val_dataset = val_dataset.map(self.tokenize, batched=True, load_from_cache_file=False)
            val_dataset = val_dataset.remove_columns(
                [f for f in val_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
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
            tokenizer=self.tokenizer,
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

    def evaluate(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        self.model.eval()
        
        print(f"Tokenizing {split} dataset")
        eval_dataset = self.dataset[split].map(self.tokenize, batched=True, load_from_cache_file=False)
        token_ids = eval_dataset["token_ids"]
        eval_dataset = eval_dataset.remove_columns(
            [f for f in eval_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
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
                preds = logits[0].argmax(2) if isinstance(logits, tuple) else logits.argmax(2)
                predictions += preds.tolist()
                labels += batch["labels"].cpu().tolist()

        token_predictions_labels = self.get_token_predictions_labels(predictions, labels, token_ids)
        f1s = self.compute_f1s(**token_predictions_labels)
        return {
            # **token_predictions_labels, 
            **f1s,
        }

    def tokenize(self, examples):
        sentences = [" ".join(tokens) for tokens in examples["tokens"]]
        tokenized_inputs = self.tokenizer(
            [sentence + self.tokenizer.eos_token for sentence in sentences],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        aligned_labels, token_ids = [], []

        for offsets, tokens, tags in zip(
            tokenized_inputs["offset_mapping"], examples["tokens"], examples["tags"]
        ):
            label_ids, token_id_per_subtoken = [], []
            word_idx, char_pos = 0, 0
            current_word = tokens[word_idx]
            current_label = tags[word_idx]

            for offset in offsets:
                if offset == (0, 0):
                    label_ids.append(-100)
                    token_id_per_subtoken.append(-100)
                    continue

                while offset[0] >= char_pos + len(current_word):
                    char_pos += len(current_word) + 1
                    word_idx += 1
                    if word_idx >= len(tokens):
                        break
                    current_word = tokens[word_idx]
                    current_label = tags[word_idx]

                if word_idx < len(tags):
                    label_ids.append(current_label)
                    token_id_per_subtoken.append(word_idx)
                else:
                    label_ids.append(-100)
                    token_id_per_subtoken.append(-100)

            aligned_labels.append(label_ids)
            token_ids.append(token_id_per_subtoken)

        tokenized_inputs["labels"] = aligned_labels
        tokenized_inputs["token_ids"] = token_ids
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs
    
    def get_token_predictions_labels(self, predictions, labels, token_ids):
        predictions_token, labels_token = [], []

        for preds, labs, tok_ids in zip(predictions, labels, token_ids):
            unique_tok_ids = sorted(list(set(tok_ids) - {-100}))
            preds_token, labs_token = [], []

            for tok_id in unique_tok_ids:
                preds_for_token = [pred for pred, _id in zip(preds, tok_ids) if _id == tok_id]
                labs_for_token = [lab for lab, _id in zip(labs, tok_ids) if _id == tok_id]
                preds_token.append(max(set(preds_for_token), key=preds_for_token.count))
                labs_token.append(labs_for_token[0])

            predictions_token.append(preds_token)
            labels_token.append(labs_token)

        return {
            "pred": predictions_token,
            "true": labels_token,
        }
    
    def compute_f1s(self, pred, true):
        f1s_i = [f1_score(t, p, average="macro") for t, p in zip(true, pred)]
        f1_avg = np.mean(f1s_i)
        return {"score": f1_avg, "scores_i": f1s_i}