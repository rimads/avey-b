from typing import Dict, List, Set

import numpy as np
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sklearn.metrics import ndcg_score
import torch
from tqdm import tqdm

from .abstract_eval import AbstractEval


class RetrievalEval(AbstractEval):
    def train(self) -> None:
        train_dataset = self.dataset["train"]

        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"]
        else:
            val_dataset = None

        self.model.tokenizer = self.tokenizer
        data_collator = SentenceTransformerDataCollator(self.tokenize)
        loss = (
            self.loss_fn(model=self.model, **self.loss_kwargs)
            if self.loss_kwargs is not None else self.loss_fn(model=self.model)
        )

        print("==== Training Arguments ====")
        print(self.tr_args)
        print("============================")

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self.tr_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            loss=loss,
        )

        print("Training model")
        trainer.train()

        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, List]:
        print("Evaluating on validation dataset")
        return self.evaluate("validation")

    def test(self) -> Dict[str, List]:
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split: str) -> Dict[str, List]:
        self.model.eval()
        queries, corpus, qrels = self.dataset["queries"], self.dataset["corpus"], self.dataset[f"qrels_{split}"]
        queries = {_id: queries[_id] for _id in qrels}
        queries_enc, docs_enc = self.encode_sequences(queries), self.encode_sequences(corpus)
        similarity_scores = self.compute_similarity_scores(queries_enc, docs_enc)
        predictions_labels_at_k = self.get_predictions_labels_at_k(qrels, similarity_scores)
        ndcgs = self.compute_ndcgs(**predictions_labels_at_k)
        return {
            # **predictions_labels_at_k, 
            **ndcgs,
        }

    def tokenize(self, sequences: List) -> torch.Tensor:
        return self.tokenizer(
            [seq + self.tokenizer.eos_token for seq in sequences],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def encode_sequences(self, sequences: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        sequence_ids = list(sequences.keys())
        sequences_enc = {}

        for i in tqdm(
            range(0, len(sequence_ids), self.tr_args.per_device_eval_batch_size),
            desc="Encoding sequences",
        ):
            sequence_ids_batch = sequence_ids[i:i+self.tr_args.per_device_eval_batch_size]
            sequences_batch = [sequences[_id] for _id in sequence_ids_batch]
            sequences_enc_batch = self.model.encode(sequences_batch, batch_size=self.tr_args.per_device_eval_batch_size)
            sequences_enc = {
                **sequences_enc, 
                **{_id: seq_enc for _id, seq_enc in zip(sequence_ids_batch, sequences_enc_batch)}
            }
    
        return sequences_enc
    
    def compute_similarity_scores(
            self, 
            queries_enc: Dict[str, np.ndarray], 
            docs_enc: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        doc_ids = list(docs_enc.keys())
        docs_enc_matrix = np.concatenate([enc.reshape(1, -1) for enc in docs_enc.values()])
        similarity_scores = {}

        for query_id, query_enc in tqdm(queries_enc.items(), desc="Computing similarity scores"):
            scores = self.model.similarity(query_enc, docs_enc_matrix).flatten().numpy()
            similarity_scores[query_id] = {doc_id: score for doc_id, score in zip(doc_ids, scores)}

        return similarity_scores

    def get_predictions_labels_at_k(
            self, 
            qrels: Dict[str, List], 
            similarity_scores: Dict[str, Dict[str, float]],
            k: int = 10,
    ) -> Dict[str, List[float]]:
        predictions_at_k, labels_at_k = [], []
        
        for query_id, query_similarity_scores in tqdm(similarity_scores.items(), desc="Evaluating"): 
            query_scores_argsort_at_k = np.argsort(list(query_similarity_scores.values()))[-k:]
            query_docs_at_k = np.array(list(query_similarity_scores.keys()))[query_scores_argsort_at_k]
            predictions_at_k.append([query_similarity_scores[doc_id].item() for doc_id in query_docs_at_k])
            labels_at_k.append([(doc_id in qrels[query_id]) * 1 for doc_id in query_docs_at_k])
            
        return {
            "pred": predictions_at_k,
            "true": labels_at_k,
        }
    
    def compute_ndcgs(self, pred, true):
        ndcgs_i = [ndcg_score([t], [p]) for t, p in zip(true, pred)]
        ndcg_avg = np.mean(ndcgs_i)
        return {"score": ndcg_avg, "scores_i": ndcgs_i}
    