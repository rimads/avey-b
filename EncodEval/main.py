import importlib
import json
import os
import subprocess

import configue
import fire
from encodeval.eval_tasks import (
    EvalConfig,
    QuestionAnsweringEval,
    RetrievalEval,
    SequenceClassificationEval,
    TokenClassificationEval,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)


def main(config_file: str = None, model_path: str = None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["EVAL_MODEL_PATH"] = model_path
    print(f"Evaluating model at path: {model_path}")
    eval_config: EvalConfig = configue.load(config_file, sub_path="eval_config")

    # Determine the evaluator based on task type
    if eval_config.task_type == "IR":
        evaluator = RetrievalEval(eval_config)
    elif eval_config.task_type == "QA":
        evaluator = QuestionAnsweringEval(eval_config)
    elif eval_config.task_type == "SC":
        evaluator = SequenceClassificationEval(eval_config)
    elif eval_config.task_type == "TC":
        evaluator = TokenClassificationEval(eval_config)

    else:
        raise ValueError(f"Invalid task type: {eval_config.task_type}")

    # Check if results file already exists
    if os.path.exists(f"{eval_config.results_dir}/results.json"):
        print(
            "A results file already exists for this configuration at "
            f"{eval_config.results_dir}/results.json, skipping evaluation"
        )
        exit()
    else:
        # Run training if needed
        if eval_config.tr_args.do_train:
            if (
                os.path.exists(eval_config.tr_args.output_dir)
                and len(os.listdir(eval_config.tr_args.output_dir)) > 0
            ):
                print(
                    "A fine-tuned model already exists for this configuration at "
                    f"{eval_config.tr_args.output_dir}, skipping training"
                )
            else:
                evaluator.train()
        else:
            print("Training disabled, skipping training")

        # Run evaluation
        results = {}
        if eval_config.tr_args.do_eval or eval_config.tr_args.do_predict:
            if eval_config.tr_args.do_eval:
                results["validation"] = evaluator.validate()
            if eval_config.tr_args.do_predict:
                results["test"] = evaluator.test()
            if os.path.exists(eval_config.tr_args.output_dir):
                subprocess.run(
                    f"rm -r {eval_config.tr_args.output_dir}", shell=True, check=True
                )
        else:
            print("Evaluation disabled, skipping evaluation")
            exit()

        # Save results to file
        os.makedirs(eval_config.results_dir, exist_ok=True)
        with open(f"{eval_config.results_dir}/results.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved at {eval_config.results_dir}")
        print("Evaluation completed")
        exit()


if __name__ == "__main__":
    fire.Fire(main)
