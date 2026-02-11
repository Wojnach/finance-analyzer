import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MAX_SEQ_LEN = 2048
SEED = 42


class ProgressCallback(TrainerCallback):
    def __init__(self, progress_file):
        self.progress_file = Path(progress_file)

    def on_log(self, args, state, control, **kwargs):
        progress = {
            "step": state.global_step,
            "total_steps": state.max_steps,
            "epoch": round(state.epoch, 2) if state.epoch else 0,
            "total_epochs": args.num_train_epochs,
            "percent": (
                round(state.global_step / state.max_steps * 100, 1)
                if state.max_steps
                else 0
            ),
            "loss": state.log_history[-1].get("loss") if state.log_history else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.progress_file.write_text(json.dumps(progress, indent=2))


def load_data(data_path, tokenizer):
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for sample in samples:
        messages = sample["messages"]
        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        prompt = f"[INST]{user_content}[/INST]"
        full_text = f"{prompt}{assistant_content}{tokenizer.eos_token}"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN
        )["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        assert len(labels) == len(full_ids)

        pad_len = MAX_SEQ_LEN - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * (MAX_SEQ_LEN - pad_len) + [0] * pad_len
        else:
            full_ids = full_ids[:MAX_SEQ_LEN]
            labels = labels[:MAX_SEQ_LEN]
            attention_mask = [1] * MAX_SEQ_LEN

        input_ids_list.append(full_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    )
    dataset.set_format("torch")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--progress-file", default="training_progress.json")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model with 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    print("Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training data from {args.data}...")
    dataset = load_data(args.data, tokenizer)
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        save_steps=50,
        logging_steps=10,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        eval_strategy="steps",
        eval_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[ProgressCallback(args.progress_file)],
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    final_dir = output_dir / "final"
    print(f"Saving final adapter to {final_dir}...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
