from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    quantization_config=bnb_config,
    device_map="auto",
    # device_map={"": 0},
    # trust_remote_code=True
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value"
    ],
)

model.config.use_cache = False
model = get_peft_model(model, peft_config)

# tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    output_dir="./results_latest_2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    save_steps=20,
    fp16=True,
    logging_steps=10,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    # max_steps=30000,
    max_steps=300,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()

model.save_pretrained("output_dir_2")
