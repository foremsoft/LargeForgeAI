# Continued pretraining script (HF Trainer)
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

args = TrainingArguments(
    output_dir="./out",
    fp16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1
)

trainer = Trainer(model=model, args=args, train_dataset=None)
trainer.train()
