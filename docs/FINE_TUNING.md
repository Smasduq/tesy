# Fine-tuning Tesy

## The short version

**OpenRouter cannot fine-tune models.** It is a routing layer: it serves 468
models that other people trained, but it exposes no fine-tuning API. Confirmed
against the live service:

- `GET https://openrouter.ai/openapi.json` → 116 documented paths, none of them
  fine-tuning (`/api/v1/fine_tuning/jobs` and friends return `404`)
- no fine-tuned model variants exist in the model catalogue
- no fine-tuning guide in the docs (17 feature guides, none about training)

So a fine-tuned model has to live somewhere else. The cheapest route that
keeps everything free:

```
free Colab GPU  →  Unsloth  →  LoRA fine-tune  →  merge to GGUF  →  Ollama
                                                                          │
                                            Tesy --provider local  ←────┘
```

Tesy already speaks to Ollama, so the only work is training and exporting.

---

## Option 1 - no training (free, instant, probably enough)

Before spending GPU hours, try the cheap route. Tesy's personality lives in
`persona.md`; the file after the `---` separator is the system prompt. Write
your companion there, and you have a "custom model" that costs nothing and
never goes stale.

```bash
python main.py "hello"                      # uses persona.md
python main.py --system "you are terse"     # one-off override
TESY_SYSTEM="..." python main.py "hello"    # environment override
```

Add a few example exchanges as examples inside the persona and the model
imitates them closely. For a companion chatbot this often gets you most of
the way to a fine-tune, and it takes a minute instead of an afternoon.

---

## Option 2 - a real fine-tune, still free

### Step 1 - pick a base model

Small enough to train on a free Colab T4 GPU:

| Base model             | Notes                                        |
| ---------------------- | -------------------------------------------- |
| `openai/gpt-oss-20b`   | strong, MoE, well supported by Unsloth        |
| `unsloth/Qwen3-4B`     | small, fast, very cheap to train              |
| `meta-llama/Llama-3.1-8B` | good general quality, gated base         |

### Step 2 - build the dataset

JSONL, one conversation per line, OpenAI chat format:

```json
{"messages":[{"role":"system","content":"You are Tesy, a warm co-worker and friend."},{"role":"user","content":"long day"},{"role":"assistant","content":"I noticed the look on you. Want to talk about it?"}]}
```

What you put in here is what you get out. For a companion:
- the user's own messages as `user`
- the replies you want as `assistant`
- 200-1000 examples is usually plenty for style and tone

Keep the `system` row consistent with `persona.md` so the fine-tune and the
prompt agree.

### Step 3 - train it (free Colab)

Open the free notebook from the Unsloth docs (they publish ready-to-run
notebooks for gpt-oss and Qwen), or run the equivalent locally:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gpt-oss-20b",
    load_in_4bit=True,
)
model, tokenizer = FastLanguageModel.from_pretrained(
    model, tokenizer, target_modules="all-linear"
)

dataset = load_dataset("json", data_files="data.jsonl", split="train")
dataset = tokenizer.apply_chat_template(dataset, add_generation_prompt=False)

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
    trainer_class=TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)
trainer.train()
```

### Step 4 - merge and export to GGUF

```python
model.save_pretrained_gguf("tesy-finetuned", tokenizer, quantization_method="q4_k_m")
```

Unsloth's Colab notebooks do this for you. You end up with something like
`tesy-finetuned.Q4_K_M.gguf`.

### Step 5 - serve it with Ollama

```bash
ollama create tesy ./tesy-finetuned.Q4_K_M.gguf
ollama list
```

Ollama also imports directly from Hugging Face:

```bash
ollama create tesy hf.co/<user>/<your-finetuned-repo>:Q4_K_M
```

### Step 6 - point Tesy at it

```bash
python main.py --provider local "hello"
```

Or make it permanent:

```bash
export TESY_PROVIDER=local
export OLLAMA_MODEL=tesy
```

Check what is available:

```bash
python main.py --list-local-models
```

Local models have no request quota at all, so this also removes the
50-requests-per-day cap that comes with the OpenRouter free tier.

---

## Option 3 - a hosted fine-tune (paid)

If you would rather not touch GPUs, the mainstream providers do it:
OpenAI (`o4-mini`, `gpt-4.1-mini`), Google (Gemini), Mistral. All need a
separate account and credits, and all need a small adapter on top of the
Tesy provider layer to translate their message format. OpenRouter's own
catalogue is not fine-tunable, so those models would have to be called
directly rather than through OpenRouter.

---

## Which one should you do?

For a voice companion: start with **Option 1** and **Option 2 in parallel**.
Get the persona right first, collect real conversations with it, then
fine-tune on those transcripts. The training data you gather while talking to
Tesy is exactly the data that makes the fine-tune good.
