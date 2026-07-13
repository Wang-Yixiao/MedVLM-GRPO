# Local VQA-RAD bridge-SFT smoke test

This is the smallest locally runnable image-question-answer experiment in the
repository. It intentionally uses only the first 100 VQA-RAD training rows.

## What it does

1. Loads local `models/Qwen2.5-VL-3B-Instruct` in FP16.
2. Freezes the base model and trains an attention LoRA on 100 multimodal rows.
3. Saves the bridge-SFT LoRA.
4. Samples four answers per question for five questions.
5. Loads the local CrossEncoder and BioGPT reward models on CPU.
6. Computes SemanticCorrectness, per-sequence BioGPT perplexity score,
   TagPresence, combined reward, group variance, normalized relative advantage,
   and output length.

The rollout stage simulates GRPO diagnostics only. It does **not** claim to be
a policy-gradient update. This keeps the smoke test runnable on native Windows
without Unsloth, vLLM, modern TRL, or bitsandbytes. The neural reward models are
loaded from `reward_model/` and do not consume GPU memory by default.

## Run

```powershell
python scripts/local_vqarad_bridge_smoke.py
```

More conservative generation:

```powershell
python scripts/local_vqarad_bridge_smoke.py `
  --num_rows 100 `
  --epochs 1 `
  --gradient_accumulation_steps 4 `
  --rollout_samples 3 `
  --num_generations 2 `
  --max_new_tokens 64
```

The defaults target the detected RTX 4080 SUPER 16 GB. If FP16 still runs out
of memory, reduce `--max_length` and `--max_new_tokens` first.

## Outputs

```text
output/vqarad-smoke/
|-- bridge_sft_lora/
|-- simulated_grpo_rollouts.jsonl
`-- simulated_grpo_summary.json
```

`simulated_grpo_rollouts.jsonl` contains every candidate and its reward
components. `neural_reward_components.jsonl` contains the detailed
CrossEncoder/BioGPT diagnostics.

For a quick dependency-light fallback only:

```powershell
python scripts/local_vqarad_bridge_smoke.py --reward_mode proxy
```
