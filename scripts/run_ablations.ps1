param(
    [string]$Model = ".\model\Qwen2.5-VL-3B-Instruct",
    [string]$Dataset = "Vqa_rad",
    [string]$Output = ".\output\ablations"
)

$ErrorActionPreference = "Stop"
foreach ($stage in @("sft", "grpo", "pipeline")) {
    python "$PSScriptRoot\..\Qwentrain.py" --model_id $Model --dataset $Dataset --stage $stage --output_dir "$Output\$stage" --load_in_4bit --max_completion_length 256 --num_generations 4
    if ($LASTEXITCODE -ne 0) { throw "Ablation $stage failed" }
}
