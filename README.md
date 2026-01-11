# Multimodal Vision-Language Assistant

课程作业：多模态视觉-语言助手（Qwen2-VL）

## Environment
- OS: Windows 11
- GPU: NVIDIA RTX 5070
- Python: Conda env: bo_env

## Base Evaluation (ChartQA)
- Model: Qwen/Qwen2-VL-2B-Instruct
- Split: val
- Samples: 200
- Metric: Exact Match (EM)
- Result: **0.23 (46/200)**

Command:
```bash
python -m src.eval.eval_chartqa \
  --eval_jsonl data/processed/chartqa_val.jsonl \
  --model_id Qwen/Qwen2-VL-2B-Instruct \
  --max_samples 200 \
  --out_path outputs/chartqa_base_eval.json


