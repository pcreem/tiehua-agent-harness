# Tiehua Tourism Agent - Evaluation Dashboard
# Hugging Face Space Deployment

This Space deploys the Tiehua Tourism Agent Evaluation Dashboard as a Streamlit app.

## Setup

1. Create a new Space on Hugging Face with SDK: Streamlit
2. Add these secrets in Space settings:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `OPENROUTER_BASE_URL`: https://openrouter.ai/api/v1 (optional)

## Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `dashboard/` - Dashboard source code

## Local Development

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

Required:
- `OPENROUTER_API_KEY`: OpenRouter API key for LLM Judge

Optional:
- `OPENROUTER_BASE_URL`: Custom base URL (default: https://openrouter.ai/api/v1)

## Data

The dashboard expects evaluation reports in `evaluation/reports/` directory.
Reports should be JSON files with the evaluation results structure.

## Features

- 📊 Overview metrics (pass rate, avg score, retries)
- 🔧 Tool usage analytics
- 🛡️ Safety metrics and violations
- 🤖 LLM Judge evaluation results
- 📅 Evaluation timeline
- 📊 Interactive visualizations with Plotly