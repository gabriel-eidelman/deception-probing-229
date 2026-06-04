# Deception Probes Survive Their Confound: Linear Structure Beyond the Elicitation Protocol

## Environment Setup

### Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Modal (Remote Compute)

This project uses [Modal](https://modal.com) to run inference on Llama-3.1-8B-Instruct. The model is downloaded to a Modal volume and accessed remotely.

**Before downloading the model**, you must request access to `meta-llama/Llama-3.1-8B-Instruct` on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and add your HF token as a Modal secret named `huggingface-secret`.

Once access is granted:

```bash
modal run setup/download_model.py
```

### Replicating Results

1. Create a `.env` file in the project root and add your OpenAI API key (used for LLM grading):

   ```
   OPENAI_API_KEY=your_key_here
   ```

2. Run Phase B — generate model responses, extract residual-stream activations, and grade with GPT-4o:

   ```bash
   modal run probe/train_probe/generate_extract_grade.py
   ```

   This writes the following artifacts to `outputs/`:

   - `scored_responses.csv` — graded labels used by the probe trainer
   - `activations/` — per-layer `.npy` activation matrices
   - `scored_responses.jsonl` and `run_metadata.json`

3. Run the full probe analysis pipeline:

   ```bash
   cd probe/train_probe
   python run_all.py
   ```

   This executes the following stages in order:

   | Stage | Script | Description |
   |-------|--------|-------------|
   | E.2 | `report_census.py` | Behavioral backbone census |
   | C.1 | `run_pipeline.py` | Decodability-by-factor; writes `peak_layer.json` |
   | C.2 | `run_c2_transfer.py` | Confound-held-fixed transfer |
   | C.3 | `run_c3_orthogonalize.py` | Orthogonalization (decisive test) |
   | D.2 | `run_d2_nonlinear.py` | Nonlinear check |
   | D.3 | `run_d3_geometry.py` | Geometry analysis |

   Outputs are written to `probe/train_probe/outputs/`.
