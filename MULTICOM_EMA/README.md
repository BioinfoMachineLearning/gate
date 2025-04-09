# MULTICOM EMA Methods for QMODE1 and QMODE3

This README outlines the commands for running **QMODE1** (global model quality estimation) and **QMODE3** (Top-5 model selection) using MULTICOM EMA methods.

---

## QMODE1: Global Model Quality Estimation

### MULTICOM\_GATE

- Runs the GATE model. Predicted scores are saved in:

```text
$OUTPUT_DIR/ensemble_nonaf.csv
```

```bash
python inference_multimer.py \
  --fasta_path $FASTA_PATH \
  --input_model_dir $INPUT_MODEL_DIR \
  --output_dir $OUTPUT_DIR
```

### MULTICOM\_LLM

#### Early Targets (T1201o, H1204, H1208)

- EnQA results are extracted from the GATE output directory:

```text
$OUTPUT_DIR/feature/enqa/enqa.csv
```

#### Remaining Targets

- Computes average pairwise similarity score (PSS). Predicted scores are saved in:

```text
$OUTPUT_DIR/PSS.csv
```

```bash
python run_pss_mmalign.py \
  --input_model_dir $INPUT_MODEL_DIR \
  --mmalign_program $MMALIGN_BINARY \
  --output_cdir $OUTPUT_DIR
```

---

## QMODE3: Top-5 Model Selection

### MULTICOM\_LLM

- Follow the same steps in **QMODE1**.

### MULTICOM\_human  

- Runs a GATE variant that incorporates AlphaFold-Multimer features

```bash
python inference_multimer.py \
  --fasta_path $FASTA_PATH \
  --input_model_dir $INPUT_MODEL_DIR \
  --output_dir $OUTPUT_DIR \
  --pkldir $PKLDIR \
  --use_af_feature True
```

- Aggregate scores and select top models:

```bash
python average_MULTICOM_human_score.py \
  --incsv $OUTPUT_DIR/gate_af_summary.csv \
  --outcsv $OUTPUT_CSV
```

---

## Notes

- Replace environment variables (e.g., `$FASTA_PATH`, `$OUTPUT_DIR`, `$INPUT_MODEL_DIR`) with actual paths.
- Ensure all required dependencies and external tools (e.g., MM-align) are installed and accessible in your environment.
