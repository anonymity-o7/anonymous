# anonymous

# O7: Open-source Foundation Model for Analog Circuit Design

## Overview

**O7** is an open-source foundation model designed to assist in analog circuit design.This project is presented at the 2024 Design Automation Conference (DAC), and the model weights are hosted on Hugging Face.

## Features

- **Open-source Foundation Model**: Pre-trained model for analog circuit design tasks.
- **Benchmarking**: Includes a benchmark suite to evaluate the performance of the model.
- **Ease of Use**: Designed for quick deployment and integration into existing workflows.
- **Hugging Face Integration**: The model weights are available for download from Hugging Face.

## Contents

1. **Model Weights**  
2. **Benchmark Files**  

---

## 1. Model Weights

The pre-trained model weights for **O7** are available for download from Hugging Face:

- [Download Model Weights from Hugging Face](https://huggingface.co/AnalogO7/O7)

Ensure you have access to Hugging Face and use the `transformers` library to load the model weights into your code.

### Loading the Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('huggingface-model-link')
tokenizer = AutoTokenizer.from_pretrained('huggingface-model-link')

# Example inference
inputs = tokenizer("Input circuit design task", return_tensors="pt")
outputs = model(**inputs)
```

## 2. Benchmark Files
The benchmarking files for evaluating the performance of the model are included in this repository.

Benchmark Files:
```bash
benchmark_data/O7_benchmark.json
```
