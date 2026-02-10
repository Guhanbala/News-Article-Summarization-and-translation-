# Text Summarization Models

A comprehensive collection of fine-tuned transformer models for document summarization in English and Indian languages. This project compares multiple state-of-the-art architectures and provides ready-to-use implementations.

## 📋 Project Overview

This repository contains Jupyter notebooks and scripts for:
- **BART Fine-tuning**: Fine-tuning BART-base on CNN/DailyMail dataset for English summarization
- **BART vs PEGASUS**: Comparative analysis of two popular summarization architectures
- **BERT-GRU Hybrid**: Custom BERT encoder with GRU decoder for summarization
- **mBART-50**: Multilingual BART model for cross-lingual summarization
- **IndicTrans2**: Fine-tuning for Indian language translation and summarization

## 📁 Repository Structure

```
.
├── notebooks/
│   ├── bart_fine_tuned-1.ipynb                  # BART fine-tuning on CNN/DailyMail
│   ├── Bart_vs_Pegasus_for_text_summarization.ipynb  # Model comparison
│   ├── bert_gru_summarization.ipynb             # BERT-GRU hybrid model
│   ├── checking_MBart_large_50_model.ipynb      # Multilingual BART testing
│   ├── indictrans2-fine-tunning-with-entam-v2_final__.ipynb  # Indian languages
│   └── test_fine_tuned_bart.ipynb               # Fine-tuned model evaluation
├── models/
│   └── fine_tuned_bart_cnn/                     # Pre-trained fine-tuned BART model
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda
- CUDA 11.0+ (for GPU support, optional but recommended)
- 4GB+ RAM (8GB+ recommended for model training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/text-summarization-models.git
   cd text-summarization-models
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv summarizer_env
   source summarizer_env/bin/activate  # On Windows: summarizer_env\Scripts\activate

   # Or using conda
   conda create -n summarizer python=3.10
   conda activate summarizer
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### Using Pre-trained Fine-tuned BART Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model
model_path = "./models/fine_tuned_bart_cnn"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Summarize text
text = "Your long document or article here..."
inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    min_length=20,
    num_beams=4,
    early_stopping=True
)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

## 📓 Notebook Descriptions

### 1. **BART Fine-tuning** (`bart_fine_tuned-1.ipynb`)
Fine-tunes BART-base model on CNN/DailyMail dataset.

**Key features:**
- Loads pre-trained BART-base from HuggingFace
- Prepares CNN/DailyMail dataset for fine-tuning
- Trains the model using Hugging Face Trainer API
- Evaluates performance on validation set
- Saves fine-tuned model for inference

**To run:**
```bash
jupyter notebook notebooks/bart_fine_tuned-1.ipynb
```

### 2. **BART vs PEGASUS Comparison** (`Bart_vs_Pegasus_for_text_summarization.ipynb`)
Comparative analysis of BART and PEGASUS models for summarization.

**Key features:**
- Loads both BART and PEGASUS models
- Tests on multiple datasets (CNN/DailyMail, XSum, etc.)
- Evaluates using ROUGE, BLEU, and METEOR metrics
- Provides visualization of model performance
- Discusses trade-offs between models

**To run:**
```bash
jupyter notebook notebooks/Bart_vs_Pegasus_for_text_summarization.ipynb
```

### 3. **BERT-GRU Hybrid** (`bert_gru_summarization.ipynb`)
Custom hybrid architecture combining BERT encoder with GRU decoder.

**Key features:**
- Implements custom encoder-decoder architecture
- Uses BERT for semantic understanding
- Implements attention mechanism
- Tests on summarization datasets
- Suitable for lightweight deployments

**To run:**
```bash
jupyter notebook notebooks/bert_gru_summarization.ipynb
```

### 4. **Multilingual BART** (`checking_MBart_large_50_model.ipynb`)
Testing mBART-50 for multilingual summarization.

**Key features:**
- Loads mBART-50 model (supports 50+ languages)
- Tests cross-lingual summarization
- Evaluates on multilingual datasets
- Demonstrates language switching

**To run:**
```bash
jupyter notebook notebooks/checking_MBart_large_50_model.ipynb
```

### 5. **IndicTrans2 Fine-tuning** (`indictrans2-fine-tunning-with-entam-v2_final__.ipynb`)
Fine-tunes IndicTrans2 for Indian language translation and summarization.

**Key features:**
- Fine-tunes on Indian language datasets
- Supports languages: Hindi, Tamil, Telugu, Kannada, Malayalam, etc.
- Implements custom training loop
- Evaluates on Indian language benchmarks

**To run:**
```bash
jupyter notebook notebooks/indictrans2-fine-tunning-with-entam-v2_final__.ipynb
```

### 6. **Fine-tuned Model Testing** (`test_fine_tuned_bart.ipynb`)
Tests the pre-trained fine-tuned BART model.

**Key features:**
- Loads saved fine-tuned model
- Provides simple summarization function
- Tests on CNN/DailyMail validation set
- Compares generated vs. original summaries

**To run:**
```bash
jupyter notebook notebooks/test_fine_tuned_bart.ipynb
```

## 🔧 Configuration

### GPU Support

Enable GPU acceleration by installing CUDA-compatible PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Model Parameters

Adjust summarization parameters in the code:

```python
outputs = model.generate(
    input_ids,
    max_length=100,          # Maximum summary length
    min_length=20,           # Minimum summary length
    num_beams=4,             # Beam search width
    early_stopping=True,     # Stop early if good solution found
    temperature=1.0,         # Sampling temperature
    top_p=0.9,              # Nucleus sampling
    top_k=50                # Top-k sampling
)
```

## 📊 Models Supported

| Model | Language | Task | Parameters |
|-------|----------|------|-----------|
| BART-base | English | Summarization | 140M |
| BART-large | English | Summarization | 400M |
| PEGASUS | English | Summarization | 568M |
| mBART-50 | 50+ Languages | Multilingual | 680M |
| IndicTrans2 | Indian Languages | Translation/Summarization | 400M+ |
| BERT-GRU | English | Summarization (Lightweight) | ~200M |

## 📈 Evaluation Metrics

Models are evaluated using:

- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
  - ROUGE-1 (unigram overlap)
  - ROUGE-2 (bigram overlap)
  - ROUGE-L (longest common subsequence)

- **BLEU**: Bilingual Evaluation Understudy Score

- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering

## 🎯 Use Cases

- **News Summarization**: Summarize news articles to key points
- **Document Summarization**: Extract summaries from long documents
- **Legal Text Summarization**: Summarize legal documents
- **Multilingual Summarization**: Summarize documents in multiple languages
- **Indian Language Processing**: Summarization for Hindi, Tamil, Telugu, etc.

## 💾 Dataset References

- **CNN/DailyMail**: https://huggingface.co/datasets/abisee/cnn_dailymail
- **XSum**: https://huggingface.co/datasets/EdinburghNLP/xsum
- **ENTAM (Indian Languages)**: https://github.com/neuralmonkey/entam

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Submit pull requests
- Suggest improvements
- Add new models or datasets

## 📝 Training Your Own Model

To fine-tune a model on your own dataset:

1. Prepare your dataset in the following format:
   ```json
   {
     "article": "Long document text...",
     "summary": "Summary text..."
   }
   ```

2. Load and preprocess:
   ```python
   from datasets import load_dataset
   dataset = load_dataset('json', data_files='your_data.json')
   ```

3. Fine-tune using the notebooks as reference

4. Evaluate using ROUGE metrics

## 📦 Requirements

See `requirements.txt` for all dependencies:
- `transformers >= 4.30.0`
- `torch >= 2.0.0`
- `datasets >= 2.10.0`
- `evaluate >= 0.4.0`
- `rouge-score >= 0.1.2`
- `nltk >= 3.8`
- `jupyter >= 1.0.0`

## ⚙️ System Requirements

**Minimum:**
- RAM: 4GB
- Storage: 10GB (for models)
- GPU: Optional (CPU will work but be slower)

**Recommended:**
- RAM: 16GB+
- Storage: 50GB+
- GPU: NVIDIA with 8GB+ VRAM

## 🔍 Troubleshooting

### Out of Memory Errors
```python
# Reduce batch size
batch_size = 4  # or 2

# Use gradient accumulation
gradient_accumulation_steps = 4
```

### Slow Generation
```python
# Use fewer beams
num_beams = 2  # Instead of 4

# Use faster sampling
do_sample = True
top_k = 50
```

**Last Updated**: February 2025
**Python Version**: 3.8+
**Status**: Active Development
