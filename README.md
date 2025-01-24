# KG_Jiangya

Knowledge Graph-based Intelligent QA System for Jiangya Hydropower Station Fault Diagnosis

## Project Overview

This project is an intelligent question-answering system based on knowledge graph technology, specifically designed for fault diagnosis at the Jiangya Hydropower Station. The system integrates advanced natural language processing techniques, including Named Entity Recognition (NER) and intent classification models, to construct and query a fault diagnosis knowledge graph.

## Project Structure

```
KG_Jiangya/
├── Pytorch_Bert_BiLSTM_CRF_NER/    # BERT-BiLSTM-CRF based Named Entity Recognition model
├── Pytorch_Bert_TextCNN_CLS/        # BERT-TextCNN based Intent Classification model
└── data.txt                         # Data file
```

## Key Features

1. **Named Entity Recognition (NER)**
   - Implementation using Pytorch BERT-BiLSTM-CRF model
   - Identifies key entities in hydropower station texts (e.g., equipment, fault types, symptoms, components)
   - Extracts structured information from maintenance records and fault reports

2. **Intent Classification**
   - BERT-TextCNN based classification model
   - Classifies user queries into different fault diagnosis intents
   - Helps route questions to appropriate knowledge graph queries

## Technology Stack

- Python
- PyTorch
- BERT
- BiLSTM
- CRF
- TextCNN

## Requirements

- Python 3.6+
- PyTorch
- transformers
- numpy
- scikit-learn

## Usage Guide

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Training**
   - Training NER Model
   ```bash
   cd Pytorch_Bert_BiLSTM_CRF_NER
   python train.py
   ```
   
   - Training Intent Classification Model
   ```bash
   cd Pytorch_Bert_TextCNN_CLS
   python train.py
   ```


## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch
3. Submit your changes
4. Create a Pull Request

 
