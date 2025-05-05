# NLP Fake News Classification Project

This project was developed as part of the **Machine Learning for Natural Language Processing** course. It explores and compares several deep learning approaches for fake news classification using text data. The models implemented range from classical RNN-based architectures to Transformers and zero-shot classification.

## 👥 Authors

- Ahmed Khairaldin  
- Imrane Zaakour

## 📁 Project Structure
.
├── data/
│ ├── raw/ # Raw datasets (ISOT, Kaggle Real or Fake)
│ └── processed/ # Preprocessed train/test data
├── preprocess_data.py # Script to preprocess raw data
├── notebook_dl_models.ipynb # RNN, LSTM, and GRU models
├── Transformer.ipynb # Transformer model
├── zero_shot_classification.ipynb # Zero-shot classification
└── README.md # This file

## 📊 Datasets

We use two publicly available datasets for fake news detection:

- [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)

Raw data is located in the `data/raw` folder. It is preprocessed using `preprocess_data.py` and stored in the `data/processed` folder as training and test splits.

## 🧠 Models

| Model Type                | Notebook File                         | Description                                  |
|---------------------------|----------------------------------------|---------------------------------------------|
| RNN, LSTM, GRU            | `notebook_dl_models.ipynb`             | RNN vs. LSTM vs. GRU                        |
| Transformer               | `Transformer.ipynb`                    | Transformer-based model                     |
| Zero-Shot Classification  | `zero_shot_classification.ipynb`       | Pretrained model used without fine-tuning   |



