# Book Recommender CF vs MF

Rating prediction and book recommendation using **User-Based Collaborative Filtering**,
**Item-Based Collaborative Filtering**, and **Matrix Factorization** on the Goodbooks-10k dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shortahairlover/BookRecommender-CFvsMF/blob/main/tugas_rekomendasi.ipynb)

Group assignment for the Recommender Systems course, Informatics Undergraduate Program, School of Computing, Telkom University.

## Group 5

| Name | Student ID |
|---|---|
| Naufal Rafif Hidayat | 103012300403 |
| Muhammad Alivio Putra | 103012300098 |
| Akbar Jundi Pradita | 103012300081 |
| Raffa Anung El Haqqi | 103012300163 |
| Naufal Thafhan | 103012300284 |

## Dataset

[Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k): about 6 million ratings (1 to 5 scale) from 53,424 Goodreads users on 10,000 books.

To keep user-based CF tractable, the notebook builds a subset:

1. 1,000 users sampled at random (seed 42) with their full rating history.
2. Iterative filtering so every user and every book has at least 10 ratings.

The resulting subset contains **84,139 ratings from 996 users on 2,618 books** (96.8% sparsity).
The dataset is downloaded automatically when the notebook runs.

## Methods

| Method | Details |
|---|---|
| User-Based CF | Pearson correlation (cosine on mean-centered data), k = 30 neighbors |
| Item-Based CF | Adjusted cosine similarity, k = 30 neighbor items |
| Matrix Factorization | Stochastic gradient descent with biases, K = 20 latent factors |

Evaluation metrics: MAE, MSE, RMSE, NMSE, and NDCG@10.

## Results

5-fold cross-validation (mean):

| Method | MAE | RMSE | NMSE | NDCG@10 |
|---|---|---|---|---|
| User-Based CF | 0.6813 | 0.8876 | 0.1970 | 0.9108 |
| Item-Based CF | **0.6597** | **0.8649** | **0.1870** | **0.9119** |
| Matrix Factorization | 0.6805 | 0.8694 | 0.1890 | 0.9103 |

Item-Based CF achieves the best score on every metric, followed closely by Matrix Factorization.
The full discussion is available in the [report](report/Laporan%20Sistem%20Rekomendasi.pdf) (in Indonesian).

## Repository Structure

```
├── tugas_rekomendasi.ipynb   Main notebook (code, results, and analysis)
├── report/                   PDF report
├── assets/                   Charts, evaluation results (CSV), diagrams, code images, and formulas
└── requirements.txt          Python dependencies
```

## How to Run

**Google Colab:** click the "Open in Colab" button above, then select Runtime > Run all.

**Local:**

```bash
pip install -r requirements.txt
jupyter notebook tugas_rekomendasi.ipynb
```

Running the whole notebook takes about 20 to 25 minutes, mostly for the parameter experiments and cross-validation.
