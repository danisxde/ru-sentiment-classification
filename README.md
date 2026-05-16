# Russian Sentiment Analysis

Анализ тональности русскоязычных текстов на датасете MonoHime/ru_sentiment_dataset.

## Содержание
- Разведывательный анализ данных (EDA)
- Тематическое моделирование (LDA)
- Классификатор тональности на базе rubert-tiny2 (3 класса: Neutral / Positive / Negative)
- Оптимизация модели для CPU-инференса (Dynamic INT8 Quantization)

## Стек
`PyTorch` `Transformers` `Gensim` `scikit-learn` `pyLDAvis`

## Результаты
| Модель | F1 macro | ROC-AUC |
|---|---|---|
| TF-IDF + LogReg (baseline) | ~0.72 | ~0.88 |
| rubert-tiny2 (fine-tuned)  | ~0.78 | ~0.93 |

## Датасет
[MonoHime/ru_sentiment_dataset](https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset)
