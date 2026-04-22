# 🖼️ Watermark Removal Autoencoder / Автоэнкодер для удаления водяных знаков

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras)](https://keras.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[🇷🇺 Русский](#-описание-проекта) · [🇬🇧 English](#-project-description)**

</div>

---

## Навигация / Navigation

**🇷🇺 Русский**
- [Описание проекта](#-описание-проекта)
- [Архитектура модели](#-архитектура-модели)
- [Структура проекта](#-структура-проекта)
- [Установка и запуск](#-установка-и-запуск)
- [Данные и препроцессинг](#-данные-и-препроцессинг)
- [Аугментации](#-аугментации)
- [Пошаговое обучение](#-пошаговое-обучение)
- [Результаты](#-результаты)

**🇬🇧 English**
- [Project Description](#-project-description)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Data & Preprocessing](#-data--preprocessing)
- [Augmentations](#-augmentations)
- [Stepwise Training](#-stepwise-training)
- [Results](#-results)

---

# 🇷🇺 Русский

## 📌 Описание проекта

Проект реализует **автоэнкодер на базе архитектуры U-Net** для автоматического удаления текстовых водяных знаков с изображений. Модель обучается восстанавливать чистое изображение из входного изображения, поверх которого наложен полупрозрачный текстовый водяной знак.

**Ключевые особенности:**

- Синтетическое нанесение водяных знаков — любой текст, настраиваемая прозрачность и плотность
- 7 видов аугментаций изображений для устойчивости модели
- Пошаговое (chunked) обучение из 3 этапов с очисткой RAM между шагами — оптимально для Google Colab Free
- U-Net с skip-connections, BatchNorm и Dropout
- Наглядная демонстрация: оригинал / с водяным знаком / предсказание / карта шума

---

## 🧠 Архитектура модели

Используется **U-Net** — сверточная сеть с энкодером, боттлнеком и декодером, соединёнными skip-connections.

```
Вход (96×96×3)  ──► Conv×2 → BN → ReLU ──► skip_1 ──┐
                         ↓ MaxPool                    │
                    Conv×2 → BN → ReLU ──► skip_2 ──┐ │
                         ↓ MaxPool                   │ │
                    Conv×2 → BN → ReLU ──► skip_3 ─┐│ │
                         ↓ MaxPool                  ││ │
                    [Боттлнек + Dropout]             ││ │
                         ↓ UpSample                  ││ │
                    Concat(skip_3) → Conv×2 ─────────┘│ │
                         ↓ UpSample                    │ │
                    Concat(skip_2) → Conv×2 ────────────┘ │
                         ↓ UpSample                        │
                    Concat(skip_1) → Conv×2 ────────────────┘
                         ↓
                    Conv 1×1 → Sigmoid
                         ↓
               Выход (96×96×3) — чистое изображение
```

| Параметр         | Значение        |
|------------------|-----------------|
| Размер входа     | 96 × 96 × 3     |
| Base filters     | 32              |
| Кол-во параметров| ~1.2M           |
| Функция потерь   | MSE             |
| Метрика          | MAE, PSNR       |
| Оптимизатор      | Adam            |

---

## 📁 Структура проекта

```
watermark-removal/
│
├── app/
│   ├── data_processing.py      # Загрузка изображений, водяные знаки, датасеты
│   ├── model.py                # Определения архитектур (U-Net, light U-Net)
│   └── loader.py               # Скачивание и распаковка данных
│
├── notebooks/
│   └── train_test_stepwise.ipynb  # Основной ноутбук: обучение и демонстрация
│
├── data/
│   └── wm-nowm/
│       ├── train/no-watermark/    # Чистые изображения для обучения
│       └── valid/no-watermark/    # Чистые изображения для валидации
│
├── results/
│   └── results_10_examples.png   # Визуализация результатов (10 примеров)
│
├── step_1.weights.h5           # Веса после шага 1
├── step_2.weights.h5           # Веса после шага 2
├── step_3.weights.h5           # Веса после шага 3
├── watermark_removal.weights.h5 # Финальные веса
│
└── README.md
```

---

## 🚀 Установка и запуск

### Требования

```bash
pip install tensorflow>=2.12 pillow numpy matplotlib
```

### Google Colab

1. Загрузите ноутбук `notebooks/train_test_stepwise.ipynb` в Google Colab
2. Подключите Google Drive с данными или используйте встроенный загрузчик
3. Запустите все ячейки последовательно (`Runtime → Run all`)

### Локальный запуск

```bash
git clone https://github.com/himynameisartem/Autoencoder_Watermarks_Removal.git
cd watermark-removal
pip install -r requirements.txt
jupyter notebook notebooks/train_test_stepwise.ipynb
```

---

## 🖼️ Данные и препроцессинг

Модель обучается на изображениях **без водяных знаков** — знаки наносятся синтетически в процессе формирования датасета.

**Правильная логика пары (вход → цель):**

```
clean → augment(clean) → watermark(clean_aug)
                ↑ цель              ↑ вход модели
```

Это гарантирует, что модель учится убирать именно тот водяной знак, который был наложен.

**Параметры:**

| Параметр         | Значение   |
|------------------|------------|
| Размер изображения | 96 × 96  |
| Batch size       | 16         |
| Train / Val / Test | 90% / val_dir / 10% train |
| Формат           | RGB float32 [0, 1] |

---

## 🔧 Аугментации

Применяется **7 стохастических трансформаций** к чистому изображению перед нанесением водяного знака:

| # | Аугментация            | Параметры                          |
|---|------------------------|------------------------------------|
| 1 | Горизонтальное отражение | p = 1.0                          |
| 2 | Изменение яркости      | max_delta = 0.25                   |
| 3 | Изменение контраста    | lower=0.6, upper=1.4               |
| 4 | Изменение насыщенности | lower=0.4, upper=1.6, p=0.6        |
| 5 | Изменение оттенка      | max_delta=0.08, p=0.4              |
| 6 | JPEG-артефакты         | quality 60–95, p=0.4               |
| 7 | Гауссов шум            | std=0.03, p=0.3                    |

---

## 📚 Пошаговое обучение

Данные разбиваются на **3 чанка**. На каждом шаге:

```
Шаг 1  lr=1e-3 │ Чанк 1/3 → 5 эпох → save_weights(step_1.h5)
                │ gc.collect() + clear_session() + rebuild model
                │
Шаг 2  lr=5e-4 │ Чанк 2/3 → 5 эпох → save_weights(step_2.h5)
                │ gc.collect() + clear_session() + rebuild model
                │
Шаг 3  lr=2.5e-4│ Чанк 3/3 → 5 эпох → save_weights(step_3.h5)
                │
                └─► watermark_removal.weights.h5  (финал)
```

Между шагами используется `ReduceLROnPlateau` для дополнительной адаптации learning rate при стагнации loss.

---

## 📊 Результаты

Каждый пример из тестовой выборки визуализируется в 4 столбца:

| Оригинал | С водяным знаком | Предсказание | Карта шума |
|:---:|:---:|:---:|:---:|
| Чистое фото | Вход модели | Результат | WM − Predicted + 0.5 |

Карта шума показывает, что именно модель «вычла» из изображения — в идеале там виден только текст водяного знака.

**Целевые метрики:**

| Метрика | Значение     |
|---------|--------------|
| MSE     | < 0.005      |
| MAE     | < 0.04       |
| PSNR    | > 30 dB      |

---

---

# 🇬🇧 English

## 📌 Project Description

This project implements a **U-Net-based autoencoder** for automatic removal of text watermarks from images. The model learns to reconstruct a clean image from an input image that has a semi-transparent text watermark overlaid on it.

**Key features:**

- Synthetic watermark generation — any text, configurable opacity and density
- 7 types of image augmentations for model robustness
- 3-step chunked training with RAM cleanup between steps — optimized for Google Colab Free tier
- U-Net with skip-connections, BatchNorm and Dropout
- Clear visualization: original / watermarked / prediction / noise map

---

## 🧠 Model Architecture

The model is a **U-Net** — a convolutional network with an encoder, bottleneck and decoder connected via skip-connections.

```
Input (96×96×3)  ──► Conv×2 → BN → ReLU ──► skip_1 ──┐
                          ↓ MaxPool                    │
                     Conv×2 → BN → ReLU ──► skip_2 ──┐ │
                          ↓ MaxPool                   │ │
                     Conv×2 → BN → ReLU ──► skip_3 ─┐│ │
                          ↓ MaxPool                  ││ │
                     [Bottleneck + Dropout]           ││ │
                          ↓ UpSample                  ││ │
                     Concat(skip_3) → Conv×2 ─────────┘│ │
                          ↓ UpSample                    │ │
                     Concat(skip_2) → Conv×2 ────────────┘ │
                          ↓ UpSample                        │
                     Concat(skip_1) → Conv×2 ────────────────┘
                          ↓
                     Conv 1×1 → Sigmoid
                          ↓
                Output (96×96×3) — clean image
```

| Parameter        | Value           |
|------------------|-----------------|
| Input size       | 96 × 96 × 3     |
| Base filters     | 32              |
| Parameters       | ~1.2M           |
| Loss function    | MSE             |
| Metrics          | MAE, PSNR       |
| Optimizer        | Adam            |

---

## 📁 Project Structure

```
watermark-removal/
│
├── app/
│   ├── data_processing.py      # Image loading, watermarking, dataset builders
│   ├── model.py                # Architecture definitions (U-Net, light U-Net)
│   └── loader.py               # Data downloading and extraction
│
├── notebooks/
│   └── train_test_stepwise.ipynb  # Main notebook: training and visualization
│
├── data/
│   └── wm-nowm/
│       ├── train/no-watermark/    # Clean training images
│       └── valid/no-watermark/    # Clean validation images
│
├── results/
│   └── results_10_examples.png   # Output visualization (10 examples)
│
├── step_1.weights.h5           # Weights after step 1
├── step_2.weights.h5           # Weights after step 2
├── step_3.weights.h5           # Weights after step 3
├── watermark_removal.weights.h5 # Final model weights
│
└── README.md
```

---

## 🚀 Installation & Usage

### Requirements

```bash
pip install tensorflow>=2.12 pillow numpy matplotlib
```

### Google Colab

1. Upload `notebooks/train_test_stepwise.ipynb` to Google Colab
2. Mount Google Drive with your data or use the built-in data loader
3. Run all cells (`Runtime → Run all`)

### Local Setup

```bash
git clone https://github.com/himynameisartem/Autoencoder_Watermarks_Removal.git
cd watermark-removal
pip install -r requirements.txt
jupyter notebook notebooks/train_test_stepwise.ipynb
```

---

## 🖼️ Data & Preprocessing

The model trains on **clean images only** — watermarks are applied synthetically during dataset construction.

**Correct (input → target) pair logic:**

```
clean → augment(clean) → watermark(clean_aug)
               ↑ target              ↑ model input
```

This ensures the model learns to remove exactly the watermark that was applied.

**Parameters:**

| Parameter          | Value      |
|--------------------|------------|
| Image size         | 96 × 96    |
| Batch size         | 16         |
| Train / Val / Test | 90% / val_dir / 10% train |
| Format             | RGB float32 [0, 1] |

---

## 🔧 Augmentations

**7 stochastic transformations** are applied to the clean image before watermarking:

| # | Augmentation           | Parameters                         |
|---|------------------------|------------------------------------|
| 1 | Horizontal flip        | p = 1.0                            |
| 2 | Random brightness      | max_delta = 0.25                   |
| 3 | Random contrast        | lower=0.6, upper=1.4               |
| 4 | Random saturation      | lower=0.4, upper=1.6, p=0.6        |
| 5 | Random hue             | max_delta=0.08, p=0.4              |
| 6 | JPEG quality degradation | quality 60–95, p=0.4             |
| 7 | Gaussian noise         | std=0.03, p=0.3                    |

---

## 📚 Stepwise Training

Training data is split into **3 chunks**. At each step:

```
Step 1  lr=1e-3  │ Chunk 1/3 → 5 epochs → save_weights(step_1.h5)
                 │ gc.collect() + clear_session() + rebuild model
                 │
Step 2  lr=5e-4  │ Chunk 2/3 → 5 epochs → save_weights(step_2.h5)
                 │ gc.collect() + clear_session() + rebuild model
                 │
Step 3  lr=2.5e-4│ Chunk 3/3 → 5 epochs → save_weights(step_3.h5)
                 │
                 └─► watermark_removal.weights.h5  (final)
```

`ReduceLROnPlateau` is used as an additional callback to adaptively lower the learning rate when validation loss stagnates.

---

## 📊 Results

Each test sample is visualized in 4 columns:

| Original | Watermarked | Prediction | Noise Map |
|:---:|:---:|:---:|:---:|
| Clean photo | Model input | Model output | WM − Predicted + 0.5 |

The noise map shows what the model subtracted from the image — ideally only the watermark text is visible there.

**Target metrics:**

| Metric  | Target value |
|---------|--------------|
| MSE     | < 0.005      |
| MAE     | < 0.04       |
| PSNR    | > 30 dB      |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
