# Spam Email Detection

A Python project to classify emails as spam or ham using a Support Vector Machine (SVM) and a Kaggle dataset (`spam_Emails_data.csv`).

---

## Project Structure

```
spam_email_detection/
├── data/
│   ├── raw/spam_Emails_data.csv         # Kaggle dataset
│   └── processed/             # Preprocessed data (optional)
├── models/                    # Trained models
├── src/                       # Python scripts
├── notebooks/                 # Jupyter notebooks for exploration
├── README.md                  # This file
├── requirements.txt           # Dependencies
└── .gitignore                 # Git ignore file
```

---

## Setup

1. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

2. **Place dataset:**

    - Copy `emails.csv` to `data/raw/`.

3. **Run preprocessing:**

    ```sh
    python src/data_preprocessing.py
    ```

4. **Train model:**

    ```sh
    python src/model_training.py
    ```

5. **Make predictions:**

    ```sh
    python src/predict.py
    ```

6. **Explore data:**
    - Open `notebooks/exploratory_analysis.ipynb` in Jupyter Notebook.

---

## Dataset

-   **Source:** [Kaggle (emails.csv)](https://drive.google.com/drive/folders/1N-PdR3u8A73PrPDhnNC-icjuT6OhtYW9?usp=sharing)
-   **Description:** Contains word frequency counts for emails, with labels (`0=ham`, `1=spam`).

---

## Usage

-   **Preprocess data:** `src/data_preprocessing.py`
-   **Train and evaluate SVM:** `src/model_training.py`
-   **Predict on new data:** `src/predict.py`
-   **Visualize data:** `notebooks/exploratory_analysis.ipynb`

---

## Notes

-   Ensure `emails.csv` is in `data/raw/`.
-   The SVM model is saved as `models/svm_spam_model.pkl`.
-   Adjust the label column in `data_preprocessing.py` if needed.
