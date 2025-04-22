# Income Classification with MLP and AuxDrop-MLP

This repository implements and compares two deep learning models — a baseline Multi-Layer Perceptron (MLP) and an **AuxDrop-enhanced MLP** — for income classification using the UCI Adult Income Dataset. The pipeline includes data cleaning, preprocessing, and evaluation.

---

## Project Structure

```
.
├── adult_income_dataset/
│   ├── adult.csv               # Original raw dataset
│   ├── adult_clean.csv         # Cleaned and preprocessed dataset
│   ├── adult_train.csv         # Training split (80%)
│   ├── adult_test.csv          # Testing split (20%)
├── dataset.py                  # Data cleaning, processing, splitting, and custom Dataset class
├── model.py                    # MLP and AuxDrop-MLP model architectures
├── main_baseline.py            # Train and test script for baseline MLP
├── main_aux.py                 # Train and test script for AuxDrop-MLP
├── requirements.txt            # List of dependencies
```

---

##  Dataset

The dataset used is the [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult), commonly used for binary classification tasks to predict whether an individual's income exceeds $50K/year based on demographic attributes.

---

##  Data Processing (`dataset.py`)

- Fills missing values.
- Applies label encoding to categorical features.
- Scales numerical features using Min-Max normalization.
- Splits data into training (80%) and testing (20%) sets.
- Defines a custom `AdultDataset` class for PyTorch `DataLoader` usage.
- Separates **base features** and **auxiliary features** for model training:
  - **Base Features:** Used by both MLP and AuxDrop-MLP.
  - **Auxiliary Features:** Exclusively used by AuxDrop-MLP for feature dropout regularization.

---

##  Models (`model.py`)

- **MLP (Multi-Layer Perceptron):**  
  A standard feedforward neural network that takes base features as input.
  
- **AuxDrop-MLP:**  
  Extends the MLP by incorporating auxiliary features along with base features and applies adaptive feature dropout to improve generalization on sparse or irregular data.

---

##  How to Run

### 1. Set Up the Environment

```bash
conda create -n income-classifier python=3.11
conda activate income-classifier
pip install -r requirements.txt
```

### 2. Preprocess the Dataset

```bash
python dataset.py
```

This will generate:
- `adult_clean.csv`
- `adult_train.csv`
- `adult_test.csv`

### 3. Train and Test the Baseline MLP

```bash
python main_baseline.py
```

### 4. Train and Test the AuxDrop-MLP

```bash
python main_aux.py
```

Both scripts will:
- Load their respective model architectures
- Train for a set number of epochs
- Evaluate accuracy on the test set

---

##  Requirements

See `requirements.txt` for dependencies:
- `torch`
- `pandas`
- `scikit-learn`
- `tqdm`

---

##  Expected Outcome

You should observe that the AuxDrop-MLP performs better (or is more robust) when auxiliary features are partially missing or noisy, thanks to the adaptive dropout mechanism.

---

##  Contact

aman48anand@gmail.com
