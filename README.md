# Neural-Network Surrogate for Airfoil Drag

A neural network that predicts the **drag coefficient (Cd)** of an airfoil from its **shape** and **angle of attack**, so you don't have to run the XFOIL solver for every case.

---

## 1. Problem statement

XFOIL computes the drag of an airfoil, but it has to be run separately for every shape and every angle, and sometimes it fails to converge.
The goal of this project is to train a model that:

- takes an airfoil shape and an angle of attack,
- predicts Cd for an airfoil it has never seen before,
- does this instantly, in a single forward pass of a neural network.

| | |
|---|---|
| **Input** | airfoil geometry + angle of attack α (−2° to 6°) |
| **Output** | drag coefficient Cd |
| **Fixed condition** | Reynolds number = 1 × 10⁶ |

---

## 2. Dataset

**Source:** airfoil shapes from the UIUC Airfoil Coordinates Database. Each shape was run in **XFOIL 6.99** at 9 angles (−2° to 6°).

| Step | Count |
|---|---|
| Coordinate files scanned | 1,011 |
| Valid, unique airfoils kept | 1,000 |
| XFOIL runs (1,000 airfoils × 9 angles) | 9,000 |
| Runs that converged (rows in the dataset) | **8,251** from **988 airfoils** |

**Data-quality issues and how they were handled**

| Issue | Count | Handling |
|---|---|---|
| Multi-element airfoils | 4 files | excluded |
| Duplicate shapes | 4 files | skipped |
| Broken geometry (open surface, implausible thickness) | 3 files | skipped |
| XFOIL did not converge | 599 runs | dropped and logged |
| XFOIL timed out | 103 runs | dropped and logged |
| XFOIL output missing or ambiguous | 47 runs | dropped and logged |

Every failure is recorded in `data/processed/dataset_failures_1000.csv`.

**Split:** the data is split **by airfoil**, so no shape appears in more than one set. Validation and test scores therefore measure performance on unseen shapes.

| Set | Airfoils | Rows |
|---|---|---|
| Train | 690 | 5,744 |
| Validation | 149 | 1,252 |
| Test | 149 | 1,255 |

The dataset was grown in three stages: 296 → 490 → 988 airfoils.

---

## 3. Methodology

1. **Clean and resample the geometry.** Coordinate files have different numbers of points, so each airfoil is normalised by its chord and interpolated onto the same 100 x-positions on the upper and lower surfaces. This gives **200 surface heights** per airfoil.
2. **Label with XFOIL.** Run XFOIL at each angle and keep only converged results.
3. **Split by airfoil** (70 / 15 / 15 %).
4. **Baseline.** Fit a linear regression to check whether a simple model is enough.
5. **Experiments.** Train 17 MLP versions, changing one thing at a time (network size, regularisation, learning rate, batch size, input features, target transform, loss function, dataset size).
6. **Fix the input representation.** Compress the 200 heights into **24 PCA modes**.
7. **Final model.** Train 5 copies with different random seeds and average their predictions. Score the test set **once**, at the end.

**Preprocessing used by the final model**

- Geometry: 200 heights → standardise → PCA (24 modes, fitted on training airfoils only)
- Inputs: 24 PCA modes + α → standardise → clamp to the training range ± 25 %
- Target: log(Cd) → standardise
- Sample weights: 1.5 / 2 / 3 for Cd ≥ 0.010 / 0.015 / 0.020, so the rare high-drag cases count more

---

## 4. Model architecture

A simple fully connected network (MLP), written in Keras:

```
Input   (25)   24 PCA shape modes + angle of attack
Dense   (256)  ReLU, L2 = 1e-3
Dropout (0.15)
Dense   (128)  ReLU, L2 = 1e-3
Dense   (1)    standardised log(Cd)
```

| Choice | Setting | Why |
|---|---|---|
| Loss | weighted MSE on standardised log(Cd) | measures relative error; high-drag cases are rare and need extra weight |
| Optimizer | Adam (learning rate 1e-3), batch 64 | stable; chosen by experiment |
| Training control | early stopping (patience 30), learning rate halved on plateaus | stops before overfitting |
| Regularisation | L2 + dropout | only 690 distinct training shapes |
| Ensemble | 5 seeds, averaged | removes seed-to-seed noise |
| Metrics | R², MAE, RMSE, and MAE on high-drag cases (Cd > 0.015) | overall fit, error in Cd units, and accuracy on the rare high-drag cases |

Trainable weights per model: **39,681**.

---

## 5. Main results

All scores are on airfoils the model never saw during training.

**Baseline vs first neural network** (500-airfoil set, validation)

| Model | R² | MAE |
|---|---|---|
| Linear regression | −0.578 | 0.00190 |
| First MLP (V1) | 0.654 | 0.00090 |

**Experiments on the 500-airfoil set** (validation R²; only the changes that were kept)

| V1 MLP | V2 scaled target | V3 L2 + dropout | V4 wider | V5 lower learning rate | V8 batch 64 |
|---|---|---|---|---|---|
| 0.654 | 0.691 | 0.707 | 0.714 | 0.735 | **0.746** |

**Experiments on the 1000-airfoil set** (validation)

| Model | R² | MAE | RMSE | High-drag MAE |
|---|---|---|---|---|
| V12: V8 recipe on new data | 0.625 | 0.000783 | 0.001742 | – |
| V14: log(Cd) target | 0.632 | 0.000746 | 0.001725 | 0.0065 |
| V15: + high-drag weights | 0.655 | 0.000759 | 0.001672 | 0.0062 |
| **Final: PCA + 5-seed ensemble** | **0.712** | **0.000661** | **0.001527** | **0.0051** |

The 1000-airfoil validation set is harder than the 500-airfoil one: it contains GOE09K, whose highest drag is above anything in training. So compare scores only within the same set.

**Final model on the test set (scored once)**

| | R² | MAE | RMSE |
|---|---|---|---|
| Test set | **0.919** | 0.00062 | 0.00119 |
| Test set without ah93w480b (48 % thick) | 0.818 | 0.00058 | 0.00098 |

**Changes that did not help (not kept):** learning-rate schedule (V6), batch size 16 (V7), weaker L2 (V9), less dropout (V10), extra thickness and camber features (V11), a smaller network (V13), α² and α³ inputs (V16), Huber loss (V17).

---

## 6. Conclusions

1. **A neural network is needed.** Linear regression scores worse than simply predicting the average (R² −0.58); the first MLP reaches 0.65.
2. **How the data is represented mattered more than tuning.** Scaling the target (+0.04 R²), using log(Cd) with high-drag weights (+0.03) and PCA geometry with an ensemble (+0.06, −13 % MAE) gave the real gains. Batch size and dropout changes were within random noise (about ±0.02–0.03 R²).
3. **Raw coordinates were the main weakness.** The 200 heights are almost identical to their neighbours (median correlation 0.9998), so the network memorised shapes: train R² 0.956 vs validation 0.632. 24 PCA modes keep 99.991 % of the shape information with far fewer inputs (201 → 25).
4. **More distinct shapes help.** Going from 172 to 690 training airfoils raised R² from 0.46 to 0.62 on the same validation set.
5. **Limitations:**
   - The model is reliable inside the range of the training data, not for extreme shapes. GOE09K at 6° is predicted at 0.026, against an XFOIL value of 0.055.
   - It covers one Reynolds number (1 × 10⁶), angles from −2° to 6°, and Cd only.
   - The labels are XFOIL results, not wind-tunnel measurements.

---

## 7. How to reproduce

### 7.1 Set up

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Option A: use the included data (quickest)

The processed datasets are already in `data/processed/`. Open the notebooks and run all cells, in order:

1. `notebooks/01_V1_to_V12_500_airfoils.ipynb`: baseline and V1–V12
2. `notebooks/02_V13_to_V17_1000_airfoils.ipynb`: error analysis, V13–V17, and the final PCA ensemble

The notebooks also run on Google Colab: upload the CSV files when the first cell asks for them.

### 7.3 Option B: rebuild the dataset from scratch

1. Download the **UIUC airfoil database** (Selig format) and put the `.dat` files in `data/raw/airfoils/coord_seligFmt/`.
2. Download **XFOIL 6.99** and place `xfoil.exe` at `~/Downloads/XFOIL6.99/xfoil.exe`, or change `XFOIL_EXECUTABLE` in `src/build_dataset.py`.
3. Generate the data and split it (this takes a while: 9,000 XFOIL runs):

```powershell
cd src
python build_dataset.py
python split_dataset_1000.py
```

4. Run the notebooks as in Option A.

`build_dataset.py` is set to 1,000 airfoils. To rebuild the 500-airfoil set used in notebook 01:
- set `MAX_AIRFOILS = 500` and the output names to `..._500.csv` in `build_dataset.py`,
- then run `split_dataset.py`.

### 7.4 Reproducibility notes

- Fixed random seeds: 42 for single runs; 42, 1, 2, 3, 4 for the final ensemble.
- Scalers and PCA are fitted on training data only.
- The split is checked in code to have no airfoil overlap.
- Notebook outputs are from the original Colab runs. Re-running on different hardware can shift single-seed R² by about ±0.02.

