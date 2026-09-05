# cd "/Users/gulcetahtasiz/Desktop/staj code/microsoft"
# source venv/bin/activate
# python sailing.py

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import predefined metric functions to evaluate performance
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_recall_fscore_support, hamming_loss
)




# ================== CONFIGURATION ==================
CSV_PATH  = "sailing_synthetic_groundtruth_150_named.csv"  # Path to the CSV dataset
EPOCHS    = 100      # Maximum number of training epochs
PATIENCE  = 5        # Stop training if validation loss doesn't improve for 5 epochs
LR        = 1e-3     # Learning rate for optimizer
BATCH     = 32       # Batch size for DataLoader
SEED      = 3        # Random seed for reproducibility
# ====================================================


# List of possible sailing crew roles 
ROLES = ["basustu","direkdibi","piyano","trim1","trim2","anayelken","dumen"]


def hr(s=""):
    """Helper function to print a visual separator in the logs."""
    print("\n" + "="*12 + f" {s} " + "="*12)






# ================== MLP MODEL DEFINITION ==================
class RoleMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for role prediction.
    Input: feature vector
    Output: multi-label prediction over crew roles
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),    # First hidden layer with 64 neurons
            nn.Linear(64, 32), nn.ReLU(),        # Second hidden layer with 32 neurons
            nn.Dropout(0.2),                     # Dropout for regularization
            nn.Linear(32, out_dim)               # Output layer for role predictions
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)







def main():

    # Set all random seeds for reproducible experiments
    torch.manual_seed(SEED); np.random.seed(SEED)


    # 1) Load
    path = Path(CSV_PATH)
    if not path.exists():
        # Fail early if the CSV is missing
        raise FileNotFoundError(f"CSV bulunamadı: {path}")


    hr("Load CSV")
    df = pd.read_csv(path)


    # 2) Features
    # Define the numeric feature columns expected in the dataset
    feature_cols = [
        "age","height_cm","weight_kg","decision_ms","attention",
        "upper_watts","grip_kg","plank_sec","agility_ms","years_exp"
    ]

    # If a categorical 'sex' column exists, map it to a binary numeric feature
    if "sex" in df.columns:
        df["sex_bin"] = df["sex"].map({"F":0, "M":1}).fillna(0).astype(float)
        feature_cols = ["sex_bin"] + feature_cols  # Prepend to keep a known order


    # Sanity check: ensure all feature columns are present
    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Eksik feature kolon(lar): {missing_feat}")


    # 3) Labels
    # Verify all label columns exist and convert to clean integer array (0/1)
    missing_lab = [c for c in ROLES if c not in df.columns]
    if missing_lab:
        raise ValueError(f"Label kolonları eksik: {missing_lab}")

    Y_raw = (
        df[ROLES]
        .apply(pd.to_numeric, errors="coerce")  # coerce invalid to NaN
        .fillna(0)                               # fill NaN with 0
        .astype(int)                             # ensure integer labels
        .values
    )

    # Extract feature matrix (X) and label matrix (Y)
    X = df[feature_cols].astype(float).values
    Y = Y_raw

    # 4) Split + scale  (train / val / test)
    hr("Train/Val/Test split + Standardize")

    VAL_SIZE  = 0.20   # validation fraction (of the full dataset)
    TEST_SIZE = 0.20   # test fraction (of the full dataset)

    # First, hold out the test set (never used for model/threshold selection)
    X_tr_full, X_te, Y_tr_full, Y_te = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=SEED
    )

    # From the remaining training pool, carve out a validation set
    # Compute relative validation size w.r.t. the remaining (1 - TEST_SIZE) portion
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)  # e.g., 0.20 / 0.80 = 0.25
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tr_full, Y_tr_full, test_size=val_rel, random_state=SEED
    )

    # Fit the scaler ONLY on the training set to avoid data leakage,
    # then apply the transform to val/test
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)


    # Convert numpy arrays to PyTorch tensors
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    Ytr = torch.tensor(Y_tr, dtype=torch.float32)

    Xva = torch.tensor(X_va, dtype=torch.float32)
    Yva = torch.tensor(Y_va, dtype=torch.float32)

    Xte = torch.tensor(X_te, dtype=torch.float32)
    Yte = torch.tensor(Y_te, dtype=torch.float32)

    # Build DataLoaders
    train_ld = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    val_ld   = DataLoader(TensorDataset(Xva, Yva), batch_size=max(64,BATCH), shuffle=False)

    # test loader is prepared for final reporting only (never used in training)
    test_ld  = DataLoader(TensorDataset(Xte, Yte), batch_size=max(64,BATCH), shuffle=False)








        
    # 5) Model + loss + opt
    hr("Train MLP")

    # Instantiate the MLP:
    #  - in_dim: number of input features
    #  - out_dim: number of labels (multi-label problem)
    model = RoleMLP(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1])

    # Use BCEWithLogitsLoss for multi-label classification:
    #   expects raw logits; applies a sigmoid internally and then computes BCE.
    crit  = nn.BCEWithLogitsLoss()

    # AdamW optimizer with a small weight decay for better generalization
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)



    # Helper: compute average loss over a DataLoader (e.g., validation set)
    def eval_loss(loader):
        model.eval()  # switch to eval mode (disables dropout, etc.)
        tot, n = 0.0, 0
        with torch.no_grad():  # no gradients needed during evaluation
            for xb, yb in loader:
                loss = crit(model(xb), yb)
                tot += loss.item() * xb.size(0)  # sum loss weighted by batch size
                n   += xb.size(0)
        return tot / max(n, 1)  # safe average

    best_val      = 1e9        # track the best (lowest) validation loss seen
    overfit_count = 0          # early-stopping counter
    best_state    = None       # to store the best model weights

    train_losses, val_losses = [], []

    for ep in range(EPOCHS):
        model.train()          # training mode
        batch_loss = 0.0

        # ----- Training loop (one epoch) -----
        for xb, yb in train_ld:
            opt.zero_grad()                    # reset gradients
            loss = crit(model(xb), yb)         # forward + loss
            loss.backward()                    # backprop
            opt.step()                         # optimizer update
            batch_loss += loss.item()

        # Average training loss for logging
        train_loss = batch_loss / len(train_ld)
        train_losses.append(train_loss)

        # ----- Validation -----
        v = eval_loss(val_ld)
        val_losses.append(v)
        print(f"Epoch {ep:03d} | train_loss={train_loss:.4f} | val_loss={v:.4f}")

        # ----- Early Stopping & Best Checkpoint -----
        # Save model weights if validation loss improved by a small margin (1e-4)
        if v < best_val - 1e-4:
            best_val = v
            overfit_count = 0
            # Deep-copy state dict tensors (clone + detach to avoid any references)
            best_state = {k: t.clone().detach() for k, t in model.state_dict().items()}
        else:
            overfit_count += 1
            # Stop if no improvement for PATIENCE epochs
            if overfit_count >= PATIENCE:
                print("Early stop.")
                break

    # Restore the best-performing weights before leaving training
    if best_state:
        model.load_state_dict(best_state)








    # 6) Validation scores 
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(Xva)).numpy()  # shape: (N_val, 7)
    
    # === Role-based threshold selection (simplified version) ===
    hr("Role-based thresholds")

    # Candidate thresholds to try (from 0.30 to 0.80, step 0.05)
    thr_grid = np.arange(0.30, 0.81, 0.05)
    best_thr_per_role = []   # will store the best threshold for each role

    for j, role in enumerate(ROLES):
        # Extract ground-truth labels and predicted scores for this role
        y_true  = Y_va[:, j]        # true binary labels (0/1)
        y_score = val_scores[:, j]  # predicted probabilities in [0,1]

        best_f1  = -1.0
        best_thr = 0.50             # default threshold

        # Loop through candidate thresholds and compute F1
        for t in thr_grid:
            y_pred = (y_score >= t).astype(int)            # classify with threshold t
            f1     = f1_score(y_true, y_pred, zero_division=0)  # compute F1 score
            if f1 > best_f1:                               # keep if better than current best
                best_f1  = f1
                best_thr = t

        # Save the best threshold for this role
        best_thr_per_role.append(best_thr)
        print(f"{role:10s} -> best_thr={best_thr:.2f} | best_F1={best_f1:.3f}")

    # Convert best thresholds into numpy vector for broadcasting
    thr_vec = np.array(best_thr_per_role)   # shape: (7,)





    # 7) Evaluation Metrics
    # === Evaluate TEST set using thresholds selected from VAL ===
    hr("TEST metrics (using thresholds chosen on VAL)")
    with torch.no_grad():
        test_scores = torch.sigmoid(model(Xte)).numpy()  # shape: (N_test, 7)

    # Apply per-role thresholds
    pred_test = (test_scores >= thr_vec).astype(int)


    # Subset accuracy: strict metric, requires all labels correct for a sample
    subset_acc_te = accuracy_score(Y_te, pred_test)
    print(f" Subset accuracy (exact match): {subset_acc_te:.3f}")


    # Per-label accuracy: report accuracy for each individual role
    label_wise_acc_te = {
        ROLES[j]: round(accuracy_score(Y_te[:, j], pred_test[:, j]), 3)
        for j in range(Y_te.shape[1])
    }
    print("[TEST] Label-wise accuracy:", label_wise_acc_te)


    # Micro-averaged precision, recall, and F1 (global over all labels/samples)
    prec_micro_te, rec_micro_te, f1_micro_te, _ = precision_recall_fscore_support(
        Y_te, pred_test, average="micro", zero_division=0
    )

    # Macro-averaged precision, recall, and F1 (average over roles equally)
    prec_macro_te, rec_macro_te, f1_macro_te, _ = precision_recall_fscore_support(
        Y_te, pred_test, average="macro", zero_division=0
    )

    print(f"Micro  Precision: {prec_micro_te:.3f}")
    print(f"Micro  Recall: {rec_micro_te:.3f}")
    print(f"Micro  F1: {f1_micro_te:.3f}")

    print(f"Macro  Precision: {prec_macro_te:.3f}")
    print(f"Macro  Recall: {rec_macro_te:.3f}")
    print(f"Macro  F1: {f1_macro_te:.3f}")


    # Compute F1 score for each role individually
    per_role_f1_te = [f1_score(Y_te[:, j], pred_test[:, j], zero_division=0)
                      for j in range(Y_te.shape[1])]
    print("Per-role F1:", dict(zip(ROLES, [round(x, 3) for x in per_role_f1_te])))


    # compute AUC for each role(how model works for different thresholds)
    try:
        per_role_auc_te = [roc_auc_score(Y_te[:, j], test_scores[:, j])
                           for j in range(Y_te.shape[1])]
        print("Per-role AUC:", dict(zip(ROLES, [round(x, 3) for x in per_role_auc_te])))
    except Exception as e:
        print("AUC not computed:", e)


    # Hamming loss: fraction of misclassified labels across all samples/roles
    hl_te = hamming_loss(Y_te, pred_test)
    print(f"Hamming loss: {hl_te:.3f}")






    # 8) Score full pool + Hungarian
    hr("Score FULL pool + Hungarian assignment")

    X_alldata = scaler.transform(df[feature_cols].astype(float).values)

    # Get sigmoid probabilities for every athlete and every role
    with torch.no_grad():
        all_scores = torch.sigmoid(model(torch.tensor(X_alldata, dtype=torch.float32))).numpy() 


    # Build a tidy scores table to export and inspect later
    score_cols = [f"score_{r}" for r in ROLES]
    scores_df = pd.DataFrame(all_scores, columns=score_cols)

    # Keep IDs/ next to scores for readability
    if "athlete_id" in df.columns:
        scores_df.insert(0, "athlete_id", df["athlete_id"].values)


    # Save raw scores to CSV for auditing / visualization
    scores_out = path.with_name("scores_from_mlp.csv")
    scores_df.to_csv(scores_out, index=False)
    print(f"Saved: {scores_out.name}")


    # Hungarian assignment (maximize total score):
    # The algorithm minimizes cost, so we convert scores to costs by negation
    cost = -scores_df[score_cols].values  # shape: (N_athletes, N_roles)
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build the team from the optimal assignment
    # (linear_sum_assignment already ensures a 1-1 matching; no need to re-check duplicates)
    used_roles, team = set(), []
    for r, c in sorted(zip(row_ind, col_ind), key=lambda x: x[1]):  # sort by role index
        if c not in used_roles and len(used_roles) < len(ROLES):
            used_roles.add(c)
            team.append({
                "role": ROLES[c],
                # safer: read id/name from scores_df (inserted above)
                "athlete_id": scores_df.iloc[r]["athlete_id"] if "athlete_id" in scores_df.columns else None,
                "score": float(scores_df.iloc[r][f"score_{ROLES[c]}"])
            })


    # Present the final recommended team
    team_df = pd.DataFrame(team).sort_values("role")
    team_out = path.with_name("team_selection.csv")
    team_df.to_csv(team_out, index=False)
    print(f"Saved: {team_out.name}")
    print("\n=== Recommended Team ===")
    print(team_df.to_string(index=False))


main()
