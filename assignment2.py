#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fisher Linear Discriminant for a 2-class classification problem with labels {-1, +1}.

Steps:
1. Load data_train, label_train, and data_test from .mat files.
2. Print shapes and, if needed, transpose data_train/data_test to have shape (d, N) and (d, M).
3. Train a Fisher linear discriminant:
   w = Sw^-1 (mu_pos - mu_neg),
   where mu_pos is the mean of samples labeled +1, mu_neg is the mean of samples labeled -1.
4. The bias w0 is chosen so that w^T x + w0 = 0 is the decision boundary, with equal priors.
5. Predict test labels using:
     if w^T x + w0 >= 0, label = +1
     else, label = -1
"""

import numpy as np
from scipy.io import loadmat


def load_data(data_train_file='data_train.mat',
              label_train_file='label_train.mat',
              data_test_file='data_test.mat'):
    """
    Loads data from .mat files and prints shapes for verification.

    Returns:
      data_train (d, N), label_train (N,), data_test (d, M)
    """
    # Load .mat files
    train_data_dict = loadmat(data_train_file)
    label_data_dict = loadmat(label_train_file)
    test_data_dict = loadmat(data_test_file)

    # Extract arrays
    # data_train originally (577, 5), label_train is (577,), data_test is (23, 5)
    data_train = train_data_dict['data_train']
    label_train = label_data_dict['label_train'].ravel()
    data_test = test_data_dict['data_test']

    # Print shapes
    print("=== Loaded Data Shapes ===")
    print("data_train.shape :", data_train.shape)
    print("label_train.shape:", label_train.shape)
    print("data_test.shape  :", data_test.shape)
    print("Unique labels    :", np.unique(label_train))
    print()

    # If data_train is (N, d) instead of (d, N), transpose it.
    # We want (d, N), i.e., columns = samples, rows = features.
    # In your case, data_train.shape is (577, 5), so we transpose to (5, 577).
    if data_train.shape[0] == label_train.shape[0] and data_train.shape[1] != label_train.shape[0]:
        print("Transposing data_train to make it (d, N).")
        data_train = data_train.T
        print("New data_train.shape :", data_train.shape)

    # Similarly ensure data_test is (d, M).
    # In your case, data_test.shape is (23, 5), so we transpose to (5, 23).
    if data_test.shape[0] != data_train.shape[0]:
        print("Transposing data_test to match dimension (d, M).")
        data_test = data_test.T
        print("New data_test.shape :", data_test.shape)

    return data_train, label_train, data_test


def fisher_linear_discriminant(data_train, label_train):
    """
    Trains a Fisher Linear Discriminant for a 2-class problem where labels are {-1, +1}.

    data_train: (d, N)
    label_train: (N,), each label is -1 or +1

    Returns:
      w   : weight vector (d,)
      w0  : bias (scalar)
    """
    data_train = data_train.astype(np.float64)

    # Indices for positive class (+1) and negative class (-1)
    pos_indices = np.where(label_train == +1)[0]
    neg_indices = np.where(label_train == -1)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        raise ValueError("One of the classes (+1 or -1) has no samples.")

    # Extract data for each class
    X_pos = data_train[:, pos_indices]  # (d, N_pos)
    X_neg = data_train[:, neg_indices]  # (d, N_neg)

    # Compute mean vectors for each class
    mu_pos = np.mean(X_pos, axis=1)  # (d,)
    mu_neg = np.mean(X_neg, axis=1)  # (d,)

    # Within-class scatter
    X_pos_centered = X_pos - mu_pos[:, None]  # (d, N_pos)
    X_neg_centered = X_neg - mu_neg[:, None]  # (d, N_neg)
    S_pos = X_pos_centered @ X_pos_centered.T  # (d, d)
    S_neg = X_neg_centered @ X_neg_centered.T  # (d, d)
    Sw = S_pos + S_neg

    # Solve for w:  Sw * w = (mu_pos - mu_neg)
    mean_diff = mu_pos - mu_neg  # (d,)
    # If Sw is singular, use np.linalg.pinv(Sw). Here we assume invertible for simplicity:
    w = np.linalg.inv(Sw) @ mean_diff

    # For classes +1 and -1 with equal priors,
    # a typical threshold sets w0 so that w^T x + w0 = 0 at the midpoint (mu_pos + mu_neg)/2
    # => w^T ((mu_pos + mu_neg)/2) + w0 = 0
    # => w0 = - w^T ((mu_pos + mu_neg)/2)
    # => w0 = -0.5 * w^T (mu_pos + mu_neg)
    w0 = -0.5 * np.dot(w, (mu_pos + mu_neg))

    return w, w0


def classify_fisher(data, w, w0):
    """
    Classifies data using the Fisher Linear Discriminant for classes {-1, +1}.

    data: (d, M)
    w   : (d,)
    w0  : scalar

    Returns:
      predicted_labels: array of shape (M,) in {-1, +1}
    """
    # Score for each sample
    scores = w.T @ data + w0  # shape (M,)

    # Decision boundary at 0
    # If score >= 0 => label = +1, else label = -1
    predicted_labels = np.where(scores >= 0, +1, -1)
    return predicted_labels


def main():
    # 1) Load data
    data_train, label_train, data_test = load_data(
        data_train_file='data_train.mat',
        label_train_file='label_train.mat',
        data_test_file='data_test.mat'
    )

    # 2) Train Fisher LDA
    w, w0 = fisher_linear_discriminant(data_train, label_train)

    print("=== Fisher Linear Discriminant Training ===")
    print(f"Weight vector w (shape={w.shape}):\n{w}")
    print(f"Bias term w0: {w0:.4f}")
    print("\nDecision rule:")
    print("Given a sample x, compute y = w^T x + w0.")
    print("If y >= 0, predict +1; otherwise, predict -1.\n")

    # 3) Predict on test data
    predicted_labels = classify_fisher(data_test, w, w0)

    print("=== Test Data Prediction ===")
    print(f"Predicted labels (for {predicted_labels.size} test samples):")
    print(predicted_labels)


if __name__ == "__main__":
    main()
