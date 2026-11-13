"""Loan Default Prediction: logistic regression from scratch."""
import argparse
import csv
import math
import random

FEATURES = ["monthly_income", "loan_amount", "tenor_months", "existing_debt", "previous_defaults", "account_age_months"]
COST_FN, COST_FP = 5.0, 1.0


def load(path):
    rows = list(csv.DictReader(open(path)))
    return [[float(r[f]) for f in FEATURES] for r in rows], [int(r["defaulted"]) for r in rows]


def standardise(X, stats=None):
    if stats is None:
        cols = list(zip(*X))
        stats = [(sum(c) / len(c), (sum((x - sum(c) / len(c)) ** 2 for x in c) / len(c)) ** 0.5 or 1) for c in cols]
    return [[(x - m) / s for x, (m, s) in zip(row, stats)] for row in X], stats


def sigmoid(z):
    return 1 / (1 + math.exp(-max(min(z, 35), -35)))


def train(X, y, lr=0.1, epochs=1500, l2=0.01):
    w, b, n = [0.0] * len(X[0]), 0.0, len(X)
    for _ in range(epochs):
        gw, gb = [0.0] * len(w), 0.0
        for xi, yi in zip(X, y):
            err = sigmoid(sum(a * c for a, c in zip(w, xi)) + b) - yi
            for j, v in enumerate(xi):
                gw[j] += err * v
            gb += err
        w = [wj - lr * (g / n + l2 * wj) for wj, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def predict(w, b, X):
    return [sigmoid(sum(a * c for a, c in zip(w, xi)) + b) for xi in X]


def auc(y, p):
    pos = [s for s, t in zip(p, y) if t == 1]
    neg = [s for s, t in zip(p, y) if t == 0]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def confusion(y, p, t):
    tp = sum(1 for a, s in zip(y, p) if a == 1 and s >= t)
    fp = sum(1 for a, s in zip(y, p) if a == 0 and s >= t)
    fn = sum(1 for a, s in zip(y, p) if a == 1 and s < t)
    tn = len(y) - tp - fp - fn
    return tp, fp, fn, tn


def split(X, y, seed=7):
    rng = random.Random(seed)
    tr, te = [], []
    for cls in (0, 1):
        idx = [i for i, v in enumerate(y) if v == cls]
        rng.shuffle(idx)
        cut = int(len(idx) * 0.75)
        tr += idx[:cut]; te += idx[cut:]
    return tr, te


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="sample_loans.csv")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    X, y = load(a.input)
    tr, te = split(X, y)
    Xtr, stats = standardise([X[i] for i in tr])
    Xte, _ = standardise([X[i] for i in te], stats)
    ytr, yte = [y[i] for i in tr], [y[i] for i in te]
    w, b = train(Xtr, ytr)
    p = predict(w, b, Xte)
    print(f"Trained on {len(tr)} loans, tested on {len(te)} (default rate {sum(y) / len(y):.0%})")
    print(f"ROC-AUC: {auc(yte, p):.3f}\n")
    print("Coefficients (per 1 SD change):")
    for f, c in sorted(zip(FEATURES, w), key=lambda x: -abs(x[1])):
        print(f"  {f:<20} {c:+.3f}  odds x{math.exp(c):.2f}")
    best = min((t / 100 for t in range(5, 96)),
               key=lambda t: (lambda c: COST_FN * c[2] + COST_FP * c[1])(confusion(yte, p, t)))
    for t, label in ((0.5, "default 0.50"), (best, f"cost-optimal {best:.2f}")):
        tp, fp, fn, tn = confusion(yte, p, t)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"\nThreshold {label}: accuracy {(tp + tn) / len(yte):.0%}, precision {prec:.0%}, recall {rec:.0%}")
        print(f"  confusion: TP {tp}  FP {fp}  FN {fn}  TN {tn}   expected cost {COST_FN * fn + COST_FP * fp:.0f}")


if __name__ == "__main__":
    main()
