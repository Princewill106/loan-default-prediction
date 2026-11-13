"""Regenerates sample_loans.csv (synthetic, reproducible, ~22% default rate)."""
import csv
import math
import random

rng = random.Random(11)
with open("sample_loans.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["monthly_income", "loan_amount", "tenor_months", "existing_debt", "previous_defaults", "account_age_months", "defaulted"])
    for _ in range(400):
        inc = rng.lognormvariate(math.log(180000), 0.5)
        amt = inc * rng.uniform(0.5, 5)
        tenor = rng.choice([3, 6, 9, 12])
        debt = inc * rng.uniform(0, 0.6)
        prev = rng.choices([0, 1, 2], [0.8, 0.15, 0.05])[0]
        age = rng.randint(1, 96)
        z = -2.2 + 0.45 * (amt / inc / tenor * 3) + 2.5 * (debt / inc) + 1.1 * prev - 0.015 * age
        w.writerow([round(inc), round(amt), tenor, round(debt), prev, age, int(rng.random() < 1 / (1 + math.exp(-z)))])
