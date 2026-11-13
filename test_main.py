import random
import subprocess
import sys
import unittest

import main


class LogisticTests(unittest.TestCase):
    def test_auc_perfect(self):
        self.assertEqual(main.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_learns_separable(self):
        rng = random.Random(0)
        X = [[rng.gauss(2 if c else -2, 1)] for c in (0, 1) * 50]
        y = [0, 1] * 50
        w, b = main.train(X, y, epochs=300)
        self.assertGreater(main.auc(y, main.predict(w, b, X)), 0.95)

    def test_sample_quality(self):
        subprocess.run([sys.executable, "make_sample.py"], check=True)
        X, y = main.load("sample_loans.csv")
        tr, te = main.split(X, y)
        Xtr, s = main.standardise([X[i] for i in tr])
        w, b = main.train(Xtr, [y[i] for i in tr], epochs=400)
        Xte, _ = main.standardise([X[i] for i in te], s)
        self.assertGreater(main.auc([y[i] for i in te], main.predict(w, b, Xte)), 0.65)


if __name__ == "__main__":
    unittest.main()
