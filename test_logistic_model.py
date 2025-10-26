import unittest
import time
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):
    """Unit-Tests für Logistic Regression Modell nach Ori Kohen Prinzipien."""

    @classmethod
    def setUpClass(cls):
        """Initiale Referenzlaufzeit (Baseline)."""
        print("\n" + "=" * 70)
        print("=== INITIALER REFERENZLAUF (setUpClass) ===")
        print("=" * 70 + "\n")

        df = load_data()
        train_model(df)
        cls.reference_time = get_last_timing('train_model')

    # ------------------------------------------------------------------
    def test_1_predict(self):
        """TESTFALL 1: Vorhersagequalität prüfen."""
        print("\n" + "=" * 70)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 70 + "\n")

        df = load_data()
        model = train_model(df)
        acc, cm, report = evaluate_model(model)

        print(f"Genauigkeit (Accuracy): {acc:.2f}")
        print("Confusion Matrix:")
        print(cm)
        print("\nKlassifikationsbericht (Auszug):")
        print(report)
        print(f"Final Accuracy: {acc:.2f}\n")

        self.assertGreaterEqual(acc, 0.9, "❌ Accuracy unter 0.9!")
        print("Ergebnis: TESTFALL 1 PASSED ✅")

    # ------------------------------------------------------------------
    def test_2_train_runtime(self):
        """TESTFALL 2: Laufzeit der Trainingsfunktion prüfen."""
        print("\n" + "=" * 70)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 70 + "\n")

        print("💬 Hinweis: Logeinträge zeigen die Abläufe beider Testfälle.")
        print("Alles vor dem Punkt ('.') gehört zu Testfall 1 (predict),")
        print("ab '.2025-…' beginnt Testfall 2 (fit/train_runtime).\n")

        df = load_data()
        train_model(df)
        runtime = get_last_timing('train_model')  # <-- direkt aus Decorator
        limit = self.reference_time * 1.2

        print("\nLaufzeitanalyse:")
        print(f" - Referenzlaufzeit: {self.reference_time:.4f} sec")
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime > limit:
            diff = runtime - limit
            print("❌ Laufzeit überschreitet das Limit!")
            print("Ergebnis: TESTFALL 2 FAILED ❌")
            self.fail(
                f"❌ Trainingslaufzeit überschreitet das erlaubte Limit: "
                f"Aktuell {runtime:.4f}s > {limit:.4f}s "
                f"(Referenz: {self.reference_time:.4f}s). Überschreitung: +{diff:.4f}s."
            )
        else:
            print("Ergebnis: TESTFALL 2 PASSED ✅")


if __name__ == "__main__":
    print("\n=== Starte Unit-Tests ===")
    unittest.main(verbosity=2)
