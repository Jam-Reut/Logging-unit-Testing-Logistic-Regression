import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

logger = logging.getLogger(__name__)


class TestLogisticRegressionModel(unittest.TestCase):
    """Variante B — Logging-basierte Version (professionell & kompakt)."""

    @classmethod
    def setUpClass(cls):
        #print("\n" + "=" * 70)
        #print("=== INITIALER REFERENZLAUF (setUpClass) ===")
        #print("=" * 70 + "\n")

        df = load_data()
        train_model(df)
        cls.reference_time = get_last_timing('train_model')

    # --------------------------------------------------------------
    def test_1_predict(self):
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

    # --------------------------------------------------------------
    def test_2_train_runtime(self):
        print("\n" + "=" * 70)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 70 + "\n")

        df = load_data()
        train_model(df)
        runtime = get_last_timing('train_model')
        limit = self.reference_time * 1.2

        status = "PASS" if runtime <= limit else "FAIL"
        logger.info(
            f"[TIMING] Aktuelle Laufzeit ={runtime:.4f}s | Limit(120%)={limit:.4f}s | Reference={self.reference_time:.4f}s | status={status}"
        )

        if runtime > limit:
            diff = runtime - limit
            self.fail(
                f"❌ Trainingslaufzeit zu hoch: {runtime:.4f}s > {limit:.4f}s "
                f"(Referenz: {self.reference_time:.4f}s, +{diff:.4f}s über Limit)"
            )
        else:
            print("✅ TESTFALL 2 PASSED — Laufzeit im Limit.")


if __name__ == "__main__":
    print("\n=== Starte Unit-Tests ===")
    unittest.main(verbosity=2)
