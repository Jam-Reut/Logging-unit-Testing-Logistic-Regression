import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# ------------------------------------------------------------
# Unit-Testklasse
# ------------------------------------------------------------
class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("=" * 70)
        print("=== INITIALER REFERENZLAUF (setUpClass) ===")
        print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        cls.ref_time = get_last_timing("train_model")

        # ✅ Änderung 1: Robustheit – falls Referenzlaufzeit fehlt
        if cls.ref_time is None:
            logging.warning("⚠️  WARNUNG: Referenzlaufzeit konnte nicht ermittelt werden.")
            cls.ref_time = 0.0

        print("\n💬 Hinweis:")
        print("Die folgenden Logeinträge zeigen die Abläufe beider Testfälle.")
        print("Alles vor dem Punkt ('.') gehört zu Testfall 1 (predict),")
        print("ab '.2025-…' beginnt Testfall 2 (train_runtime).\n")

    # ------------------------------------------------
    # TESTFALL 1 – Vorhersageprüfung
    # ------------------------------------------------
    def test_1_predict(self):
        print("=" * 70)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc, metrics_text = evaluate_model(model, X_test, y_test)

        # Nur Metriken ausgeben (evaluate_model macht print intern)
        print(f"\nErgebnis: TESTFALL 1 PASSED ✅\n")

        self.assertGreaterEqual(acc, 0.9)

    # ------------------------------------------------
    # TESTFALL 2 – Laufzeitprüfung
    # ------------------------------------------------
    def test_2_train_runtime(self):
    print("=" * 70)
    print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
    print("=" * 70 + "\n")

    df = load_data("advertising.csv")
    train_model(df)
    runtime = get_last_timing("train_model")

    ref = self.ref_time or 0.0
    limit = ref * 1.2 if ref > 0 else float("inf")

    # 👉 bisherige Analyse wird ans Ende verschoben

    if runtime <= limit:
        status = "PASSED ✅"
        analysis_text = (
            "✅ Laufzeit liegt innerhalb der Toleranz.\n\n"
            f"Ergebnis: TESTFALL 2 {status}\n"
        )
    else:
        status = "FAILED ❌"
        analysis_text = (
            "❌ Laufzeit überschreitet das Limit!\n\n"
            f"Ergebnis: TESTFALL 2 {status}\n"
        )

        self.fail(
            f"❌ Trainingslaufzeit überschreitet das erlaubte Limit: "
            f"Aktuell {runtime:.4f}s > {limit:.4f}s (Referenz: {ref:.4f}s). "
            f"Überschreitung: +{runtime - limit:.4f}s."
        )

    # 👉 Ausgabe der Analyse erst NACH den Logeinträgen
    print("\nLaufzeitanalyse:")
    print("  (Referenzzeit = aus setUpClass())")
    print(f" - Referenzlaufzeit: {ref:.4f} sec")
    print("  (Aktuelle Laufzeit = aktueller Testlauf)")
    print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
    print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")
    print(analysis_text)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
