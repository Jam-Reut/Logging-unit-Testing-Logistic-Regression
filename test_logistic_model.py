import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):
    """Unit-Tests für das ML-System nach Ori Cohen Struktur."""

    @classmethod
    def setUpClass(cls):
        """Referenzlauf für Laufzeitmessung."""
        print("=== Starte Unit-Tests ===\n")
        df = load_data("advertising.csv")
        train_model(df)
        cls.ref_time = get_last_timing("train_model")

    # ----------------------------------------------------------
    # TESTFALL 1 – Vorhersageprüfung
    # ----------------------------------------------------------
    def test_1_predict(self):
        print("=" * 70)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc, _ = evaluate_model(model, X_test, y_test)

        # Nur prüfen, keine doppelte Ausgabe
        self.assertGreaterEqual(acc, 0.9)
        print("Ergebnis: TESTFALL 1 PASSED ✅\n")

    # ----------------------------------------------------------
    # TESTFALL 2 – Laufzeitprüfung
    # ----------------------------------------------------------
    def test_2_train_runtime(self):
        print("=" * 70)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 70 + "\n")

        # Aktueller Lauf
        df = load_data("advertising.csv")
        train_model(df)
        runtime = get_last_timing("train_model")

        ref = self.ref_time or 0.0
        if ref == 0.0:
            print("⚠️  WARNUNG: Referenzlaufzeit konnte nicht ermittelt werden.\n")

        limit = ref * 1.2 if ref > 0 else float("inf")

        print("Laufzeitanalyse:")
        print("  (Referenzzeit = aus setUpClass())")
        print(f" - Referenzlaufzeit: {ref:.4f} sec")
        print("  (Aktuelle Laufzeit = aktueller Testlauf)")
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime > limit:
            print(f"❌ Laufzeit überschreitet das Limit!\n")
            print("Ergebnis: TESTFALL 2 FAILED ❌\n")
            self.fail(
                f"Trainingslaufzeit zu hoch: {runtime:.4f}s "
                f"(Limit: {limit:.4f}s, Referenz: {ref:.4f}s) – "
                f"{runtime - limit:.4f}s über Limit."
            )
        else:
            print(f"✅  Aktuelle Laufzeit ({runtime:.4f} sec) liegt unter dem Limit ({limit:.4f} sec).\n")
            print("Laufzeit liegt innerhalb der Toleranz.\n")
            print("Ergebnis: TESTFALL 2 PASSED ✅\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
