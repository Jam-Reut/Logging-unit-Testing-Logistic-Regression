import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):
    """Unit-Tests für das ML-System nach Ori Cohen Struktur."""

    @classmethod
    def setUpClass(cls):
        """Einmaliger Referenzlauf für Laufzeitmessung."""
        print("\n" + "=" * 70)
        print("=== INITIALER REFERENZLAUF (setUpClass) ===")
        print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        cls.ref_time = get_last_timing("train_model")

        print("\n" + "-" * 70)
        print(f"Referenzlauf abgeschlossen – gemessene Zeit: {cls.ref_time:.4f} sec")
        print("-" * 70 + "\n")

    # ----------------------------------------------------------
    # TESTFALL 1 – Vorhersageprüfung
    # ----------------------------------------------------------
    def test_1_predict(self):
        print("\n" + "=" * 70)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 70)
        print("\n📘 [TEST 1 LOGGING] – Technische Abläufe unten:\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc, _ = evaluate_model(model, X_test, y_test)

        print("\n📊 Testergebnisse:")
        print(f"Final Accuracy: {acc:.2f}\n")

        self.assertGreaterEqual(acc, 0.9)
        print("✅ Ergebnis: TESTFALL 1 PASSED\n")

        print("-" * 70)
        print("ENDE TESTFALL 1")
        print("-" * 70 + "\n")

    # ----------------------------------------------------------
    # TESTFALL 2 – Laufzeitprüfung
    # ----------------------------------------------------------
    def test_2_train_runtime(self):
        print("\n" + "=" * 70)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 70)
        print("\n⚙️ [TEST 2 LOGGING] – Technische Abläufe unten:\n")

        df = load_data("advertising.csv")
        train_model(df)
        runtime = get_last_timing("train_model")

        ref = self.ref_time or 0.0
        if ref == 0.0:
            print("⚠️ WARNUNG: Referenzlaufzeit konnte nicht ermittelt werden.\n")

        limit = ref * 1.2 if ref > 0 else float("inf")

        print("\n🕒 Laufzeitanalyse:")
        print("  (Referenzzeit = aus setUpClass)")
        print(f" - Referenzlaufzeit: {ref:.4f} sec")
        print("  (Aktuelle Laufzeit = aktueller Testlauf)")
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime > limit:
            delta = runtime - limit
            print(f"❌ Laufzeit überschreitet das Limit um {delta:.4f} Sekunden!")
            print("Ergebnis: TESTFALL 2 FAILED ❌\n")
            self.fail(
                f"Trainingslaufzeit überschreitet Limit: {runtime:.4f}s "
                f"(Limit {limit:.4f}s, Referenz {ref:.4f}s, Überschreitung {delta:.4f}s)"
            )
        else:
            print(f"✅ Laufzeit ({runtime:.4f}s) liegt innerhalb des erlaubten Limits ({limit:.4f}s).")
            print("Ergebnis: TESTFALL 2 PASSED ✅\n")

        print("-" * 70)
        print("ENDE TESTFALL 2")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
