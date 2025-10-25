import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# ------------------------------------------------------------
# Unit-Testklasse
# ------------------------------------------------------------
class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #print("=" * 70)
        #print("=== INITIALER REFERENZLAUF (setUpClass) ===")
        #print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        cls.ref_time = get_last_timing("train_model")

        # Schutz gegen NoneType (robuste Testinitialisierung)
        if cls.ref_time is None:
            logging.warning("⚠️  WARNUNG: Referenzlaufzeit konnte nicht ermittelt werden.")
            cls.ref_time = 0.0

        #print("\n💬 Hinweis:")
        #print("Die folgenden Logeinträge zeigen die Abläufe beider Testfälle.")
        #print("Alles vor dem Punkt ('.') gehört zu Testfall 1 (predict),")
        #print("ab '.2025-…' beginnt Testfall 2 (train_runtime).\n")

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
        passed = runtime <= limit

        # --- Analyse und Bewertung jetzt über Logger statt print ---
        logger = logging.getLogger()

        logger.info("")
        logger.info("=== Laufzeitanalyse ===")
        logger.info("  (Referenzzeit = aus setUpClass())")
        logger.info(f" - Referenzlaufzeit: {ref:.4f} sec")
        logger.info("  (Aktuelle Laufzeit = aktueller Testlauf)")
        logger.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        logger.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec")

        if passed:
            logger.info("✅ Laufzeit liegt innerhalb der Toleranz.")
            logger.info("Ergebnis: TESTFALL 2 PASSED ✅\n")
        else:
            logger.error("❌ Laufzeit überschreitet das Limit!")
            logger.error("Ergebnis: TESTFALL 2 FAILED ❌\n")
            self.fail(
                f"❌ Trainingslaufzeit überschreitet das erlaubte Limit: "
                f"Aktuell {runtime:.4f}s > {limit:.4f}s (Referenz: {ref:.4f}s). "
                f"Überschreitung: +{runtime - limit:.4f}s."
            )



if __name__ == "__main__":
    print("=== Starte Unit-Tests ===")
    unittest.main(argv=[""], exit=False)
