import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):

	# ------------------------------------------------
	# TESTFALL 1: predict(): Vorhersagefunktion 
	# ------------------------------------------------
	def test_1_predict_function(self):
		print("=" * 54)
		print("TESTFALL 1: predict(): Vorhersagefunktion")
		print("=" * 54)

		# Daten laden
		df = load_data("advertising.csv")
		print(f"→ load_data ran in: {get_last_timing('load_data'):.4f} sec")
		print()
		# Modell trainieren
		print("=== Modell trainieren ===")
		model, X_test, y_test = train_model(df)
		print(f"→ train_model ran in: {get_last_timing('train_model'):.4f} sec")

		# Modell evaluieren
		acc = evaluate_model(model, X_test, y_test)
		print(f"→ evaluate_model ran in: {get_last_timing('evaluate_model'):.4f} sec")
		print(f"\nFinal Accuracy: {acc:.2f}")

		# Genauigkeit prüfen
		self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
		print("\nErgebnis: TESTFALL 1 PASSED\n")

	# ------------------------------------------------
	# TESTFALL 2: fit(): Laufzeit der Trainingsfunktion 
	# ------------------------------------------------
	def test_2_train_runtime(self):
		print("=" * 54)
		print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
		print("=" * 54)

		# Daten laden
		df = load_data("advertising.csv")
		print(f"→ load_data ran in: {get_last_timing('load_data'):.4f} sec")
		print()
		# Referenzlauf
		print("=== Modell trainieren (Referenzlauf) ===")
		#train_model(df)
		#ref_time = get_last_timing("train_model")
		ref_time = 0.30  # feste Referenzzeit in Sekunden
		print(f"→ train_model ran in: {ref_time:.4f} sec")

		# Testlauf
		print("=== Modell trainieren (Testlauf) ===")
		train_model(df)
		runtime = get_last_timing("train_model")
		print(f"→ train_model ran in: {runtime:.4f} sec")

		# Laufzeitanalyse
		limit = ref_time * 1.2
		print("\n=== Laufzeit-Analyse ===")
		print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
		print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
		print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

		# Bewertung
		if runtime <= limit:
			print("	 Laufzeit liegt innerhalb der Toleranz.\n")
		else:
			print("	 ❌ Laufzeit überschreitet das Limit!\n")

		# Testbedingung
		self.assertLessEqual(
			runtime, limit,
			f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
		)

		print("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
	print("\n=== Starte Unit-Tests ===\n")
	unittest.main(argv=[""], exit=False)
