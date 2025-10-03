# Logistic Regression mit Logging und Unit Tests

Dieses Projekt demonstriert ein ML-basiertes System (Logistic Regression) mit integriertem **Logging** und **automatischen Unit Tests**.

## 📦 Projektstruktur
- `advertising.csv` – Beispieldatensatz
- `logistic_model.py` – Modell mit Logging und Timer
- `test_logistic_model.py` – Unit Tests für `fit()` und `predict()`
- `README.md` – Diese Dokumentation

## ▶️ Ausführung in Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/USERNAME/REPO/main?filepath=logistic-regression.ipynb)

---

## 1️⃣ Modell mit Logging ausführen
```bash
python logistic_model.py
```

## 2️⃣ Unit-Tests ausführen
```bash
python -m unittest test_logistic_model.py
```

---

## 🧪 Testfälle
- **Test `predict()`**: Überprüft Accuracy und Confusion Matrix auf Testdaten.
- **Test `fit()`**: Misst die Laufzeit und prüft, ob diese unter 120% einer Referenzzeit liegt.

---

## 📄 Testdaten
Die Datei `advertising.csv` muss im Projektverzeichnis vorhanden sein.

---

## 🔍 Anleitung für den Prüfer

1. **Projekt klonen oder in Binder öffnen**
   - GitHub Repo klonen:  
     ```bash
     git clone https://github.com/USERNAME/REPO.git
     cd REPO
     ```
   - Oder in Binder öffnen: Klick auf den Badge oben.

2. **Modell ausführen**  
   ```bash
   python logistic_model.py
   ```

3. **Tests ausführen**  
   ```bash
   python -m unittest test_logistic_model.py
   ```

4. **Erwartetes Verhalten**  
   - Beim Modellstart: Logging-Ausgaben zu Datenladen, Training, Evaluation.
   - Bei Tests: Meldung `OK`, wenn alle Tests bestanden.

---

## ⚙️ Abhängigkeiten
```bash
pip install pandas scikit-learn
```
