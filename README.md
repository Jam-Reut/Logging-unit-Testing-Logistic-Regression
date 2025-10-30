Logistic Regression - Logging und Unit-Tests

Online Ausführen via Binder:

Klicke auf den Button, um das Jupyter Notebook `auto_start.ipynb` interaktiv online zu starten.  
Das Notebook führt automatisch das Training mit Logging durch und führt anschließend die Unit-Tests aus.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=auto_start.ipynb)
---

Projektbeschreibung:
Dieses Projekt zeigt ein Beispiel für maschinelles Lernen mit **Logistic Regression**, kombiniert mit **Logging** und **automatisierten Unit-Tests**.

Das Ziel ist es, zu demonstrieren, wie man Modelltraining und -vorhersage systematisch testet und dokumentiert.

---

Projektstruktur:
- **`logistic_model.py`** – Trainiert und bewertet das Modell mit `my_logger` und `my_timer` für Logging & Zeitmessung.  
- **`test_logistic_model.py`** – Enthält Unit-Tests für:
  - `fit()` → prüft, dass die Trainingszeit im erlaubten Rahmen bleibt  
  - `predict()` → prüft die Vorhersagegenauigkeit (Accuracy + Confusion Matrix)  
- **`advertising.csv`** – Datensatz für Training & Tests  
- **`auto_start.ipynb`** – Führt automatisch Training & Tests aus  

---

Logging und Timer:
- **`my_logger`**: schreibt Start, Ende und Ergebnis jeder Funktion ins Logfile und in die Konsole.  
- **`my_timer`**: misst die Laufzeit jeder Funktion und loggt diese.  

Beispielhafte Logausgabe:

### 1 predict()
Prüft:
- Accuracy ≥ 0.90  
- Confusion Matrix + Klassifikationsbericht  

Beispielausgabe:

Modellevaluierung:
  Genauigkeit (Accuracy): 0.97
  Confusion Matrix:
[[143   3]
 [  7 147]]
Final Accuracy: 0.97
Ergebnis: TESTFALL 1 PASSED

### 2 fit()
Prüft:
- Trainingslaufzeit ≤ 120 % der Referenzzeit  
- Laufzeit wird per `my_timer` geloggt  

Beispielausgabe:

<img width="1645" height="849" alt="thumbnail_image" src="https://github.com/user-attachments/assets/4569880d-4ab0-46ab-9476-268acb54b6cb" />



Logging & Testergebnisse:

Beim Ausführen erhältst du eine ausführliche Ausgabe mit:  
- Ladeprozess der Daten  
- Laufzeitmessung einzelner Schritte  
- Genauigkeit (Accuracy) und Konfusionsmatrix der Vorhersagen  
- Ausgabe der Testergebnisse mit Erfolgsmeldung oder Fehlerdetails 
