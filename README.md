
# Logistic Regression mit Logging und Unit-Tests

## Auto Start (Modell & Testing)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=auto_start.ipynb)

## Separat staten ( Modell | Testing)
[![Modell starten](https://img.shields.io/badge/Run-Logistic%20Model-blue?logo=python)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=logistic_model.py)

[![Unit Tests starten](https://img.shields.io/badge/Run-Unit%20Tests-green?logo=pytest)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=test_logistic_model.py)



# Projektbeschreibung

Dieses Projekt zeigt ein Beispiel für maschinelles Lernen mit ausführlichem Logging und automatischen Unit-Tests.

---

## Projektstruktur

- `logistic_model.py`  
  Trainiert und bewertet das Modell mit ausführlichem Logging und Zeitmessung.  
- `test_logistic_model.py`  
  Enthält Unit-Tests für die Funktionen `fit()` (Laufzeit) und `predict()` (Vorhersagequalität).  
- `advertising.csv`  
  Beispiel-Datensatz für das Training und die Tests.  
- `auto_start.ipynb`  
  Notebook zum automatischen Ausführen von Training und Unit-Tests.  

---

## Online Ausführen via Binder

[![Open In Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deinGitHubUser/deinRepoName/HEAD?filepath=auto_start.ipynb)

Klicke auf den Button, um das Jupyter Notebook `auto_start.ipynb` interaktiv online zu starten.  
Das Notebook führt automatisch das Training mit Logging durch und führt anschließend die Unit-Tests aus.


## Logging & Testergebnisse

Beim Ausführen erhältst du eine ausführliche Ausgabe mit:  
- Ladeprozess der Daten  
- Laufzeitmessung einzelner Schritte  
- Genauigkeit (Accuracy) und Konfusionsmatrix der Vorhersagen  
- Ausgabe der Testergebnisse mit Erfolgsmeldung oder Fehlerdetails  

---





