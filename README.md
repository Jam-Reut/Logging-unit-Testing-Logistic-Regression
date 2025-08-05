# Logistic Regression mit Logging und Unit-Tests


## Online Ausführen via Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=auto_start.ipynb)

Klicke auf den Button, um das Jupyter Notebook `auto_start.ipynb` interaktiv online zu starten.  
Das Notebook führt automatisch das Training mit Logging durch und führt anschließend die Unit-Tests aus.
## Auto Start (Modell & Testing)


# Projektbeschreibung

Dieses Projekt zeigt ein Beispiel für maschinelles Lernen mit ausführlichem Logging und automatischen Unit-Tests.

## Projektstruktur

- `logistic_model.py`  
  Trainiert und bewertet das Modell mit ausführlichem Logging und Zeitmessung.  
- `test_logistic_model.py`  
  Enthält Unit-Tests für die Funktionen `fit()` (Laufzeit) und `predict()` (Vorhersagequalität).  
- `advertising.csv`  
  Beispiel-Datensatz für das Training und die Tests.  
- `auto_start.ipynb`  
  Notebook zum automatischen Ausführen von Training und Unit-Tests.  


## Logging & Testergebnisse

Beim Ausführen erhältst du eine ausführliche Ausgabe mit:  
- Ladeprozess der Daten  
- Laufzeitmessung einzelner Schritte  
- Genauigkeit (Accuracy) und Konfusionsmatrix der Vorhersagen  
- Ausgabe der Testergebnisse mit Erfolgsmeldung oder Fehlerdetails  






