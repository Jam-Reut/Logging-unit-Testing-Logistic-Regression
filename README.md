
# Logging-unit-Testing-Logistic-Regression

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/HEAD?labpath=logistic-regression.ipynb)

## Beschreibung

Dieses Projekt verwendet logistische Regression, um vorherzusagen, ob ein Benutzer auf eine Online-Werbeanzeige klickt. Es enthält Logging zur Nachvollziehbarkeit sowie zwei Unit-Tests zur Sicherstellung der Funktionalität.

## Ausführung

1. Stelle sicher, dass `advertising.csv` im gleichen Verzeichnis liegt.
2. Führe `logistic_model.py` aus, um das Modell zu trainieren und zu evaluieren.
3. Starte die Unit-Tests mit:

```
python -m unittest test_logistic_model.py
```

## Erwartetes Ergebnis

- Ein trainiertes Modell
- Ausgabe des `classification_report`
- Log-Datei `logistic_model.log` mit den Prozessschritten

[![Modell starten](https://img.shields.io/badge/Run-Logistic%20Model-blue?logo=python)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=logistic_model.py)

[![Unit Tests starten](https://img.shields.io/badge/Run-Unit%20Tests-green?logo=pytest)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=test_logistic_model.py)


## Binder-Button
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jam-Reut/Logging-unit-Testing-Logistic-Regression/main?labpath=auto_start.ipynb)



