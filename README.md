
# Logistic Regression

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/username/logistic-regression/HEAD)

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
