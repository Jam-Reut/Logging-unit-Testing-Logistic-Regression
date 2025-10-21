import logging
import time
import functools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Logging-Konfiguration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# internes Dictionary für Laufzeiten

__timings = {}

# Neue Decorators (nur diese geändert)

from functools import wraps

def my_logger(func):
import logging
logging.basicConfig(filename="{}.log".format(func.**name**), level=logging.INFO)

```
@wraps(func)
def wrapper(*args, **kwargs):
    logging.info("Ran with args: {}, and kwargs: {}".format(args, kwargs))
    return func(*args, **kwargs)

return wrapper
```

def my_timer(func):
import time

```
@wraps(func)
def wrapper(*args, **kwargs):
    t1 = time.time()
    result = func(*args, **kwargs)
    t2 = time.time() - t1
    print("{} ran in: {} sec".format(func.__name__, t2))
    return result

return wrapper
```

def get_last_timing(func_name: str):
return __timings.get(func_name)

@my_logger
@my_timer
def load_data(file_path: str):
logging.info(f"Lade Daten aus {file_path}")
df = pd.read_csv(file_path)
log
