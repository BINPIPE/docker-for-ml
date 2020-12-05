## Realtime prediction request

```
curl -i -H "Content-Type: application/json" -X POST -d '{"CRIM": 15.02, "ZN": 0.0, "INDUS": 18.1, "CHAS": 0.0, "NOX": 0.614, "RM": 5.3, "AGE": 97.3, "DIS": 2.1, "RAD": 24.0, "TAX": 666.0, "PTRATIO": 20.2, "B": 349.48, "LSTAT": 24.9}' 127.0.0.1:5000/predict
```
