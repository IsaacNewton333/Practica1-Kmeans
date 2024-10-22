def calcular_matriz_confusion(y_true, y_pred):
    """
    Calcula la matriz de confusión para un problema de clasificación binaria.
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)  # Verdaderos positivos
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)  # Verdaderos negativos
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)  # Falsos positivos
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)  # Falsos negativos
    
    return tp, tn, fp, fn

def accuracy(y_true, y_pred):
    """
    Calcula la precisión (accuracy).
    """
    return sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)

def error(y_true, y_pred):
    """
    Calcula la tasa de error.
    """
    return 1 - accuracy(y_true, y_pred)

def precision(tp, fp):
    """
    Calcula la precisión.
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    """
    Calcula el recall.
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def positive_predictive_value(tp, fp):
    """
    Calcula el valor predictivo positivo (PPV).
    """
    return precision(tp, fp)

def true_positive_rate(tp, fn):
    """
    Calcula la tasa de verdaderos positivos (TPR).
    """
    return recall(tp, fn)

def true_negative_rate(tn, fp):
    """
    Calcula la tasa de verdaderos negativos (TNR).
    """
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def false_positive_rate(fp, tn):
    """
    Calcula la tasa de falsos positivos (FPR).
    """
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def false_negative_rate(fn, tp):
    """
    Calcula la tasa de falsos negativos (FNR).
    """
    return fn / (fn + tp) if (fn + tp) > 0 else 0

def f1_score(precision, recall):
    """
    Calcula el F1-Score.
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Ejemplo de uso
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

tp, tn, fp, fn = calcular_matriz_confusion(y_true, y_pred)

print("Accuracy:", accuracy(y_true, y_pred))
print("Error:", error(y_true, y_pred))
print("Precision:", precision(tp, fp))
print("Recall:", recall(tp, fn))
print("Positive Predictive Value:", positive_predictive_value(tp, fp))
print("True Positive Rate:", true_positive_rate(tp, fn))
print("True Negative Rate:", true_negative_rate(tn, fp))
print("False Positive Rate:", false_positive_rate(fp, tn))
print("False Negative Rate:", false_negative_rate(fn, tp))
print("F1-Score:", f1_score(precision(tp, fp), recall(tp, fn)))