from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Datos de ejemplo
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

# Calcular la matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calcular métricas
accuracy = accuracy_score(y_true, y_pred)
error = 1 - accuracy
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Imprimir resultados
print("Matriz de Confusión:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print("Accuracy:", accuracy)
print("Error:", error)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)