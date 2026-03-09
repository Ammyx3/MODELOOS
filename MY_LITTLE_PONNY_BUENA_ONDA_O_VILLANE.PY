import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# DATASET "MY LITTLE PONY"

X, y = make_classification(
    n_samples=800,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    random_state=seed
)

X = X.astype(np.float32)

print(np.bincount(y))

# SPLIT

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=seed
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
)

# ESCALAMIENTO

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# MODELO MLP

def create_model(input_dim):

    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(16, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

model = create_model(X_train.shape[1])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision")
    ]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    verbose=1
)

# PREDICCIONES

pred_test = model.predict(X_test).ravel()

auc_roc = roc_auc_score(y_test, pred_test)
auc_pr = average_precision_score(y_test, pred_test)

print("AUC ROC:", auc_roc)
print("AUC PR:", auc_pr)

# BUSCAR THRESHOLD

thresholds = np.linspace(0.05,0.95,50)

def evaluate_threshold(t):

    y_pred = (pred_test >= t).astype(int)

    cm = confusion_matrix(y_test,y_pred)

    tn,fp,fn,tp = cm.ravel()

    recall = tp/(tp+fn+1e-12)
    precision = tp/(tp+fp+1e-12)

    return cm,precision,recall

target_precision = 0.95
best_threshold = 0.5
best_recall = -1

for t in thresholds:

    cm,precision,recall = evaluate_threshold(t)

    if precision >= target_precision and recall > best_recall:
        best_recall = recall
        best_threshold = t

cm,precision,recall = evaluate_threshold(best_threshold)

print("Threshold:",best_threshold)
print("Confusion Matrix:\n",cm)
print("Precision:",precision)
print("Recall:",recall)

y_pred = (pred_test >= best_threshold).astype(int)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Villain Pony","Hero Pony"]
))

#  GRÁFICA LOSS

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])
plt.show()
