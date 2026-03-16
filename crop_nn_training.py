# crop_nn_training.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")
X = data.drop("label", axis=1).values
y = data["label"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalize
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train_norm = (X_train - X_min) / (X_max - X_min)
X_test_norm = (X_test - X_min) / (X_max - X_min)

# Neural network
num_features = X_train.shape[1]
num_classes = len(np.unique(y_encoded))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=100, batch_size=32)

# Save model and encoders
model.save("model/crop_nn_model.h5")
joblib.dump(le, "model/crop_label_encoder.pkl")
joblib.dump((X_min, X_max), "model/crop_minmax.pkl")