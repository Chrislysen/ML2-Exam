import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. Configuration & Constants
# ==========================================
IMG_DIR = 'cnn_dataset/images/'
LABELS_FILE = 'labels.csv'
IMG_SIZE = (32, 32)
BATCH_SIZE = 64  # [cite: 87]
EPOCHS = 40  # [cite: 88]
LEARNING_RATE = 1e-3  # [cite: 84]
SEED = 42

print("NumPy Version:", np.__version__)
print("TensorFlow Version:", tf.__version__)


# ==========================================
# Step 1: Data Loading & Exploration [cite: 66]
# ==========================================
def load_dataset(csv_path, img_dir):
    print(f"\n--- Loading Data from {csv_path} ---")
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for index, row in df.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        # Load as grayscale (color_mode='grayscale') to get (32, 32, 1)
        img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(row['label'])

    return np.array(images), np.array(labels)


X, y = load_dataset(LABELS_FILE, IMG_DIR)
print(f"Total samples loaded: {X.shape[0]}")  # Should be 1800 [cite: 64]
print(f"Image shape: {X.shape[1:]}")  # Should be (32, 32, 1)

# Display random samples per class [cite: 66]
classes = {0: 'Circle', 1: 'Square', 2: 'Triangle'}
plt.figure(figsize=(10, 3))
for i in range(3):
    # Find indices for this class
    idxs = np.where(y == i)[0]
    rand_idx = np.random.choice(idxs)
    plt.subplot(1, 3, i + 1)
    plt.imshow(X[rand_idx].reshape(32, 32), cmap='gray')
    plt.title(f"Label: {classes[i]}")
    plt.axis('off')
plt.suptitle("Random Samples per Class")
plt.tight_layout()
plt.show()

# ==========================================
# Step 2: Preprocessing [cite: 67]
# ==========================================
print("\n--- Preprocessing ---")

# 1. Normalize pixel values to [0, 1] [cite: 68]
X_norm = X.astype('float32') / 255.0

# 2. One-hot encode labels [cite: 70]
y_encoded = to_categorical(y, num_classes=3)

# 3. Split: Train (70%), Validation (15%), Test (15%) [cite: 69]
# First split: 70% Train, 30% Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y_encoded, test_size=0.30, random_state=SEED, stratify=y_encoded
)
# Second split: Split the 30% Temp into 50/50 (which is 15% / 15% of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape:     {X_test.shape}")

# ==========================================
# Step 3: Model Design [cite: 71]
# ==========================================
print("\n--- Building Model ---")

model = Sequential([
    # Input: (32, 32, 1) [cite: 73]
    # Conv2D (32, kernel=3, ReLU) [cite: 74]
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),

    # MaxPool2D (2) [cite: 75]
    MaxPool2D(pool_size=(2, 2)),

    # Conv2D (64, kernel=3, ReLU) [cite: 76]
    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    # MaxPool2D (2) [cite: 77]
    MaxPool2D(pool_size=(2, 2)),

    # Conv2D (128, kernel=3, ReLU) [cite: 78]
    Conv2D(128, kernel_size=(3, 3), activation='relu'),

    # Flatten [cite: 79]
    Flatten(),

    # Dense (64, ReLU) [cite: 80]
    Dense(64, activation='relu'),

    # Dropout (0.3) [cite: 81]
    Dropout(0.3),

    # Dense (3, Softmax) [cite: 82]
    Dense(3, activation='softmax')
])

model.summary()

# ==========================================
# Step 4: Training [cite: 83]
# ==========================================
print("\n--- Training Model ---")

# Optimizer: Adam (lr=1e-3) [cite: 84]
# Loss: Categorical Crossentropy [cite: 86]
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping (patience=5) on validation loss [cite: 89]
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Record training time [cite: 90]
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,  # [cite: 88]
    batch_size=BATCH_SIZE,  # [cite: 87]
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal Training Time: {total_time:.2f} seconds")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curves')
plt.legend()
plt.show()

# ==========================================
# Step 5: Evaluation & Visualization [cite: 91]
# ==========================================
print("\n--- Evaluation ---")

# 1. Compute metrics (Accuracy, Precision, Recall, F1) [cite: 92]
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Circle', 'Square', 'Triangle']))

# 2. Plot Confusion Matrix [cite: 93]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Circle', 'Square', 'Triangle'],
            yticklabels=['Circle', 'Square', 'Triangle'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# 3. Display 6 random test images with predicted and true labels [cite: 94]
plt.figure(figsize=(12, 6))
rand_indices = np.random.choice(len(X_test), 6, replace=False)

for i, idx in enumerate(rand_indices):
    plt.subplot(2, 3, i + 1)
    img = X_test[idx].reshape(32, 32)
    plt.imshow(img, cmap='gray')

    true_cls = classes[y_true[idx]]
    pred_cls = classes[y_pred[idx]]
    color = 'green' if true_cls == pred_cls else 'red'

    plt.title(f"True: {true_cls}\nPred: {pred_cls}", color=color)
    plt.axis('off')

plt.suptitle("Model Predictions on Test Set")
plt.tight_layout()
plt.show()