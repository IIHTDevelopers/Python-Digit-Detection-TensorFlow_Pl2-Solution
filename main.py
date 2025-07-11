import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Step 1: Load and preprocess dataset
def load_and_preprocess():
    (X, y), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten images and normalize
    X = X.reshape(-1, 28 * 28).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # Use a subset for faster training
    X_train, _, y_train, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


# Step 2: Build model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes (0-9)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 3: Train model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model


# Step 4: Evaluate model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")


# Step 5: Predict a sample
def predict_sample(model):
    sample = tf.keras.datasets.mnist.load_data()[1][0][0:1]
    sample_image = sample.reshape(1, 28 * 28).astype("float32") / 255.0
    prediction = model.predict(sample_image)
    predicted_class = np.argmax(prediction)
    print(f"Predicted digit: {predicted_class}")


# Step 6: Main function
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    model = build_model()
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    predict_sample(model)


# Run
if __name__ == "__main__":
    main()
