# Deep Learning Method: CNN Classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

Y_train_cnn = to_categorical(Y_train, 10)
Y_test_cnn = to_categorical(Y_test, 10)

# Build CNN model
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train_cnn, Y_train_cnn, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate the model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, Y_test_cnn)
cnn_predictions = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

# Display classification report and F1-score
print("CNN Classification Report:")
print(classification_report(Y_test, cnn_predictions, digits=4))

# Calculate F1-score
cnn_f1 = f1_score(Y_test, cnn_predictions, average='weighted')
print(f"CNN Accuracy: {cnn_accuracy * 100:.2f}%")
print(f"CNN Weighted F1 Score: {cnn_f1:.4f}")
