import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Loading the data
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalizing 
X_train = X_train.astype(float) / 255.0
X_test = X_test.astype(float) / 255.0

# Binarizing using threshold of 232 (232 gives best result, after a number of iterations on different values )
threshold = 232/ 255.0
X_train = np.where(X_train > threshold, 1.0, 0.0)
X_test = np.where(X_test > threshold, 1.0, 0.0)

# Flattening images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

def hopfield_update_rule(X, z, beta):
    
   
    softmax_values = np.exp(beta * X @ z)
    softmax_values -= np.max(softmax_values)  
    softmax_values = np.exp(softmax_values)
    softmax_values /= np.sum(softmax_values)
    return X.T @ softmax_values

# Preparing stable patterns for each digit
def get_stored_patterns(X, Y):
    
    patterns = []
    for digit in range(10):
        # Average of all images for each digit
        pattern = np.mean(X[Y == digit], axis=0)
        patterns.append(pattern)
    return np.array(patterns)

# Getting the stored patterns
stored_patterns = get_stored_patterns(X_train, Y_train)

# Classifier function
def classify(img, stored_patterns, beta):
    """Classify an image using the Hopfield Network."""
    # Reshaping image to a column vector
    img_vector = img.reshape(-1, 1)
    # Getting the pattern using update rule
    output = hopfield_update_rule(stored_patterns, img_vector, beta)
    # Calculating similarity with each stored pattern using cosine similarity
    similarities = [np.dot(output.flatten(), pattern.flatten()) / (np.linalg.norm(output) * np.linalg.norm(pattern)) for pattern in stored_patterns]
    # Returning the index of the highest similarity, which corresponds to the predicted digit
    return np.argmax(similarities)


  

# Evaluation of the classifier 
beta = 1
def evaluate_classifier(X_test, stored_patterns, beta):
    predictions = []
    for i in range(len(X_test)):
        prediction = classify(X_test[i], stored_patterns, beta)
        predictions.append(prediction)
    return np.array(predictions)

# Getting predictions
predictions = evaluate_classifier(X_test, stored_patterns, beta)

# Calculating accuracy
accuracy = accuracy_score(Y_test, predictions) * 100
print(f"Accuracy : {accuracy:.2f}%")

# classification report for each digit
print(classification_report(Y_test, predictions, digits=4))

# F1-score 
f1 = f1_score(Y_test, predictions, average='weighted')
print(f"F1 Score for Hopfield Network: {f1:.4f}")

# Visualization of some of the results 
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {Y_test[i]}")
    plt.axis('off')
    plt.subplot(2, 5, i + 6)
    reconstructed = hopfield_update_rule(stored_patterns, X_test[i].reshape(-1, 1), beta)
    plt.imshow(reconstructed.reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {classify(X_test[i], stored_patterns, beta)}")
    plt.axis('off')
plt.show()
