# Classical Method: SVM Classification
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca = PCA(n_components=50)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)

# Train SVM classifier
svm_clf = SVC(kernel='rbf', gamma='scale')  
svm_clf.fit(X_train_scaled, Y_train)

# Predictions and evaluation
svm_predictions = svm_clf.predict(X_test_scaled)

# Display classification report and F1-score
print("SVM Classification Report:")
print(classification_report(Y_test, svm_predictions, digits=4))

# Calculate accuracy and F1-score
svm_accuracy = accuracy_score(Y_test, svm_predictions)
svm_f1 = f1_score(Y_test, svm_predictions, average='weighted')
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print(f"SVM Weighted F1 Score: {svm_f1:.4f}")
