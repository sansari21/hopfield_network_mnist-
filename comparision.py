import matplotlib.pyplot as plt



models = ['Hopfield Network', 'SVM', 'CNN']
accuracy = [67.58, 98.35, 99.08]
f1_scores = [0.6620, 0.9835, 0.9908]


plt.figure(figsize=(12, 6))

# Accuracy 
plt.subplot(1, 2, 1)
plt.bar(models, accuracy, color=['blue', 'green', 'red'])
plt.xlabel('Classification model')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Classification models')

# F1 Score 
plt.subplot(1, 2, 2)
plt.bar(models, f1_scores, color=['blue', 'green', 'red'])
plt.xlabel('Classification model')
plt.ylabel('F1 Score')
plt.title('F1 Score of Classification models')

plt.tight_layout()
plt.show()
