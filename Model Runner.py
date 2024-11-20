import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Load the trained model
model_path = 'model_epoch_17.h5'  # Path to your saved model
model = tf.keras.models.load_model(model_path, compile=False)

# Define image dimensions and batch size
img_height, img_width = 512, 512  # Adjust based on your model input
batch_size = 32

# Load the test dataset without augmentation
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'C://Users//ryang//VSCode//JSTest//dataset',  # Path to your test dataset
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Change if using binary classification
    shuffle=False  # Important for confusion matrix
)

# Make predictions on the test set
predictions = model.predict(test_generator)
predicted_classes = tf.argmax(predictions, axis=1)

# Get true classes
true_classes = test_generator.classes

# Stats
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(
    true_classes, predicted_classes, average="weighted", zero_division=0
)
recall = recall_score(
    true_classes, predicted_classes, average="weighted", zero_division=0
)
f1 = f1_score(true_classes, predicted_classes, average="weighted", zero_division=0)


# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
