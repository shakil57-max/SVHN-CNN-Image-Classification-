import matplotlib.pyplot as plt
import numpy as np

def plot_results(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()

def plot_sample_predictions(x_test, y_test, model, num_samples=10):
    print("\nSample Predictions:")
    sample_images = x_test[:num_samples]
    sample_labels = y_test[:num_samples]
    predictions = model.predict(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(20, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(sample_images[i])
        color = 'green' if sample_labels[i] == predicted_classes[i] else 'red'
        plt.title(f"True: {sample_labels[i]}\nPred: {predicted_classes[i]}", color=color)
        plt.axis('off')
    plt.suptitle('Sample Predictions (Green = Correct, Red = Wrong)')
    plt.show()
