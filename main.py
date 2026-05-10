import tensorflow as tf
import numpy as np
from src.data_loader import load_svhn
from src.model import build_model
from src.visualize import plot_results, plot_sample_predictions

tf.random.set_seed(42)
np.random.seed(42)

print("=== SVHN CNN Classification Started ===\n")

# Load Data
x_train, y_train, x_test, y_test = load_svhn()

# Build Model
model = build_model()
model.summary()

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
print("\nTraining started...\n")
history = model.fit(x_train, y_train, 
                    epochs=20,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                    verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Visualization
plot_results(history)
plot_sample_predictions(x_test, y_test, model)

# Save Model
model.save('svhn_cnn_model.h5')
print("\nModel saved as svhn_cnn_model.h5")
print("Project Completed Successfully!")
