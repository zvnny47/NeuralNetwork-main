import numpy as np
import os
from Layer_Dense import Layer_Dense
from Activation_Relu import Activation_ReLU
from LossFunction import Loss_CategoricalCrossentropy
from Softmax import Activation_Softmax
from Optimizer import Optimizer_SGD
from tensorflow import keras as data
from PIL import Image
from DigitRecGUI import DigitRecGUI
import tkinter as tk

# Create the layers
def CreateModel(_X_train, _Y_train, _epochs, version):
  layer1 = Layer_Dense(784, 128)  # First dense layer with 784 inputs (28x28 pixels), and 128 outputs
  activation1 = Activation_ReLU()  # ReLU activation for the first layer

  layer2 = Layer_Dense(128, 64)  # Second dense layer with 128 inputs, and 64 outputs
  activation2 = Activation_ReLU()  # ReLU activation for the second layer

  layer3 = Layer_Dense(64, 10)  # Third dense layer with 64 inputs, and 10 outputs (for the 10 classes)
  activation3 = Activation_Softmax()  # Softmax activation for the final layer

  # Create a loss function instance
  loss_function = Loss_CategoricalCrossentropy()

  #Create optimizer
  optimizer = Optimizer_SGD()

  # Convert labels to one-hot vectors
  y_train_one_hot = np.eye(10)[_Y_train]

  # Define the batch size
  batch_size = 32

  # Compute the number of batches
  n_batches = int(np.ceil(_X_train.shape[0] / batch_size))

  # Number of epochs
  n_epochs = _epochs

  for epoch in range(n_epochs):
      # Initialize list to store batch losses
      batch_losses = []

      for i in range(n_batches):
          # Get batch data
          start = i * batch_size
          end = start + batch_size
          X_batch = _X_train[start:end]
          y_batch = y_train_one_hot[start:end]

          # Forward pass
          layer1.forward(X_batch)
          activation1.forward(layer1.output)

          layer2.forward(activation1.output)
          activation2.forward(layer2.output)

          layer3.forward(activation2.output)
          activation3.forward(layer3.output)

          # Compute loss
          loss = loss_function.forward(activation3.output, y_batch)
          batch_losses.append(loss)

          # Add a small epsilon to avoid division by zero
          epsilon = 1e-7
          activation3.output += epsilon

          # Backward pass
          loss_function.backward(activation3.output, y_batch)

          activation3.backward(loss_function.dinputs)
          layer3.backward(activation3.dinputs)

          activation2.backward(layer3.dinputs)
          layer2.backward(activation2.dinputs)

          activation1.backward(layer2.dinputs)
          layer1.backward(activation1.dinputs)

          # Update weights and biases
          optimizer.update_params(layer1)
          optimizer.update_params(layer2)
          optimizer.update_params(layer3)

      # Compute and print average loss for this epoch
      avg_loss = np.mean(batch_losses)
      print(f'Epoch {epoch+1}, average loss: {avg_loss}')
      
  save_dir = 'Models/V' + str(version) + '/'
  os.makedirs(f'Models', exist_ok=True)
  os.makedirs(save_dir, exist_ok=True)

  layer1.save(save_dir + 'v' + str(version) + '_layer1.npz')
  layer2.save(save_dir + 'v' + str(version) + '_layer2.npz')
  layer3.save(save_dir + 'v' + str(version) + '_layer3.npz')
    
def RunModel(_X_test, _Y_test, version):
  y_test_one_hot = np.eye(10)[_Y_test]

  layer1 = Layer_Dense(n_inputs=784, n_neurons=128)
  layer2 = Layer_Dense(n_inputs=128, n_neurons=64)
  layer3 = Layer_Dense(n_inputs=64, n_neurons=10)

  load_dir = 'Models/V' + str(version) + '/'

  layer1.load(load_dir + 'v' + str(version) + '_layer1.npz')
  layer2.load(load_dir + 'v' + str(version) + '_layer2.npz')
  layer3.load(load_dir + 'v' + str(version) + '_layer3.npz')

  # Forward pass on the test data
  layer1.forward(_X_test)
  layer2.forward(layer1.output)
  layer3.forward(layer2.output)

  # Compute test loss
  loss_function = Loss_CategoricalCrossentropy()
  test_loss = loss_function.forward(layer3.output, y_test_one_hot)

  print(f'Test Loss: {np.average(test_loss)}')

  # Compute test accuracy
  test_predictions = np.argmax(layer3.output, axis=1)
  test_accuracy = np.mean(test_predictions == _Y_test)
  print(f'Test accuracy: {test_accuracy * 100}%')

#CreateModel(X_train, Y_train, 10, 1)
def PredictDigit(_X, version):
  layer1 = Layer_Dense(n_inputs=784, n_neurons=128)
  layer2 = Layer_Dense(n_inputs=128, n_neurons=64)
  layer3 = Layer_Dense(n_inputs=64, n_neurons=10)

  load_dir = 'Models/V' + str(version) + '/'

  layer1.load(load_dir + 'v' + str(version) + '_layer1.npz')
  layer2.load(load_dir + 'v' + str(version) + '_layer2.npz')
  layer3.load(load_dir + 'v' + str(version) + '_layer3.npz')

  # Forward pass on the test data
  layer1.forward(_X)
  layer2.forward(layer1.output)
  layer3.forward(layer2.output)

  # Compute predictions
  predictions = np.argmax(layer3.output, axis=1)
  print(predictions[0])

if __name__ == '__main__':
  window = tk.Tk()
  
  def predict_digit_callback(image):
     PredictDigit(image, version=1)

  gui = DigitRecGUI(window, predict_digit_callback)

  window.mainloop()