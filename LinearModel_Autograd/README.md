Building a Linear Model with Autograd


#Building a Simple Linear Model with one single neuron and no activation function, and use AutoGrad package to train neural network weights
X_train has features to train the model and Y train corresponds to target values.

This project is a foundational example of how neural networks learn by minimizing loss 
and updating weights through gradient descent — all implemented without any PyTorch high-level abstractions.
This project demonstrates a simple linear regression model using a basic neural network architecture 
(1 input, 1 hidden, 1 output) implemented from scratch using PyTorch. 
The model is trained using Mean Squared Error (MSE) as the loss function and updated via gradient descent.

x_train / y_train: Sample dataset for training (1D regression)
Manual implementation of:
    Forward pass using torch.mm
    Loss computation using .pow(2).sum()
    Backward pass using .backward()
    Parameter update with torch.no_grad()
