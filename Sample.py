#!/usr/bin/env python
# coding: utf-8

# In[24]:


data = {
    'AND': {
        'inputs': [(0, 0), (0, 1), (1, 0), (1, 1)],
        'targets': [0, 0, 0, 1]
    },
    'OR': {
        'inputs': [(0, 0), (0, 1), (1, 0), (1, 1)],
        'targets': [0, 1, 1, 1]
    }
}


# In[25]:


import random

weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
bias = random.uniform(-1, 1)

print(F"Initial weights: {weights}, Initial bias: {bias}")


# In[26]:


def step_function(x):
    return 1 if x >= 0 else 0


def predict(inputs, weights, bias):
    """Predicts the output using the perceptron model."""
    total_activation = sum(w * i for w, i in zip(weights, inputs)) + bias
    return 1 if total_activation > 0 else 0


# In[27]:


def train_perceptron(inputs, targets, weights, bias, learning_rate=0.1, epochs=20):
    for epoch in range(epochs):
        total_error = 0
        for input_vec, target in zip(inputs, targets):
            # Calculate weighted sum
            weighted_sum = sum(w * inp for w, inp in zip(weights, input_vec)) + bias
            # Get the prediction
            prediction = step_function(weighted_sum)
            # Calculate error
            error = target - prediction
            total_error += abs(error)
            # Update weights and bias
            weights = [w + learning_rate * error * inp for w, inp in zip(weights, input_vec)]
            bias += learning_rate * error
            print(f"Epoch: {epoch+1}, Input: {input_vec}, Target: {target}, Prediction: {prediction}, Error: {error}, Updated weights: {weights}, Updated bias: {bias}")
        # Print the total error at each epoch
        print(f'Epoch {epoch+1}, Total Error: {total_error}')
    return weights, bias


# In[ ]:





# In[28]:


# Cell 6: Train and Test for AND Gate

# Re-initialize weights and bias for a fresh start
weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
bias = random.uniform(-1, 1)

print("Training for AND gate")
weights, bias = train_perceptron(data['AND']['inputs'], data['AND']['targets'], weights, bias)

# Testing the trained model
print("Testing AND gate")
for input_vec in data['AND']['inputs']:
    output = predict(input_vec, weights, bias)
    print(f'Input: {input_vec}, Predicted Output: {output}')


# In[31]:


weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
bias = random.uniform(-1, 1)

print("\nTraining for OR gate")
weights, bias = train_perceptron(data['OR']['inputs'], data['OR']['targets'], weights, bias)

print("Testing OR gate")
for input_vec in data['OR']['inputs']:
    output = predict(input_vec, weights, bias)
    print(f'Input: {input_vec}, Predicted Output: {output}')


# In[ ]:




