import numpy as np
import matplotlib.pyplot as plt
import math


def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w*x[i]+b
        cost = (f_wb - y[i])**2
        cost_sum += cost
        total_cost = (1/(2*m))*cost_sum
    return total_cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i]+b
        dj_dw += (f_wb - y[i])*x[i]
        dj_db += (f_wb - y[i])
        dj_dw /= m
        dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        cost = cost_function(x,y,w,b)
        cost_history.append(cost)
    return w,b


    
# Taking the training data 
# note that x scale is in 1000 sq ft and y scale is in 1000s of dollars
x_train = np.array([1.0 , 2.0, 3.0])
y_train = np.array([300,500,700])
cost_history = []

# Initial parameters
w_init = 0
b_init = 0
iterations = 10000
alpha = 1.0e-2

# Run gradient descent
W_final, b_final = gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations,compute_cost,compute_gradient)
print(f'(w,b) found the gradient descent: ({W_final:8.4f}, {b_final:8.4f})')

#plot the data and prediction line
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual values')
x_pred = np.linspace(0, 3, 100)
y_pred = W_final * x_pred + b_final
plt.plot(x_pred, y_pred, label='Model Prediction', color='lime')
plt.title("House Price Prediction")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.legend()
plt.show()

# Plot cost history
plt.plot(range(iterations), cost_history, color='magenta', label='Cost over iterations')
plt.title("Cost Over Time")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()

#taking the human input for prediction
while True:
    try:
        sqft = float(input("Enter house size in 1000 sqft (or -1 to exit): "))
        if sqft == -1:
            break
        predicted_price = W_final * sqft + b_final
        print(f"Predicted Price: {predicted_price:.2f}k dollars")
    except ValueError:
        print("Please enter a valid number.")


print("\nTraining Summary:")
print(f"Final weight (w): {W_final:.4f}")
print(f"Final bias (b):   {b_final:.4f}")
print(f"Training examples: {x_train.shape[0]}")
print(f"Learning rate: {alpha}")
print(f"Total iterations: {iterations}")