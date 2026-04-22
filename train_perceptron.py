import numpy as np
L_ideal = np.array([
    1, 0, 0,
    1, 0, 0,
    1, 1, 1
])
T_ideal = np.array([
    1, 1, 1,
    0, 1, 0,
    0, 1, 0
])

def augment_pattern(pattern):
    new_pattern = pattern.copy()
    num_flips = np.random.randint(1, 3)
    flip_indices = np.random.choice(9, num_flips, replace=False)

    for idx in flip_indices:
        new_pattern[idx] = 1 - new_pattern[idx]

    return new_pattern


def shift_pattern(pattern):
    grid = pattern.reshape(3, 3)

    if np.random.rand() > 0.5:
        grid = np.roll(grid, 1, axis=1)

    if np.random.rand() > 0.5:
        grid = np.roll(grid, 1, axis=0)

    return grid.flatten()

def generate_dataset(count_per_class=60):
    X = []
    y = []

    for _ in range(count_per_class):
        sample_L = L_ideal.copy()
        flip_idx = np.random.randint(0, 9)
        sample_L[flip_idx] = 1 - sample_L[flip_idx]
        X.append(sample_L)
        y.append(0)
        sample_T = T_ideal.copy()
        flip_idx = np.random.randint(0, 9)
        sample_T[flip_idx] = 1 - sample_T[flip_idx]
        X.append(sample_T)
        y.append(1)
    return np.array(X), np.array(y)


X_train, y_train = generate_dataset(60)
indices = np.random.permutation(len(X_train))
X_train = X_train[indices]
y_train = y_train[indices]

def fit(X_train, y_train):
    weights = np.zeros(9)
    bias = 0
    learning_rate = 0.01

    for _ in range(1000):
        for idx, x_i in enumerate(X_train):
            linear_product = np.dot(x_i, weights) + bias
            y_pred = 1 if linear_product >= 0 else 0

            update = learning_rate * (y_pred - y_train[idx])
            weights = weights - update * x_i
            bias = bias - update

    return weights, bias


final_weights, final_bias = fit(X_train, y_train)

print("Final Weights:", list(final_weights))
print("Final Bias:", final_bias)

def predict(x, weights, bias):
    return 1 if np.dot(x, weights) + bias >= 0 else 0

correct = 0
for i in range(len(X_train)):
    pred = predict(X_train[i], final_weights, final_bias)
    if pred == y_train[i]:
        correct += 1

accuracy = correct / len(X_train)
print(f"Training Accuracy: {accuracy * 100:.2f}%")