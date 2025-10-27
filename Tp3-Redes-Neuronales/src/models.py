import numpy as np

#inicialización y forward

# --- Funciones de activación ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x_shifted = x - np.max(x, axis=0, keepdims=True)  
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# --- Inicialización de pesos ---
def inicializar_pesos(dims):
    """
    dims = [n_entrada, n1, n2, ..., nL, n_salida]
    """
    np.random.seed(42)
    parametros = {}
    for l in range(1, len(dims)):
        parametros[f"W{l}"] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2. / dims[l-1])  # Uso He initialization como recomienda el Bishop
        parametros[f"b{l}"] = np.zeros((dims[l], 1)) # Inicializo biases en cero
    return parametros

# --- Forward propagation ---
def forward_propagation(X, parametros):
    """
    X: entrada (n_entrada, n_muestras)
    """
    caches = {}
    A = X
    L = len(parametros) // 2

    for l in range(1, L):
        Z = parametros[f"W{l}"] @ A + parametros[f"b{l}"]
        A = relu(Z)
        caches[f"Z{l}"], caches[f"A{l}"] = Z, A

    # Capa de salida con softmax
    ZL = parametros[f"W{L}"] @ A + parametros[f"b{L}"]
    AL = softmax(ZL)
    caches[f"Z{L}"], caches[f"A{L}"] = ZL, AL

    return AL, caches

# Backpropagation y entrenamiento

def cross_entropy_loss(Y, Y_hat):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(Y_hat + 1e-9)) / m
    return loss


def backward_propagation(X, Y, parametros, caches):
    grads = {}
    L = len(parametros) // 2
    m = X.shape[1]
    A_prev = X

    # --- Capa de salida (softmax + cross-entropy) ---
    dZL = caches[f"A{L}"] - Y
    grads[f"dW{L}"] = (1/m) * dZL @ caches[f"A{L-1}"].T
    grads[f"db{L}"] = (1/m) * np.sum(dZL, axis=1, keepdims=True)

    dA_prev = parametros[f"W{L}"].T @ dZL

    # --- Capas ocultas ---
    for l in reversed(range(1, L)):
        Z = caches[f"Z{l}"]
        dZ = dA_prev * relu_derivative(Z)
        A_prev = X if l == 1 else caches[f"A{l-1}"]
        grads[f"dW{l}"] = (1/m) * dZ @ A_prev.T
        grads[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA_prev = parametros[f"W{l}"].T @ dZ

    return grads

def actualizar_parametros(parametros, grads, lr):
    L = len(parametros) // 2
    for l in range(1, L+1):
        parametros[f"W{l}"] -= lr * grads[f"dW{l}"]
        parametros[f"b{l}"] -= lr * grads[f"db{l}"]
    return parametros


# Entrenamiento del modelo

def entrenar_red(X, Y, dims, epochs=50, lr=0.01):
    parametros = inicializar_pesos(dims)
    for epoch in range(epochs):
        # Forward
        Y_hat, caches = forward_propagation(X, parametros)
        loss = cross_entropy_loss(Y, Y_hat)

        # Backward
        grads = backward_propagation(X, Y, parametros, caches)

        # Update
        parametros = actualizar_parametros(parametros, grads, lr)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Época {epoch+1}/{epochs} - Pérdida: {loss:.4f}")

    return parametros

# --- Entrenamiento con registro de pérdidas ---
def entrenar_red_con_validacion(X_train, Y_train, X_val, Y_val, dims, epochs=50, lr=0.01):
    parametros = inicializar_pesos(dims)
    losses_train = []
    losses_val = []

    for epoch in range(epochs):
        # === Forward ===
        Y_hat_train, caches_train = forward_propagation(X_train, parametros)
        loss_train = cross_entropy_loss(Y_train, Y_hat_train)

        # === Forward validación ===
        Y_hat_val, _ = forward_propagation(X_val, parametros)
        loss_val = cross_entropy_loss(Y_val, Y_hat_val)

        # === Backward ===
        grads = backward_propagation(X_train, Y_train, parametros, caches_train)

        # === Update ===
        parametros = actualizar_parametros(parametros, grads, lr)

        # === Registro ===
        losses_train.append(loss_train)
        losses_val.append(loss_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Época {epoch+1}/{epochs} | Loss Train: {loss_train:.4f} | Loss Val: {loss_val:.4f}")

    return parametros, losses_train, losses_val
