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
def entrenar_red_con_validacion(
    X_train, Y_train, X_val, Y_val, dims,
    epochs=50, lr=0.01, batch_size=5000
):
    parametros = inicializar_pesos(dims)
    losses_train, losses_val = [], []
    m = X_train.shape[1]  # número total de muestras

    for epoch in range(epochs):
        # Mezclar los datos
        indices = np.random.permutation(m)
        X_shuffled = X_train[:, indices]
        Y_shuffled = Y_train[:, indices]

        # Inicializar gradientes acumulados
        grads_acum = {f"dW{l}": np.zeros_like(parametros[f"W{l}"]) for l in range(1, len(dims))}
        grads_acum.update({f"db{l}": np.zeros_like(parametros[f"b{l}"]) for l in range(1, len(dims))})

        epoch_loss = 0
        num_batches = int(np.ceil(m / batch_size))

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]

            # Forward y pérdida
            Y_hat, caches = forward_propagation(X_batch, parametros)
            loss = cross_entropy_loss(Y_batch, Y_hat)
            epoch_loss += loss

            # Backward
            grads = backward_propagation(X_batch, Y_batch, parametros, caches)

            # Acumular gradientes
            for l in range(1, len(dims)):
                grads_acum[f"dW{l}"] += grads[f"dW{l}"] * (X_batch.shape[1] / m)
                grads_acum[f"db{l}"] += grads[f"db{l}"] * (X_batch.shape[1] / m)

        # Actualizar una vez por epoch (equivalente a full batch)
        parametros = actualizar_parametros(parametros, grads_acum, lr)

        # Evaluar validación
        Y_hat_val, _ = forward_propagation(X_val, parametros)
        loss_val = cross_entropy_loss(Y_val, Y_hat_val)

        losses_train.append(epoch_loss / num_batches)
        losses_val.append(loss_val)

        #if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época {epoch+1}/{epochs} | Loss Train: {epoch_loss/num_batches:.4f} | Loss Val: {loss_val:.4f}")

    return parametros, losses_train, losses_val

# --- Predicciones ---
def predict(X, parametros):
    Y_hat, _ = forward_propagation(X, parametros)
    return np.argmax(Y_hat, axis=0)  # clase más probable

# IMPLEMENTACIONES PARA LA RED AVANZADA

# --- Rate Scheduling ---

def lr_linear_schedule(epoch, lr_init, lr_final, total_epochs):
    """
    Decrece linealmente el learning rate desde lr_init hasta lr_final.
    Se mantiene constante en lr_final tras alcanzar el valor mínimo.
    """
    lr = lr_init - (lr_init - lr_final) * (epoch / total_epochs)
    return max(lr, lr_final)  # saturación inferior


def lr_exponential_schedule(epoch, lr_init, decay_rate, total_epochs):
    """
    Decrece exponencialmente el learning rate según:
        lr = lr_init * exp(-decay_rate * epoch)
    """
    return lr_init * np.exp(-decay_rate * epoch)


# Modificacion de la funcion de entrenamiento anterior para incluir schedulers
def entrenar_red_con_validacion_scheduler(
    X_train, Y_train, X_val, Y_val, dims,
    epochs=50, lr_init=0.01,
    scheduler="none", lr_final=0.001, decay_rate=0.05, batch_size=5000
):
    """
    scheduler: 'none', 'linear', 'exponential'
    """
    parametros = inicializar_pesos(dims)
    losses_train, losses_val = [], []
    m = X_train.shape[1]

    lrs = []

    for epoch in range(epochs):
          # --- Actualizar learning rate según scheduler ---
        if scheduler == "linear":
            lr = lr_linear_schedule(epoch, lr_init, lr_final, epochs)
        elif scheduler == "exponential":
            lr = lr_exponential_schedule(epoch, lr_init, decay_rate, epochs)
        else:
            lr = lr_init
        lrs.append(lr)
        # Mezclar los datos
        indices = np.random.permutation(m)
        X_shuffled = X_train[:, indices]
        Y_shuffled = Y_train[:, indices]

        # Inicializar gradientes acumulados
        grads_acum = {f"dW{l}": np.zeros_like(parametros[f"W{l}"]) for l in range(1, len(dims))}
        grads_acum.update({f"db{l}": np.zeros_like(parametros[f"b{l}"]) for l in range(1, len(dims))})

        epoch_loss = 0
        num_batches = int(np.ceil(m / batch_size))

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]

            # Forward y pérdida
            Y_hat, caches = forward_propagation(X_batch, parametros)
            loss = cross_entropy_loss(Y_batch, Y_hat)
            epoch_loss += loss

            # Backward
            grads = backward_propagation(X_batch, Y_batch, parametros, caches)

            # Acumular gradientes
            for l in range(1, len(dims)):
                grads_acum[f"dW{l}"] += grads[f"dW{l}"] * (X_batch.shape[1] / m)
                grads_acum[f"db{l}"] += grads[f"db{l}"] * (X_batch.shape[1] / m)

        # Actualizar una vez por epoch (equivalente a full batch)
        parametros = actualizar_parametros(parametros, grads_acum, lr)

        # Evaluar validación
        Y_hat_val, _ = forward_propagation(X_val, parametros)
        loss_val = cross_entropy_loss(Y_val, Y_hat_val)

        losses_train.append(epoch_loss / num_batches)
        losses_val.append(loss_val)

        #if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época {epoch+1}/{epochs} | Loss Train: {epoch_loss/num_batches:.4f} | Loss Val: {loss_val:.4f}")

    return parametros, losses_train, losses_val, lrs

# Modificacion de la funcion de entrenamiento para incluir Mini-batch stochastic gradient descent.

import numpy as np
import time
import gc

def entrenar_red_minibatch_optimizado(
    X_train, Y_train, X_val, Y_val, dims,
    epochs=50, lr=0.1, batch_size=256, verbose=True, val_interval=5
):
    """
    Entrenamiento mediante Mini-batch SGD optimizado para grandes datasets.
    - Sin rate scheduling.
    - Sin duplicar arrays en memoria.
    - Usa float32 para reducir uso de RAM.
    - Evalúa validación cada `val_interval` epochs.
    """

    # --- Convertir a float32 si no lo está ---
    X_train = X_train.astype(np.float32, copy=False)
    X_val   = X_val.astype(np.float32, copy=False)
    for key in ["W", "b"]:
        pass  # parámetros se crean en float32 más abajo

    parametros = inicializar_pesos(dims)
    for l in range(1, len(dims)):
        parametros[f"W{l}"] = parametros[f"W{l}"].astype(np.float32)
        parametros[f"b{l}"] = parametros[f"b{l}"].astype(np.float32)

    losses_train, losses_val = [], []
    m = X_train.shape[1]
    start_time = time.time()

    for epoch in range(epochs):
        # Mezclar índices sin copiar matrices completas
        indices = np.random.permutation(m)
        epoch_loss = 0.0
        num_batches = int(np.ceil(m / batch_size))

        for i in range(0, m, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[:, batch_idx]
            Y_batch = Y_train[:, batch_idx]

            # --- Forward ---
            Y_hat, caches = forward_propagation(X_batch, parametros)
            loss = cross_entropy_loss(Y_batch, Y_hat)
            epoch_loss += loss

            # --- Backward ---
            grads = backward_propagation(X_batch, Y_batch, parametros, caches)

            # --- Actualización inmediata (mini-batch SGD) ---
            parametros = actualizar_parametros(parametros, grads, lr)

            # Liberar memoria intermedia
            del X_batch, Y_batch, Y_hat, grads, caches
            gc.collect()

        # Promedio de pérdida por epoch
        epoch_loss /= num_batches
        losses_train.append(epoch_loss)

        # Evaluar validación cada `val_interval` epochs
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            Y_hat_val, _ = forward_propagation(X_val, parametros)
            loss_val = cross_entropy_loss(Y_val, Y_hat_val)
            losses_val.append(loss_val)
            if verbose:
                print(f"Época {epoch+1}/{epochs} | Loss Train: {epoch_loss:.4f} | Loss Val: {loss_val:.4f}")
            del Y_hat_val
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Época {epoch+1}/{epochs} | Loss Train: {epoch_loss:.4f}")

        gc.collect()  # limpieza entre epochs

    total_time = time.time() - start_time
    print(f"\n✅ Entrenamiento completado en {total_time:.2f} segundos.")
    return parametros, losses_train, losses_val

import numpy as np
import time
import gc

def entrenar_red_adam(
    X_train, Y_train, X_val, Y_val, dims,
    epochs=50, lr=0.001, batch_size=256,
    beta1=0.9, beta2=0.999, epsilon=1e-8,
    verbose=True, val_interval=5
):
    """
    Entrenamiento con Mini-batch + Adam Optimizer.
    Incluye optimizaciones de memoria.
    """
    # --- Preparación de datos ---
    X_train = X_train.astype(np.float32, copy=False)
    X_val   = X_val.astype(np.float32, copy=False)

    parametros = inicializar_pesos(dims)
    for l in range(1, len(dims)):
        parametros[f"W{l}"] = parametros[f"W{l}"].astype(np.float32)
        parametros[f"b{l}"] = parametros[f"b{l}"].astype(np.float32)

    # Inicializar momentos Adam
    v = {}
    s = {}
    for l in range(1, len(dims)):
        v[f"dW{l}"] = np.zeros_like(parametros[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(parametros[f"b{l}"])
        s[f"dW{l}"] = np.zeros_like(parametros[f"W{l}"])
        s[f"db{l}"] = np.zeros_like(parametros[f"b{l}"])

    losses_train, losses_val = [], []
    m = X_train.shape[1]
    start_time = time.time()

    # --- Entrenamiento principal ---
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        epoch_loss = 0.0
        num_batches = int(np.ceil(m / batch_size))

        for i in range(0, m, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[:, batch_idx]
            Y_batch = Y_train[:, batch_idx]

            # Forward y pérdida
            Y_hat, caches = forward_propagation(X_batch, parametros)
            loss = cross_entropy_loss(Y_batch, Y_hat)
            epoch_loss += loss

            # Backpropagation
            grads = backward_propagation(X_batch, Y_batch, parametros, caches)

            # --- Actualización Adam ---
            t = epoch + 1  # para corrección de sesgo
            for l in range(1, len(dims)):
                # Promedios móviles
                v[f"dW{l}"] = beta1 * v[f"dW{l}"] + (1 - beta1) * grads[f"dW{l}"]
                v[f"db{l}"] = beta1 * v[f"db{l}"] + (1 - beta1) * grads[f"db{l}"]
                s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * (grads[f"dW{l}"] ** 2)
                s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * (grads[f"db{l}"] ** 2)

                # Correcciones de sesgo
                v_corr_dW = v[f"dW{l}"] / (1 - beta1 ** t)
                v_corr_db = v[f"db{l}"] / (1 - beta1 ** t)
                s_corr_dW = s[f"dW{l}"] / (1 - beta2 ** t)
                s_corr_db = s[f"db{l}"] / (1 - beta2 ** t)

                # Actualización de parámetros
                parametros[f"W{l}"] -= lr * v_corr_dW / (np.sqrt(s_corr_dW) + epsilon)
                parametros[f"b{l}"] -= lr * v_corr_db / (np.sqrt(s_corr_db) + epsilon)

            # Liberar memoria
            del X_batch, Y_batch, Y_hat, grads, caches
            gc.collect()

        # Promedio de pérdida por epoch
        epoch_loss /= num_batches
        losses_train.append(epoch_loss)

        # Validación cada val_interval epochs
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            Y_hat_val, _ = forward_propagation(X_val, parametros)
            loss_val = cross_entropy_loss(Y_val, Y_hat_val)
            losses_val.append(loss_val)
            if verbose:
                print(f"Época {epoch+1}/{epochs} | Loss Train: {epoch_loss:.4f} | Loss Val: {loss_val:.4f}")
            del Y_hat_val
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Época {epoch+1}/{epochs} | Loss Train: {epoch_loss:.4f}")

        gc.collect()

    print(f"\n✅ Entrenamiento con Adam completado en {time.time()-start_time:.2f} segundos.")
    return parametros, losses_train, losses_val

import numpy as np
import time
import gc
import copy

def entrenar_red_adam_regularizado(
    X_train, Y_train, X_val, Y_val, dims,
    epochs=100, lr=0.001, batch_size=256,
    beta1=0.9, beta2=0.999, epsilon=1e-8,
    lambda_l2=1e-4, patience=10, verbose=True, val_interval=5
):
    """
    Adam con regularización L2 + Early Stopping.
    """
    # --- Preparación de datos ---
    X_train = X_train.astype(np.float32, copy=False)
    X_val   = X_val.astype(np.float32, copy=False)

    parametros = inicializar_pesos(dims)
    for l in range(1, len(dims)):
        parametros[f"W{l}"] = parametros[f"W{l}"].astype(np.float32)
        parametros[f"b{l}"] = parametros[f"b{l}"].astype(np.float32)

    # Inicialización Adam
    v, s = {}, {}
    for l in range(1, len(dims)):
        v[f"dW{l}"] = np.zeros_like(parametros[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(parametros[f"b{l}"])
        s[f"dW{l}"] = np.zeros_like(parametros[f"W{l}"])
        s[f"db{l}"] = np.zeros_like(parametros[f"b{l}"])

    losses_train, losses_val = [], []
    best_loss = float('inf')
    best_params = copy.deepcopy(parametros)
    epochs_no_improve = 0

    m = X_train.shape[1]
    start_time = time.time()

    # --- Entrenamiento ---
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        epoch_loss = 0.0
        num_batches = int(np.ceil(m / batch_size))

        for i in range(0, m, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[:, batch_idx]
            Y_batch = Y_train[:, batch_idx]

            # --- Forward y pérdida ---
            Y_hat, caches = forward_propagation(X_batch, parametros)
            loss = cross_entropy_loss(Y_batch, Y_hat)

            # --- Agregar término L2 ---
            for l in range(1, len(dims)):
                loss += (lambda_l2 / (2 * batch_size)) * np.sum(np.square(parametros[f"W{l}"]))
            epoch_loss += loss

            # --- Backpropagation ---
            grads = backward_propagation(X_batch, Y_batch, parametros, caches)

            # --- Agregar derivada de regularización L2 ---
            for l in range(1, len(dims)):
                grads[f"dW{l}"] += (lambda_l2 / batch_size) * parametros[f"W{l}"]

            # --- Actualización Adam ---
            t = epoch + 1
            for l in range(1, len(dims)):
                v[f"dW{l}"] = beta1 * v[f"dW{l}"] + (1 - beta1) * grads[f"dW{l}"]
                v[f"db{l}"] = beta1 * v[f"db{l}"] + (1 - beta1) * grads[f"db{l}"]
                s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * (grads[f"dW{l}"] ** 2)
                s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * (grads[f"db{l}"] ** 2)

                v_corr_dW = v[f"dW{l}"] / (1 - beta1 ** t)
                v_corr_db = v[f"db{l}"] / (1 - beta1 ** t)
                s_corr_dW = s[f"dW{l}"] / (1 - beta2 ** t)
                s_corr_db = s[f"db{l}"] / (1 - beta2 ** t)

                parametros[f"W{l}"] -= lr * v_corr_dW / (np.sqrt(s_corr_dW) + epsilon)
                parametros[f"b{l}"] -= lr * v_corr_db / (np.sqrt(s_corr_db) + epsilon)

            del X_batch, Y_batch, Y_hat, grads, caches
            gc.collect()

        # --- Promedio por epoch ---
        epoch_loss /= num_batches
        losses_train.append(epoch_loss)

        # --- Validación ---
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            Y_hat_val, _ = forward_propagation(X_val, parametros)
            loss_val = cross_entropy_loss(Y_val, Y_hat_val)
            losses_val.append(loss_val)
            del Y_hat_val

            # Early stopping
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                best_params = copy.deepcopy(parametros)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️  Early stopping en epoch {epoch+1} | Mejor val_loss: {best_loss:.4f}")
                    parametros = best_params
                    break

            if verbose:
                print(f"Época {epoch+1}/{epochs} | Train: {epoch_loss:.4f} | Val: {loss_val:.4f}")

        gc.collect()

    print(f"\n✅ Entrenamiento completado en {time.time()-start_time:.2f} s.")
    return parametros, losses_train, losses_val
