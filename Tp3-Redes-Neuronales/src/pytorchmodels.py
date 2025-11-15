import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

# ---------------------------------------------------------------
# Función 1: crear dataset de PyTorch a partir de NumPy
# ---------------------------------------------------------------
def prepare_dataloaders(X_train_flat, y_train, X_val_flat, y_val, batch_size=256):
    """Convierte los datos numpy en DataLoaders de PyTorch."""
    X_train_t = torch.tensor(X_train_flat.T, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val_flat.T,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ---------------------------------------------------------------
# Función 2: definir el modelo según arquitectura dada
# ---------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dims, activation="ReLU", dropout_p=0.0):
        """dims: lista de dimensiones, ej [784,256,128,47]
           activation: nombre ('ReLU','LeakyReLU','SiLU','Swish','GELU'),
                       una clase de nn.Module (e.g. nn.ReLU) o una instancia/fábrica.
           dropout_p: probabilidad de dropout entre capas ocultas (0.0 = sin dropout)
        """
        super().__init__()

        act_map = {
            "ReLU": lambda: nn.ReLU(),
            "LeakyReLU": lambda: nn.LeakyReLU(0.1),
            "SiLU": lambda: nn.SiLU(),
            "Swish": lambda: nn.SiLU(),
            "GELU": lambda: nn.GELU()
        }

        # Determinar fábrica de activaciones
        if isinstance(activation, str):
            act_factory = act_map.get(activation, lambda: nn.ReLU())
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            act_factory = lambda: activation()
        elif isinstance(activation, nn.Module):
            # crear nuevas instancias de la misma clase que la instancia dada
            act_factory = lambda: activation.__class__()
        elif callable(activation):
            # asumir que activation es una fábrica callable que devuelve un nn.Module
            act_factory = lambda: activation()
        else:
            act_factory = lambda: nn.ReLU()

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_factory())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------------
# Función 3: entrenamiento del modelo
# ---------------------------------------------------------------
def train_pytorch_model(X_train_flat, y_train, X_val_flat, y_val,
                        dims=[784,256,128,47],
                        lr=0.002, lambda_l2=1e-5,
                        batch_size=256, epochs=20, verbose=True, activation="ReLU", dropout_p=0.0):
    """
    Entrena una red MLP con PyTorch usando Adam y CrossEntropyLoss.
    Devuelve el modelo entrenado y las listas de pérdidas.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_dataloaders(X_train_flat, y_train, X_val_flat, y_val, batch_size)

    model = MLP(dims, activation=activation, dropout_p=dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(epochs):
        # ---- Entrenamiento ----
        model.train()
        run_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        train_losses.append(run_loss / len(train_loader))

        # ---- Validación ----
        model.eval()
        vloss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vloss += criterion(out, yb).item()
                _, pred = torch.max(out, 1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()
        val_losses.append(vloss / len(val_loader))
        val_acc = correct / total

        if verbose:
            print(f"Época {epoch+1}/{epochs} | Train {train_losses[-1]:.4f} | "
                  f"Val {val_losses[-1]:.4f} | Val Acc {val_acc:.4f}")

    print(f"\n✅ Entrenamiento completado en {time.time()-t0:.2f}s")
    return model, train_losses, val_losses

# ---------------------------------------------------------------
# Función 4: evaluación final del modelo
# ---------------------------------------------------------------
def evaluate_model(model, X_val_flat, y_val, batch_size=256):
    """Evalúa accuracy del modelo entrenado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_val_t = torch.tensor(X_val_flat.T, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, pred = torch.max(out, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
    return correct / total


# ===============================================================
#  MODEL M3
# ===============================================================

import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# ======================================================
# Función auxiliar: crear modelo flexible
# ======================================================
def build_mlp(input_dim, hidden_layers, output_dim,
              activation="ReLU", dropout_p=0.0):
    act_map = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(0.1),
        "SiLU": nn.SiLU(),
        "Swish": nn.SiLU(),
        "GELU": nn.GELU()
    }
    layers = []
    dims = [input_dim] + hidden_layers
    for i in range(len(hidden_layers)):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(act_map.get(activation, nn.ReLU()))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)

# ======================================================
# Función de entrenamiento genérica
# ======================================================
def train_model(model, X_train, y_train, X_val, y_val,
                lr=0.001, lambda_l2=1e-4, batch_size=256, epochs=10, verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear DataLoaders
    X_train_t = torch.tensor(X_train.T, dtype=torch.float32)   
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val.T, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)

    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluación final
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
    val_loss /= len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc

# ======================================================
# Script de búsqueda M3
# ======================================================

def search_pytorch_M3(X_train, y_train, X_val, y_val,
                      architectures, activations, dropouts,
                      lr=0.001, lambda_l2=1e-4, batch_size=256, epochs=10):

    results = []
    model_id = 1

    for arch in architectures:
        for act in activations:
            for d in dropouts:
                print(f"\n🧠 Entrenando Modelo {model_id}: {arch}, act={act}, dropout={d}")
                dims = [784] + arch + [47]  # entrada y salida fijas para EMNIST
                start = time.time()

                try:
                    model = build_mlp(784, arch, 47, activation=act, dropout_p=d)
                    val_loss, val_acc = train_model(
                        model, X_train, y_train, X_val, y_val,
                        lr=lr, lambda_l2=lambda_l2, batch_size=batch_size, epochs=epochs
                    )
                    elapsed = time.time() - start

                    results.append({
                        "Modelo": model_id,
                        "Capas ocultas": arch,
                        "Activación": act,
                        "Dropout": d,
                        "Learning rate": lr,
                        "L2": lambda_l2,
                        "Batch": batch_size,
                        "Epochs": epochs,
                        "Val Loss": val_loss,
                        "Val Acc": val_acc,
                        "Tiempo (s)": round(elapsed, 2)
                    })

                except Exception as e:
                    print(f"❌ Error en modelo {model_id}: {e}")
                    results.append({
                        "Modelo": model_id,
                        "Capas ocultas": arch,
                        "Activación": act,
                        "Dropout": d,
                        "Error": str(e)
                    })

                model_id += 1

    # Exportar resultados
    df = pd.DataFrame(results)
    df.sort_values(by="Val Loss", inplace=True)
    df.to_csv("M3_search_results.csv", index=False)
    print("\n✅ Búsqueda completada. Resultados guardados en 'M3_search_results.csv'")
    return df

# ===============================================================

def predict_pytorch_model(model, X):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_t)
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()


def f1_score_macro(model, X_val_flat, y_val):
    y_val_pred = predict_pytorch_model(model, X_val_flat.T)
    num_classes = np.unique(y_val).size
    f1_per_class = []
    for c in range(num_classes):
        tp = np.sum((y_val == c) & (y_val_pred == c))
        fp = np.sum((y_val != c) & (y_val_pred == c))
        fn = np.sum((y_val == c) & (y_val_pred != c))

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_per_class.append(f1)
    return np.mean(f1_per_class)