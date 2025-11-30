import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

# --- 1. ARQUITECTURA (NO TOCAR) ---
class SignClassifierCNN_LSTM(nn.Module):
    def __init__(self, num_classes, cnn_embed_dim=512, lstm_hidden_size=256, lstm_layers=1, dropout=0.5):
        super(SignClassifierCNN_LSTM, self).__init__()
        self.cnn_embed_dim = cnn_embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.cnn_extractor = nn.Sequential(*modules)
        self.lstm = nn.LSTM(input_size=cnn_embed_dim, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden_size, 128), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x_flat = x.view(batch_size * seq_len, C, H, W)
        cnn_out = self.cnn_extractor(x_flat)
        cnn_out_flat = cnn_out.view(batch_size * seq_len, -1)
        lstm_input = cnn_out_flat.view(batch_size, seq_len, self.cnn_embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        final_lstm_output = lstm_out[:, -1, :]
        output_logits = self.classifier(final_lstm_output)
        return output_logits

# --- 2. CONFIGURACIÓN ---
MODEL_PATH = 'modelo_tpi_ultimo.pth' 
SEQ_LENGTH = 20
IMG_SIZE = 224
NUM_CLASSES = 37 # <--- CONFIRMADO QUE ES 37

device = torch.device("cpu") # Forzamos CPU para evitar errores raros
print(f"Modo Optimizado CPU.")

# --- 3. CARGAR MODELO ---
model = SignClassifierCNN_LSTM(num_classes=NUM_CLASSES)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Modelo cargado OK.")
except Exception as e:
    print(f"Error: {e}")
    exit()

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. BUCLE OPTIMIZADO ---
# Aquí usa el índice que te funcionó (0 o 1)
cap = cv2.VideoCapture(1) 

buffer_frames = []
prediction_text = "Cargando..."
frame_count = 0

print("Iniciando... Haz movimientos claros.")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        display_frame = frame.copy()

        # Procesamos la imagen SIEMPRE para llenar el buffer
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor_img = transform(pil_img)
        buffer_frames.append(tensor_img)
        if len(buffer_frames) > SEQ_LENGTH:
            buffer_frames.pop(0)

        # --- TRUCO DE OPTIMIZACIÓN ---
        # Solo predecimos 1 de cada 5 cuadros.
        # Si el buffer está lleno Y es el turno de predecir:
        if len(buffer_frames) == SEQ_LENGTH and frame_count % 5 == 0:
            
            input_sequence = torch.stack(buffer_frames).unsqueeze(0).to(device)
            outputs = model(input_sequence)
            
            # Obtenemos probabilidades para ver si duda
            probs = torch.softmax(outputs, 1)
            confidence, predicted_idx = torch.max(probs, 1)
            
            clase_actual = predicted_idx.item()
            confianza = confidence.item() * 100
            
            print(f"Predicción: {clase_actual} | Confianza: {confianza:.1f}%") # Debug en consola
            
            prediction_text = f"Clase: {clase_actual} ({confianza:.0f}%)"

        # Dibujar (Esto es liviano)
        cv2.rectangle(display_frame, (0,0), (350, 60), (0,0,0), -1)
        cv2.putText(display_frame, prediction_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('TPI Optimizado', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()