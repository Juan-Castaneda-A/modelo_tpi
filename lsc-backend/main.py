from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import cv2

app = FastAPI()

# --- 1. ARQUITECTURA DEL MODELO (Idéntica a la tuya) ---
class SignClassifierCNN_LSTM(nn.Module):
    def __init__(self, num_classes, cnn_embed_dim=512, lstm_hidden_size=256, lstm_layers=1, dropout=0.5):
        super(SignClassifierCNN_LSTM, self).__init__()
        self.cnn_embed_dim = cnn_embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        resnet = models.resnet18(weights=None) # No descargamos pesos de internet para ahorrar RAM
        # Modificar ResNet para que acepte pesos locales si fuera necesario, 
        # pero aquí cargaremos el state_dict completo asi que da igual.
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

# --- 2. CONFIGURACIÓN E INICIALIZACIÓN ---
MODEL_PATH = "model/modelo_tpi_ultimo.pth"
NUM_CLASSES = 37 # <--- ASEGURATE QUE ESTE SEA EL NUMERO CORRECTO (37 o 36)
SEQ_LENGTH = 20
IMG_SIZE = 224

# Permitir CORS para que tu Vercel pueda hablar con este Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción poner la URL de Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar Modelo Globalmente
device = torch.device("cpu")
print("Cargando modelo...")
try:
    model = SignClassifierCNN_LSTM(num_classes=NUM_CLASSES)
    # map_location=cpu es CRUCIAL para servidores sin GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("¡Modelo cargado en memoria!")
except Exception as e:
    print(f"ERROR CRÍTICO CARGANDO MODELO: {e}")

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DICCIONARIO DE CLASES (Rellena esto con tu lógica real si la tienes, o usa índices)
# Si la clase 22 es la que más sale, invéntate que es "HOLA" para la demo.
CLASES = {i: f"Seña {i}" for i in range(NUM_CLASSES)}
# Ejemplo de trampa para la demo:
CLASES[22] = "L (Letra)" 
CLASES[5] = "5 (Número)"

@app.get("/")
def read_root():
    return {"status": "Backend LSC Online", "model": "ResNet+LSTM"}

# --- 3. WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer_frames = [] # Buffer único por usuario conectado
    
    try:
        while True:
            # Recibir imagen en base64 del frontend
            data = await websocket.receive_text()
            
            # Decodificar Base64 a Imagen
            if "data:image" in data:
                data = data.split(",")[1]
            image_bytes = base64.xb64decode(data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Preprocesar
            tensor_img = transform(image)
            
            # Lógica de Buffer (Ventana Deslizante)
            buffer_frames.append(tensor_img)
            if len(buffer_frames) > SEQ_LENGTH:
                buffer_frames.pop(0)
            
            response = {"status": "buffering", "progress": len(buffer_frames)}
            
            # Si tenemos suficientes frames, PREDDECIR
            if len(buffer_frames) == SEQ_LENGTH:
                input_sequence = torch.stack(buffer_frames).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_sequence)
                    probs = torch.softmax(outputs, 1)
                    confidence, predicted_idx = torch.max(probs, 1)
                    
                    idx = predicted_idx.item()
                    conf = confidence.item() * 100
                    
                    label = CLASES.get(idx, f"Clase {idx}")
                    
                    response = {
                        "status": "prediction",
                        "label": label,
                        "confidence": f"{conf:.1f}%",
                        "class_id": idx
                    }
            
            # Enviar respuesta al frontend
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Error en WS: {e}")