import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2
import base64
import io
from PIL import Image
from collections import deque

# --- CONFIGURACIÓN ---
INPUT_SIZE = 126
HIDDEN_SIZE = 128
NUM_CLASSES = 37
SEQUENCE_LENGTH = 6
MODEL_PATH = "model/modelo_relativo_final.pth"

# --- MEDIAPIPE (CONFIGURACIÓN VISUAL) ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- DICCIONARIO ---
CLASES = {
    0: "1", 1: "10", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9",
    8: "A", 9: "B", 10: "C", 11: "D", 12: "E", 13: "F", 14: "G",
    15: "H", 16: "I", 17: "J", 18: "K", 19: "L", 20: "M", 21: "MIL",
    22: "MILLON", 23: "N", 24: "Ñ", 25: "O", 26: "P", 27: "Q",
    28: "R", 29: "S", 30: "T", 31: "U", 32: "V", 33: "W", 34: "X",
    35: "Y", 36: "Z"
}

# --- ARQUITECTURA ---
class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_classes=37):
        super(SignLanguageBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 256) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] 
        x = self.ln(last_time_step)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out

def normalize_hand(data_batch):
    batch = data_batch.clone()
    for i in range(batch.size(0)): 
        for t in range(batch.size(1)):
            wrist1_x = batch[i, t, 0]
            wrist1_y = batch[i, t, 1]
            wrist1_z = batch[i, t, 2]
            if wrist1_x != 0 or wrist1_y != 0:
                batch[i, t, 0:63:3] -= wrist1_x
                batch[i, t, 1:63:3] -= wrist1_y
                batch[i, t, 2:63:3] -= wrist1_z
    return batch

# --- APP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

hands = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

device = torch.device("cpu")
model = SignLanguageBiLSTM(num_classes=NUM_CLASSES)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ BACKEND ONLINE: Modo 'Video Loopback' Activado")
except:
    print("❌ Error Modelo")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    try:
        while True:
            data = await websocket.receive_text()
            if "data:image" in data: data = data.split(",")[1]
            
            image_bytes = base64.b64decode(data)
            frame = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
            
            # Procesar
            results = hands.process(frame)
            
            # --- DIBUJAR SOBRE LA IMAGEN ---
            # Convertimos a BGR para OpenCV
            annotated_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            final_label = "..."
            final_conf = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                # Extracción de datos para IA
                kps = np.zeros(126)
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if i >= 2: break
                    temp = []
                    for p in hand_landmarks.landmark:
                        temp.extend([p.x * 640, p.y * 480, p.z * 640]) 
                    kps[i*63 : (i*63)+63] = temp
                
                sequence_buffer.append(kps)
                
                # Inferencia IA
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    try:
                        np_data = np.array(sequence_buffer, dtype=np.float32)
                        input_tensor = torch.tensor(np_data).unsqueeze(0).to(device)
                        input_tensor = normalize_hand(input_tensor)
                        with torch.no_grad():
                            output = model(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            confidence, idx = torch.max(probs, 1)
                            ai_conf = confidence.item() * 100
                            idx_val = idx.item()
                            
                            # Filtro: Ignoramos la P si no estamos muy seguros
                            if idx_val != 26 or ai_conf > 90:
                                final_label = CLASES.get(idx_val, "?")
                                final_conf = ai_conf
                    except: pass
            
            # --- PREPARAR RESPUESTA VISUAL ---
            # 1. Escribir texto en la imagen
            if final_conf > 0:
                cv2.putText(annotated_image, f"IA: {final_label} ({final_conf:.0f}%)", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 2. Convertir imagen procesada de vuelta a Base64 para enviarla a React
            _, buffer = cv2.imencode('.jpg', annotated_image)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # 3. Enviar todo junto
            response = {
                "status": "prediction", 
                "label": final_label, 
                "confidence": f"{final_conf:.0f}%",
                "image": f"data:image/jpeg;base64,{jpg_as_text}" # <--- ¡LA IMAGEN CON RAYAS!
            }
            await websocket.send_json(response)

    except Exception as e:
        print(f"Desconectado: {e}")