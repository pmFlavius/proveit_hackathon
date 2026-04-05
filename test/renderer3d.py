from ursina import *
import socket
import json
import threading
import time

# --- SETUP RETEA ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5006

state_lock = threading.Lock()
state_data = {
    "risk_level": "scazut",
    "speed": 60,
    "brake": "fara frana",
    "objects": []
}

def udp_listener():
    """Ascultă JSON-ul trimis de server_logic.py și actualizează starea."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    print("[RENDER 3D] Ascult date pe portul 5005...")
    
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            packet = json.loads(data.decode('utf-8'))
            ego = packet.get("ego_vehicle", {})
            
            with state_lock:
                state_data["risk_level"] = ego.get("risk_level", "scazut")
                state_data["speed"] = ego.get("speed_kmh", 0)
                state_data["brake"] = ego.get("decisions", {}).get("brake", "fara frana")
                state_data["objects"] = packet.get("detected_objects", [])
        except Exception as e:
            pass

# Pornim firul de execuție pentru rețea
threading.Thread(target=udp_listener, daemon=True).start()

# --- SETUP ENGINE URSINA ---
# --- SETUP ENGINE URSINA ---
app = Ursina(title="Drive Assist 3D", borderless=False)

# 1. SCOATE BUTOANELE GENERATE AIUREA DE URSINA
window.exit_button.visible = False
window.fps_counter.enabled = False # Ascunde și contorul de FPS din colț dacă vrei ecran curat

# Camera
camera.position = (0, 3, -10)
camera.rotation_x = 10

# Mediul 3D
Sky() 
road = Entity(model='cube', color=color.dark_gray, scale=(12, 0.1, 200), position=(0, 0, 50))
linie_st = Entity(model='cube', color=color.white, scale=(0.2, 0.11, 200), position=(-2, 0, 50))
linie_dr = Entity(model='cube', color=color.white, scale=(0.2, 0.11, 200), position=(2, 0, 50))

# Mașina Noastră
ego_car = Entity(model='cube', color=color.azure, scale=(1.8, 1.2, 4), position=(0, 0.6, 0))

# 2. FIXARE HUD (Fără ieșit din ecran)
# Am mutat position X la -0.80 (mai spre centru) 
# origin=(-0.5, 0.5) ancorează textul stânga-sus, așa că dacă e lung, crește spre dreapta, nu iese din ecran.
hud_viteza = Text(text="0 km/h", position=(-0.80, 0.45), origin=(-0.5, 0.5), scale=1.8, color=color.white)
hud_risc   = Text(text="RISC: -", position=(-0.80, 0.38), origin=(-0.5, 0.5), scale=1.3)
hud_frana  = Text(text="FRANA: -", position=(-0.80, 0.33), origin=(-0.5, 0.5), scale=1.3, color=color.orange)

masini_trafic = {}

def update():
    """Această funcție este apelată automat de Ursina la fiecare cadru (de ex. 60 FPS)"""
    with state_lock:
        obiecte_curente = list(state_data["objects"])
        risc = state_data["risk_level"]
        viteza = state_data["speed"]
        frana = state_data["brake"]

    # 1. Update HUD
    hud_viteza.text = f"{int(viteza)} km/h"
    hud_frana.text = f"FRANA: {frana.upper()}"
    hud_risc.text = f"RISC: {risc.upper()}"
    
    if risc == "ridicat": hud_risc.color = color.red
    elif risc == "mediu": hud_risc.color = color.yellow
    else: hud_risc.color = color.green

    # 2. Gestionare Mașini din Trafic
    id_uri_primite = []
    
    for obj in obiecte_curente:
        obj_id = obj["id"]
        id_uri_primite.append(obj_id)
        
        # Extragem datele din JSON
        distanta = float(obj.get("distance", 30))
        banda = obj.get("assigned_lane", "dreapta")
        tip = obj.get("type", "car")
        
        # Calculăm coordonata X în funcție de bandă
        target_x = 0
        if banda == "stanga": target_x = -3.5
        elif banda == "dreapta": target_x = 3.5
        
        # Calculăm coordonata Z (adâncimea / distanța în față)
        target_z = distanta * 0.8  # Factor de scalare pentru a arăta bine pe ecran
        
        # Stabilim culoarea
        c = color.red if obj.get("is_in_my_lane") and distanta < 25 else color.light_gray
        if tip == "person": c = color.magenta
        elif tip == "stop sign": c = color.red
        
        # Dacă mașina nu există în scena 3D, o creăm
        if obj_id not in masini_trafic:
            forma = 'sphere' if tip == 'person' else 'cube'
            dimensiune = (1, 1.8, 1) if tip == 'person' else (1.8, 1.2, 4)
            
            noua_masina = Entity(
                model=forma,
                color=c,
                scale=dimensiune,
                position=(target_x, 0.6, target_z)
            )
            masini_trafic[obj_id] = noua_masina
        else:
            # Dacă există, îi actualizăm poziția fluid (lerp) și culoarea
            entitate = masini_trafic[obj_id]
            entitate.color = c
            # Lerp face mișcarea lină, ca un joc video real, chiar dacă JSON-ul vine greu
            entitate.x = lerp(entitate.x, target_x, time.dt * 5)
            entitate.z = lerp(entitate.z, target_z, time.dt * 10)

    # 3. Curățenie: Ștergem mașinile care nu mai sunt în JSON (au ieșit din cadru)
    id_uri_de_sters = []
    for vechi_id in masini_trafic.keys():
        if vechi_id not in id_uri_primite:
            destroy(masini_trafic[vechi_id]) # O ștergem din engine-ul 3D
            id_uri_de_sters.append(vechi_id)
            
    for id_sters in id_uri_de_sters:
        del masini_trafic[id_sters]

    # Efect vizual opțional: mișcăm liniile drumului înapoi pentru iluzia de viteză
    viteza_iluzie = (viteza / 3.6) * time.dt # m/s
    # Putem adăuga detalii de asfalt mai târziu aici

# Pornim engine-ul 3D
app.run()