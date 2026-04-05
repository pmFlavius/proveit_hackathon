import cv2
import numpy as np
import queue
import threading
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Geometrie pentru video-ul tău
_Y_JOS      = 720
_Y_ORIZONT  = 490
_ST_X_JOS, _ST_X_SUS = 300, 455
_DR_X_JOS, _DR_X_SUS = 650, 590

_POLY_LEFT  = np.poly1d(np.polyfit([_Y_JOS, _Y_ORIZONT], [_ST_X_JOS, _ST_X_SUS], 1))
_POLY_RIGHT = np.poly1d(np.polyfit([_Y_JOS, _Y_ORIZONT], [_DR_X_JOS, _DR_X_SUS], 1))

_FOCAL_LENGTH = 600
_KNOWN_WIDTH  = 1.8

_tracking = {}
_TRACK_TIMEOUT = 2.0

def _estimeaza_viteza_relativa(track_key, dist_curenta):
    global _tracking
    now = time.time()
    if track_key in _tracking:
        dist_ant, timp_ant, viteza_ant = _tracking[track_key]
        dt = now - timp_ant
        if 0.05 < dt < _TRACK_TIMEOUT:
            delta = dist_ant - dist_curenta 
            viteza_raw = (delta / dt) * 3.6 
            viteza_smooth = 0.6 * viteza_raw + 0.4 * viteza_ant
            _tracking[track_key] = (dist_curenta, now, viteza_smooth)
            return round(viteza_smooth, 1)

    _tracking[track_key] = (dist_curenta, now, 0.0)
    return 0.0

def obtine_limite_banda(image):
    h, w = image.shape[:2]
    if w == 1280 and h == 720:
        return _POLY_LEFT, _POLY_RIGHT
    sx, sy = w / 1280.0, h / 720.0
    yj = _Y_JOS * sy
    yo = _Y_ORIZONT * sy
    pl = np.poly1d(np.polyfit([yj, yo], [_ST_X_JOS*sx, _ST_X_SUS*sx], 1))
    pr = np.poly1d(np.polyfit([yj, yo], [_DR_X_JOS*sx, _DR_X_SUS*sx], 1))
    return pl, pr

def motor_inteligenta_artificiala(interfata, coada):
    obj_id_counter = 0

    while interfata.running:
        try:
            cadru_original, frame_idx = coada.get(timeout=0.1)
        except queue.Empty:
            continue

        cadru = cadru_original.copy()
        height, width = cadru.shape[:2]

        poly_left, poly_right = obtine_limite_banda(cadru)
        results = model(cadru, verbose=False)
        obiecte_detectate = []
        cadru_adnotat = results[0].plot()

        try:
            y_b = height
            y_t = int(height * (_Y_ORIZONT / 720.0))
            xbl = int(np.clip(poly_left(y_b),  0, width-1))
            xtl = int(np.clip(poly_left(y_t),  0, width-1))
            xbr = int(np.clip(poly_right(y_b), 0, width-1))
            xtr = int(np.clip(poly_right(y_t), 0, width-1))
            overlay = cadru_adnotat.copy()
            pts = np.array([[xbl,y_b],[xtl,y_t],[xtr,y_t],[xbr,y_b]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cadru_adnotat = cv2.addWeighted(cadru_adnotat, 0.78, overlay, 0.22, 0)
            cv2.line(cadru_adnotat, (xbl,y_b), (xtl,y_t), (0,255,0), 4)
            cv2.line(cadru_adnotat, (xbr,y_b), (xtr,y_t), (0,255,0), 4)
        except Exception:
            pass

        for r in results:
            for box in r.boxes:
                cls_id     = int(box.cls[0])
                nume_clasa = model.names[cls_id]
                if nume_clasa not in ["car","truck","bus","person", "stop sign","traffic light","motorcycle"]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w_box    = x2 - x1
                x_center = (x1 + x2) / 2
                y_rot    = y2

                distanta = round((_KNOWN_WIDTH * _FOCAL_LENGTH) / w_box, 1) if w_box > 0 else 80.0

                lim_st = float(poly_left(y_rot))
                lim_dr = float(poly_right(y_rot))
                is_in_my_lane = bool(lim_st < x_center < lim_dr)

                banda_approx = "ego" if is_in_my_lane else ("dr" if x_center > lim_dr else "st")
                track_key    = f"{nume_clasa}_{banda_approx}_{obj_id_counter}" # ID adăugat pt unicitate
                viteza_rel   = _estimeaza_viteza_relativa(track_key, distanta)

                obiecte_detectate.append({
                    "id":              obj_id_counter,
                    "type":            nume_clasa,
                    "x_center":        x_center,
                    "distanta":        distanta,
                    "is_in_my_lane":   is_in_my_lane,
                    "lim_stanga":      lim_st,
                    "lim_dreapta":     lim_dr,
                    "frame_width":     float(width),
                    "viteza_relativa": viteza_rel,
                })
                obj_id_counter += 1

        with interfata.data_lock:
            interfata.last_detections = obiecte_detectate
            interfata.frame_count     = frame_idx

        interfata.actualizeaza_imagine_ui(cadru_adnotat)

def process_and_parse_video(cale_video, interfata):
    cap   = cv2.VideoCapture(cale_video)
    coada = queue.Queue(maxsize=3)
    threading.Thread(target=motor_inteligenta_artificiala, args=(interfata, coada), daemon=True).start()
    nr_cadru = 0
    
    while cap.isOpened() and interfata.running:
        succes, cadru = cap.read()
        if not succes:
            break
        nr_cadru += 1
        cadru_redimensionat = cv2.resize(cadru, (1280, 720))
        try:
            coada.put((cadru_redimensionat, nr_cadru), timeout=0.1)
            time.sleep(0.01)
        except queue.Full:
            pass
            
    cap.release()