import cv2
import numpy as np
import queue
import threading
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# --- Fallback pentru Benzi (în caz că detecția dinamică eșuează) ---
_Y_JOS      = 720
_Y_ORIZONT  = 490
_ST_X_JOS, _ST_X_SUS = 300, 455
_DR_X_JOS, _DR_X_SUS = 650, 590

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

def estimeaza_vreme_si_drum(cadru):
    hsv = cv2.cvtColor(cadru, cv2.COLOR_BGR2HSV)
    v_mean = hsv[:,:,2].mean()
    
    if v_mean < 60:
        return {"time_of_day": "night", "weather": "clear", "surface": "asfalt_rece_aderenta_scazuta"}
    elif v_mean > 200:
        return {"time_of_day": "day", "weather": "fog", "surface": "asfalt_umed"}
    return {"time_of_day": "day", "weather": "clear", "surface": "asfalt_uscat_aderenta_optima"}

def obtine_limite_banda(image):
    """Lane Detection Dinamic + Fallback pe coordonate fixe"""
    h, w = image.shape[:2]
    
    # 1. Procesare imagine pentru detectarea marginilor
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 2. Creare mască (Region of Interest)
    poligons = np.array([[(100, h), (w//2 - 50, int(h*0.6)), (w//2 + 50, int(h*0.6)), (w-100, h)]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, poligons, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 3. Detectie Linii cu Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=80, maxLineGap=50)
    
    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            slope = (y2 - y1) / (x2 - x1)
            # Filtram liniile prea orizontale
            if slope < -0.4: left_lines.append((x1, y1, x2, y2))
            elif slope > 0.4: right_lines.append((x1, y1, x2, y2))
            
    # Fallback default (polinoamele tale vechi, updatate la rezoluția curentă)
    sx, sy = w / 1280.0, h / 720.0
    pl_default = np.poly1d(np.polyfit([_Y_JOS * sy, _Y_ORIZONT * sy], [_ST_X_JOS * sx, _ST_X_SUS * sx], 1))
    pr_default = np.poly1d(np.polyfit([_Y_JOS * sy, _Y_ORIZONT * sy], [_DR_X_JOS * sx, _DR_X_SUS * sx], 1))

    # Construim noul polinom stanga daca avem linii
    if len(left_lines) > 0:
        lx = [pt[0] for pt in left_lines] + [pt[2] for pt in left_lines]
        ly = [pt[1] for pt in left_lines] + [pt[3] for pt in left_lines]
        try:
            pl_default = np.poly1d(np.polyfit(ly, lx, 1))
        except: pass

    # Construim noul polinom dreapta daca avem linii
    if len(right_lines) > 0:
        rx = [pt[0] for pt in right_lines] + [pt[2] for pt in right_lines]
        ry = [pt[1] for pt in right_lines] + [pt[3] for pt in right_lines]
        try:
            pr_default = np.poly1d(np.polyfit(ry, rx, 1))
        except: pass

    return pl_default, pr_default

def motor_inteligenta_artificiala(interfata, coada):
    obj_id_counter = 0

    while interfata.running:
        try:
            cadru_original, frame_idx = coada.get(timeout=0.1)
        except queue.Empty:
            continue

        cadru = cadru_original.copy()
        height, width = cadru.shape[:2]

        # Extragem vremea și o actualizăm în interfață
        mediu_estimat = estimeaza_vreme_si_drum(cadru)
        with interfata.data_lock:
            interfata.mediu_curent = mediu_estimat

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
            cv2.fillPoly(overlay, [pts], (0, 150, 0)) # Am facut verdele mai discret
            cadru_adnotat = cv2.addWeighted(cadru_adnotat, 0.78, overlay, 0.22, 0)
            cv2.line(cadru_adnotat, (xbl,y_b), (xtl,y_t), (0,255,0), 3)
            cv2.line(cadru_adnotat, (xbr,y_b), (xtr,y_t), (0,255,0), 3)
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
                track_key    = f"{nume_clasa}_{banda_approx}_{obj_id_counter}" 
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