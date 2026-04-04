import cv2
import numpy as np
import math
import threading
import queue
from ultralytics import YOLO

memorie_banda = None
istoric_banda = []  
memorie_linie_stanga = None
memorie_linie_dreapta = None

def extrage_zona_interes(imagine, puncte):
    masca = np.zeros_like(imagine)
    cv2.fillPoly(masca, puncte, 255)
    return cv2.bitwise_and(imagine, masca)

def calculeaza_distanta(latime_obiect_pixeli, clasa_obiect):
    focal_length = 600
    latimi_reale = {
        'car': 1.8, 'truck': 2.5, 'bus': 2.8, 
        'person': 0.5, 'stop sign': 0.6, 'traffic light': 0.4
    }
    latime_reala = latimi_reale.get(clasa_obiect, 1.5)
    return (latime_reala * focal_length) / latime_obiect_pixeli if latime_obiect_pixeli > 0 else 0

def analizeaza_conditii_mediu(imagine):
    conditii = {"time_of_day": "day", "weather": "clear"}
    
    hsv = cv2.cvtColor(imagine, cv2.COLOR_BGR2HSV)
    luminozitate_medie = np.mean(hsv[:, :, 2])
    
    if luminozitate_medie < 60:
        conditii["time_of_day"] = "night"
        
    gray = cv2.cvtColor(imagine, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    
    if contrast < 40 and luminozitate_medie > 90:
        conditii["weather"] = "fog"
        
    return conditii

def detecteaza_marcaje_banda(imagine):
    hls = cv2.cvtColor(imagine, cv2.COLOR_BGR2HLS)
    
    masca_alb = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    masca_galben = cv2.inRange(hls, np.array([15, 50, 100]), np.array([35, 255, 255]))
    
    return cv2.bitwise_or(masca_alb, masca_galben)

def ferestra_glisanta(imagine_binara, n_ferestre=10, latime_fereastra=100, prag_pixeli=50):
    h, w = imagine_binara.shape
    roi_jos = int(h * 0.5)
    imagine_roi = imagine_binara[roi_jos:, :]

    histograma = np.sum(imagine_roi, axis=0)
    punct_mijloc = w // 2

    stanga_x = np.argmax(histograma[:punct_mijloc])
    dreapta_x = np.argmax(histograma[punct_mijloc:]) + punct_mijloc

    if histograma[stanga_x] < prag_pixeli: stanga_x = int(w * 0.25)
    if histograma[dreapta_x] < prag_pixeli: dreapta_x = int(w * 0.75)

    inaltime_fereastra = h // n_ferestre
    puncte_stanga, puncte_dreapta = [], []
    x_stanga_curent, x_dreapta_curent = stanga_x, dreapta_x

    for fereastra in range(n_ferestre):
        y_top = h - (fereastra + 1) * inaltime_fereastra
        y_bottom = h - fereastra * inaltime_fereastra

        if y_bottom < h * 0.45: continue

        x_stanga_min = max(0, int(x_stanga_curent - latime_fereastra))
        x_stanga_max = min(w, int(x_stanga_curent + latime_fereastra))
        x_dreapta_min = max(0, int(x_dreapta_curent - latime_fereastra))
        x_dreapta_max = min(w, int(x_dreapta_curent + latime_fereastra))

        fereastra_stanga = imagine_binara[y_top:y_bottom, x_stanga_min:x_stanga_max]
        fereastra_dreapta = imagine_binara[y_top:y_bottom, x_dreapta_min:x_dreapta_max]

        ys_stanga, xs_stanga = np.where(fereastra_stanga > 0)
        ys_dreapta, xs_dreapta = np.where(fereastra_dreapta > 0)

        if len(xs_stanga) > prag_pixeli:
            x_stanga_curent = np.mean(xs_stanga) + x_stanga_min
            puncte_stanga.append((x_stanga_curent, (y_top + y_bottom) / 2))

        if len(xs_dreapta) > prag_pixeli:
            x_dreapta_curent = np.mean(xs_dreapta) + x_dreapta_min
            puncte_dreapta.append((x_dreapta_curent, (y_top + y_bottom) / 2))

    return puncte_stanga, puncte_dreapta

def ajusteaza_linie(puncte, grad=2):
    if len(puncte) < 3: return None
    puncte = np.array(puncte)
    try:
        return np.poly1d(np.polyfit(puncte[:, 1], puncte[:, 0], grad))
    except:
        return None

def detecteaza_banda_vizual(imagine):
    global memorie_banda, istoric_banda, memorie_linie_stanga, memorie_linie_dreapta
    h, w = imagine.shape[:2]

    banda_implicita = np.array([[(int(w * 0.20), h), (int(w * 0.40), int(h * 0.65)), (int(w * 0.60), int(h * 0.65)), (int(w * 0.80), h)]], dtype=np.int32)
    if memorie_banda is None: memorie_banda = banda_implicita.copy()

    puncte_roi = np.array([[(int(w * 0.05), h), (int(w * 0.45), int(h * 0.55)), (int(w * 0.55), int(h * 0.55)), (int(w * 0.95), h)]], np.int32)

    masca_marcaje = detecteaza_marcaje_banda(imagine)
    masca_roi = np.zeros_like(masca_marcaje)
    cv2.fillPoly(masca_roi, puncte_roi, 255)
    
    kernel = np.ones((5, 5), np.uint8)
    masca_curata = cv2.morphologyEx(cv2.bitwise_and(masca_marcaje, masca_roi), cv2.MORPH_CLOSE, kernel)
    masca_curata = cv2.morphologyEx(masca_curata, cv2.MORPH_OPEN, kernel)

    puncte_stanga, puncte_dreapta = ferestra_glisanta(masca_curata, n_ferestre=12, latime_fereastra=80, prag_pixeli=25)

    linie_stanga = ajusteaza_linie(puncte_stanga) if len(puncte_stanga) >= 4 else None
    linie_dreapta = ajusteaza_linie(puncte_dreapta) if len(puncte_dreapta) >= 4 else None
    gasit_linii = False
    poligon_nou = None

    if linie_stanga is not None:
        if memorie_linie_stanga is not None:
            linie_stanga = np.poly1d(0.6 * memorie_linie_stanga.coefficients + 0.4 * linie_stanga.coefficients)
        memorie_linie_stanga = linie_stanga

    if linie_dreapta is not None:
        if memorie_linie_dreapta is not None:
            linie_dreapta = np.poly1d(0.6 * memorie_linie_dreapta.coefficients + 0.4 * linie_dreapta.coefficients)
        memorie_linie_dreapta = linie_dreapta

    y_sus, y_jos = int(h * 0.58), h

    if linie_stanga is not None and linie_dreapta is not None:
        y_vals = np.linspace(y_sus, y_jos, 30)
        x_s_jos, x_d_jos = int(linie_stanga(y_jos)), int(linie_dreapta(y_jos))
        x_s_sus, x_d_sus = int(linie_stanga(y_sus)), int(linie_dreapta(y_sus))

        if x_s_jos < x_d_jos and x_s_sus < x_d_sus and 80 < (x_d_jos - x_s_jos) < w * 0.8:
            pts_s = [(int(linie_stanga(y)), int(y)) for y in y_vals]
            pts_d = [(int(linie_dreapta(y)), int(y)) for y in y_vals]
            poligon_nou = np.array([pts_s + pts_d[::-1]], dtype=np.float32)
            gasit_linii = True

    if not gasit_linii and memorie_linie_stanga is not None and memorie_linie_dreapta is not None:
        try:
            y_vals = np.linspace(y_sus, y_jos, 30)
            pts_s = [(int(memorie_linie_stanga(y)), int(y)) for y in y_vals]
            pts_d = [(int(memorie_linie_dreapta(y)), int(y)) for y in y_vals]
            poligon_nou = np.array([pts_s + pts_d[::-1]], dtype=np.float32)
            gasit_linii = True
        except: pass

    if gasit_linii and poligon_nou is not None:
        istoric_banda.append(poligon_nou.copy())
        if len(istoric_banda) > 5: istoric_banda.pop(0)
        banda_medie = sum(istoric_banda) / len(istoric_banda)
        memorie_banda = (memorie_banda.astype(np.float32) * 0.5 + banda_medie * 0.5).astype(np.int32)

    strat_banda = np.zeros_like(imagine)
    if memorie_banda is not None: cv2.fillPoly(strat_banda, memorie_banda, [0, 255, 0])

    return cv2.addWeighted(imagine, 1.0, strat_banda, 0.35, 0), memorie_banda

def calculeaza_risk_level(distanta, pe_banda, latime, tip_obiect):
    if tip_obiect == "person":
        if pe_banda and distanta < 30: return "red"
        elif pe_banda and distanta < 50: return "yellow"
        return "green"

    if tip_obiect == "stop sign":
        return "red" if pe_banda and distanta < 20 else "yellow"

    if pe_banda:
        if distanta < 15 or latime > 250: return "red"
        elif distanta < 30 or latime > 150: return "yellow"
    else:
        if distanta < 20: return "yellow"
        
    return "green"

def motor_inteligenta_artificiala(interfata, coada_cadre):
    print(">>> AI pregatit. Incep detectia pe video...")
    model = YOLO("yolov8n.pt")
    obiecte_importante = ['car', 'truck', 'bus', 'person', 'stop sign', 'traffic light']

    latime_vid, inaltime_vid = 1280, 720
    poligon_banda_mea = np.array([[(int(latime_vid * 0.35), int(inaltime_vid * 0.65)), (int(latime_vid * 0.65), int(inaltime_vid * 0.65)), (int(latime_vid * 0.85), inaltime_vid), (int(latime_vid * 0.15), inaltime_vid)]], np.int32).reshape((-1, 1, 2))

    while interfata.running:
        try:
            cadru, nr_cadru = coada_cadre.get(timeout=1.0)
        except queue.Empty: continue

        interfata.mediu_curent = analizeaza_conditii_mediu(cadru)
        imagine_banda, poligon_siguranta = detecteaza_banda_vizual(cadru)
        
        rezultate_yolo = model.track(source=cadru, persist=True, imgsz=480, tracker="bytetrack.yaml", verbose=False)

        if rezultate_yolo:
            detectii_cadru = []
            for box in rezultate_yolo[0].boxes:
                if box.conf < 0.3: continue
                nume_clasa = rezultate_yolo[0].names[int(box.cls)]
                if nume_clasa not in obiecte_importante: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                id_obiect = int(box.id) if box.id is not None else -1
                latime = x2 - x1

                distanta = calculeaza_distanta(latime, nume_clasa)
                punct_baza = (int((x1 + x2) / 2), y2)

                pe_banda_vizual = poligon_siguranta is not None and cv2.pointPolygonTest(poligon_siguranta, punct_baza, False) >= 0
                is_in_my_lane = cv2.pointPolygonTest(poligon_banda_mea, punct_baza, False) >= 0
                risk_level = calculeaza_risk_level(distanta, is_in_my_lane, latime, nume_clasa)

                detectii_cadru.append({
                    "type": nume_clasa, "id": id_obiect, "distanta": round(distanta, 1),
                    "is_in_my_lane": is_in_my_lane, "risk_level": risk_level, "w": latime,
                    "x_center": punct_baza[0], "y_baza": y2
                })

                culoare = (0, 0, 255) if risk_level == "red" else (0, 255, 255) if risk_level == "yellow" else (0, 255, 0)
                cv2.rectangle(imagine_banda, (x1, y1), (x2, y2), culoare, 2)
                cv2.putText(imagine_banda, f"{nume_clasa} | {distanta:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, culoare, 2)

            interfata.actualizeaza_imagine_ui(imagine_banda)

            if nr_cadru % 5 == 0:
                with interfata.data_lock:
                    interfata.last_detections = detectii_cadru
                    interfata.frame_count = nr_cadru

        coada_cadre.task_done()

def process_and_parse_video(cale_video, interfata):
    cap = cv2.VideoCapture(cale_video)
    coada = queue.Queue(maxsize=2)
    threading.Thread(target=motor_inteligenta_artificiala, args=(interfata, coada), daemon=True).start()

    nr_cadru = 0
    while cap.isOpened() and interfata.running:
        succes, cadru = cap.read()
        if not succes: break
        nr_cadru += 1
        try: coada.put((cadru.copy(), nr_cadru), timeout=0.1)
        except queue.Full: pass

    cap.release()