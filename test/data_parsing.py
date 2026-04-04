import cv2
import numpy as np
import math
import threading
import queue
from ultralytics import YOLO

# Memorie pentru a menține banda stabilă pe drum
memorie_banda = None
istoric_banda = []  # Istoric pentru smoothing temporal
memorie_linie_stanga = None
memorie_linie_dreapta = None

def extrage_zona_interes(imagine, puncte):
    masca = np.zeros_like(imagine)
    cv2.fillPoly(masca, puncte, 255)
    return cv2.bitwise_and(imagine, masca)

def calculeaza_distanta(latime_obiect):
    focal_length = 600
    latime_reala_masina = 1.8
    return (latime_reala_masina * focal_length) / latime_obiect if latime_obiect > 0 else 0

def detecteaza_marcaje_banda(imagine):
    """Detectează marcajele de bandă (albe și galbene) folosind filtru de culoare."""
    h, w = imagine.shape[:2]

    # Convertim la HLS pentru detectare mai bună a culorilor
    hls = cv2.cvtColor(imagine, cv2.COLOR_BGR2HLS)

    # Masca pentru marcaje albe (luminozitate ridicată)
    alb_jos = np.array([0, 200, 0])
    alb_sus = np.array([180, 255, 255])
    masca_alb = cv2.inRange(hls, alb_jos, alb_sus)

    # Masca pentru marcaje galbene
    galben_jos = np.array([15, 50, 100])
    galben_sus = np.array([35, 255, 255])
    masca_galben = cv2.inRange(hls, galben_jos, galben_sus)

    # Combinăm măștile
    masca_combinata = cv2.bitwise_or(masca_alb, masca_galben)

    return masca_combinata

def ferestra_glisanta(imagine_binara, n_ferestre=10, latime_fereastra=100, prag_pixeli=50):
    """Detectează linia de bandă folosind tehnica ferestrei glisante."""
    h, w = imagine_binara.shape

    # ROI - partea inferioară a imaginii (ultimele 50%)
    roi_jos = int(h * 0.5)
    imagine_roi = imagine_binara[roi_jos:, :]

    # Histogramă pe axa orizontală pentru a găsi poziția inițială a liniilor
    histograma = np.sum(imagine_roi, axis=0)

    # Punctul de plecare pentru linia stângă și dreaptă
    punct_mijloc = w // 2

    # Căutăm peak-ul în jumătatea stângă și dreaptă
    stanga_x = np.argmax(histograma[:punct_mijloc])
    dreapta_x = np.argmax(histograma[punct_mijloc:]) + punct_mijloc

    # Verificăm dacă avem suficientă bază
    if histograma[stanga_x] < prag_pixeli:
        stanga_x = int(w * 0.25)
    if histograma[dreapta_x] < prag_pixeli:
        dreapta_x = int(w * 0.75)

    inaltime_fereastra = h // n_ferestre

    puncte_stanga = []
    puncte_dreapta = []

    x_stanga_curent = stanga_x
    x_dreapta_curent = dreapta_x

    for fereastra in range(n_ferestre):
        # Coordonatele ferestrei (y_top < y_bottom în sistemul OpenCV)
        y_top = h - (fereastra + 1) * inaltime_fereastra
        y_bottom = h - fereastra * inaltime_fereastra

        # Verificăm că nu mergem prea sus (evităm cerul)
        if y_bottom < h * 0.45:
            continue

        # Fereastra pentru linia stângă
        x_stanga_min = max(0, int(x_stanga_curent - latime_fereastra))
        x_stanga_max = min(w, int(x_stanga_curent + latime_fereastra))

        # Fereastra pentru linia dreaptă
        x_dreapta_min = max(0, int(x_dreapta_curent - latime_fereastra))
        x_dreapta_max = min(w, int(x_dreapta_curent + latime_fereastra))

        # Extragem pixelii activi din ferestre
        fereastra_stanga = imagine_binara[y_top:y_bottom, x_stanga_min:x_stanga_max]
        fereastra_dreapta = imagine_binara[y_top:y_bottom, x_dreapta_min:x_dreapta_max]

        # Coordonatele pixelilor activi (relativ la fereastră)
        ys_stanga, xs_stanga = np.where(fereastra_stanga > 0)
        ys_dreapta, xs_dreapta = np.where(fereastra_dreapta > 0)

        # Actualizăm poziția centrului ferestrei dacă găsim pixeli
        if len(xs_stanga) > prag_pixeli:
            x_stanga_curent = np.mean(xs_stanga) + x_stanga_min
            puncte_stanga.append((x_stanga_curent, (y_top + y_bottom) / 2))

        if len(xs_dreapta) > prag_pixeli:
            x_dreapta_curent = np.mean(xs_dreapta) + x_dreapta_min
            puncte_dreapta.append((x_dreapta_curent, (y_top + y_bottom) / 2))

    return puncte_stanga, puncte_dreapta

def ajusteaza_linie(puncte, grad=2):
    """Ajustează o curbă polinomială prin punctele detectate."""
    if len(puncte) < 3:
        return None

    puncte = np.array(puncte)
    xs = puncte[:, 0]
    ys = puncte[:, 1]

    try:
        coeficienti = np.polyfit(ys, xs, grad)
        return np.poly1d(coeficienti)
    except:
        return None

def detecteaza_banda_vizual(imagine):
    global memorie_banda, istoric_banda, memorie_linie_stanga, memorie_linie_dreapta
    h, w = imagine.shape[:2]

    # Bandă implicită de siguranță
    banda_implicita = np.array([[
        (int(w * 0.20), h), (int(w * 0.40), int(h * 0.65)),
        (int(w * 0.60), int(h * 0.65)), (int(w * 0.80), h)
    ]], dtype=np.int32)

    if memorie_banda is None:
        memorie_banda = banda_implicita.copy()

    # ROI trapezoidal - exclude cerul
    puncte_roi = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.45), int(h * 0.55)),
        (int(w * 0.55), int(h * 0.55)),
        (int(w * 0.95), h)
    ]], np.int32)

    # Detectăm marcajele de bandă (albe și galbene)
    masca_marcaje = detecteaza_marcaje_banda(imagine)

    # Aplicăm ROI
    masca_roi = np.zeros_like(masca_marcaje)
    cv2.fillPoly(masca_roi, puncte_roi, 255)
    masca_aplicata = cv2.bitwise_and(masca_marcaje, masca_roi)

    # Aplicăm morfologie pentru a curăța zgomotul
    kernel = np.ones((5, 5), np.uint8)
    masca_curata = cv2.morphologyEx(masca_aplicata, cv2.MORPH_CLOSE, kernel)
    masca_curata = cv2.morphologyEx(masca_curata, cv2.MORPH_OPEN, kernel)

    # Detectăm liniile cu fereastră glisantă
    puncte_stanga, puncte_dreapta = ferestra_glisanta(
        masca_curata, n_ferestre=12, latime_fereastra=80, prag_pixeli=25
    )

    linie_stanga = None
    linie_dreapta = None
    gasit_linii = False
    poligon_nou = None

    # Ajustăm curbele polinomiale (grad 2 pentru curburi)
    if len(puncte_stanga) >= 4:
        linie_stanga = ajusteaza_linie(puncte_stanga, grad=2)
    if len(puncte_dreapta) >= 4:
        linie_dreapta = ajusteaza_linie(puncte_dreapta, grad=2)

    # Smoothing pentru liniile individuale
    if linie_stanga is not None:
        if memorie_linie_stanga is not None:
            coef_nou = linie_stanga.coefficients
            coef_vechi = memorie_linie_stanga.coefficients
            coef_smooth = 0.6 * coef_vechi + 0.4 * coef_nou
            linie_stanga = np.poly1d(coef_smooth)
        memorie_linie_stanga = linie_stanga

    if linie_dreapta is not None:
        if memorie_linie_dreapta is not None:
            coef_nou = linie_dreapta.coefficients
            coef_vechi = memorie_linie_dreapta.coefficients
            coef_smooth = 0.6 * coef_vechi + 0.4 * coef_nou
            linie_dreapta = np.poly1d(coef_smooth)
        memorie_linie_dreapta = linie_dreapta

    # Generăm poligonul benzii
    y_sus = int(h * 0.58)
    y_jos = h

    if linie_stanga is not None and linie_dreapta is not None:
        # Generăm punctele pentru desenare
        y_vals = np.linspace(y_sus, y_jos, 30)

        puncte_stanga_draw = [(int(linie_stanga(y)), int(y)) for y in y_vals]
        puncte_dreapta_draw = [(int(linie_dreapta(y)), int(y)) for y in y_vals]

        # Verificăm validitatea geometrică
        x_stanga_jos = int(linie_stanga(y_jos))
        x_dreapta_jos = int(linie_dreapta(y_jos))
        x_stanga_sus = int(linie_stanga(y_sus))
        x_dreapta_sus = int(linie_dreapta(y_sus))

        # Validare: liniile nu trebuie să se intersecteze și banda trebuie să fie rezonabilă
        latime_jos = x_dreapta_jos - x_stanga_jos
        latime_sus = x_dreapta_sus - x_stanga_sus

        if (x_stanga_jos < x_dreapta_jos and
            x_stanga_sus < x_dreapta_sus and
            latime_jos > 80 and latime_jos < w * 0.8 and
            latime_sus > 30):

            # Creăm poligonul
            poligon_nou = np.array(
                [puncte_stanga_draw + puncte_dreapta_draw[::-1]],
                dtype=np.float32
            )
            gasit_linii = True

    # Dacă nu am găsit linii, păstrăm memoria anterioară
    if not gasit_linii and memorie_linie_stanga is not None and memorie_linie_dreapta is not None:
        # Folosim memoria pentru a genera poligonul
        y_vals = np.linspace(y_sus, y_jos, 30)
        try:
            puncte_stanga_draw = [(int(memorie_linie_stanga(y)), int(y)) for y in y_vals]
            puncte_dreapta_draw = [(int(memorie_linie_dreapta(y)), int(y)) for y in y_vals]
            poligon_nou = np.array(
                [puncte_stanga_draw + puncte_dreapta_draw[::-1]],
                dtype=np.float32
            )
            gasit_linii = True
        except:
            pass

    # Smoothing temporal
    if gasit_linii and poligon_nou is not None:
        istoric_banda.append(poligon_nou.copy())
        if len(istoric_banda) > 5:
            istoric_banda.pop(0)

        banda_medie = np.zeros_like(poligon_nou, dtype=np.float32)
        for b in istoric_banda:
            banda_medie += b
        banda_medie /= len(istoric_banda)

        memorie_banda = (memorie_banda.astype(np.float32) * 0.5 + banda_medie * 0.5).astype(np.int32)

    # Desenăm linia benzii detectate
    strat_banda = np.zeros_like(imagine)

    # Desenăm liniile individuale (roșu = stânga, albastru = dreapta)
    if memorie_linie_stanga is not None:
        try:
            y_vals = np.linspace(y_sus, y_jos, 50)
            pts_stanga = np.array([[int(memorie_linie_stanga(y)), int(y)] for y in y_vals], dtype=np.int32)
            cv2.polylines(strat_banda, [pts_stanga], False, (0, 0, 255), 4)
        except:
            pass

    if memorie_linie_dreapta is not None:
        try:
            y_vals = np.linspace(y_sus, y_jos, 50)
            pts_dreapta = np.array([[int(memorie_linie_dreapta(y)), int(y)] for y in y_vals], dtype=np.int32)
            cv2.polylines(strat_banda, [pts_dreapta], False, (255, 0, 0), 4)
        except:
            pass

    # Umplem banda cu verde transparent
    if memorie_banda is not None:
        cv2.fillPoly(strat_banda, memorie_banda, [0, 255, 0])

    rezultat_vizual = cv2.addWeighted(imagine, 1.0, strat_banda, 0.35, 0)

    return rezultat_vizual, memorie_banda

def calculeaza_risk_level(distanta, pe_banda, latime, tip_obiect):
    """Calculează nivelul de risc: green, yellow, red."""
    # Persoanele și semnele de stop au risc implicit mai mare
    if tip_obiect == "person":
        if pe_banda and distanta < 30:
            return "red"
        elif pe_banda and distanta < 50:
            return "yellow"
        return "green"

    if tip_obiect == "stop sign":
        if pe_banda and distanta < 20:
            return "red"
        return "yellow"

    # Pentru vehicule
    if pe_banda:
        if distanta < 15 or latime > 250:
            return "red"
        elif distanta < 30 or latime > 150:
            return "yellow"
    else:
        # Obiecte în alte benzi
        if distanta < 20:
            return "yellow"
    return "green"

def motor_inteligenta_artificiala(interfata, coada_cadre):
    print(">>> Copilot: Sistemul de analiză video este gata. Monitorizez drumul...")
    model = YOLO("yolov8n.pt")
    obiecte_importante = ['car', 'truck', 'bus', 'person', 'stop sign', 'traffic light']

    # Definim poligonul benzii proprii (același ca în server_logic)
    latime_vid, inaltime_vid = 1280, 720
    poligon_banda_mea = np.array([[
        (int(latime_vid * 0.35), int(inaltime_vid * 0.65)),
        (int(latime_vid * 0.65), int(inaltime_vid * 0.65)),
        (int(latime_vid * 0.85), inaltime_vid),
        (int(latime_vid * 0.15), inaltime_vid)
    ]], np.int32).reshape((-1, 1, 2))

    while interfata.running:
        try:
            cadru, nr_cadru = coada_cadre.get(timeout=1.0)
        except queue.Empty:
            continue

        imagine_banda, poligon_siguranta = detecteaza_banda_vizual(cadru)
        rezultate_yolo = model.track(source=cadru, persist=True, imgsz=480, verbose=False)

        if rezultate_yolo:
            detectii_cadru = []
            for box in rezultate_yolo[0].boxes:
                if box.conf < 0.3: continue
                nume_clasa = rezultate_yolo[0].names[int(box.cls)]
                if nume_clasa not in obiecte_importante: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                id_obiect = int(box.id) if box.id is not None else -1

                distanta = calculeaza_distanta(x2 - x1)
                punct_baza = (int((x1 + x2) / 2), y2)
                latime = x2 - x1

                # Verificăm dacă obiectul ne stă în cale (pe banda detectată vizual)
                pe_banda_vizual = poligon_siguranta is not None and \
                                   cv2.pointPolygonTest(poligon_siguranta, punct_baza, False) >= 0

                # Verificăm dacă este în banda noastră definită (poligon fix)
                is_in_my_lane = cv2.pointPolygonTest(poligon_banda_mea, punct_baza, False) >= 0

                # Calculăm nivelul de risc
                risk_level = calculeaza_risk_level(distanta, is_in_my_lane, latime, nume_clasa)

                detectii_cadru.append({
                    "type": nume_clasa,
                    "id": id_obiect,
                    "distanta": round(distanta, 1),
                    "is_in_my_lane": is_in_my_lane,
                    "risk_level": risk_level,
                    "in_lane": pe_banda_vizual,  # păstrăm compatibilitatea
                    "w": latime,
                    "x_center": punct_baza[0],
                    "y_baza": y2
                })

                # Alegem culoarea în funcție de nivelul de risc
                if risk_level == "red":
                    culoare = (0, 0, 255)      # Roșu
                elif risk_level == "yellow":
                    culoare = (0, 255, 255)    # Galben
                else:
                    culoare = (0, 255, 0)      # Verde

                cv2.rectangle(imagine_banda, (x1, y1), (x2, y2), culoare, 2)
                cv2.putText(imagine_banda, f"{nume_clasa} | {distanta:.1f}m | {risk_level}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, culoare, 2)

            interfata.actualizeaza_imagine_ui(imagine_banda)

            if nr_cadru % 5 == 0:
                with interfata.data_lock:
                    interfata.last_detections = detectii_cadru
                    interfata.frame_count = nr_cadru

        coada_cadre.task_done()

def process_and_parse_video(cale_video, interfata):
    cap = cv2.VideoCapture(cale_video)
    coada = queue.Queue(maxsize=2)

    thread_ai = threading.Thread(target=motor_inteligenta_artificiala, args=(interfata, coada), daemon=True)
    thread_ai.start()

    print(f">>> Copilot: Am început procesarea fișierului: {cale_video}")
    nr_cadru = 0
    while cap.isOpened() and interfata.running:
        succes, cadru = cap.read()
        if not succes: break
        nr_cadru += 1
        try:
            coada.put((cadru.copy(), nr_cadru), timeout=0.1)
        except queue.Full: pass

    cap.release()
    print(">>> Copilot: Călătoria s-a încheiat. Toate datele au fost procesate.")
