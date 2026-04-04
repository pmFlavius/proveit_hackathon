import time
import socket
import json
import cv2
import numpy as np

class ADAS_Subject:
    """Clasa care monitorizează starea drumului și notifică observatorii."""
    def __init__(self):
        self._observers = []

    def ataseaza_observer(self, observer):
        self._observers.append(observer)

    def notifica_observers(self, packet):
        for obs in self._observers:
            obs.update(packet)

class UDP_Observer:
    """Observatorul care ia datele decizionale și le trimite pe UDP."""
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (ip, port)

    def update(self, packet):
        # Trimite JSON-ul complet conform cerintei
        self.sock.sendto(json.dumps(packet).encode('utf-8'), self.dest)


def run_server_side(interfata):
    print("[SERVER] Sistemul decizional OOP a pornit.")
    

    sistem_adas = ADAS_Subject()
    modul_retea = UDP_Observer(interfata.udp_ip, interfata.udp_port)
    sistem_adas.ataseaza_observer(modul_retea)

    latime_vid, inaltime_vid = 1280, 720
    poligon_banda = np.array([[
        (int(latime_vid * 0.35), int(inaltime_vid * 0.65)),
        (int(latime_vid * 0.65), int(inaltime_vid * 0.65)),
        (int(latime_vid * 0.85), inaltime_vid),
        (int(latime_vid * 0.15), inaltime_vid)
    ]], np.int32).reshape((-1, 1, 2))

    while interfata.running:
        time.sleep(0.1) 

        with interfata.data_lock:
            obiecte = list(interfata.last_detections)
            cadru_numar = interfata.frame_count
            viteza_mea = getattr(interfata, 'viteza_video', 60) 
        decizie_finala = "DRUM LIBER"
        nivel_risc = 0
        lista_vehicule_json = []

        if viteza_mea < 5:
            decizie_finala = "STATIONARE"
        else:
            for obj in obiecte:
                tip = obj["type"]
                latime = obj["w"]
                x_center = obj["x_center"]
                y_baza = obj["y_baza"]
                is_in_my_lane = obj.get("is_in_my_lane", False)
                risk_level = obj.get("risk_level", "green")

                # Salvăm informațiile complete despre vehicul
                lista_vehicule_json.append({
                    "id": obj["id"],
                    "type": tip,
                    "is_in_my_lane": is_in_my_lane,
                    "risk_level": risk_level,
                    "distance": obj.get("distanta", 0)
                })

                if tip == "stop sign":
                    decizie_finala = "OPRIRE LA STOP"; nivel_risc = 3; break
                elif tip == "traffic light" and latime > 40:
                    decizie_finala = "OPRIRE SEMAFOR"; nivel_risc = 3; break
                elif tip == "person" and is_in_my_lane:
                    decizie_finala = "FRANA URGENTA - PIETON"; nivel_risc = 3; break

                elif tip in ["car", "truck", "bus"]:
                    if is_in_my_lane:
                        prag_critic = 250 if viteza_mea < 70 else 180

                        if latime > prag_critic or risk_level == "red":
                            decizie_finala = "FRANA URGENTA"
                            nivel_risc = 3
                            break
                        elif latime > 120 or risk_level == "yellow":
                            if nivel_risc < 2:
                                decizie_finala = "REDU VITEZA"
                                nivel_risc = 2
                    else:
                        if nivel_risc < 1:
                            decizie_finala = "PUTETI MERGE NORMAL"
                            nivel_risc = 1

        if interfata.render_activ:
            packet = {
                "timestamp": time.strftime('%H:%M:%S.%f')[:-3],
                "frame": cadru_numar,
                "decizie": decizie_finala,
                "risc": nivel_risc,
                "viteza_ego_vehicul": viteza_mea,
                "tip_masini_detectate": lista_vehicule_json
            }
            
            sistem_adas.notifica_observers(packet)
            with open("activitate_sistem.log", "a") as f:
                f.write(f"Frame {cadru_numar} | Decizie: {decizie_finala} | Vehicule: {len(lista_vehicule_json)}\n")