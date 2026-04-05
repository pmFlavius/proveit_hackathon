import threading
import time
import socket
import json

class ADAS_Subject:
    def __init__(self):
        self._observers = []
    def ataseaza_observer(self, o):
        self._observers.append(o)
    def notifica_observers(self, packet):
        for o in self._observers:
            o.update(packet)

class UDP_Observer:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (ip, port)
    def update(self, packet):
        try:
            self.sock.sendto(json.dumps(packet).encode('utf-8'), self.dest)
        except Exception:
            pass

def _determina_banda(obj):
    x_center = obj.get("x_center", 640)
    if obj.get("is_in_my_lane", False):
        return "banda ta"
    lim_st = obj.get("lim_stanga", None)
    lim_dr = obj.get("lim_dreapta", None)
    if lim_st is not None and lim_dr is not None:
        return "stanga" if x_center < lim_st else "dreapta"
    return "stanga" if x_center < obj.get("frame_width", 1280) / 2 else "dreapta"

def _analizeaza_risc(obj, viteza_ego_kmh=60):
    dist     = float(obj.get("distanta", 50))
    tip      = obj.get("type", "car")
    vrel     = float(obj.get("viteza_relativa", 0))  
    in_lane  = obj.get("is_in_my_lane", False)

    vrel_ms = vrel / 3.6

    # Protecție la împarțirea cu zero / numere negative
    if vrel_ms > 0.1:
        ttc = dist / vrel_ms
    else:
        ttc = 999  

    rezultat = {"nivel": 0, "ttc": round(ttc, 1), "vrel": vrel}

    if not in_lane:
        return rezultat  

    if tip == "person":
        if dist < 25: rezultat["nivel"] = 3
        elif dist < 45: rezultat["nivel"] = 2
        elif dist < 60: rezultat["nivel"] = 1
        return rezultat

    if dist < 10:
        rezultat["nivel"] = 3
    elif dist < 20 and ttc < 3.5:
        rezultat["nivel"] = 3
    elif dist < 35 and ttc < 6.0:
        rezultat["nivel"] = 2
    elif dist < 50 and ttc < 10.0 and vrel > 5:
        rezultat["nivel"] = 1
    elif dist < 25 and vrel < -3:
        rezultat["nivel"] = 0

    return rezultat

def run_server_side(interfata):
    print("[SERVER] Modulul ADAS a pornit.")

    sistem_adas = ADAS_Subject()
    modul_retea = UDP_Observer(interfata.udp_ip, interfata.udp_port)
    sistem_adas.ataseaza_observer(modul_retea)

    ultimul_timp = time.time()
    interval     = 0.3
    istoric_json = []

    while interfata.running:
        now = time.time()
        if now - ultimul_timp < interval:
            time.sleep(0.01)
            continue
        ultimul_timp = now

        with interfata.data_lock:
            obiecte   = list(interfata.last_detections)
            cadru_nr  = interfata.frame_count
            viteza_ego = getattr(interfata, 'viteza_video', 60)
            mediu     = getattr(interfata, 'mediu_curent', {"time_of_day": "day", "weather": "clear"})

        speed_dec  = "mentinere"
        lane_dec   = "mentinerea benzii"
        brake_dec  = "fara frana"
        risk_str   = "scazut"
        reasoning  = "Drum liber."
        nivel_max  = 0
        ttc_critic = 999

        lista_json = []

        for obj in obiecte:
            banda = _determina_banda(obj)
            risc  = _analizeaza_risc(obj, viteza_ego)
            nivel = risc["nivel"]
            ttc   = risc["ttc"]
            vrel  = risc["vrel"]
            dist  = float(obj.get("distanta", 50))
            tip   = obj["type"]

            lista_json.append({
                "id":            obj["id"],
                "type":          tip,
                "distance":      dist,
                "is_in_my_lane": obj.get("is_in_my_lane", False),
                "assigned_lane": banda,
                "viteza_rel_kmh": vrel,
                "ttc_sec":       ttc if ttc < 99 else None,
            })

            if nivel > nivel_max:
                nivel_max  = nivel
                ttc_critic = ttc

                if nivel == 3:
                    speed_dec  = "scadere"
                    brake_dec  = "puternica"
                    risk_str   = "ridicat"
                    if tip == "person":
                        reasoning = f"PERICOL: pieton la {dist:.0f}m pe banda ta! Oprire!"
                    elif ttc < 99:
                        reasoning = f"COLIZIUNE in {ttc:.1f}s cu {tip} la {dist:.0f}m. Franare de urgenta!"
                    else:
                        reasoning = f"Obstacol la {dist:.0f}m pe banda ta. Franare puternica!"

                elif nivel == 2:
                    speed_dec  = "scadere"
                    brake_dec  = "usoara"
                    risk_str   = "mediu"
                    if ttc < 99:
                        reasoning = f"{tip.capitalize()} la {dist:.0f}m, TTC={ttc:.1f}s. Reduce viteza."
                    else:
                        reasoning = f"{tip.capitalize()} la {dist:.0f}m in fata. Mentine distanta."

                elif nivel == 1:
                    risk_str  = "mediu"
                    reasoning = f"{tip.capitalize()} la {dist:.0f}m. Monitorizare atenta."

        packet = {
            "timestamp": time.strftime('%H:%M:%S'),
            "frame":     cadru_nr,
            "environment": mediu,
            "ego_vehicle": {
                "speed_kmh":  viteza_ego,
                "risk_level": risk_str,
                "reasoning":  reasoning,
                "decisions": {
                    "speed": speed_dec,
                    "lane":  lane_dec,
                    "brake": brake_dec,
                },
            },
            "detected_objects": lista_json,
        }

        if interfata.render_activ:
            sistem_adas.notifica_observers(packet)

        with open("activitate_sistem.log", "a", encoding="utf-8") as f:
            f.write(f"[{packet['timestamp']}] F:{cadru_nr} | {risk_str} | {reasoning}\n")

        istoric_json.append(packet)
        if len(istoric_json) > 100:
            istoric_json.pop(0)

        class NumpySafe(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                return super().default(obj)

        with open("decizii_sistem.json", "w", encoding="utf-8") as jf:
            json.dump(istoric_json, jf, indent=4, ensure_ascii=False, cls=NumpySafe)