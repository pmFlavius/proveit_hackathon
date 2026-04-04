import time
import socket
import json
import numpy as np

class ADAS_Subject:
    def __init__(self):
        self._observers = []

    def ataseaza_observer(self, observer):
        self._observers.append(observer)

    def notifica_observers(self, packet):
        for obs in self._observers:
            obs.update(packet)

class UDP_Observer:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (ip, port)

    def update(self, packet):
        try:
            self.sock.sendto(json.dumps(packet).encode('utf-8'), self.dest)
        except Exception as e:
            pass

def run_server_side(interfata):
    print("[SERVER] Modulul decizional si retea a pornit.")
    sistem_adas = ADAS_Subject()
    modul_retea = UDP_Observer(interfata.udp_ip, interfata.udp_port)
    sistem_adas.ataseaza_observer(modul_retea)

    ultimul_timp_trimis = time.time()
    interval_trimitere = 0.5
    
    # Inițializăm o listă globală pentru a stoca tot istoricul pachetelor JSON
    istoric_pachete_json = []

    while interfata.running:
        timp_curent = time.time()
        
        if timp_curent - ultimul_timp_trimis < interval_trimitere:
            time.sleep(0.01)
            continue
            
        ultimul_timp_trimis = timp_curent

        with interfata.data_lock:
            obiecte = list(interfata.last_detections)
            cadru_numar = interfata.frame_count
            viteza_mea = getattr(interfata, 'viteza_video', 60)
            mediu = getattr(interfata, 'mediu_curent', {"time_of_day": "day", "weather": "clear"})

        speed_dec = "mentinere"
        lane_dec = "mentinerea benzii"
        brake_dec = "fara frana"
        risk_str = "scazut"
        reasoning = "Drum liber, se mentin parametrii actuali."
        nivel_risc_int = 1
        
        lista_vehicule_json = []

        if viteza_mea < 5:
            speed_dec = "mentinere"; brake_dec = "puternica"
            reasoning = "Vehiculul se afla in stationare."
        else:
            for obj in obiecte:
                tip = obj["type"]
                is_in_my_lane = obj.get("is_in_my_lane", False)
                risk_level = obj.get("risk_level", "green")
                dist = obj.get("distanta", 0)

                lista_vehicule_json.append({
                    "id": obj["id"], "type": tip, 
                    "is_in_my_lane": is_in_my_lane, 
                    "risk_level": risk_level, "distance": dist
                })

                if tip == "stop sign" and dist < 30:
                    speed_dec = "scadere"; brake_dec = "puternica"; risk_str = "ridicat"; nivel_risc_int = 3
                    reasoning = f"Indicator STOP la {dist}m."
                    break 
                
                elif tip == "person" and is_in_my_lane:
                    speed_dec = "scadere"; brake_dec = "puternica"; risk_str = "ridicat"; nivel_risc_int = 3
                    reasoning = f"Pieton detectat pe banda la {dist}m."
                    break

                elif tip in ["car", "truck", "bus"] and is_in_my_lane:
                    if risk_level == "red" or dist < 15:
                        speed_dec = "scadere"; brake_dec = "puternica"; risk_str = "ridicat"; nivel_risc_int = 3
                        reasoning = f"Coliziune iminenta cu {tip} la {dist}m."
                        break
                    elif risk_level == "yellow" or dist < 40:
                        if nivel_risc_int < 2:
                            speed_dec = "scadere"; brake_dec = "usoara"; risk_str = "mediu"; nivel_risc_int = 2
                            reasoning = f"Vehicul lent detectat ({tip} la {dist}m)."

        if mediu["weather"] == "fog" and nivel_risc_int < 3:
            speed_dec = "scadere"
            reasoning += " Vizibilitate limitata (Ceata)."
            
        if mediu["time_of_day"] == "night" and nivel_risc_int < 3:
            reasoning += " Rulare pe timp de noapte."

        if interfata.render_activ:
            packet = {
                "timestamp": time.strftime('%H:%M:%S.%f')[:-3],
                "frame": cadru_numar,
                "environment": mediu,
                "ego_vehicle": {
                    "speed_kmh": viteza_mea,
                    "decisions": {
                        "speed": speed_dec,
                        "lane": lane_dec,
                        "brake": brake_dec
                    },
                    "risk_level": risk_str,
                    "reasoning": reasoning
                },
                "detected_objects": lista_vehicule_json
            }
            
            sistem_adas.notifica_observers(packet)
            log_entry = (f"[{packet['timestamp']}] Frame {cadru_numar} | Risc: {risk_str.upper()} | "
                         f"Mediu: {mediu['time_of_day']},{mediu['weather']} | "
                         f"Actiune: Viteza {speed_dec}, Banda {lane_dec}, Frana {brake_dec} | "
                         f"Motiv: {reasoning}\n")
                         
            with open("activitate_sistem.log", "a", encoding="utf-8") as f:
                f.write(log_entry)
            istoric_pachete_json.append(packet)
            with open("decizii_sistem.json", "w", encoding="utf-8") as json_file:
                json.dump(istoric_pachete_json, json_file, indent=4)