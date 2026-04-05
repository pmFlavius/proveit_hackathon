import pygame
import socket
import json
import threading
import time
from dataclasses import dataclass, field

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
W, H     = 620, 780
FPS      = 30

BG          = (15,  17,  24)
ROAD        = (38,  42,  52)
LANE_DASH   = (200, 190,  80)
LANE_SOLID  = (230, 230, 230)
EGO         = ( 55, 145, 235)
OBJ_SAME    = (225,  55,  55)
OBJ_WARN    = (235, 155,  35)
OBJ_OTHER   = ( 70, 175, 120)
PERSON_COL  = (195,  85, 235)
SIGN_COL    = (215,  45,  45)
WHITE       = (240, 240, 245)
GRAY        = (110, 115, 130)
LGRAY       = (175, 180, 195)
DARK        = ( 10,  12,  18)

@dataclass
class State:
    risk_level:       str   = "scazut"
    reasoning:        str   = "Astept date..."
    speed_kmh:        float = 0.0
    decision_speed:   str   = "mentinere"
    decision_brake:   str   = "fara frana"
    detected_objects: list  = field(default_factory=list)
    frame:            int   = 0
    environment:      dict  = field(default_factory=lambda: {"surface": "necunoscut"}) # NOU
    last_update:      float = field(default_factory=time.time)

state      = State()
state_lock = threading.Lock()

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)
    print(f"[RENDER 2D] Ascult pe {UDP_IP}:{UDP_PORT}...")
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            p    = json.loads(data.decode('utf-8'))
            ego  = p.get("ego_vehicle", {})
            dec  = ego.get("decisions", {})
            with state_lock:
                state.risk_level       = ego.get("risk_level", "scazut")
                state.reasoning        = ego.get("reasoning", "Fara detalii.")
                state.speed_kmh        = ego.get("speed_kmh", 60)
                state.decision_speed   = dec.get("speed", "mentinere")
                state.decision_brake   = dec.get("brake", "fara frana")
                state.detected_objects = p.get("detected_objects", [])
                state.frame            = p.get("frame", 0)
                state.environment      = p.get("environment", {}) # NOU
                state.last_update      = time.time()
        except socket.timeout:
            continue
        except Exception:
            pass

ROAD_LEFT  = int(W * 0.12)
ROAD_RIGHT = int(W * 0.88)
ROAD_W     = ROAD_RIGHT - ROAD_LEFT
CENTER_X   = W // 2
LANE_W     = ROAD_W // 3  

LANE_CENTERS = {
    "stanga":   ROAD_LEFT  + LANE_W // 2,
    "banda ta": CENTER_X,
    "dreapta":  ROAD_RIGHT - LANE_W // 2,
}
LANE_BOUNDS = {
    "stanga":   (ROAD_LEFT,           ROAD_LEFT  + LANE_W),
    "banda ta": (ROAD_LEFT + LANE_W,  ROAD_LEFT  + LANE_W * 2),
    "dreapta":  (ROAD_LEFT + LANE_W * 2, ROAD_RIGHT),
}

OBJ_TOP    = 80
OBJ_BOTTOM = H - 185
EGO_Y      = H - 140
EGO_W, EGO_H = 40, 64
MAX_DIST   = 150.0   
HUD_H = 160  

def dist_to_y(dist):
    t = max(0.0, min(1.0, float(dist) / MAX_DIST))
    return int(OBJ_BOTTOM - (OBJ_BOTTOM - OBJ_TOP) * t)

def obj_x(obj):
    banda = obj.get("assigned_lane", "dreapta").strip().lower()
    if banda not in LANE_CENTERS: banda = "dreapta"
    cx = LANE_CENTERS[banda]
    offset = (hash(str(obj.get("id", 0))) % 21) - 10
    lbound, rbound = LANE_BOUNDS[banda]
    return max(lbound + 25, min(rbound - 25, cx + offset))

def obj_color(obj):
    typ     = obj.get("type", "car")
    in_lane = obj.get("is_in_my_lane", False)
    dist    = float(obj.get("distance", 50))
    if typ == "person": return PERSON_COL
    if typ == "stop sign": return SIGN_COL
    if in_lane: return OBJ_SAME if dist < 20 else OBJ_WARN
    return OBJ_OTHER

def obj_size(dist, typ="car"):
    t = max(0.0, min(1.0, float(dist) / MAX_DIST))
    base = int(22 + (48 - 22) * (1.0 - t))
    if typ in ("person", "stop sign", "traffic light"): return int(base * 0.7)
    return base

def draw_road(surf, font_tiny):
    pygame.draw.rect(surf, ROAD, (ROAD_LEFT, 0, ROAD_W, H - HUD_H + 5))
    pygame.draw.line(surf, LANE_SOLID, (ROAD_LEFT,  0), (ROAD_LEFT,  H), 3)
    pygame.draw.line(surf, LANE_SOLID, (ROAD_RIGHT, 0), (ROAD_RIGHT, H), 3)

    x_sep1 = ROAD_LEFT + LANE_W
    x_sep2 = ROAD_LEFT + LANE_W * 2
    offset  = int(time.time() * 55) % 44

    for y in range(-44, H - HUD_H, 44):
        pygame.draw.line(surf, LANE_DASH, (x_sep1, y + offset), (x_sep1, y + offset + 24), 2)
        pygame.draw.line(surf, LANE_DASH, (x_sep2, y + offset), (x_sep2, y + offset + 24), 2)

    for label, cx in [("STANGA",  LANE_CENTERS["stanga"]), ("BANDA TA", LANE_CENTERS["banda ta"]), ("DREAPTA",  LANE_CENTERS["dreapta"])]:
        t = font_tiny.render(label, True, GRAY)
        surf.blit(t, (cx - t.get_width() // 2, 12))

def draw_markers(surf, font_tiny):
    for dist in [15, 30, 50, 80, 120]:
        y = dist_to_y(dist)
        pygame.draw.line(surf, (55, 60, 75), (ROAD_LEFT, y), (ROAD_RIGHT, y), 1)
        lbl = font_tiny.render(f"{dist}m", True, GRAY)
        surf.blit(lbl, (ROAD_LEFT - lbl.get_width() - 6, y - 6))

def draw_objects(surf, objects, font_sm):
    for obj in sorted(objects, key=lambda o: -float(o.get("distance", 0))):
        dist  = float(obj.get("distance", 30))
        typ   = obj.get("type", "car")
        x   = obj_x(obj)
        y   = dist_to_y(dist)
        sz  = obj_size(dist, typ)
        col = obj_color(obj)
        r = pygame.Rect(x - sz // 2, y - sz // 2, sz, sz)

        if typ == "person":
            pygame.draw.circle(surf, col, (x, y), sz // 2)
            pygame.draw.circle(surf, WHITE, (x, y), sz // 2, 2)
        elif typ == "stop sign":
            pts = [(x, y - sz // 2), (x + sz // 2, y), (x, y + sz // 2), (x - sz // 2, y)]
            pygame.draw.polygon(surf, col, pts)
            pygame.draw.polygon(surf, WHITE, pts, 2)
        else:
            pygame.draw.rect(surf, col, r, border_radius=6)
            pygame.draw.rect(surf, WHITE, r, width=2, border_radius=6)

        label_txt = f"{typ[:4].upper()} {dist:.0f}m"
        lbl = font_sm.render(label_txt, True, WHITE)
        bg_r = pygame.Rect(x - lbl.get_width() // 2 - 2, y - sz // 2 - 19, lbl.get_width() + 4, lbl.get_height() + 2)
        pygame.draw.rect(surf, DARK, bg_r)
        surf.blit(lbl, (x - lbl.get_width() // 2, y - sz // 2 - 18))

def draw_ego(surf, font_sm):
    r = pygame.Rect(CENTER_X - EGO_W // 2, EGO_Y - EGO_H // 2, EGO_W, EGO_H)
    pygame.draw.rect(surf, EGO, r, border_radius=8)
    pygame.draw.rect(surf, WHITE, r, width=2, border_radius=8)
    glass = pygame.Rect(CENTER_X - EGO_W // 2 + 4, EGO_Y - EGO_H // 2 + 10, EGO_W - 8, 14)
    pygame.draw.rect(surf, (30, 195, 255), glass, border_radius=3)
    lbl = font_sm.render("TU", True, WHITE)
    surf.blit(lbl, (CENTER_X - lbl.get_width() // 2, EGO_Y + 12))

def draw_hud(surf, st, font_title, font_sm, font_tiny, connected):
    py = H - HUD_H
    panel = pygame.Surface((W, HUD_H), pygame.SRCALPHA)
    panel.fill((12, 15, 22, 250))
    surf.blit(panel, (0, py))
    pygame.draw.line(surf, EGO, (0, py), (W, py), 3)

    suprafata = st.environment.get("surface", "asfalt_uscat").replace("_", " ").upper()
    suprafata_txt = font_tiny.render(f"DRUM: {suprafata}", True, (200, 200, 200))
    surf.blit(suprafata_txt, (W - suprafata_txt.get_width() - 18, py + 44))

    risk_col = OBJ_SAME if st.risk_level == "ridicat" else OBJ_WARN if st.risk_level == "mediu" else (70, 215, 110)
    surf.blit(font_title.render(f"RISC: {st.risk_level.upper()}", True, risk_col), (18, py + 14))

    spd = font_title.render(f"{int(st.speed_kmh)} km/h", True, WHITE)
    surf.blit(spd, (W - spd.get_width() - 18, py + 14))

    c_col = (70, 215, 110) if connected else OBJ_SAME
    c_txt = "● LIVE" if connected else "● ASTEPT DATE..."
    c = font_sm.render(c_txt, True, c_col)
    surf.blit(c, (W // 2 - c.get_width() // 2, py + 14))

    dec_txt = f"Viteza: [{st.decision_speed.upper()}]  |  Frana: [{st.decision_brake.upper()}]"
    dec = font_sm.render(dec_txt, True, LGRAY)
    surf.blit(dec, (W // 2 - dec.get_width() // 2, py + 44))

    surf.blit(font_tiny.render("LOGICA:", True, GRAY), (18, py + 76))
    words, line, lines = st.reasoning.split(), [], []
    for w in words:
        test = " ".join(line + [w])
        if font_sm.size(test)[0] < W - 36: line.append(w)
        else: lines.append(" ".join(line)); line = [w]
    if line: lines.append(" ".join(line))
    for i, ln in enumerate(lines[:3]):
        surf.blit(font_sm.render(ln, True, WHITE), (18, py + 96 + i * 19))


def main():
    pygame.init()
    surf  = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Drive Assist — 2D Top-Down")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("consolas", 17, bold=True)
    font_sm    = pygame.font.SysFont("consolas", 13, bold=True)
    font_tiny  = pygame.font.SysFont("consolas", 11)

    threading.Thread(target=udp_listener, daemon=True).start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); return

        with state_lock:
            st = State(
                risk_level       = state.risk_level,
                reasoning        = state.reasoning,
                speed_kmh        = state.speed_kmh,
                decision_speed   = state.decision_speed,
                decision_brake   = state.decision_brake,
                detected_objects = list(state.detected_objects),
                frame            = state.frame,
                last_update      = state.last_update,
                environment      = state.environment
            )

        connected = (time.time() - st.last_update) < 2.5
        surf.fill(BG)
        draw_road(surf, font_tiny)
        draw_markers(surf, font_tiny)
        draw_objects(surf, st.detected_objects, font_sm)
        draw_ego(surf, font_sm)
        draw_hud(surf, st, font_title, font_sm, font_tiny, connected)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()