import tkinter as tk
from tkinter import filedialog
import os
import threading
import socket
import cv2
import queue
from PIL import Image, ImageTk
from data_parsing import process_and_parse_video
from server_logic import run_server_side

class InterfataDriveAssist:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1280x800")
        self.root.title("Drive Assist")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(800, 600)

        self.data_lock = threading.Lock()
        self.last_detections = []
        self.frame_count = 0
        self.running = False
        self.render_activ = False
        self.cale_video_selectata = None
        self.viteza_video = 60
        self.mediu_curent = {"time_of_day": "day", "weather": "clear"}

        self.udp_ip = "127.0.0.1"
        self.udp_port = 5005

        # Dimensiunile fixe pentru zona video - nu se mai extinde infinit
        self.VIDEO_W = 1240
        self.VIDEO_H = 620

        # Coada pentru transferul sigur al cadrelor între fire de execuție
        self.frame_queue = queue.Queue(maxsize=2)

        self.construieste_ui()

    def construieste_ui(self):
        # ── Bara de sus (titlu) ──
        header = tk.Frame(self.root, bg="#0f3460", height=40)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="⬡  DRIVE ASSIST  ⬡", bg="#0f3460", fg="#e94560",
                 font=("Consolas", 14, "bold")).pack(side=tk.LEFT, padx=20, pady=8)

        # ── Bara de jos cu butoane (FIXĂ - 100px) ──
        self.frame_jos = tk.Frame(self.root, bg="#16213e", height=100)
        self.frame_jos.pack(side=tk.BOTTOM, fill=tk.X)
        self.frame_jos.pack_propagate(False)  # ← CHEIE: nu lasă butoanele să crească

        btn_cfg = {"font": ("Consolas", 12, "bold"), "relief": "flat",
                   "bd": 0, "padx": 20, "pady": 10, "cursor": "hand2"}

        tk.Button(self.frame_jos, text="📂  LOAD VIDEO",
                  bg="#4CAF50", fg="white", activebackground="#45a049",
                  command=self.incarca_video, **btn_cfg
                  ).pack(side=tk.LEFT, padx=30, pady=15)

        self.buton_switch = tk.Button(self.frame_jos, text="○  RENDER OFF",
                                      bg="#555", fg="white", activebackground="#666",
                                      command=self.toggle_render, **btn_cfg)
        self.buton_switch.pack(side=tk.LEFT, padx=10, pady=15)

        self.label_status = tk.Label(self.frame_jos, text="Niciun video selectat",
                                     bg="#16213e", fg="#888",
                                     font=("Consolas", 10))
        self.label_status.pack(side=tk.LEFT, padx=20)

        tk.Button(self.frame_jos, text="▶  START",
                  bg="#e94560", fg="white", activebackground="#c73652",
                  command=self.start_sistem, **btn_cfg
                  ).pack(side=tk.RIGHT, padx=30, pady=15)

        # ── Zona video (restul spațiului, dar cu dimensiune FIXĂ a imaginii) ──
        self.frame_video = tk.Frame(self.root, bg="black")
        self.frame_video.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Label-ul video cu dimensiuni fixe - nu se mai extinde
        self.label_video = tk.Label(
            self.frame_video,
            text="[ SISTEM INACTIV ]\n\nÎncarcă un videoclip pentru a începe.",
            bg="black", fg="#555",
            font=("Consolas", 16),
            width=self.VIDEO_W,   # ← lățime fixă în pixeli (via image)
        )
        self.label_video.pack(expand=True)

        # Pornește bucla de actualizare a imaginii din firul principal
        self._process_frame_queue()

    def toggle_render(self):
        self.render_activ = not self.render_activ
        if self.render_activ:
            self.buton_switch.config(text="● RENDER ON", bg="#FF9800")
        else:
            self.buton_switch.config(text="○ RENDER OFF", bg="#555")

    def incarca_video(self):
        cale = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            filetypes=(("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*"))
        )
        if cale:
            self.cale_video_selectata = cale
            nume = os.path.basename(cale)
            self.label_status.config(text=f"✓ {nume}", fg="#4CAF50")
            self.label_video.config(
                text=f"Video pregătit:\n{nume}\n\nApasă START pentru procesare.",
                fg="#4CAF50", image=""
            )

    def actualizeaza_imagine_ui(self, frame):
        """Apune cadrul în coada pentru a fi procesat în firul principal (thread-safe)."""
        try:
            # Încercăm să punem în coadă fără blocare
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Dacă e plină, ignorăm cadrul curent

    def _process_frame_queue(self):
        """Procesează cadrele din coadă în firul principal Tkinter."""
        try:
            # Procesăm toate cadrele disponibile, păstrând doar ultimul
            frame = None
            while True:
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            if frame is not None:
                self._display_frame(frame)
        except Exception:
            pass

        # Programează următoarea verificare în 16ms (~60fps)
        self.root.after(16, self._process_frame_queue)

    def _display_frame(self, frame):
        """Afișează cadrul în GUI - apelat doar din firul principal."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Calculăm dimensiunea disponibilă
            avail_w = self.frame_video.winfo_width() or self.VIDEO_W
            avail_h = self.frame_video.winfo_height() or self.VIDEO_H

            # Menținem aspect ratio
            fh, fw = frame_rgb.shape[:2]
            scale = min(avail_w / fw, avail_h / fh, 1.0)
            new_w = int(fw * scale)
            new_h = int(fh * scale)

            if new_w > 0 and new_h > 0:
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

            img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.label_video.config(image=img_tk, text="")
            self.label_video.image = img_tk  # păstrăm referința
        except Exception:
            pass

    def start_sistem(self):
        if not self.cale_video_selectata:
            self.label_status.config(text="⚠ Selectează un video mai întâi!", fg="#e94560")
            return
        self.running = True
        self.label_status.config(text="▶ Sistem activ...", fg="#FF9800")

        threading.Thread(
            target=process_and_parse_video,
            args=(self.cale_video_selectata, self),
            daemon=True
        ).start()
        threading.Thread(
            target=run_server_side,
            args=(self,),
            daemon=True
        ).start()

    def on_closing(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = InterfataDriveAssist(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()