import tkinter as tk
from tkinter import filedialog
import os
import threading
import socket
import cv2
from PIL import Image, ImageTk
from data_parsing import process_and_parse_video
from server_logic import run_server_side

class InterfataDriveAssist:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1280x720")
        self.root.title("Drive Assist")
        self.root.configure(bg="#2b2b2b")
        
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

        self.construieste_ui()

    def construieste_ui(self):
        self.frame_sus = tk.Frame(self.root, bg="black")
        self.frame_sus.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=20, pady=20)
        self.label_video = tk.Label(self.frame_sus, text="[ SISTEM INACTIV ]\n\nINCARCA UN VIDEO", bg="black", fg="white", font=("Consolas", 20))
        self.label_video.pack(expand=True, fill=tk.BOTH)

        self.frame_jos = tk.Frame(self.root, bg="#2b2b2b", height=100)
        self.frame_jos.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        self.frame_jos.pack_propagate(False)

        tk.Button(self.frame_jos, text="LOAD VIDEO", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", width=15, command=self.incarca_video).pack(side=tk.LEFT, padx=50)
        
        self.buton_switch = tk.Button(self.frame_jos, text="ON RENDER", font=("Arial", 14, "bold"), bg="#FF9800", fg="white", width=15, command=self.toggle_render)
        self.buton_switch.pack(side=tk.LEFT, padx=10)

        tk.Button(self.frame_jos, text="START RENDER", font=("Arial", 14, "bold"), bg="#F44336", fg="white", width=15, command=self.start_sistem).pack(side=tk.RIGHT, padx=50)

    def toggle_render(self):
        self.render_activ = not self.render_activ
        self.buton_switch.config(text="OFF RENDER" if self.render_activ else "ON RENDER", bg="#9E9E9E" if self.render_activ else "#FF9800")

    def incarca_video(self):
        cale = filedialog.askopenfilename(initialdir=os.getcwd(), filetypes=(("Video", "*.mp4 *.avi"), ("All", "*.*")))
        if cale:
            self.cale_video_selectata = cale
            self.label_video.config(text=f"Video pregatit: {os.path.basename(cale)}", fg="green", image="")

    def actualizeaza_imagine_ui(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.label_video.config(image=img_tk)
        self.label_video.image = img_tk

    def start_sistem(self):
        if not self.cale_video_selectata: return
        self.running = True
        
        threading.Thread(target=process_and_parse_video, args=(self.cale_video_selectata, self), daemon=True).start()
        threading.Thread(target=run_server_side, args=(self, ), daemon=True).start()

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfataDriveAssist(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()