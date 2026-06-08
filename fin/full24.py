import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Папка для моделей
MODELS_DIR = "tournament_models"
if not os.path.exists(MODELS_DIR): 
    os.makedirs(MODELS_DIR)

CLASSES = ('літак', 'авто', 'птах', 'кіт', 'олень', 'собака', 'жаба', 'кінь', 'корабель', 'вантажівка')

def get_resnet_model(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True  # Розморожуємо для Fine-tuning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

class TournamentLab:
    def __init__(self, root, device):
        self.root = root
        self.device = device
        self.history = {}
        
        self.root.title("Deep Learning Tournament: Diploma Edition (ResNet-18)")
        self.root.geometry("1250x850")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.train_tab = ttk.Frame(self.notebook)
        self.chart_tab = ttk.Frame(self.notebook)
        self.battle_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.train_tab, text=" 1. Тренування  ")
        self.notebook.add(self.chart_tab, text=" 2. Аналітика (Train vs Val) ")
        self.notebook.add(self.battle_tab, text=" 3. ЗМАГАННЯ МОДЕЛЕЙ ")

        self.setup_train_ui()
        self.setup_chart_ui()
        self.setup_battle_ui()

    def log(self, text):
        self.root.after(0, self._safe_log_update, text)

    def _safe_log_update(self, text):
        self.log_area.config(state="normal")
        self.log_area.insert("end", text + "\n")
        self.log_area.see("end")
        self.log_area.config(state="disabled")

    def setup_train_ui(self):
        panel = tk.Frame(self.train_tab, width=320, padx=20, pady=20, bg="#f8f9fa")
        panel.pack(side="left", fill="y")
        tk.Label(panel, text="ПАРАМЕТРИ ", font=("Arial", 12, "bold"), bg="#f8f9fa").pack(pady=10)
        
        tk.Label(panel, text="Назва (без пробілів):", bg="#f8f9fa").pack(anchor="w")
        self.model_name_entry = tk.Entry(panel)
        self.model_name_entry.insert(0, "ResNet_Adam_Exp1")
        self.model_name_entry.pack(fill="x", pady=5)

        tk.Label(panel, text="Оптимізатор:", bg="#f8f9fa").pack(anchor="w")
        self.opt_var = tk.StringVar(value="Adam")
        ttk.Combobox(panel, textvariable=self.opt_var, values=["Adam", "SGD", "RMSprop"], state="readonly").pack(fill="x", pady=5)

        tk.Label(panel, text="LR (рекомендовано 0.0001):", bg="#f8f9fa").pack(anchor="w")
        self.lr_entry = tk.Entry(panel)
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.pack(fill="x", pady=5)

        tk.Label(panel, text="Епохи (5-10 для результату):", bg="#f8f9fa").pack(anchor="w")
        self.epoch_entry = tk.Entry(panel)
        self.epoch_entry.insert(0, "5")
        self.epoch_entry.pack(fill="x", pady=5)

        self.btn_train = tk.Button(panel, text="🔥 ЗАПУСТИТИ FINE-TUNING", command=self.start_train_thread, bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), height=2)
        self.btn_train.pack(fill="x", pady=25)

        self.log_area = tk.Text(self.train_tab, height=15, state="disabled", font=("Consolas", 10))
        self.log_area.pack(side="bottom", fill="x", padx=10, pady=10)

    def setup_chart_ui(self):
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(11, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        tk.Button(self.chart_tab, text="🔄 ОНОВИТИ ГРАФІКИ МЕТРИК", command=self.draw_plots, bg="#3498db", fg="white", font=("Arial", 10, "bold"), height=2).pack(pady=10)

    def setup_battle_ui(self):
        top = tk.Frame(self.battle_tab, pady=15)
        top.pack(side="top", fill="x")
        tk.Button(top, text="📸 Обрати фото для тесту", command=self.load_battle_image, font=("Arial", 10)).pack()
        self.battle_img_view = tk.Label(self.battle_tab)
        self.battle_img_view.pack(pady=10)
        
        self.tree = ttk.Treeview(self.battle_tab, columns=("Model", "Opt", "Pred", "Conf"), show="headings", height=10)
        self.tree.heading("Model", text="Модель")
        self.tree.heading("Opt", text="Оптимізатор")
        self.tree.heading("Pred", text="Прогноз (Клас)")
        self.tree.heading("Conf", text="Впевненість %")
        self.tree.pack(fill="both", expand=True, padx=20)
        
        tk.Button(self.battle_tab, text="⚔️ ЗАПУСТИТИ ТУРНІРНЕ ЗМАГАННЯ ⚔️", command=self.run_tournament_battle, bg="#2c3e50", fg="white", font=("Arial", 12, "bold"), height=2).pack(pady=20)

    def start_train_thread(self):
        threading.Thread(target=self.train_process, daemon=True).start()

    def train_process(self):
        self.root.after(0, lambda: self.btn_train.config(state="disabled"))
        name, opt_name = self.model_name_entry.get(), self.opt_var.get()
        lr, epochs = float(self.lr_entry.get()), int(self.epoch_entry.get())

        model = get_resnet_model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr) if opt_name == "Adam" else \
                    optim.SGD(model.parameters(), lr=lr, momentum=0.9) if opt_name == "SGD" else \
                    optim.RMSprop(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()
        
        # Аугментація для тренувальних даних
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Трансформація для валідаційних даних (без аугментації)
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Завантаження датасетів (Train + Validation)
        self.log("[DATA] Завантаження наборів даних CIFAR-10...")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=0)

        # Створення структури історії для повноцінного аналізу
        self.history[name] = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "opt": opt_name}
        self.log(f"[START] Експеримент: {name}. Оптимізатор: {opt_name}. Початок Fine-tuning...")

        for epoch in range(epochs):
            # --- ЕТАП НАВЧАННЯ ---
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

            train_acc = 100. * correct / total
            avg_train_loss = running_loss / len(train_loader)

            # --- ЕТАП ВАЛІДАЦІЇ (ПЕРЕВІРКИ) ---
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += pred.eq(labels).sum().item()

            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            # Збереження метрик в історію
            self.history[name]["loss"].append(avg_train_loss)
            self.history[name]["acc"].append(train_acc)
            self.history[name]["val_loss"].append(avg_val_loss)
            self.history[name]["val_acc"].append(val_acc)

            self.log(f"Епоха {epoch+1}/{epochs} | "
                     f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
                     f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Збереження стану моделі на диск
        torch.save({'state_dict': model.state_dict(), 'optimizer': opt_name}, os.path.join(MODELS_DIR, f"{name}.pth"))
        self.log(f"[DONE]  {name} успішно збережений на диск.")
        self.root.after(0, lambda: self.btn_train.config(state="normal"))
        self.root.after(0, lambda: messagebox.showinfo("Готово", f"Навчання моделі {name} завершено!"))

    def draw_plots(self):
        self.ax_loss.clear()
        self.ax_acc.clear()
        
        # Якщо ще немає жодних натренованих моделей — просто очищуємо графіки і виходимо
        if not self.history:
            self.ax_loss.set_title("Функція втрат (Немає даних)")
            self.ax_acc.set_title("Точність класифікації (Немає даних)")
            self.canvas.draw()
            return

        for name, data in self.history.items():
            epochs_range = range(1, len(data["loss"]) + 1)
            opt = data['opt']
            
            # Графіки Loss
            self.ax_loss.plot(epochs_range, data["loss"], label=f"{name} ({opt}) Train", marker='o')
            self.ax_loss.plot(epochs_range, data["val_loss"], label=f"{name} ({opt}) Val", marker='x', linestyle='--')
            
            # Графіки Accuracy
            self.ax_acc.plot(epochs_range, data["acc"], label=f"{name} ({opt}) Train", marker='s')
            self.ax_acc.plot(epochs_range, data["val_acc"], label=f"{name} ({opt}) Val", marker='d', linestyle='--')
        
        self.ax_loss.set_title("Функція втрат (Cross Entropy Loss)")
        self.ax_loss.set_xlabel("Епохи")
        self.ax_loss.set_ylabel("Значення Loss")
        self.ax_loss.legend()
        self.ax_loss.grid(True)

        self.ax_acc.set_title("Точність класифікації (Accuracy)")
        self.ax_acc.set_xlabel("Епохи")
        self.ax_acc.set_ylabel("Точність %")
        self.ax_acc.legend()
        self.ax_acc.grid(True)
        
        self.canvas.draw()
    def load_battle_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            img = Image.open(path).convert('RGB')
            tk_img = ImageTk.PhotoImage(img.resize((250, 250)))
            self.battle_img_view.config(image=tk_img)
            self.battle_img_view.image = tk_img
            self.battle_path = path

    def run_tournament_battle(self):
        if not hasattr(self, 'battle_path'): 
            messagebox.showwarning("Увага", "Будь ласка, оберіть тестове зображення перед початком турніру.")
            return
            
        for i in self.tree.get_children(): 
            self.tree.delete(i)
            
        img = Image.open(self.battle_path).convert('RGB')
        tensor = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(img).unsqueeze(0).to(self.device)
        
        model = get_resnet_model().to(self.device)
        model.eval()
        
        saved_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
        if not saved_files:
            messagebox.showwarning("Помилка", f"У папці '{MODELS_DIR}' немає збережених моделей (.pth). Спочатку натренуйте їх!")
            return

        for f in saved_files:
            ckpt = torch.load(os.path.join(MODELS_DIR, f), map_location=self.device)
            model.load_state_dict(ckpt['state_dict'])
            with torch.no_grad():
                out = model(tensor)
                prob = torch.nn.functional.softmax(out, dim=1)
                conf, pred = torch.max(prob, 1)
                self.tree.insert("", "end", values=(
                    f.replace('.pth', ''), 
                    ckpt['optimizer'], 
                    CLASSES[pred.item()].upper(), 
                    f"{conf.item()*100:.1f}%"
                ))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Запущено обчислення на: {device}")
    root = tk.Tk()
    app = TournamentLab(root, device)
    root.mainloop()