import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import tensorflow as tf
import numpy as np
import matplotlib
import threading
import psutil
import time




# Configuracion de  Matplotlib para usar el backend TkAgg para graficos en la interfaz
matplotlib.use("TkAgg")




# Configuración básica de CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# Definicion de Colores Personalizados
COLORS = {
    "primary": "#2D5BA3",
    "secondary": "#A32D2D",
    "accent": "#55A868",
    "background": "#2B2B2B",
    "text": "#CCCCCC",
    "success": "#90EE90",
    "warning": "#FFD700",
    "error": "#FF6347"
}




# Clase Principal de la Aplicación
class GPUvsCPUApp(ctk.CTk):





    # Inicialización de la Aplicación
    def __init__(self):

        super().__init__()
        self.title("Análisis de Rendimiento CPU vs GPU")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.matrix_size = ctk.IntVar(value=3000)
        self.iterations = ctk.IntVar(value=3)
        self.is_running = False
        self.historical_data = {
            "cpu_times": [],
            "gpu_times": [],
            "cpu_gflops": [],
            "gpu_gflops": [],
            "cpu_utilization": [],
            "gpu_utilization": [],
            "sizes": []
        }

        self.create_ui()


    # Crear interfaz de usuario
    def create_ui(self):

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        control_frame = ctk.CTkFrame(main_frame, width=300)  # Panel más ancho
        control_frame.pack(side="left", fill="y", padx=(0, 10), pady=0)
        title_frame = ctk.CTkFrame(control_frame)
        title_frame.pack(fill="x", padx=15, pady=(15, 5))

        title_label = ctk.CTkLabel(
            title_frame,
            text="Rendimiento CPU vs GPU",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=5)

        # Resultados de la última prueba con diseño mejorado (movido arriba)
        results_frame = ctk.CTkFrame(control_frame)
        results_frame.pack(fill="x", padx=15, pady=10)

        results_title = ctk.CTkLabel(
            results_frame,
            text="Últimos resultados",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        results_title.pack(pady=5)

        # Crear subframes para cada tipo de resultado
        time_frame = ctk.CTkFrame(results_frame)
        time_frame.pack(fill="x", pady=2, padx=5)

        self.cpu_time_label = ctk.CTkLabel(time_frame, text="CPU: -", anchor="w")
        self.cpu_time_label.pack(fill="x", pady=1)

        self.gpu_time_label = ctk.CTkLabel(time_frame, text="GPU: -", anchor="w")
        self.gpu_time_label.pack(fill="x", pady=1)

        perf_frame = ctk.CTkFrame(results_frame)
        perf_frame.pack(fill="x", pady=2, padx=5)

        self.speedup_label = ctk.CTkLabel(perf_frame, text="Aceleración: -", anchor="w")
        self.speedup_label.pack(fill="x", pady=1)

        self.cpu_gflops_label = ctk.CTkLabel(perf_frame, text="CPU GFLOPS: -", anchor="w")
        self.cpu_gflops_label.pack(fill="x", pady=1)

        self.gpu_gflops_label = ctk.CTkLabel(perf_frame, text="GPU GFLOPS: -", anchor="w")
        self.gpu_gflops_label.pack(fill="x", pady=1)

        # Información de hardware con diseño mejorado
        hardware_frame = ctk.CTkFrame(control_frame)
        hardware_frame.pack(fill="x", padx=15, pady=10)

        hardware_title = ctk.CTkLabel(
            hardware_frame,
            text="Información del Hardware",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        hardware_title.pack(pady=5)

        # Crear subframes para CPU y GPU
        cpu_frame = ctk.CTkFrame(hardware_frame)
        cpu_frame.pack(fill="x", pady=2, padx=5)

        cpu_info = f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} threads)"
        self.cpu_label = ctk.CTkLabel(
            cpu_frame,
            text=cpu_info,
            anchor="w"
        )
        self.cpu_label.pack(fill="x", pady=1)

        # Información adicional de CPU
        cpu_usage_label = ctk.CTkLabel(
            cpu_frame,
            text=f"Uso actual: {psutil.cpu_percent()}%",
            anchor="w"
        )
        cpu_usage_label.pack(fill="x", pady=1)

        # Frame para GPU
        gpu_frame = ctk.CTkFrame(hardware_frame)
        gpu_frame.pack(fill="x", pady=2, padx=5)

        # Detectar GPU disponible
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info = f"GPU: {'Disponible' if gpus else 'No disponible'}"
        if gpus:
            try:
                gpu_info = f"GPU: {tf.test.gpu_device_name()}"
            except:
                pass

        self.gpu_label = ctk.CTkLabel(
            gpu_frame,
            text=gpu_info,
            anchor="w"
        )
        self.gpu_label.pack(fill="x", pady=1)

        # Información adicional de GPU
        gpu_status = "Activa" if gpus else "No disponible"
        gpu_status_label = ctk.CTkLabel(
            gpu_frame,
            text=f"Estado: {gpu_status}",
            anchor="w"
        )
        gpu_status_label.pack(fill="x", pady=1)

        # Parámetros de prueba con diseño mejorado
        params_frame = ctk.CTkFrame(control_frame)
        params_frame.pack(fill="x", padx=15, pady=10)

        params_title = ctk.CTkLabel(
            params_frame,
            text="Parámetros de Prueba",
            font=ctk.CTkFont(weight="bold")
        )
        params_title.pack(pady=5)

        # Tamaño de matriz con entrada numérica
        size_frame = ctk.CTkFrame(params_frame)
        size_frame.pack(fill="x", pady=5)

        size_label = ctk.CTkLabel(size_frame, text="Tamaño de matriz:", anchor="w")
        size_label.pack(side="left", padx=5)

        size_entry = ctk.CTkEntry(
            size_frame,
            width=70,
            textvariable=self.matrix_size
        )
        size_entry.pack(side="right", padx=5)

        size_slider = ctk.CTkSlider(
            params_frame,
            from_=1000,
            to=5000,
            number_of_steps=8,
            variable=self.matrix_size,
            command=self.update_size_label
        )
        size_slider.pack(fill="x", pady=5, padx=5)

        self.size_value_label = ctk.CTkLabel(
            params_frame,
            text=f"{self.matrix_size.get()}x{self.matrix_size.get()}"
        )
        self.size_value_label.pack(pady=2)

        # Iteraciones con entrada numérica
        iter_frame = ctk.CTkFrame(params_frame)
        iter_frame.pack(fill="x", pady=5)

        iter_label = ctk.CTkLabel(iter_frame, text="Iteraciones:", anchor="w")
        iter_label.pack(side="left", padx=5)

        iter_entry = ctk.CTkEntry(
            iter_frame,
            width=70,
            textvariable=self.iterations
        )
        iter_entry.pack(side="right", padx=5)

        iter_slider = ctk.CTkSlider(
            params_frame,
            from_=1,
            to=10,
            number_of_steps=9,
            variable=self.iterations,
            command=self.update_iter_label
        )
        iter_slider.pack(fill="x", pady=5, padx=5)

        self.iter_value_label = ctk.CTkLabel(params_frame, text=f"{self.iterations.get()}")
        self.iter_value_label.pack(pady=2)

        # Toggles de visualización mejorados
        toggle_frame = ctk.CTkFrame(control_frame)
        toggle_frame.pack(fill="x", padx=15, pady=10)

        self.show_history = ctk.BooleanVar(value=True)
        history_toggle = ctk.CTkSwitch(
            toggle_frame,
            text="Mostrar historial",
            variable=self.show_history,
            command=self.update_graphics,
            progress_color=COLORS["accent"]
        )
        history_toggle.pack(anchor="w", pady=5)

        # Botones con diseño mejorado (movidos a la parte inferior)
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(side="bottom", fill="x", padx=15, pady=(0, 15))

        self.run_button = ctk.CTkButton(
            button_frame,
            text="Ejecutar prueba",
            command=self.run_test_thread,
            fg_color=COLORS["primary"],
            hover_color="#1E3F75",
            height=40
        )
        self.run_button.pack(pady=5, fill="x")

        self.clear_button = ctk.CTkButton(
            button_frame,
            text="Limpiar historial",
            command=self.clear_history,
            fg_color=COLORS["secondary"],
            hover_color="#751E1E",
            height=40
        )
        self.clear_button.pack(pady=5, fill="x")

        # Estado con diseño mejorado (movido al final)
        self.status_label = ctk.CTkLabel(
            control_frame,
            text="Listo para comenzar",
            text_color=COLORS["text"],
            fg_color=COLORS["background"],
            corner_radius=6
        )
        self.status_label.pack(side="bottom", pady=(0, 15), padx=15, fill="x")

        # Panel principal (gráficos) con diseño mejorado
        graph_frame = ctk.CTkFrame(main_frame)
        graph_frame.pack(side="right", fill="both", expand=True)

        # Grid para los gráficos (2x2)
        self.graph_frames = []
        for i in range(4):
            row = i // 2
            col = i % 2
            frame = ctk.CTkFrame(graph_frame)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            self.graph_frames.append(frame)

        # Configurar grid
        for i in range(2):
            graph_frame.grid_rowconfigure(i, weight=1)
            graph_frame.grid_columnconfigure(i, weight=1)

        # Inicializar gráficos
        self.canvases = [None, None, None, None]
        self.create_empty_plots()


    # Actualizar Etiqueta del Tamaño de Matriz
    def update_size_label(self, value=None):

        size = self.matrix_size.get()
        self.size_value_label.configure(text=f"{size}x{size}")


    # Actualizar Etiqueta del Número de Iteraciones
    def update_iter_label(self, value=None):

        self.iter_value_label.configure(text=f"{self.iterations.get()}")


    # Iniciar Prueba en un Hilo Separado
    def run_test_thread(self):

        if self.is_running:
            return

        self.is_running = True
        self.run_button.configure(state="disabled", text="Ejecutando...")
        self.status_label.configure(text="Preparando prueba...", text_color="#FFD700")

        thread = threading.Thread(target=self.run_test)
        thread.daemon = True
        thread.start()


    # Limpiar Historial
    def clear_history(self):

        self.historical_data = {
            "cpu_times": [],
            "gpu_times": [],
            "cpu_gflops": [],
            "gpu_gflops": [],
            "cpu_utilization": [],
            "gpu_utilization": [],
            "sizes": []
        }
        self.update_graphics()
        self.status_label.configure(text="Historial limpiado", text_color="#87CEEB")


    # Ejecutar Prueba
    def run_test(self):

        try:
            matrix_size = self.matrix_size.get()
            iterations = self.iterations.get()

            # Actualizar UI
            self.status_label.configure(text=f"Generando matrices {matrix_size}x{matrix_size}...", text_color="#FFD700")

            # Generar matrices
            A = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)
            B = tf.random.uniform((matrix_size, matrix_size), dtype=tf.float32)

            # Prueba en CPU
            self.status_label.configure(text="Ejecutando prueba en CPU...", text_color="#FFD700")
            cpu_times = []
            cpu_util_data = []

            with tf.device('/CPU:0'):
                # Medición de utilización de CPU
                for _ in range(iterations):
                    cpu_util_start = psutil.cpu_percent(interval=0.1)
                    start_time = time.time()
                    tf.matmul(A, B)
                    end_time = time.time()
                    cpu_util_end = psutil.cpu_percent(interval=0.1)

                    cpu_times.append(end_time - start_time)
                    cpu_util_data.append((cpu_util_start + cpu_util_end) / 2)

            # Calcular media y desviación
            avg_cpu_time = np.mean(cpu_times)
            std_cpu_time = np.std(cpu_times)
            avg_cpu_util = np.mean(cpu_util_data)

            # Prueba en GPU si está disponible
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.status_label.configure(text="Ejecutando prueba en GPU...", text_color="#FFD700")
                gpu_times = []

                with tf.device('/GPU:0'):
                    # Calentamiento
                    tf.matmul(A, B)

                    # Medición real
                    for _ in range(iterations):
                        start_time = time.time()
                        tf.matmul(A, B)
                        end_time = time.time()
                        gpu_times.append(end_time - start_time)

                # Calcular media y desviación
                avg_gpu_time = np.mean(gpu_times)
                std_gpu_time = np.std(gpu_times)

                # GPU utilization (aproximado ya que no podemos medirlo directamente sin librerías específicas)
                gpu_utilization = 95.0  # Valor fijo aproximado
            else:
                avg_gpu_time = None
                std_gpu_time = None
                gpu_utilization = 0

            # Calcular GFLOPS
            operations = 2 * (matrix_size ** 3)  # Aproximación para multiplicación de matrices
            cpu_gflops = operations / (avg_cpu_time * 1e9)
            gpu_gflops = operations / (avg_gpu_time * 1e9) if avg_gpu_time else 0

            # Calcular speedup
            speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time else 0

            # Guardar en historial
            self.historical_data["cpu_times"].append(avg_cpu_time)
            self.historical_data["gpu_times"].append(avg_gpu_time if avg_gpu_time else 0)
            self.historical_data["cpu_gflops"].append(cpu_gflops)
            self.historical_data["gpu_gflops"].append(gpu_gflops)
            self.historical_data["cpu_utilization"].append(avg_cpu_util)
            self.historical_data["gpu_utilization"].append(gpu_utilization)
            self.historical_data["sizes"].append(matrix_size)

            # Actualizar etiquetas de resultados
            self.update_result_labels(avg_cpu_time, avg_gpu_time, speedup, cpu_gflops, gpu_gflops)

            # Actualizar gráficos
            self.update_graphics()

            self.status_label.configure(text="Prueba completada con éxito", text_color="#90EE90")

        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="#FF6347")
            print(f"Error: {str(e)}")

        finally:
            self.is_running = False
            self.run_button.configure(state="normal", text="Ejecutar prueba")


    # Actualizar Etiquetas de Resultados
    def update_result_labels(self, cpu_time, gpu_time, speedup, cpu_gflops, gpu_gflops):

        self.after(0, lambda: self._update_labels(cpu_time, gpu_time, speedup, cpu_gflops, gpu_gflops))


    # Actualizar Etiquetas de Resultados
    def _update_labels(self, cpu_time, gpu_time, speedup, cpu_gflops, gpu_gflops):

        self.cpu_time_label.configure(
            text=f"CPU: {cpu_time:.4f} s (±{np.std(self.historical_data['cpu_times']):.4f} s)")

        if gpu_time is not None:
            self.gpu_time_label.configure(
                text=f"GPU: {gpu_time:.6f} s (±{np.std(self.historical_data['gpu_times']):.6f} s)")
            self.speedup_label.configure(text=f"Aceleración: {speedup:.2f}x")
        else:
            self.gpu_time_label.configure(text="GPU: No disponible")
            self.speedup_label.configure(text="Aceleración: N/A")

        self.cpu_gflops_label.configure(text=f"CPU GFLOPS: {cpu_gflops:.2f}")
        self.gpu_gflops_label.configure(text=f"GPU GFLOPS: {gpu_gflops:.2f}" if gpu_time else "GPU GFLOPS: N/A")


    # Crear Gráficos Vacíos
    def create_empty_plots(self):

        # Plot 1: Tiempos de ejecución
        self._create_plot(0, "Tiempo de ejecución (s)", "Tiempo (segundos)", "Comparación de tiempos")

        # Plot 2: Aceleración
        self._create_plot(1, "Aceleración CPU vs GPU", "Veces más rápido", "Factor de aceleración")

        # Plot 3: Rendimiento computacional
        self._create_plot(2, "Rendimiento computacional", "GFLOPS", "Operaciones por segundo")

        # Plot 4: Utilización de recursos
        self._create_plot(3, "Utilización de recursos", "% Utilización", "CPU y GPU")


    # Crear Gráficos
    def _create_plot(self, idx, title, ylabel, xlabel):

        if self.canvases[idx]:
            self.canvases[idx].get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
        fig.patch.set_facecolor('#2B2B2B')
        ax.set_facecolor('#2B2B2B')

        # Configuración de colores para tema oscuro
        ax.spines['bottom'].set_color('#999999')
        ax.spines['top'].set_color('#999999')
        ax.spines['left'].set_color('#999999')
        ax.spines['right'].set_color('#999999')
        ax.tick_params(axis='x', colors='#CCCCCC')
        ax.tick_params(axis='y', colors='#CCCCCC')
        ax.set_title(title, color='#CCCCCC')
        ax.set_ylabel(ylabel, color='#CCCCCC')
        ax.set_xlabel(xlabel, color='#CCCCCC')
        ax.grid(True, linestyle='--', alpha=0.3, color='#999999')

        plt.tight_layout()

        self.canvases[idx] = FigureCanvasTkAgg(fig, master=self.graph_frames[idx])
        self.canvases[idx].draw()
        self.canvases[idx].get_tk_widget().pack(fill="both", expand=True)

        return ax


    # Actualizar Gráficos
    def update_graphics(self):

        self.after(0, self._update_graphics)


    # Actualizar Gráficos
    def _update_graphics(self):

        if not self.historical_data["sizes"]:
            return

        # Gráfico 1: Tiempos de ejecución
        ax1 = self._create_plot(0, "Tiempo de ejecución (s)", "Tiempo (segundos)", "Tamaño de matriz")

        x = np.arange(len(self.historical_data["sizes"]))
        width = 0.35

        ax1.bar(x - width / 2, self.historical_data["cpu_times"], width, label='CPU', color='#4C72B0')
        ax1.bar(x + width / 2, self.historical_data["gpu_times"], width, label='GPU', color='#55A868')

        # Si hay un historial, mostrar los tamaños de matriz como etiquetas del eje x
        if self.show_history.get():
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(s) for s in self.historical_data["sizes"]])
        else:
            # Mostrar solo la última prueba
            latest_idx = len(self.historical_data["sizes"]) - 1
            ax1.clear()
            ax1.bar(["CPU"], [self.historical_data["cpu_times"][latest_idx]], color='#4C72B0')
            ax1.bar(["GPU"], [self.historical_data["gpu_times"][latest_idx]], color='#55A868')

        ax1.legend(loc='upper left', facecolor='#2B2B2B', labelcolor='#CCCCCC')

        if self.historical_data["gpu_times"] and max(self.historical_data["gpu_times"]) > 0:
            # Usar escala logarítmica si la diferencia es muy grande
            if max(self.historical_data["cpu_times"]) / max(self.historical_data["gpu_times"]) > 10:
                ax1.set_yscale('log')

        self.canvases[0].draw()

        # Gráfico 2: Factor de aceleración
        ax2 = self._create_plot(1, "Aceleración CPU vs GPU", "Veces más rápido", "Tamaño de matriz")

        speedups = []
        for cpu_t, gpu_t in zip(self.historical_data["cpu_times"], self.historical_data["gpu_times"]):
            if gpu_t > 0:
                speedups.append(cpu_t / gpu_t)
            else:
                speedups.append(0)

        if self.show_history.get():
            ax2.plot(self.historical_data["sizes"], speedups, 'o-', color='#DD8452')
            ax2.set_xticks(self.historical_data["sizes"])
        else:
            if speedups:
                ax2.bar(["Aceleración"], [speedups[-1]], color='#DD8452')

        self.canvases[1].draw()

        # Gráfico 3: Rendimiento computacional (GFLOPS)
        ax3 = self._create_plot(2, "Rendimiento computacional", "GFLOPS", "Tamaño de matriz")

        if self.show_history.get():
            ax3.plot(self.historical_data["sizes"], self.historical_data["cpu_gflops"], 'o-', label='CPU',
                     color='#4C72B0')
            ax3.plot(self.historical_data["sizes"], self.historical_data["gpu_gflops"], 'o-', label='GPU',
                     color='#55A868')
            ax3.set_xticks(self.historical_data["sizes"])
        else:
            latest_idx = len(self.historical_data["sizes"]) - 1
            ax3.bar(["CPU"], [self.historical_data["cpu_gflops"][latest_idx]], color='#4C72B0')
            ax3.bar(["GPU"], [self.historical_data["gpu_gflops"][latest_idx]], color='#55A868')

        ax3.legend(loc='upper left', facecolor='#2B2B2B', labelcolor='#CCCCCC')
        self.canvases[2].draw()

        # Gráfico 4: Utilización de recursos
        ax4 = self._create_plot(3, "Utilización de recursos", "% Utilización", "Prueba")

        if self.show_history.get():
            ax4.plot(self.historical_data["sizes"], self.historical_data["cpu_utilization"], 'o-', label='CPU',
                     color='#4C72B0')
            ax4.plot(self.historical_data["sizes"], self.historical_data["gpu_utilization"], 'o-', label='GPU',
                     color='#55A868')
            ax4.set_xticks(self.historical_data["sizes"])
        else:
            latest_idx = len(self.historical_data["sizes"]) - 1
            ax4.bar(["CPU"], [self.historical_data["cpu_utilization"][latest_idx]], color='#4C72B0')
            ax4.bar(["GPU"], [self.historical_data["gpu_utilization"][latest_idx]], color='#55A868')

        ax4.set_ylim(0, 100)
        ax4.legend(loc='upper left', facecolor='#2B2B2B', labelcolor='#CCCCCC')
        self.canvases[3].draw()




# Ejecutar la aplicación
if __name__ == "__main__":
    app = GPUvsCPUApp()
    app.mainloop()