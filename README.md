# 🧠 Análisis de Rendimiento CPU vs GPU con Python

<p align="center">
  <img src="Imagenes/1.png" alt="Interfaz de la aplicación" width="700"/>
</p>


## GPU vs CPU Performance Analyzer 🚀

Una aplicación de análisis de rendimiento que compara el poder computacional entre GPU y CPU utilizando operaciones de multiplicación de matrices. Esta herramienta proporciona métricas detalladas de rendimiento, visualizaciones en tiempo real y análisis comparativo para evaluar la eficiencia de ambos procesadores. La herramienta proporciona métricas detalladas, visualizaciones en tiempo real y análisis comparativo para evaluar la eficiencia de ambos procesadores. Desarrollar este tipo de experimentos forma parte de mi formación profesional y también de un interés personal por comprender a fondo el impacto que tiene la computación acelerada en tareas analíticas complejas.

## ✨ Características Principales


### 🎯 **Comparación de Rendimiento**
- **Medición precisa** de tiempos de ejecución CPU vs GPU
- **Cálculo de factor de aceleración** (speedup) en tiempo real
- **Métricas GFLOPS** para evaluar el rendimiento computacional
- **Monitoreo de utilización** de recursos del sistema


### 📊 **Visualización Avanzada**
- **Gráficos en tiempo real** con cuatro paneles informativos:
  - Comparación de tiempos de ejecución
  - Factor de aceleración GPU vs CPU
  - Rendimiento computacional (GFLOPS)
  - Utilización de recursos del sistema


### ⚙️ **Configuración Flexible**
- **Tamaño de matriz configurable** (1000x1000 hasta 5000x5000)
- **Número de iteraciones ajustable** (1-10 repeticiones)
- **Historial de pruebas** con opción de limpieza
- **Detección automática** de hardware GPU

## 🔧 Cómo Funciona

La aplicación cuenta con una interfaz gráfica moderna y funcional que muestra:
- Panel de control con parámetros configurables
- Información del hardware detectado
- Resultados de rendimiento en tiempo real
- Cuatro gráficos interactivos para análisis visual



### Principio de Funcionamiento

La aplicación utiliza **multiplicación de matrices** como benchmark para comparar el rendimiento entre CPU y GPU. Este tipo de operación es ideal para la comparación porque:

1. **Paralelización masiva**: Las GPUs están optimizadas para operaciones paralelas
2. **Carga computacional intensiva**: Permite medir diferencias significativas de rendimiento
3. **Operaciones floating-point**: Evalúa la capacidad de cálculo científico


### Proceso de Comparación

```
1. Generación de Matrices → Creación de matrices aleatorias NxN
2. Prueba CPU → Ejecución secuencial en procesador principal
3. Prueba GPU → Ejecución paralela en tarjeta gráfica
4. Análisis de Resultados → Cálculo de métricas y speedup
5. Visualización → Actualización de gráficos en tiempo real
```


### Algoritmo de Benchmark

La aplicación ejecuta las siguientes operaciones:

- **Matriz A**: Matriz aleatoria de dimensiones N×N
- **Matriz B**: Matriz aleatoria de dimensiones N×N  
- **Operación**: `C = A × B` (multiplicación matricial)
- **Medición**: Tiempo promedio de múltiples iteraciones



## 🚀 Instalación y Uso


### Requisitos Previos

```bash
pip install tensorflow
pip install customtkinter
pip install matplotlib
pip install psutil
pip install numpy
```

### Ejecución

```bash
python "GPU vs CPU 5.py"
```

### Configuración de Parámetros

1. **Ajustar tamaño de matriz**: Utiliza el slider o ingresa el valor directamente
2. **Establecer iteraciones**: Define el número de repeticiones para mayor precisión
3. **Ejecutar prueba**: Presiona "Ejecutar prueba" para iniciar el benchmark
4. **Visualizar resultados**: Los gráficos se actualizan automáticamente



## 📈 Métricas de Rendimiento


### Indicadores Principales

- **Tiempo de Ejecución**: Medición en segundos con desviación estándar
- **Factor de Aceleración**: Ratio de velocidad GPU/CPU (ej: 15.6x más rápido)
- **GFLOPS**: Giga-operaciones de punto flotante por segundo
- **Utilización**: Porcentaje de uso de recursos del sistema


### Interpretación de Resultados

| Métrica | Descripción | Valor Ideal |
|---------|-------------|-------------|
| **Speedup** | Aceleración GPU vs CPU | > 10x |
| **CPU GFLOPS** | Rendimiento procesador | 50-200 GFLOPS |
| **GPU GFLOPS** | Rendimiento tarjeta gráfica | 1000-5000+ GFLOPS |
| **Utilización** | Eficiencia de recursos | 80-95% |



## 🛠️ Detalles Técnicos


### Arquitectura del Software

- **Framework GUI**: CustomTkinter (tema oscuro moderno)
- **Backend Computacional**: TensorFlow 2.x
- **Visualización**: Matplotlib con backend TkAgg
- **Monitoreo Sistema**: PSUtil para métricas de hardware
- **Threading**: Ejecución asíncrona para UI responsiva


### Implementación de Benchmark

```python
# Configuración de dispositivos
with tf.device('/CPU:0'):  # Forzar ejecución en CPU
    result_cpu = tf.matmul(matrix_a, matrix_b)

with tf.device('/GPU:0'):  # Forzar ejecución en GPU  
    result_gpu = tf.matmul(matrix_a, matrix_b)
```


### Cálculo de Métricas

- **Operaciones por matriz**: `2 × N³` (aproximación para multiplicación matricial)
- **GFLOPS**: `Operaciones / (Tiempo × 10⁹)`
- **Speedup**: `Tiempo_CPU / Tiempo_GPU`
- **Utilización CPU**: Medición con `psutil.cpu_percent()`


### Compatibilidad de Hardware

- **CPU**: Cualquier procesador x86/x64
- **GPU**: NVIDIA con soporte CUDA (compute capability ≥ 3.5)
- **RAM**: Mínimo 4GB (recomendado 8GB+)
- **SO**: Windows, Linux, macOS


### Limitaciones Conocidas

- La aplicación requiere TensorFlow con soporte GPU para aprovechar la aceleración
- Matrices muy grandes pueden causar errores de memoria insuficiente
- La medición de utilización GPU es aproximada (valor fijo del 95%)


## 📝 **Licencia**

Este proyecto está licenciado bajo la [Licencia GNU](LICENSE).

---
