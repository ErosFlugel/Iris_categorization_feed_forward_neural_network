# Walkthrough: Motor de Inferencia de Red Neuronal Feed-Forward (Iris Classifier)

**Autores:** Luis Eulacio, Eros Flugel  
**Fecha:** Julio 2026  
**Objetivo:** Documentación explicativa completa del código para un estudiante universitario.

---

## Tabla de Contenidos

1. [Visión General del Proyecto](#1-visión-general-del-proyecto)
2. [Arquitectura del Código](#2-arquitectura-del-código)
3. [Archivo `network.h` - Definiciones y TADs](#3-archivo-networkh---definiciones-y-tads)
4. [Archivo `network.c` - Implementaciones](#4-archivo-networkc---implementaciones)
5. [Archivo `main.c` - Punto de Entrada](#5-archivo-mainc---punto-de-entrada)
6. [Algoritmos Clave](#6-algoritmos-clave)
7. [Flujo del Programa Completo](#7-flujo-del-programa-completo)
8. [Compilación y Ejecución](#8-compilación-y-ejecución)

---

## 1. Visión General del Proyecto

Este programa implementa una **Red Neuronal Feed-Forward** completa en C puro (sin librerías externas de ML) para clasificar flores del dataset Iris en tres especies:

- **Iris Setosa** (clase 0)
- **Iris Versicolor** (clase 1)  
- **Iris Virginica** (clase 2)

### ¿Qué hace el programa?

1. **Lee** 130 muestras desde `flores-dataset.txt` (4 características + 1 clase)
2. **Normaliza** los datos al rango [0, 1] usando min-max scaling
3. **Divide** el dataset: 80% entrenamiento, 20% prueba (estratificado)
4. **Pide al usuario** el Learning Rate y número de Épocas
5. **Crea** una red neuronal con topología 4 → 8 → 3
6. **Entrena** la red usando backpropagation
7. **Evalúa** la precisión y muestra una matriz de confusión
8. **Mide** el tiempo de inferencia (benchmarking)
9. **Libera** toda la memoria dinámica

### Topología de la Red

```
Entrada (4)    Oculta (8)    Salida (3)
  [Largo Sépalo]  ──┐
  [Ancho Sépalo]  ──┼──→ [Neuronas] ──→ [Setosa]
  [Largo Pétalo]  ──┤     (8)         [Versicolor]
  [Ancho Pétalo]  ──┘                [Virginica]
```

---

## 2. Arquitectura del Código

El proyecto está dividido en 4 archivos modulares:

| Archivo | Responsabilidad |
|---------|-----------------|
| `network.h` | Definiciones de estructuras (TADs), constantes, prototipos de funciones |
| `network.c` | Implementación de todos los algoritmos (Strassen, QuickSort, red neuronal) |
| `main.c` | Punto de entrada, interfaz de usuario, flujo principal |
| `Makefile` | Instrucciones de compilación |

### Principios de Diseño

- **Modularidad:** Cada función tiene una responsabilidad clara
- **Aritmética de punteros:** Se evita `array[i][j]`, se usa `*(array + i*n + j)`
- **Memoria dinámica:** Todo se reserva con `malloc()`, `calloc()`, `realloc()`
- **Gestión exhaustiva:** Cada `malloc()` tiene su `free()` correspondiente

---

## 3. Archivo `network.h` - Definiciones y TADs

### Constantes

```c
#define INPUT_SIZE   4    // 4 features del dataset Iris
#define OUTPUT_SIZE  3    // 3 clases de flores
#define HIDDEN_SIZE  8    // 8 neuronas en capa oculta
```

### Estructura `Layer` (Capa)

Cada capa de la red neuronal contiene:

```c
typedef struct {
    int num_neurons;      // Cuántas neuronas tiene esta capa
    int num_inputs;       // Cuántas entradas recibe cada neurona
    double *weights;      // Matriz de pesos [num_neurons × num_inputs]
    double *biases;       // Vector de sesgos [num_neurons]
    double *activations;  // Salidas de las neuronas después de activar
    double *z;            // Suma ponderada antes de la activación
} Layer;
```

**¿Por qué punteros?** Para usar almacenamiento contiguo en memoria (mejora la localidad de datos y el rendimiento de la CPU).

### Estructura `FeedForwardNN` (Red neuronal)

```c
typedef struct {
    int num_layers;        // Número total de capas (3 en nuestro caso)
    Layer *layers;         // Array dinámico de capas
    double learning_rate;  // Tasa de aprendizaje (configurable)
    int epochs;            // Épocas de entrenamiento (configurable)
} FeedForwardNN;
```

### Estructura `Dataset`

```c
typedef struct {
    double **features;      // Matriz de features [num_samples × 4]
    int *labels;            // Vector de clases [num_samples]
    int num_samples;        // Total de muestras (130)
    int num_train;          // Muestras de entrenamiento (~104)
    int num_test;           // Muestras de prueba (~26)
    double **train_features;
    int *train_labels;
    double **test_features;
    int *test_labels;
} Dataset;
```

---

## 4. Archivo `network.c` - Implementaciones

### 4.1 Funciones de Activación

#### `sigmoid(double x)`
```
σ(x) = 1 / (1 + e^(-x))
```

- **¿Qué hace?** Convierte cualquier número real a un valor entre 0 y 1
- **Complejidad:** O(1) - operación constante
- **Uso:** Se aplica en cada neurona después de calcular la suma ponderada

#### `sigmoid_derivative(double x)`
```
σ'(x) = σ(x) × (1 - σ(x))
```

- **¿Qué hace?** Calcula la pendiente de la sigmoide en un punto
- **Uso:** Se necesita en el backpropagation para calcular cuánto ajustar los pesos

---

### 4.2 Algoritmo de Strassen (Multiplicación de Matrices)

#### Problema
Multiplicar dos matrices A×B → C usando el algoritmo naive tiene complejidad **O(n³)**.

#### Solución de Strassen
El algoritmo de Strassen reduce esto a **O(n^2.807)** dividiendo las matrices en subbloques y calculando solo **7 productos intermedios** en lugar de 8.

#### Paso a paso:

1. **Dividir** cada matriz en 4 submatrices:
   ```
   A = | A11  A12 |    B = | B11  B12 |
       | A21  A22 |        | B21  B22 |
   ```

2. **Calcular 7 productos intermedios:**
   ```
   M1 = (A11 + A22)(B11 + B22)
   M2 = (A21 + A22)(B11)
   M3 = (A11)(B12 - B22)
   M4 = (A22)(B21 - B11)
   M5 = (A11 + A12)(B22)
   M6 = (A21 - A11)(B11 + B12)
   M7 = (A12 - A22)(B21 + B22)
   ```

3. **Reconstruir el resultado:**
   ```
   C11 = M1 + M4 - M5 + M7
   C12 = M3 + M5
   C21 = M2 + M4
   C22 = M1 - M2 + M3 + M6
   ```

4. **Caso base:** Para matrices pequeñas (n < 64), se usa el algoritmo naive porque el overhead de la recursión supera el beneficio.

#### ¿Por qué Strassen?
- Para matrices grandes, es significativamente más rápido
- En nuestro caso (redes 4-8-3), las capas son pequeñas, así que usamos `matrix_vector_multiply` directamente
- Strassen se activa cuando `num_neurons >= 64`

#### Aritmética de Punteros
En lugar de `A[i][j]`, usamos `*(A + i*n + j)`:
```c
// Acceso a matriz usando punteros
*(C + i * n + j) += *(A + i * n + k) * *(B + k * n + j);
```

---

### 4.3 QuickSort con Partición de Lomuto

#### Problema
Necesitamos ordenar las muestras por clase para hacer un split estratificado equilibrado.

#### Algoritmo QuickSort

**Complejidad promedio:** O(n log(n))  
**Peor caso:** O(n²)  

#### Partición de Lomuto

El pivot es el **último elemento** del arreglo:

```c
int lomuto_partition(double *arr, int *indices, int low, int high) {
    double pivot = *(arr + high);  // Último elemento es el pivot
    int i = low - 1;              // Borde de elementos <= pivot

    for (j = low; j < high; j++) {
        if (*(arr + j) <= pivot) {
            i++;
            // Intercambiar arr[i] con arr[j]
            // También intercambiar indices[i] con indices[j]
        }
    }
    // Colocar pivot en su posición final
    return i + 1;
}
```

**¿Por qué Lomuto?** Es más simple de implementar que la partición de Hoare y更容易 de entender.

#### QuickSort Recursivo

```c
void quicksort(double *arr, int *indices, int low, int high) {
    if (low < high) {
        int pi = lomuto_partition(arr, indices, low, high);
        quicksort(arr, indices, low, pi - 1);   // Ordenar izquierda
        quicksort(arr, indices, pi + 1, high);   // Ordenar derecha
    }
}
```

**Uso en el proyecto:** Se ordenan los índices por clase para separar equilibradamente entrenamiento y prueba.

---

### 4.4 Funciones de la Red Neuronal

#### `nn_create()` - Crear la Red

1. Reserva memoria para la estructura `FeedForwardNN`
2. Reserva memoria para cada `Layer`
3. Para cada capa, reserva memoria para pesos, sesgos, activaciones y z
4. **Inicializa pesos aleatoriamente** en [-0.5, 0.5]

**¿Por qué [-0.5, 0.5]?** Valores muy grandes pueden causar saturación de la sigmoide, valores muy pequeños pueden causar que la red no aprenda.

#### `nn_forward()` - Propagación hacia Adelante

Para cada capa:
1. Calcular `z = W × activations + bias` (suma ponderada)
2. Aplicar activación: `a = sigmoid(z)`

```
Entrada → [Capa 1: z = W1×input + b1, a = σ(z)] → [Capa 2: z = W2×a + b2, a = σ(z)] → Salida
```

#### `nn_backward_fixed()` - Retropropagación

1. **Calcular error en la salida:** `error = output - target`
2. **Calcular delta en salida:** `delta = error × sigmoid'(output)`
3. **Propagar error hacia atrás:** Para cada capa oculta:
   ```
   error = W^T × delta_siguiente
   delta = error × sigmoid'(activacion)
   ```
4. **Actualizar pesos:** `w_nuevo = w_viejo - learning_rate × delta × input`

**Fórmula de actualización:**
```
W_ij = W_ij - lr × delta_j × a_i
b_j = b_j - lr × delta_j
```

#### `nn_train()` - Entrenamiento

Para cada época:
1. Para cada muestra de entrenamiento:
   - Forward propagation
   - Calcular pérdida (MSE)
   - Backward propagation
2. Imprimir loss cada 100 épocas

#### `nn_predict()` - Predicción

1. Ejecutar forward propagation con la entrada
2. Encontrar la neurona de salida con mayor valor
3. Retornar esa clase como predicción

#### `nn_evaluate()` - Evaluación

Compara predicciones con etiquetas reales y calcula:
```
Accuracy = muestras_correctas / total_muestras
```

---

### 4.5 Funciones de Dataset

#### `dataset_load()` - Cargar Datos

1. Abrir `flores-dataset.txt` con `fopen()`
2. Contar líneas para reservar memoria exacta
3. Leer cada línea con `fscanf()`:
   ```
   fscanf(file, "%lf %lf %lf %lf %d", &f1, &f2, &f3, &f4, &clase)
   ```
4. Reservar memoria dinámica para cada fila de features

#### `dataset_split()` - División Estratificada

**¿Qué es un split estratificado?** En lugar de tomar los primeros 80% y los últimos 20%, se toma proporcionalmente de cada clase:

```
Clase 0 (50 muestras): 40 train, 10 test
Clase 1 (40 muestras): 32 train, 8 test  
Clase 2 (40 muestras): 31 train, 9 test
```

Esto garantiza que el conjunto de prueba tenga representación de todas las clases.

#### `dataset_normalize()` - Normalización Min-Max

Para cada feature:
```
normalized = (value - min) / (max - min)
```

Esto escala todos los valores al rango [0, 1], lo cual:
- Mejora la convergencia del entrenamiento
- Evita que features con rangos grandes dominen sobre otros

---

### 4.6 Métricas y Evaluación

#### Matriz de Confusión

Muestra en una tabla:
- **Filas:** Clases reales
- **Columnas:** Clases predichas
- **Diagonal:** Predicciones correctas

```
           Setosa  Versicolor  Virginica
Setosa     [  9  ] [    0    ] [    0    ]
Versicolor [  0  ] [    9    ] [    0    ]
Virginica  [  0  ] [    0    ] [    9    ]
```

#### Benchmarking

Mide el tiempo de ejecución del forward propagation usando `clock()`:
```c
start = clock();
// Ejecutar forward propagation
end = clock();
time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
```

---

## 5. Archivo `main.c` - Punto de Entrada

### Flujo del Programa

```
main()
  │
  ├── print_banner()           // Mostrar información del proyecto
  │
  ├── dataset_load()           // Leer flores-dataset.txt
  │
  ├── dataset_normalize()      // Escalar features a [0, 1]
  │
  ├── dataset_split(0.8)       // 80% train, 20% test
  │
  ├── get_hyperparameters()    // Pedir LR y épocas al usuario
  │
  ├── nn_create()              // Crear red 4-8-3
  │
  ├── nn_train()               // Entrenar la red
  │
  ├── benchmark_forward()      // Medir tiempo de inferencia
  │
  ├── print_evaluation_metrics() // Mostrar accuracy y matriz de confusión
  │
  ├── nn_free()                // Liberar memoria de la red
  │
  └── dataset_free()           // Liberar memoria del dataset
```

### Interfaz de Usuario

El programa solicita interactivamente:
1. **Learning Rate:** Tamaño del paso en el descenso de gradiente (ej: 0.1, 0.5, 1.0)
2. **Épocas:** Número de veces que se recorre todo el dataset de entrenamiento (ej: 500, 1000)

### Salida Esperada

```
Epoca     1/1000 | Loss (MSE): 0.391145
Epoca   100/1000 | Loss (MSE): 0.014760
...
Accuracy Global: 100.00%
```

---

## 6. Algoritmos Clave - Resumen de Complejidad

| Algoritmo | Complejidad | Uso en el Proyecto |
|-----------|-------------|-------------------|
| Strassen | O(n^2.807) | Multiplicación de matrices (capas grandes) |
| Naive Multiply | O(n³) | Multiplicación de matrices (capas pequeñas) |
| QuickSort | O(n log(n)) promedio | Ordenar muestras por clase |
| Sigmoide | O(1) | Función de activación |
| Forward Prop | O(L × n²) | Propagación hacia adelante |
| Backpropagation | O(N × M) | Ajuste de pesos |

Donde:
- L = número de capas
- n = número de neuronas por capa
- N = muestras de entrenamiento
- M = total de conexiones en la red

---

## 7. Gestión de Memoria

### Principios

1. **Cada `malloc()` tiene su `free()` correspondiente**
2. **Se libera en orden inverso** a la asignación
3. **Se verifica** que los punteros no sean NULL antes de liberar

### Funciones de Limpieza

- `nn_free()`: Libera la red neuronal completa
  - Pesos, sesgos, activaciones de cada capa
  - Array de capas
  - Estructura principal

- `dataset_free()`: Libera el dataset
  - Cada fila de features individualmente
  - Vectores de labels
  - Arrays de train/test

### Prevención de Memory Leaks

```c
void nn_free(FeedForwardNN *nn) {
    if (!nn) return;
    
    for (i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers + i;
        free(layer->weights);    // Liberar pesos
        free(layer->biases);     // Liberar sesgos
        free(layer->activations);// Liberar activaciones
        free(layer->z);          // Liberar sumas ponderadas
    }
    free(nn->layers);  // Liberar array de capas
    free(nn);          // Liberar estructura principal
}
```

---

## 8. Compilación y Ejecución

### Con Makefile

```bash
# Compilar
make

# Ejecutar
make run

# Limpiar
make clean
```

### Sin Makefile (manual)

```bash
gcc -Wall -Wextra -std=c99 -O2 -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -O2 -c network.c -o network.o
gcc -Wall -Wextra -std=c99 -O2 -o iris_nn.exe main.o network.o -lm
./iris_nn.exe
```

### Parámetros de Compilación

- `-Wall -Wextra`: Habilitar todas las advertencias
- `-std=c99`: Usar estándar C99
- `-O2`: Optimización de nivel 2
- `-lm`: Enlazar librería matemática (necesaria para `exp()`)

### Ejemplo de Ejecución

```
$ ./iris_nn.exe

================================================================
  MOTOR DE INFERENCIA - RED NEURONAL FEED-FORWARD
  Clasificador del Dataset Iris
================================================================

Dataset cargado exitosamente: 130 muestras.
Dataset normalizado al rango [0, 1].
Dataset dividido (estratificado): 103 entrenamiento, 27 prueba.

  Ingrese el Learning Rate: 0.5
  Ingrese el numero de Epocas: 1000

Epoca     1/1000 | Loss (MSE): 0.391145
Epoca   100/1000 | Loss (MSE): 0.014760
...
Epoca  1000/1000 | Loss (MSE): 0.000121

Accuracy Global: 100.00%

  +---------------+----------+----------+----------+
  | Matriz de Confusion                        |
  +---------------+----------+----------+----------+
  | Setosa        |        9 |        0 |        0 |
  | Versicolor    |        0 |        9 |        0 |
  | Virginica     |        0 |        0 |        9 |
  +---------------+----------+----------+----------+

  Memoria liberada exitosamente.
  Cero memory leaks.
```

---

## 9. Conceptos Clave para Recordar

### Red Neuronal Feed-Forward
- La información fluye de izquierda a derecha (sin ciclos)
- Cada neurona recibe entradas, las multiplica por pesos, suma el sesgo, y aplica una función de activación

### Backpropagation
- Algoritmo para entrenar la red
- Calcula cuánto contribuye cada peso al error
- Actualiza los pesos en la dirección que reduce el error

### Learning Rate
- Controla qué tan grandes son los pasos de actualización
- **Muy alto:** La red puede "pasarse" del mínimo óptimo
- **Muy bajo:** La red aprende muy lentamente

### Épocas
- Una época = recorrer todo el dataset de entrenamiento una vez
- Más épocas = más oportunidades de aprender, pero riesgo de overfitting

### Overfitting vs Underfitting
- **Overfitting:** La red memoriza los datos de entrenamiento pero falla con datos nuevos
- **Underfitting:** La red no aprende lo suficiente, tiene bajo rendimiento en ambos conjuntos

---

## 10. Preguntas Frecuentes

### ¿Por qué no se usa una librería de ML?
El objetivo es entender a bajo nivel cómo funciona una red neuronal, desde la gestión de memoria hasta los algoritmos matemáticos.

### ¿Por qué Strassen si las capas son pequeñas?
Se implementa como requisito académico. En la práctica, para capas pequeñas (como 4-8-3), el algoritmo naive es más rápido debido al overhead de la recursión.

### ¿Por qué C y no Python?
C permite control total de la memoria y el rendimiento, lo cual es educativo para entender qué sucede "bajo el capó" de las librerías de ML.

### ¿Cómo se controla el overfitting?
- Usando un conjunto de prueba separado (20% del dataset)
- Monitoreando la curva de loss durante el entrenamiento
- Ajustando el número de épocas y learning rate

---

**Fin del Walkthrough**
