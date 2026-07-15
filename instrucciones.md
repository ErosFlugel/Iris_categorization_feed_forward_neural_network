Tu tarea es desarrollar el código fuente completo para un "Motor de Inferencia de Red Neuronal Feed-Forward", diseñado para resolver el problema de clasificación botánica del Dataset Iris de Fisher. El proyecto debe entregarse en un plazo estricto y cumplir con rigurosos criterios académicos de manejo de memoria y complejidad algorítmica.

El sistema debe programarse EXCLUSIVAMENTE en C nativo, sin utilizar ninguna librería externa de Machine Learning. Solo puedes usar <stdio.h>, <stdlib.h>, <math.h> y <time.h>.

DEBES implementar estrictamente los siguientes requerimientos técnicos, algorítmicos y de arquitectura:

1. Arquitectura de la Red y TAD (Tipos Abstractos de Datos):

- Diseña estructuras (`struct`) modulares para las Capas (`Layer`) y la Red (`FeedForwardNN`).
- La topología debe ser completamente configurable dinámicamente en tiempo de ejecución (número de capas y nodos variables), utilizando obligatoriamente `malloc()`, `calloc()` o `realloc()`. No uses arreglos estáticos de tamaño fijo para la red.
- Implementa la función de activación Sigmoide.

2. Algoritmos Obligatorios:

- Multiplicación de Matrices: Debes implementar el Algoritmo de Strassen para optimizar la multiplicación de las matrices de pesos por las entradas durante el Forward Propagation.
- Pre-procesamiento/Ordenamiento: Implementa Quick Sort (utilizando el esquema de partición de Lomuto) o Merge Sort para pre-procesar y normalizar el dataset de entrada antes de pasarlo a la red, o para ordenar las probabilidades de salida.
- Aritmética de Punteros: Manipula todas las matrices y arreglos de la red neuronal utilizando aritmética de punteros pura, evitando el uso de índices tradicionales (como `array[i][j]`).

3. Procesamiento de Archivos e Interfaz:

- El programa debe leer los datos de entrenamiento desde un archivo `flores-dataset.txt` utilizando flujos de archivos (`fopen`, `fscanf`, `fclose`) y parsear estos registros en estructuras en C.
- Cada línea del archivo contendrá 4 medidas numéricas continuas y un entero (LargoSépalo, AnchoSépalo, LargoPétalo, AnchoPétalo y ClaseEsperada la clase real: 0, 1 o 2 para Setosa, Versicolor y Virgínica respectivamente).

4. Gestión de Memoria y Rendimiento:

- Es crítico que el programa incluya una función de limpieza exhaustiva que utilice `free()` para liberar absolutamente toda la memoria dinámica asignada, garantizando cero fugas de memoria (memory leaks).
- Implementa la librería `<time.h>` para realizar benchmarking y medir el tiempo exacto de ejecución del bloque de inferencia (Forward Propagation).

FORMATO DE SALIDA ESPERADO:
Por favor, genera el código completo dividido de forma modular:

1. `network.h` (Definición de las estructuras, TADs y prototipos de funciones).
2. `network.c` (Implementación de los algoritmos como Strassen, Quick Sort, y la lógica de la red).
3. `main.c` (Punto de entrada, lectura del archivo `.txt`, ejecución del benchmarking y limpieza de memoria).
4. El contenido de un archivo `Makefile` básico para compilar el proyecto fácilmente.

Asegúrate de que el código esté abundantemente comentado, explicando la complejidad teórica (Notación Big-O) de los algoritmos clave elegidos.

Al finalizar crea un arhivo "walkthrough.md" que contenga un recorrido explicativo sobre todo el funcionamiento en cada paso y función para que un estudiante universitario pueda entender todo el código, la implementación y la lógica que se realizó
