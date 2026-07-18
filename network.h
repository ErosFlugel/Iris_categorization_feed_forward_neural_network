/**
 * =============================================================================
 * ARCHIVO: network.h
 * DESCRIPCION: Definicion de estructuras (TAD) y prototipos de funciones para
 *              una Red Neuronal Feed-Forward con algorítmos optimizados.
 *
 * PROYECTO: Clasificador del Dataset Iris
 * AUTORES: Luis Eulacio, Eros Flugel
 * FECHA: Julio 2026
 * =============================================================================
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* =============================================================================
 * CONSTANTES DE CONFIGURACIÓN DE LA RED
 * La topología es 4 -> HIDDEN_SIZE -> 3 (Entrada, Oculta, Salida)
 * =============================================================================
 */
#define INPUT_SIZE   4    /* 4 features: LargoSepalo, AnchoSepalo, LargoPetalo, AnchoPetalo */
#define OUTPUT_SIZE  3    /* 3 clases: Setosa(0), Versicolor(1), Virginica(2)              */
#define HIDDEN_SIZE  8    /* Neuronas en capa oculta (valor fijo optimizado para Iris)      */

/* =============================================================================
 * ESTRUCTURA: Layer (Capa de la Red Neuronal)
 * ---------------------------------------------------------------------------
 * Representa una capa con sus pesos sinápticos, sesgos y activaciones.
 * Dimensiones: [num_neurons x num_inputs] para pesos, [num_neurons] para sesgos.
 *
 * Almacenamiento contiguo: se usa un único bloque malloc para la matriz de pesos,
 * y se accede mediante aritmética de punteros pura (sin array[i][j]).
 * =============================================================================
 */
typedef struct {
    int num_neurons;    /* Número de neuronas en esta capa    */
    int num_inputs;     /* Número de entradas a esta capa     */
    double *weights;    /* Matriz de pesos [num_neurons x num_inputs], almacenamiento contiguo */
    double *biases;     /* Vector de sesgos [num_neurons]     */
    double *activations;/* Vector de activaciones de salida [num_neurons] */
    double *z;          /* Suma ponderada antes de activación [num_neurons] */
} Layer;

/* =============================================================================
 * ESTRUCTURA: FeedForwardNN (Red Neuronal Completa)
 * ---------------------------------------------------------------------------
 * Contiene el array de capas y dimensiones. La red es completamente configurable
 * en tiempo de ejecución mediante malloc().
 * =============================================================================
 */
typedef struct {
    int num_layers;     /* Número total de capas (incluyendo entrada y salida) */
    Layer *layers;      /* Array dinámico de capas [num_layers]                */
    double learning_rate; /* Tasa de aprendizaje (configurable)               */
    int epochs;           /* Número de epocas de entrenamiento (configurable) */
} FeedForwardNN;

/* =============================================================================
 * ESTRUCTURA: Dataset
 * ---------------------------------------------------------------------------
 * Almacena las muestras leídas del archivo flores-dataset.txt.
 * Se reserva memoria dinámicamente y se separa en entrenamiento (80%) y
 * prueba (20%) para validación.
 * =============================================================================
 */
typedef struct {
    double **features;   /* Matriz de features [num_samples x INPUT_SIZE]  */
    int *labels;         /* Vector de etiquetas de clase [num_samples]     */
    int num_samples;     /* Total de muestras en el dataset               */
    int num_train;       /* Muestras de entrenamiento (80%)               */
    int num_test;        /* Muestras de prueba (20%)                      */
    double **train_features; /* Features de entrenamiento                 */
    int *train_labels;       /* Etiquetas de entrenamiento                */
    double **test_features;  /* Features de prueba                        */
    int *test_labels;        /* Etiquetas de prueba                       */
} Dataset;

/* =============================================================================
 * PROTOTIPOS DE FUNCIONES - network.c
 * =============================================================================
 */

/* --- Funciones de la Red Neuronal --- */

/*
 * Crea y retorna una red neuronal con la topología configurada.
 * Reserva toda la memoria dinámicamente con malloc/calloc.
 * Inicializa pesos con valores aleatorios en [-0.5, 0.5].
 */
FeedForwardNN* nn_create(int num_layers, int *layer_sizes, double lr, int epochs);

/*
 * Función de activación Sigmoide: sigma(x) = 1 / (1 + e^(-x))
 * Complejidad: O(1) por neurona.
 */
double sigmoid(double x);

/*
 * Derivada de la Sigmoide: sigma'(x) = sigma(x) * (1 - sigma(x))
 * Complejidad: O(1) por neurona.
 */
double sigmoid_derivative(double x);

/*
 * Forward Propagation: propaga una entrada a traves de la red.
 * Utiliza el Algorítmo de Strassen para multiplicaciones de matrices.
 * Complejidad teórica de Strassen: O(n^2.807) vs O(n^3) del naive.
 */
void nn_forward(FeedForwardNN *nn, double *input);

/*
 * Backpropagation: calcula gradientes y actualiza pesos.
 * Utiliza la derivada de la Sigmoide.
 * Complejidad: O(N * M) donde N=muestras, M=total conexiones.
 */
void nn_backward(FeedForwardNN *nn, double *target);

/*
 * Entrena la red con el dataset completo por el número de epocas.
 * Imprime la perdida (MSE) cada 100 epocas.
 */
void nn_train(FeedForwardNN *nn, Dataset *dataset);

/*
 * Predice la clase de una entrada. Retorna el indice con mayor probabilidad.
 */
int nn_predict(FeedForwardNN *nn, double *input);

/*
 * Evalua la red con un conjunto de datos y retorna la precision (accuracy).
 */
double nn_evaluate(FeedForwardNN *nn, double **features, int *labels, int num_samples);

/*
 * Libera toda la memoria asignada a la red neuronal.
 */
void nn_free(FeedForwardNN *nn);

/* --- Algoritmo de Strassen para Multiplicación de Matrices --- */

/*
 * Multiplica dos matrices cuadradas A x B -> C usando Strassen.
 * Tamaño debe ser potencia de 2 (se redondea hacia arriba internamente).
 * Complejidad: O(n^2.807) - significativamente mejor que O(n^3) para matrices grandes.
 */
void strassen_multiply(double *A, double *B, double *C, int n);

/*
 * Multiplicación naive de matrices para comparación o fallback.
 * Complejidad: O(n^3) - algorítmo clásico triple bucle anidado.
 */
void naive_multiply(double *A, double *B, double *C, int n);

/*
 * Multiplica matriz [m x k] por vector [k] -> resultado [m].
 * Utiliza aritmética de punteros pura.
 */
void matrix_vector_multiply(double *matrix, double *vector, double *result,
                            int rows, int cols);

/* --- Algorítmo Quick Sort (Esquema de Lomuto) --- */

/*
 * Ordena un array de doubles y reordena un array de índices asociados.
 * Se usa para ordenar probabilidades de salida o normalizar datos.
 * Complejidad promedio: O(n log(n)) - peor caso: O(n^2).
 * Se usa partición de Lomuto como requiere el enunciado.
 */
void quicksort(double *arr, int *indices, int low, int high);

/*
 * Función de partición de Lomuto para QuickSort.
 * Elige el último elemento como pivot.
 * Complejidad: O(n) por partición.
 */
int lomuto_partition(double *arr, int *indices, int low, int high);

/* --- Funciones de Dataset --- */

/*
 * Lee el dataset desde un archivo de texto.
 * Cada línea: 4 floats + 1 entero (clase).
 * Retorna un puntero a Dataset con memoria reservada.
 */
Dataset* dataset_load(const char *filename);

/*
 * Separa el dataset en entrenamiento (80%) y prueba (20%).
 * Emplea QuickSort para ordenar los índices antes de la separación.
 */
void dataset_split(Dataset *dataset, double train_ratio);

/*
 * Normaliza los features usando min-max scaling por columna.
 * Cada feature se escala al rango [0, 1].
 */
void dataset_normalize(Dataset *dataset);

/*
 * Libera toda la memoria del dataset.
 */
void dataset_free(Dataset *dataset);

/* --- Funciones de Metricas y Evaluación --- */

/*
 * Imprime una matriz de confusión textual en la terminal.
 */
void print_confusion_matrix(FeedForwardNN *nn, Dataset *dataset);

/*
 * Imprime métricas de evaluación: precisión por clase y accuracy global.
 */
void print_evaluation_metrics(FeedForwardNN *nn, Dataset *dataset);

/* --- Benchmarking --- */

/*
 * Mide el tiempo de ejecución del Forward Propagation sobre todo el dataset
 * de prueba y retorna el tiempo en milisegundos.
 */
double benchmark_forward(FeedForwardNN *nn, Dataset *dataset);

#endif /* NETWORK_H */
