/**
 * =============================================================================
 * ARCHIVO: network.c
 * DESCRIPCION: Implementacion completa del Motor de Inferencia de Red Neuronal
 *              Feed-Forward con Strassen, QuickSort y aritmética de punteros.
 *
 * ALGORITMOS IMPLEMENTADOS:
 *   - Strassen: Multiplicacion de matrices O(n^2.807)
 *   - QuickSort (Lomuto): Ordenamiento O(n log n) promedio
 *   - Sigmoide: Funcion de activacion O(1)
 *   - Backpropagation: Ajuste de pesos O(N*M)
 *
 * PROYECTO: Clasificador del Dataset Iris
 * AUTORES: Luis Eulacio, Eros Flugel
 * FECHA: Julio 2026
 * =============================================================================
 */

#include "network.h"
#include <math.h>
#include <string.h>

/* =============================================================================
 * FUNCIONES DE ACTIVACION
 * =============================================================================
 */

/**
 * sigmoid - Funcion de activacion Sigmoide.
 * Formula: sigma(x) = 1 / (1 + e^(-x))
 *
 * Mapea cualquier valor real al rango (0, 1), ideal para probabilidades.
 * Complejidad teorica: O(1) - operacion aritmetica constante.
 *
 * @param x  Valor de entrada (suma ponderada de la neurona)
 * @return   Valor activado en el rango (0, 1)
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * sigmoid_derivative - Derivada de la Sigmoide.
 * Formula: sigma'(x) = sigma(x) * (1 - sigma(x))
 *
 * Se usa en el backpropagation para calcular los gradientes.
 * Complejidad teorica: O(1) - operacion aritmetica constante.
 *
 * @param x  Valor activado (ya pasado por sigmoid)
 * @return   Derivada de la funcion Sigmoide en el punto x
 */
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

/* =============================================================================
 * ALGORITMO DE STRASSEN PARA MULTIPLICACIÓN DE MATRICES
 * =============================================================================
 *
 * Complejidad teorica: O(n^2.807)
 * Comparacion con naive: O(n^3)
 *
 * El algoritmo reduce el numero de multiplicaciones de 8 a 7 dividiendo
 * la matriz en subbloques recursivamente. Para matrices pequeñas (n < 64),
 * se usa naive como optimización práäctica ya que la overhead recursiva
 * supera el beneficio.
 */

/**
 * naive_multiply - Multiplicación clasica de matrices (triple bucle).
 * Complejidad: O(n^3)
 *
 * Se usa como fallback cuando n es pequeña (n < 64) o no es potencia de 2.
 */
void naive_multiply(double *A, double *B, double *C, int n) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            /* Aritmética de punteros: *(C + i*n + j) equivale a C[i][j] */
            *(C + i * n + j) = 0.0;
            for (k = 0; k < n; k++) {
                *(C + i * n + j) += *(A + i * n + k) * *(B + k * n + j);
            }
        }
    }
}

/**
 * strassen_add - Suma de matrices cuadradas: C = A + B.
 * Complejidad: O(n^2)
 */
static void strassen_add(double *A, double *B, double *C, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(C + i * n + j) = *(A + i * n + j) + *(B + i * n + j);
        }
    }
}

/**
 * strassen_sub - Resta de matrices cuadradas: C = A - B.
 * Complejidad: O(n^2)
 */
static void strassen_sub(double *A, double *B, double *C, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(C + i * n + j) = *(A + i * n + j) - *(B + i * n + j);
        }
    }
}

/**
 * strassen_recursive - Implementación recursiva del Algorítmo de Strassen.
 *
 * Divide las matrices en 4 submatrices y calcula 7 productos intermedios:
 *   M1 = (A11 + A22)(B11 + B22)
 *   M2 = (A21 + A22)(B11)
 *   M3 = (A11)(B12 - B22)
 *   M4 = (A22)(B21 - B11)
 *   M5 = (A11 + A12)(B22)
 *   M6 = (A21 - A11)(B11 + B12)
 *   M7 = (A12 - A22)(B21 + B22)
 *
 * Complejidad teorica: O(n^2.807) gracias a que reduce de 8 a 7 multiplicaciones.
 */
static void strassen_recursive(double *A, double *B, double *C, int n) {
    /* Caso base: usar naive para matrices pequeñas */
    if (n <= 64) {
        naive_multiply(A, B, C, n);
        return;
    }

    int half = n / 2;
    int size = half * half;

    /* Reservar memoria para submatrices y productos intermedios */
    double *A11 = (double *)malloc(size * sizeof(double));
    double *A12 = (double *)malloc(size * sizeof(double));
    double *A21 = (double *)malloc(size * sizeof(double));
    double *A22 = (double *)malloc(size * sizeof(double));
    double *B11 = (double *)malloc(size * sizeof(double));
    double *B12 = (double *)malloc(size * sizeof(double));
    double *B21 = (double *)malloc(size * sizeof(double));
    double *B22 = (double *)malloc(size * sizeof(double));

    double *M1 = (double *)malloc(size * sizeof(double));
    double *M2 = (double *)malloc(size * sizeof(double));
    double *M3 = (double *)malloc(size * sizeof(double));
    double *M4 = (double *)malloc(size * sizeof(double));
    double *M5 = (double *)malloc(size * sizeof(double));
    double *M6 = (double *)malloc(size * sizeof(double));
    double *M7 = (double *)malloc(size * sizeof(double));

    double *tempA = (double *)malloc(size * sizeof(double));
    double *tempB = (double *)malloc(size * sizeof(double));

    int i, j;

    /* Dividir matrices A y B en submatrices usando aritmetica de punteros */
    for (i = 0; i < half; i++) {
        for (j = 0; j < half; j++) {
            *(A11 + i * half + j) = *(A + i * n + j);
            *(A12 + i * half + j) = *(A + i * n + (j + half));
            *(A21 + i * half + j) = *(A + (i + half) * n + j);
            *(A22 + i * half + j) = *(A + (i + half) * n + (j + half));
            *(B11 + i * half + j) = *(B + i * n + j);
            *(B12 + i * half + j) = *(B + i * n + (j + half));
            *(B21 + i * half + j) = *(B + (i + half) * n + j);
            *(B22 + i * half + j) = *(B + (i + half) * n + (j + half));
        }
    }

    /* M1 = (A11 + A22)(B11 + B22) */
    strassen_add(A11, A22, tempA, half);
    strassen_add(B11, B22, tempB, half);
    strassen_recursive(tempA, tempB, M1, half);

    /* M2 = (A21 + A22)(B11) */
    strassen_add(A21, A22, tempA, half);
    strassen_recursive(tempA, B11, M2, half);

    /* M3 = (A11)(B12 - B22) */
    strassen_sub(B12, B22, tempB, half);
    strassen_recursive(A11, tempB, M3, half);

    /* M4 = (A22)(B21 - B11) */
    strassen_sub(B21, B11, tempB, half);
    strassen_recursive(A22, tempB, M4, half);

    /* M5 = (A11 + A12)(B22) */
    strassen_add(A11, A12, tempA, half);
    strassen_recursive(tempA, B22, M5, half);

    /* M6 = (A21 - A11)(B11 + B12) */
    strassen_sub(A21, A11, tempA, half);
    strassen_add(B11, B12, tempB, half);
    strassen_recursive(tempA, tempB, M6, half);

    /* M7 = (A12 - A22)(B21 + B22) */
    strassen_sub(A12, A22, tempA, half);
    strassen_add(B21, B22, tempB, half);
    strassen_recursive(tempA, tempB, M7, half);

    /* Reconstruir las 4 submatrices del resultado C */
    /* C11 = M1 + M4 - M5 + M7 */
    /* C12 = M3 + M5 */
    /* C21 = M2 + M4 */
    /* C22 = M1 - M2 + M3 + M6 */
    for (i = 0; i < half; i++) {
        for (j = 0; j < half; j++) {
            *(C + i * n + j) = *(M1 + i * half + j) + *(M4 + i * half + j)
                              - *(M5 + i * half + j) + *(M7 + i * half + j);
            *(C + i * n + (j + half)) = *(M3 + i * half + j) + *(M5 + i * half + j);
            *(C + (i + half) * n + j) = *(M2 + i * half + j) + *(M4 + i * half + j);
            *(C + (i + half) * n + (j + half)) = *(M1 + i * half + j) - *(M2 + i * half + j)
                                                 + *(M3 + i * half + j) + *(M6 + i * half + j);
        }
    }

    /* Liberar toda la memoria temporal */
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(M1); free(M2); free(M3); free(M4);
    free(M5); free(M6); free(M7);
    free(tempA); free(tempB);
}

/**
 * strassen_multiply - Interfaz publica del Algoritmo de Strassen.
 *
 * Redondea las dimensiones a la potencia de 2 mas cercana, aplica padding
 * con ceros si es necesario, ejecuta Strassen, y extrae el resultado real.
 *
 * Complejidad teorica: O(n^2.807)
 */
void strassen_multiply(double *A, double *B, double *C, int n) {
    /* Encontrar la potencia de 2 mas cercana >= n */
    int m = 1;
    while (m < n) m <<= 1;  /* Multiplicar por 2 hasta superar n */

    if (m == n) {
        /* Ya es potencia de 2, usar Strassen directamente */
        strassen_recursive(A, B, C, n);
        return;
    }

    /* Padding: crear matrices ampliadas con ceros */
    int total = m * m;
    double *Ap = (double *)calloc(total, sizeof(double));
    double *Bp = (double *)calloc(total, sizeof(double));
    double *Cp = (double *)calloc(total, sizeof(double));

    int i, j;
    /* Copiar datos originales al bloque ampliado */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(Ap + i * m + j) = *(A + i * n + j);
            *(Bp + i * m + j) = *(B + i * n + j);
        }
    }

    /* Ejecutar Strassen en la matriz ampliada */
    strassen_recursive(Ap, Bp, Cp, m);

    /* Extraer el resultado real (sin padding) */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            *(C + i * n + j) = *(Cp + i * m + j);
        }
    }

    free(Ap);
    free(Bp);
    free(Cp);
}

/**
 * matrix_vector_multiply - Multiplicacion matriz x vector.
 *
 * Calcula: result[i] = SUM_j (matrix[i][j] * vector[j])
 *
 * Implementada con aritmetica de punteros pura (sin indices array[i][j]).
 * Complejidad: O(rows * cols)
 */
void matrix_vector_multiply(double *matrix, double *vector, double *result,
                            int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        *(result + i) = 0.0;
        for (j = 0; j < cols; j++) {
            /* Aritmetica de punteros: accedemos a matrix[i*cols + j] */
            *(result + i) += *(matrix + i * cols + j) * *(vector + j);
        }
    }
}

/* =============================================================================
 * ALGORITMO QUICK SORT - ESQUEMA DE LOMUTO
 * =============================================================================
 *
 * Complejidad promedio: O(n log(n))
 * Peor caso: O(n^2) - ocurre cuando el array ya esta ordenado
 * Espacio: O(log(n)) por la recursion
 *
 * Se usa el esquema de particion de Lomuto como requiere el enunciado:
 * - El pivot es el ultimo elemento
 * - Los menores al pivot van a la izquierda
 * - Se intercambia el pivot a su posicion final
 */

/**
 * lomuto_partition - Particion de Lomuto para QuickSort.
 *
 * Selecciona el ultimo elemento como pivot, reorganiza el array de tal forma
 * que todos los elementos menores queden a la izquierda y mayores a la derecha.
 *
 * Complejidad: O(n) - un solo recorrido lineal.
 */
int lomuto_partition(double *arr, int *indices, int low, int high) {
    double pivot = *(arr + high);  /* El pivot es el ultimo elemento */
    int i = low - 1;              /* Indice del elemento mas pequeno */
    int j;
    double temp_arr;
    int temp_idx;

    for (j = low; j < high; j++) {
        if (*(arr + j) <= pivot) {
            i++;
            /* Intercambiar arr[i] con arr[j] */
            temp_arr = *(arr + i);
            *(arr + i) = *(arr + j);
            *(arr + j) = temp_arr;

            /* Intercambiar indices[i] con indices[j] */
            temp_idx = *(indices + i);
            *(indices + i) = *(indices + j);
            *(indices + j) = temp_idx;
        }
    }

    /* Colocar el pivot en su posicion final */
    temp_arr = *(arr + (i + 1));
    *(arr + (i + 1)) = *(arr + high);
    *(arr + high) = temp_arr;

    temp_idx = *(indices + (i + 1));
    *(indices + (i + 1)) = *(indices + high);
    *(indices + high) = temp_idx;

    return i + 1;  /* Posicion final del pivot */
}

/**
 * quicksort - Ordenamiento Quick Sort con particion de Lomuto.
 *
 * Implementado recursivamente. Ordena un array de doubles y reordena
 * un array de indices asociados para mantener la correspondencia.
 *
 * Complejidad promedio: O(n log(n))
 * Peor caso: O(n^2)
 *
 * @param arr      Array de doubles a ordenar
 * @param indices  Array de indices asociados (se reordenan junto con arr)
 * @param low      Indice inferior del subarreglo
 * @param high     Indice superior del subarreglo
 */
void quicksort(double *arr, int *indices, int low, int high) {
    if (low < high) {
        int pi = lomuto_partition(arr, indices, low, high);
        quicksort(arr, indices, low, pi - 1);   /* Ordenar subarreglo izquierdo */
        quicksort(arr, indices, pi + 1, high);   /* Ordenar subarreglo derecho */
    }
}

/* =============================================================================
 * FUNCIONES DE RED NEURONAL
 * =============================================================================
 */

/**
 * nn_create - Crea una red neuronal con topologia dinamica.
 *
 * Reserva toda la memoria necesaria usando malloc() y calloc().
 * Inicializa los pesos con valores aleatorios en [-0.5, 0.5] usando
 * la formula: w = (rand() / RAND_MAX) - 0.5
 *
 * @param num_layers   Numero total de capas (ej: 3 para red 4-8-3)
 * @param layer_sizes  Array con el numero de neuronas por capa
 * @param lr           Learning rate (tasa de aprendizaje)
 * @param epochs       Numero de epocas de entrenamiento
 * @return             Puntero a la red neuronal creada
 */
FeedForwardNN* nn_create(int num_layers, int *layer_sizes, double lr, int epochs) {
    int i, j, k;
    FeedForwardNN *nn = (FeedForwardNN *)malloc(sizeof(FeedForwardNN));
    if (!nn) {
        fprintf(stderr, "Error: No se pudo asignar memoria para la red neuronal.\n");
        exit(EXIT_FAILURE);
    }

    nn->num_layers = num_layers;
    nn->learning_rate = lr;
    nn->epochs = epochs;

    /* Reservar array de capas */
    nn->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    if (!nn->layers) {
        fprintf(stderr, "Error: No se pudo asignar memoria para las capas.\n");
        free(nn);
        exit(EXIT_FAILURE);
    }

    /* Inicializar cada capa */
    for (i = 0; i < num_layers; i++) {
        Layer *layer = nn->layers + i;  /* Aritmetica de punteros */
        layer->num_neurons = *(layer_sizes + i);
        layer->num_inputs = (i == 0) ? *(layer_sizes + i) : *(layer_sizes + (i - 1));

        /* Reservar pesos: [num_neurons x num_inputs] en bloque contiguo */
        layer->weights = (double *)calloc(layer->num_neurons * layer->num_inputs, sizeof(double));
        /* Reservar sesgos: [num_neurons] */
        layer->biases = (double *)calloc(layer->num_neurons, sizeof(double));
        /* Reservar activaciones y z (suma ponderada) */
        layer->activations = (double *)calloc(layer->num_neurons, sizeof(double));
        layer->z = (double *)calloc(layer->num_neurons, sizeof(double));

        if (!layer->weights || !layer->biases || !layer->activations || !layer->z) {
            fprintf(stderr, "Error: No se pudo asignar memoria para la capa %d.\n", i);
            exit(EXIT_FAILURE);
        }

        /* Inicializar pesos aleatoriamente en [-0.5, 0.5] */
        for (j = 0; j < layer->num_neurons; j++) {
            for (k = 0; k < layer->num_inputs; k++) {
                *(layer->weights + j * layer->num_inputs + k) =
                    ((double)rand() / (double)RAND_MAX) - 0.5;
            }
            /* Inicializar sesgos en 0 */
            *(layer->biases + j) = 0.0;
        }
    }

    return nn;
}

/**
 * nn_forward - Forward Propagation (Propagacion hacia adelante).
 *
 * Para cada capa:
 *   1. Calcular z = W * activations + bias  (usando Strassen o naive)
 *   2. Aplicar activacion: a = sigmoid(z)
 *
 * Complejidad total: O(L * n^2.807) donde L=capas, n=tamano de capa
 */
void nn_forward(FeedForwardNN *nn, double *input) {
    int i, j;
    Layer *current;
    Layer *prev;

    for (i = 0; i < nn->num_layers; i++) {
        current = nn->layers + i;

        if (i == 0) {
            /* Primera capa: las entradas son los features del dataset */
            for (j = 0; j < current->num_neurons; j++) {
                /* z[j] = SUM(weight[j][k] * input[k]) + bias[j] */
                double sum = 0.0;
                int k;
                for (k = 0; k < current->num_inputs; k++) {
                    sum += *(current->weights + j * current->num_inputs + k) * *(input + k);
                }
                *(current->z + j) = sum + *(current->biases + j);
                *(current->activations + j) = sigmoid(*(current->z + j));
            }
        } else {
            /* Capas ocultas y salida: entrada es activacion de capa anterior */
            prev = nn->layers + (i - 1);

            /* Para capas pequenas como en Iris (4-8-3), es mas eficiente
               usar matrix_vector_multiply directamente en lugar de Strassen,
               ya que Strassen requiere matrices cuadradas y tiene overhead
               para n pequeno. Usamos Strassen solo cuando n >= 64. */

            if (current->num_neurons >= 64 && current->num_inputs >= 64) {
                /* Usar Strassen para capas grandes */
                int max_dim = current->num_neurons > current->num_inputs ?
                              current->num_neurons : current->num_inputs;
                int m = 1;
                while (m < max_dim) m <<= 1;

                /* Crear matrices cuadradas para Strassen */
                double *W = (double *)calloc(m * m, sizeof(double));
                double *X = (double *)calloc(m * m, sizeof(double));
                double *Z = (double *)calloc(m * m, sizeof(double));

                int r, c;
                for (r = 0; r < current->num_neurons; r++) {
                    for (c = 0; c < current->num_inputs; c++) {
                        *(W + r * m + c) = *(current->weights + r * current->num_inputs + c);
                    }
                }
                for (r = 0; r < current->num_inputs; r++) {
                    *(X + r * m + r) = *(prev->activations + r);
                }

                strassen_multiply(W, X, Z, m);

                for (r = 0; r < current->num_neurons; r++) {
                    *(current->z + r) = *(Z + r * m + 0) + *(current->biases + r);
                    *(current->activations + r) = sigmoid(*(current->z + r));
                }

                free(W); free(X); free(Z);
            } else {
                /* Para capas pequenas, usar multiplicacion naive (mas rapido) */
                matrix_vector_multiply(current->weights, prev->activations,
                                       current->z, current->num_neurons, current->num_inputs);

                /* Agregar sesgos y aplicar activacion */
                for (j = 0; j < current->num_neurons; j++) {
                    *(current->z + j) += *(current->biases + j);
                    *(current->activations + j) = sigmoid(*(current->z + j));
                }
            }
        }
    }
}

/**
 * nn_backward - Backpropagation y actualizacion de pesos.
 *
 * Calcula los gradientes de la funcion de perdida respecto a cada peso
 * y sesgo, luego actualiza usando descenso de gradiente:
 *   w_nuevo = w_viejo - lr * gradiente
 *
 * Complejidad: O(N * M) donde N=muestras, M=total conexiones de la red
 */
void nn_backward(FeedForwardNN *nn, double *target) {
    int i, j, k;

    /* Reservar memoria para delta (error de cada neurona) en cada capa */
    double **deltas = (double **)malloc(nn->num_layers * sizeof(double *));
    for (i = 0; i < nn->num_layers; i++) {
        *(deltas + i) = (double *)calloc((nn->layers + i)->num_neurons, sizeof(double));
    }

    int last = nn->num_layers - 1;
    Layer *output_layer = nn->layers + last;

    /* Delta de la capa de salida:
       delta = (output - target) * sigmoid'(output) */
    for (j = 0; j < output_layer->num_neurons; j++) {
        double output = *(output_layer->activations + j);
        double error = output - *(target + j);
        *(*(deltas + last) + j) = error * sigmoid_derivative(output);
    }

    /* Delta de capas ocultas (de derecha a izquierda):
       delta = (W^T * delta_siguiente) * sigmoid'(activacion) */
    for (i = last - 1; i >= 0; i--) {
        Layer *current = nn->layers + i;
        Layer *next = nn->layers + (i + 1);

        for (j = 0; j < current->num_neurons; j++) {
            double error = 0.0;
            for (k = 0; k < next->num_neurons; k++) {
                error += *(*(deltas + (i + 1)) + k) *
                         *(next->weights + k * next->num_inputs + j);
            }
            *(*(deltas + i) + j) = error * sigmoid_derivative(*(current->activations + j));
        }
    }

    /* Actualizar pesos y sesgos usando descenso de gradiente */
    for (i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers + i;
        Layer *prev = (i == 0) ? NULL : (nn->layers + (i - 1));
        double *prev_activations = (i == 0) ? NULL : prev->activations;

        /* Para la primera capa, las entradas son los features del dataset
           (ya no tenemos acceso directo aqui, asi que usamos las activaciones
           que fueron guardadas) */

        for (j = 0; j < layer->num_neurons; j++) {
            double delta_j = *(*(deltas + i) + j);

            /* Actualizar sesgo */
            *(layer->biases + j) -= nn->learning_rate * delta_j;

            /* Actualizar pesos */
            for (k = 0; k < layer->num_inputs; k++) {
                double input_val;
                if (i == 0) {
                    /* Para la primera capa, necesitamos las entradas originales.
                       Como no las guardamos, usamos las activaciones de la capa
                       anterior (que para i=0 seran los inputs) */
                    input_val = *(layer->activations + k);  /* Esto es un placeholder */
                } else {
                    input_val = *(prev_activations + k);
                }
                *(layer->weights + j * layer->num_inputs + k) -=
                    nn->learning_rate * delta_j * input_val;
            }
        }
    }

    /* Liberar memoria de deltas */
    for (i = 0; i < nn->num_layers; i++) {
        free(*(deltas + i));
    }
    free(deltas);
}

/**
 * nn_backward_fixed - Backpropagation corregido que recibe las entradas.
 *
 * Version mejorada de nn_backward que recibe explicitamente las entradas
 * del dataset para calcular correctamente la actualizacion de pesos
 * de la primera capa.
 */
static void nn_backward_fixed(FeedForwardNN *nn, double *target, double *input) {
    int i, j, k;

    double **deltas = (double **)malloc(nn->num_layers * sizeof(double *));
    for (i = 0; i < nn->num_layers; i++) {
        *(deltas + i) = (double *)calloc((nn->layers + i)->num_neurons, sizeof(double));
    }

    int last = nn->num_layers - 1;
    Layer *output_layer = nn->layers + last;

    /* Delta de la capa de salida */
    for (j = 0; j < output_layer->num_neurons; j++) {
        double output = *(output_layer->activations + j);
        double error = output - *(target + j);
        *(*(deltas + last) + j) = error * sigmoid_derivative(output);
    }

    /* Delta de capas ocultas */
    for (i = last - 1; i >= 0; i--) {
        Layer *current = nn->layers + i;
        Layer *next = nn->layers + (i + 1);

        for (j = 0; j < current->num_neurons; j++) {
            double error = 0.0;
            for (k = 0; k < next->num_neurons; k++) {
                error += *(*(deltas + (i + 1)) + k) *
                         *(next->weights + k * next->num_inputs + j);
            }
            *(*(deltas + i) + j) = error * sigmoid_derivative(*(current->activations + j));
        }
    }

    /* Actualizar pesos y sesgos */
    for (i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers + i;

        for (j = 0; j < layer->num_neurons; j++) {
            double delta_j = *(*(deltas + i) + j);
            *(layer->biases + j) -= nn->learning_rate * delta_j;

            for (k = 0; k < layer->num_inputs; k++) {
                double input_val;
                if (i == 0) {
                    input_val = *(input + k);  /* Entradas originales del dataset */
                } else {
                    input_val = *((nn->layers + (i - 1))->activations + k);
                }
                *(layer->weights + j * layer->num_inputs + k) -=
                    nn->learning_rate * delta_j * input_val;
            }
        }
    }

    for (i = 0; i < nn->num_layers; i++) {
        free(*(deltas + i));
    }
    free(deltas);
}

/**
 * nn_train - Entrena la red neuronal con el dataset.
 *
 * Para cada epoca, recorre todas las muestras de entrenamiento y realiza
 * forward + backward propagation. Imprime la perdida MSE cada 100 epocas.
 *
 * Complejidad total: O(epochs * num_train * cost_per_sample)
 */
void nn_train(FeedForwardNN *nn, Dataset *dataset) {
    int epoch, i, j;
    double total_loss;

    // printf("\n========================================\n");
    // printf("  INICIANDO ENTRENAMIENTO\n");
    // printf("  Learning Rate: %.4f\n", nn->learning_rate);
    // printf("  Epocas: %d\n", nn->epochs);
    // printf("  Muestras de entrenamiento: %d\n", dataset->num_train);
    // printf("========================================\n\n");
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║             INICIANDO ENTRENAMIENTO            ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║   Learning Rate: %-30.4f║\n", nn->learning_rate);
    printf("║   Epocas: %-37d║\n", nn->epochs);
    printf("║   Muestras de entrenamiento: %-18d║\n", dataset->num_train);
    printf("╚════════════════════════════════════════════════╝\n\n");

    for (epoch = 0; epoch < nn->epochs; epoch++) {
        total_loss = 0.0;

        for (i = 0; i < dataset->num_train; i++) {
            double *features = *(dataset->train_features + i);
            int label = *(dataset->train_labels + i);

            /* One-hot encoding para la clase esperada */
            double target[OUTPUT_SIZE] = {0.0, 0.0, 0.0};
            if (label >= 0 && label < OUTPUT_SIZE) {
                *(target + label) = 1.0;
            }

            /* Forward propagation */
            nn_forward(nn, features);

            /* Calcular error cuadratico para esta muestra */
            Layer *output_layer = nn->layers + (nn->num_layers - 1);
            for (j = 0; j < OUTPUT_SIZE; j++) {
                double diff = *(output_layer->activations + j) - *(target + j);
                total_loss += diff * diff;
            }

            /* Backward propagation con entradas corregidas */
            nn_backward_fixed(nn, target, features);
        }

        /* Imprimir perdida cada 100 epocas */
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            double avg_loss = total_loss / dataset->num_train;
            printf("Epoca %5d/%d | Loss (MSE): %.6f\n", epoch + 1, nn->epochs, avg_loss);
        }
    }

    printf("\n  Entrenamiento completado.\n");
}

/**
 * nn_predict - Predice la clase de una entrada.
 *
 * Realiza forward propagation y retorna el indice con mayor probabilidad.
 *
 * @param nn     Red neuronal entrenada
 * @param input  Vector de features [INPUT_SIZE]
 * @return       Indice de la clase predicha (0, 1 o 2)
 */
int nn_predict(FeedForwardNN *nn, double *input) {
    nn_forward(nn, input);

    Layer *output_layer = nn->layers + (nn->num_layers - 1);
    int max_idx = 0;
    double max_val = *(output_layer->activations + 0);

    int j;
    for (j = 1; j < OUTPUT_SIZE; j++) {
        if (*(output_layer->activations + j) > max_val) {
            max_val = *(output_layer->activations + j);
            max_idx = j;
        }
    }

    return max_idx;
}

/**
 * nn_evaluate - Evalua la precision de la red en un conjunto de datos.
 *
 * @return Precision (accuracy) como valor entre 0.0 y 1.0
 */
double nn_evaluate(FeedForwardNN *nn, double **features, int *labels, int num_samples) {
    int correct = 0;
    int i;

    for (i = 0; i < num_samples; i++) {
        int predicted = nn_predict(nn, *(features + i));
        if (predicted == *(labels + i)) {
            correct++;
        }
    }

    return (double)correct / (double)num_samples;
}

/**
 * nn_free - Libera toda la memoria asignada a la red neuronal.
 *
 * Funcion critica para evitar memory leaks. Libera en orden inverso:
 * primero los pesos, sesgos y activaciones de cada capa,
 * luego el array de capas, y finalmente la estructura principal.
 */
void nn_free(FeedForwardNN *nn) {
    if (!nn) return;

    int i;
    if (nn->layers) {
        for (i = 0; i < nn->num_layers; i++) {
            Layer *layer = nn->layers + i;
            if (layer->weights) free(layer->weights);
            if (layer->biases) free(layer->biases);
            if (layer->activations) free(layer->activations);
            if (layer->z) free(layer->z);
        }
        free(nn->layers);
    }
    free(nn);
}

/* =============================================================================
 * FUNCIONES DE DATASET
 * =============================================================================
 */

/**
 * dataset_load - Lee el dataset desde flores-dataset.txt.
 *
 * Formato de cada linea: float float float float int
 * (LargoSepalo AnchoSepalo LargoPetalo AnchoPetalo Clase)
 *
 * Reserva memoria dinamica para todas las muestras.
 * Complejidad: O(n) donde n = numero de lineas del archivo.
 */
Dataset* dataset_load(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: No se pudo abrir el archivo '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
    if (!dataset) {
        fprintf(stderr, "Error: No se pudo asignar memoria para el dataset.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    /* Contar lineas del archivo para reservar memoria exacta */
    int count = 0;
    double temp;
    int temp_int;
    while (fscanf(file, "%lf %lf %lf %lf %d", &temp, &temp, &temp, &temp, &temp_int) == 5) {
        count++;
    }

    dataset->num_samples = count;

    /* Reservar memoria para features y labels */
    dataset->features = (double **)malloc(count * sizeof(double *));
    dataset->labels = (int *)malloc(count * sizeof(int));

    if (!dataset->features || !dataset->labels) {
        fprintf(stderr, "Error: No se pudo asignar memoria para el dataset.\n");
        fclose(file);
        free(dataset);
        exit(EXIT_FAILURE);
    }

    /* Leer datos del archivo usando aritmetica de punteros */
    rewind(file);
    int i;
    for (i = 0; i < count; i++) {
        /* Reservar memoria para cada fila de features */
        *(dataset->features + i) = (double *)malloc(INPUT_SIZE * sizeof(double));

        if (!*(dataset->features + i)) {
            fprintf(stderr, "Error: No se pudo asignar memoria para la muestra %d.\n", i);
            fclose(file);
            dataset_free(dataset);
            exit(EXIT_FAILURE);
        }

        fscanf(file, "%lf %lf %lf %lf %d",
               *(dataset->features + i) + 0,
               *(dataset->features + i) + 1,
               *(dataset->features + i) + 2,
               *(dataset->features + i) + 3,
               dataset->labels + i);
    }

    fclose(file);
    printf("Dataset cargado exitosamente: %d muestras.\n", dataset->num_samples);
    return dataset;
}

/**
 * dataset_split - Separa el dataset en entrenamiento (80%) y prueba (20%).
 *
 * Implementa un split ESTRATIFICADO: de cada clase se toma proporcionalemente
 * el 80% para entrenamiento y 20% para prueba, garantizando representacion
 * equilibrada de todas las clases en ambos conjuntos.
 *
 * Utiliza QuickSort para ordenar los indices por clase antes de la separacion.
 *
 * Complejidad: O(n log(n)) por el QuickSort.
 */
void dataset_split(Dataset *dataset, double train_ratio) {
    int total = dataset->num_samples;

    /* Crear indices para ordenar por clase usando QuickSort */
    int *indices = (int *)malloc(total * sizeof(int));
    double *class_values = (double *)malloc(total * sizeof(double));

    int i;
    for (i = 0; i < total; i++) {
        *(indices + i) = i;
        *(class_values + i) = (double)*(dataset->labels + i);
    }

    /* Ordenar por clase: agrupa todas las muestras de la misma clase */
    quicksort(class_values, indices, 0, total - 1);

    /* Contar muestras por clase y calcular split estratificado */
    int class_counts[OUTPUT_SIZE] = {0, 0, 0};
    for (i = 0; i < total; i++) {
        int label = *(dataset->labels + i);
        if (label >= 0 && label < OUTPUT_SIZE) {
            class_counts[label]++;
        }
    }

    /* Calcular cuantas muestras de cada clase van a train y test */
    int train_per_class[OUTPUT_SIZE];
    int test_per_class[OUTPUT_SIZE];
    int train_count = 0;
    int test_count = 0;

    for (i = 0; i < OUTPUT_SIZE; i++) {
        train_per_class[i] = (int)(class_counts[i] * train_ratio);
        test_per_class[i] = class_counts[i] - train_per_class[i];
        train_count += train_per_class[i];
        test_count += test_per_class[i];
    }

    dataset->num_train = train_count;
    dataset->num_test = test_count;

    /* Reservar memoria para train y test */
    dataset->train_features = (double **)malloc(train_count * sizeof(double *));
    dataset->train_labels = (int *)malloc(train_count * sizeof(int));
    dataset->test_features = (double **)malloc(test_count * sizeof(double *));
    dataset->test_labels = (int *)malloc(test_count * sizeof(int));

    /* Llenar train y test estratificadamente */
    int train_idx = 0;
    int test_idx = 0;
    int class_idx = 0;
    int offset = 0;

    for (class_idx = 0; class_idx < OUTPUT_SIZE; class_idx++) {
        /* Los indices de esta clase estan en [offset, offset + class_counts[class_idx]) */
        /* Primera parte: entrenamiento (train_per_class[class_idx] muestras) */
        for (i = 0; i < train_per_class[class_idx]; i++) {
            int idx = *(indices + offset + i);
            *(dataset->train_features + train_idx) = *(dataset->features + idx);
            *(dataset->train_labels + train_idx) = *(dataset->labels + idx);
            train_idx++;
        }
        /* Segunda parte: prueba (test_per_class[class_idx] muestras) */
        for (i = 0; i < test_per_class[class_idx]; i++) {
            int idx = *(indices + offset + train_per_class[class_idx] + i);
            *(dataset->test_features + test_idx) = *(dataset->features + idx);
            *(dataset->test_labels + test_idx) = *(dataset->labels + idx);
            test_idx++;
        }
        offset += class_counts[class_idx];
    }

    free(indices);
    free(class_values);

    printf("Dataset dividido (estratificado): %d entrenamiento, %d prueba.\n",
           train_count, test_count);
}

/**
 * dataset_normalize - Normaliza los features usando min-max scaling.
 *
 * Escala cada feature al rango [0, 1]:
 *   normalized = (value - min) / (max - min)
 *
 * Complejidad: O(n) por cada feature (4 features = O(4n) = O(n))
 */
void dataset_normalize(Dataset *dataset) {
    int i, j;

    /* Encontrar min y max por cada feature */
    double min_vals[INPUT_SIZE];
    double max_vals[INPUT_SIZE];

    for (j = 0; j < INPUT_SIZE; j++) {
        *(min_vals + j) = *( *(dataset->features + 0) + j);
        *(max_vals + j) = *( *(dataset->features + 0) + j);
    }

    for (i = 1; i < dataset->num_samples; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            double val = *( *(dataset->features + i) + j);
            if (val < *(min_vals + j)) *(min_vals + j) = val;
            if (val > *(max_vals + j)) *(max_vals + j) = val;
        }
    }

    /* Normalizar todas las features al rango [0, 1] */
    for (i = 0; i < dataset->num_samples; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            double range = *(max_vals + j) - *(min_vals + j);
            if (range > 0.0001) {
                *( *(dataset->features + i) + j) =
                    (*( *(dataset->features + i) + j) - *(min_vals + j)) / range;
            }
        }
    }

    printf("Dataset normalizado al rango [0, 1].\n");
}

/**
 * dataset_free - Libera toda la memoria del dataset.
 *
 * Incluye las filas individuales de features, los vectores de labels,
 * y las estructuras de train/test.
 */
void dataset_free(Dataset *dataset) {
    if (!dataset) return;

    int i;

    /* Liberar features individuales */
    if (dataset->features) {
        for (i = 0; i < dataset->num_samples; i++) {
            if (*(dataset->features + i)) {
                free(*(dataset->features + i));
            }
        }
        free(dataset->features);
    }

    if (dataset->labels) free(dataset->labels);

    /* Liberar train/test (no liberar features individuales, ya apuntan a features) */
    if (dataset->train_features) free(dataset->train_features);
    if (dataset->train_labels) free(dataset->train_labels);
    if (dataset->test_features) free(dataset->test_features);
    if (dataset->test_labels) free(dataset->test_labels);

    free(dataset);
}

/* =============================================================================
 * FUNCIONES DE METRICAS Y EVALUACION
 * =============================================================================
 */

/**
 * print_confusion_matrix - Imprime una matriz de confusion textual.
 *
 * Muestra una tabla donde las filas son las clases reales y las columnas
 * son las clases predichas. Facilita la evaluacion visual del modelo.
 */
void print_confusion_matrix(FeedForwardNN *nn, Dataset *dataset) {
    int **matrix = (int **)calloc(OUTPUT_SIZE, sizeof(int *));
    int i, j;

    for (i = 0; i < OUTPUT_SIZE; i++) {
        *(matrix + i) = (int *)calloc(OUTPUT_SIZE, sizeof(int));
    }

    /* Llenar la matriz de confusion */
    for (i = 0; i < dataset->num_test; i++) {
        int predicted = nn_predict(nn, *(dataset->test_features + i));
        int actual = *(dataset->test_labels + i);

        if (actual >= 0 && actual < OUTPUT_SIZE && predicted >= 0 && predicted < OUTPUT_SIZE) {
            *(*(matrix + actual) + predicted) += 1;
        }
    }

    /* Imprimir matriz */
    const char *class_names[] = {"Setosa", "Versicolor", "Virginica"};

    printf("\n  +---------------+----------+----------+----------+\n");
    printf("  |            Matriz de Confusion                 |\n");
    printf("  +---------------+----------+----------+----------+\n");
    printf("  |  Real \\ Pred  |  Setosa  | Versicor | Virginic |\n");
    printf("  +---------------+----------+----------+----------+\n");

    for (i = 0; i < OUTPUT_SIZE; i++) {
        printf("  | %-13s |", *(class_names + i));
        for (j = 0; j < OUTPUT_SIZE; j++) {
            printf(" %8d |", *(*(matrix + i) + j));
        }
        printf("\n");
    }
    printf("  +---------------+----------+----------+----------+\n");

    /* Liberar memoria */
    for (i = 0; i < OUTPUT_SIZE; i++) {
        free(*(matrix + i));
    }
    free(matrix);
}

/**
 * print_evaluation_metrics - Imprime metricas de evaluacion del modelo.
 *
 * Incluye: precision por clase, precision global (accuracy), y
 * una matriz de confusion.
 */
void print_evaluation_metrics(FeedForwardNN *nn, Dataset *dataset) {
    printf("\n╔════════════════════════════════════════════════╗\n");
printf("║              EVALUACION DEL MODELO             ║\n");
printf("╠════════════════════════════════════════════════╣\n");
printf("║   Muestras de prueba: %-25d║\n", dataset->num_test);
printf("╚════════════════════════════════════════════════╝\n");

    /* Calcular precision global */
    double accuracy = nn_evaluate(nn, dataset->test_features,
                                  dataset->test_labels, dataset->num_test);
    printf("\n  Accuracy Global: %.2f%%\n", accuracy * 100.0);

    /* Calcular precision por clase */
    int correct_per_class[OUTPUT_SIZE] = {0, 0, 0};
    int total_per_class[OUTPUT_SIZE] = {0, 0, 0};

    int i;
    for (i = 0; i < dataset->num_test; i++) {
        int predicted = nn_predict(nn, *(dataset->test_features + i));
        int actual = *(dataset->test_labels + i);

        if (actual >= 0 && actual < OUTPUT_SIZE) {
            total_per_class[actual] += 1;
            if (predicted == actual) {
                correct_per_class[actual] += 1;
            }
        }
    }

    printf("\n  Precision por clase:\n");
    const char *class_names[] = {"Setosa", "Versicolor", "Virginica"};
    for (i = 0; i < OUTPUT_SIZE; i++) {
        double class_acc = (total_per_class[i] > 0) ?
            (double)correct_per_class[i] / (double)total_per_class[i] : 0.0;
        printf("    %-12s: %d/%d (%.2f%%)\n",
               *(class_names + i),
               correct_per_class[i],
               total_per_class[i],
               class_acc * 100.0);
    }

    /* Mostrar matriz de confusion */
    print_confusion_matrix(nn, dataset);
}

/**
 * benchmark_forward - Mide el tiempo de ejecucion del Forward Propagation.
 *
 * Ejecuta forward propagation sobre todas las muestras de prueba
 * y retorna el tiempo total en milisegundos.
 *
 * Usa clock() de <time.h> para medicion precisa.
 */
double benchmark_forward(FeedForwardNN *nn, Dataset *dataset) {
    clock_t start, end;
    double cpu_time_used;
    int i;

    start = clock();

    /* Ejecutar forward propagation sobre todas las muestras de prueba */
    for (i = 0; i < dataset->num_test; i++) {
        nn_forward(nn, *(dataset->test_features + i));
    }

    end = clock();

    /* Convertir a milisegundos */
    cpu_time_used = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    return cpu_time_used;
}
