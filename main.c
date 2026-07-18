/**
 * =============================================================================
 * ARCHIVO: main.c
 * DESCRIPCION: Punto de entrada del Motor de Inferencia de Red Neuronal.
 *              Implementa la interfaz interactiva, carga del dataset,
 *              configuracion de hiperparametros, entrenamiento, benchmarking
 *              y liberacion exhaustiva de memoria.
 *
 * FLUJO DEL PROGRAMA:
 *   1. Cargar dataset desde flores-dataset.txt
 *   2. Normalizar y dividir (80% train / 20% test)
 *   3. Solicitar hiperparametros (Learning Rate y Epocas)
 *   4. Crear red neuronal con topologia 4-8-3
 *   5. Entrenar la red
 *   6. Evaluar precision y mostrar matriz de confusion
 *   7. Medir tiempo de inferencia (benchmarking)
 *   8. Liberar toda la memoria
 *
 * PROYECTO: Clasificador del Dataset Iris
 * AUTORES: Luis Eulacio, Eros Flugel
 * FECHA: Julio 2026
 * =============================================================================
 */
#ifdef _WIN32
#include <windows.h>
#endif

#include "network.h"
#include "network.h"

/* =============================================================================
 * PRINT_BANNER - Muestra informacion del proyecto al iniciar
 * =============================================================================
 */

static void print_banner(void) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       MOTOR DE INFERENCIA - RED NEURONAL FEED-FORWARD        ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Clasificador del Dataset Iris                               ║\n");
    printf("║  Autores: Luis Eulacio, Eros Flugel                          ║\n");
    printf("║  Arquitectura: %-2d - %-2d - %-2d (Entrada - Oculta - Salida)      ║\n",
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
}

/* =============================================================================
 * GET_HYPERPARAMETERS - Solicita al usuario los hiperparametros
 *
 * Learning Rate y Epocas son los unicos valores configurables.
 * La topologia de la red es fija por disenio.
 * =============================================================================
 */
static void get_hyperparameters(double *lr, int *epochs) {
    printf("\n--- CONFIGURACION DE HIPERPARAMETROS ---\n\n");

    do {
        printf("  Ingrese el Learning Rate (ej: 0.1, 0.5, 1.0, 2.3): ");
        if (scanf("%lf", lr) != 1 || *lr <= 0.0 || *lr > 10.0) {
            printf("  [!] Valor invalido. Use un numero positivo entre 0.01 y 10.0\n");
            while (getchar() != '\n'); /* Limpiar buffer */
        } else {
            break;
        }
    } while (1);

    do {
        printf("  Ingrese el numero de Epocas (ej: 100, 500, 1000): ");
        if (scanf("%d", epochs) != 1 || *epochs <= 0) {
            printf("  [!] Valor invalido. Use un entero positivo.\n");
            while (getchar() != '\n'); /* Limpiar buffer */
        } else {
            break;
        }
    } while (1);

    printf("\n  Configuracion seleccionada:\n");
    printf("    Learning Rate = %.4f\n", *lr);
    printf("    Epocas       = %d\n", *epochs);
}

/* =============================================================================
 * MAIN - Punto de entrada del programa
 * =============================================================================
 */
int main(void) {
    // Set console output codepage to UTF-8
    #ifdef _WIN32
        SetConsoleOutputCP(65001);
    #endif

    double learning_rate;
    int epochs;
    double benchmark_ms;

    /* Semilla aleatoria para inicializar pesos */
    srand((unsigned int) time(NULL));

    /* Mostrar banner informativo */
    print_banner();

    /* =====================================================================
     * PASO 1: Cargar el dataset desde flores-dataset.txt
     * ===================================================================== */
    printf("\n--- CARGANDO DATASET ---\n\n");
    Dataset *dataset = dataset_load("flores-dataset.txt");

    /* =====================================================================
     * PASO 2: Normalizar los features al rango [0, 1]
     * ===================================================================== */
    dataset_normalize(dataset);

    /* =====================================================================
     * PASO 3: Dividir en entrenamiento (80%) y prueba (20%)
     * ===================================================================== */
    dataset_split(dataset, 0.8);

    /* =====================================================================
     * PASO 4: Solicitar hiperparametros al usuario
     * ===================================================================== */
    get_hyperparameters(&learning_rate, &epochs);

    /* =====================================================================
     * PASO 5: Crear la red neuronal con topologia 4-8-3
     * ===================================================================== */
    printf("\n--- CREANDO RED NEURONAL ---\n\n");

    int layer_sizes[3] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    FeedForwardNN *nn = nn_create(3, layer_sizes, learning_rate, epochs);

    printf("  Red creada exitosamente.\n");
    printf("  Topologia: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    /* =====================================================================
     * PASO 6: Entrenar la red neuronal
     * ===================================================================== */
    nn_train(nn, dataset);

    /* =====================================================================
     * PASO 7: Evaluar el modelo con el conjunto de prueba
     * ===================================================================== */
    printf("\n--- BENCHMARKING: Midiendo tiempo de inferencia ---\n\n");

    /* Medir tiempo de forward propagation sobre el conjunto de prueba */
    benchmark_ms = benchmark_forward(nn, dataset);

    printf("  Tiempo de inferencia (%d muestras): %.4f ms\n",
           dataset->num_test, benchmark_ms);
    printf("  Tiempo promedio por muestra: %.6f ms\n",
           benchmark_ms / dataset->num_test);

    /* =====================================================================
     * PASO 8: Mostrar metricas de evaluacion
     * ===================================================================== */
    print_evaluation_metrics(nn, dataset);

    /* =====================================================================
     * PASO 9: Ejemplo de prediccion individual
     * ===================================================================== */
    printf("\n--- EJEMPLO DE PREDICCION ---\n\n");

    if (dataset->num_test > 0) {
        double *sample = *(dataset->test_features + 0);
        int actual = *(dataset->test_labels + 0);
        int predicted = nn_predict(nn, sample);

        const char *class_names[] = {"Setosa", "Versicolor", "Virginica"};

        printf("  Muestra de prueba #0:\n");
        printf("    Features: [%.2f, %.2f, %.2f, %.2f]\n",
               *(sample + 0), *(sample + 1), *(sample + 2), *(sample + 3));
        printf("    Clase real:    %s (%d)\n", *(class_names + actual), actual);
        printf("    Clase predicha: %s (%d)\n", *(class_names + predicted), predicted);

        if (predicted == actual) {
            printf("    Resultado: CORRECTO\n");
        } else {
            printf("    Resultado: INCORRECTO\n");
        }
    }

    /* =====================================================================
     * PASO 10: Liberar toda la memoria (prevencion de memory leaks)
     * ===================================================================== */
    printf("\n--- LIBERANDO MEMORIA ---\n\n");

    nn_free(nn);
    dataset_free(dataset);

    printf("  Memoria liberada exitosamente.\n");
    printf("  Cero memory leaks.\n");

    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║               PROGRAMA FINALIZADO              ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    return 0;
}
