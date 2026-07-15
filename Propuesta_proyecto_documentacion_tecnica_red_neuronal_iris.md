# **Propuesta de Proyecto y Documentación Técnica: Red Neuronal Artificial para la Clasificación del Dataset Iris (Hiper Parámetros Configurables)**

**Estudiantes:** Luis Eulacio, Eros Flugel  
**Carrera:** Ingeniería en Ciencia de Datos  
**Fecha:** Julio de 2026  
A continuación se presenta el documento unificado de propuesta formal y documentación técnica para el desarrollo del proyecto de aula correspondiente a la implementación de una Red Neuronal Artificial (ANN) básica en C/C++, aplicada al conjunto de datos clásico "Iris flower". El enfoque combina el rigor académico con una arquitectura de software donde la topología estructural de la red permanece estática y la experimentación se centra en el ajuste dinámico de los hiper parámetros de optimización durante el entrenamiento.

## **1\. Descripción del Problema y Justificación**

El conjunto de datos Iris es el estándar de referencia para evaluar algoritmos de clasificación de patrones. El problema consiste en clasificar tres especies de plantas del género Iris (Iris setosa, Iris versicolor e Iris virginica) a partir de cuatro características morfológicas fijas: longitud y ancho del sépalo, y longitud y ancho del pétalo.  
La justificación de este proyecto radica en la necesidad de comprender de manera matemática y empírica cómo influyen los hiper parámetros de control en el descenso del gradiente utilizando la función de activación Sigmoide. Al mantener una arquitectura estructural fija y conocida, podemos concentrarnos rigurosamente en analizar los fenómenos de convergencia, sub ajuste (underfitting) o sobreajuste (overfitting) manipulando exclusivamente la tasa de aprendizaje (Learning Rate) y la cantidad de ciclos de entrenamiento (Épocas) en un entorno de bajo nivel nativo en C/C++.

## **2\. Objetivos del Proyecto**

### **Objetivo General**

Desarrollar e implementar una Red Neuronal Artificial nativa en C/C++ con topología estructural fija y comportamiento de entrenamiento configurable (Learning Rate y Épocas), utilizando la función de activación Sigmoide para la clasificación precisa de las especies en el dataset Iris.

### **Objetivos Específicos**

* Implementar una arquitectura rígida de red neuronal en C/C++ con dimensiones estáticas optimizadas para el dataset Iris (4 neuronas de entrada, un ancho determinado en capas ocultas y 3 de salida).  
* Diseñar un módulo de configuración interactivo que permita parametrizar dinámicamente la tasa de aprendizaje (Learning Rate) y el número total de épocas de entrenamiento antes de la ejecución del algoritmo.  
* Implementar de forma modular la función de activación Sigmoide y su derivada exacta para procesar el avance y calcular los ajustes en el backpropagation.  
* Evaluar y reportar el impacto del Learning Rate y la cantidad de épocas en la curva de pérdida (Loss) y la precisión final (Accuracy) del modelo.

## **3\. Sustento Lógico de las Decisiones de Diseño (Rationale)**

Para garantizar el rendimiento, el control absoluto del hardware y la coherencia metodológica, se han tomado decisiones de diseño críticas fundamentadas en los siguientes principios:

* **Aislamiento de Variables Experimentales:** Al congelar las dimensiones morfológicas de la red (capas y neuronas), se reduce el ruido en la experimentación científica. Cualquier fluctuación en la convergencia del error o en el porcentaje de aciertos se atribuye unívocamente a los cambios introducidos en el Learning Rate (LT) y las épocas.  
* **Gestión de Memoria y Eficiencia:** Al no requerir configurabilidad en la estructura, se evita la asignación dinámica de memoria en tiempo de ejecución (mediante malloc o new). Toda la matriz de pesos sinápticos y vectores de activación se gestionan en bloques contiguos de memoria (Stack o estructuras estáticas definidas en compilación). Esto elimina la sobrecarga de software y la fragmentación de memoria, garantizando un rendimiento óptimo en microprocesadores.  
* **Localidad de Datos:** La disposición de matrices continuas aprovecha al máximo la memoria caché del procesador, acelerando los bucles intensivos de multiplicaciones requeridos tanto en la propagación hacia adelante como en el cálculo de gradientes.

## **4\. Estructuras de Datos y Algoritmos Utilizados**

Para modelar los componentes matemáticos de la red de manera nativa sin recurrir a bibliotecas externas de álgebra lineal, se definen estructuras compactas de almacenamiento contiguo para los pesos sinápticos y los vectores de sesgo del sistema. Las dimensiones se rigen por constantes fijas en tiempo de compilación asociadas directamente a la morfología del conjunto de datos Iris (4 características morfológicas de entrada y 3 clases de salida correspondientes a las especies del género Iris).

### **Algoritmos Core**

* Algoritmo de Activación (Función Sigmoide y su Derivada Matemática)  
* Algoritmo de Propagación hacia Adelante (Forward Pass)  
* Algoritmo de Retropropagación (Backpropagation) y Actualización de Pesos

## **5\. Especificaciones de Configurabilidad Técnica**

La siguiente tabla resume qué componentes permanecen fijos por diseño y cuáles se exponen al usuario para cumplir con los objetivos del experimento:

| Componente del Sistema | Descripción Técnica | Estado / Criterio de Configuración   |
| :---- | :---- | :---- |
| **Tasa de Aprendizaje (Learning Rate / LT)** | Escalar que determina la magnitud del paso que da el optimizador en cada iteración hacia el mínimo global de la función de pérdida. | **Completamente Configurable** (Ingresado por el usuario en tiempo de ejecución). |
| **Épocas de Entrenamiento** | Número total de pasadas completas que el algoritmo realizará sobre el conjunto de datos de entrenamiento (80% del dataset). | **Completamente Configurable** (Ingresado por el usuario en tiempo de ejecución). |
| **Topología y Ancho de Capas** | Estructura interna de la red neuronal. El ancho de la capa oculta y el número de capas intermedias se definen de manera estática en el código base. | Estático / No Configurable (Ej. Red fija 4-8-3). |
| **Función de Activación** | Operación matemática no lineal aplicada en la suma ponderada de cada neurona oculta. | Fijo Estático: Función Sigmoide (1 / (1 \+ e^(-x))). |
| **Capa de Entrada y Salida** | Dimensiones inmutables dictadas estrictamente por la naturaleza del dataset Iris. | Fijo Estático: Entrada \= 4 neuronas. Salida \= 3 neuronas. |

## **6\. Evidencia de Funcionamiento y Producto Final (Estrategia de Demostración)**

Para la defensa y presentación oral del proyecto ante el jurado evaluador, el producto final constará de una aplicación de consola interactiva nativa que servirá como evidencia empírica inmediata de funcionamiento:

* **Interfaz de Configuración en Vivo:** Al arrancar, el programa cargará el dataset de forma estática en memoria y congelará el hilo de ejecución para solicitar de forma interactiva que el usuario escriba los valores deseados de LT y Épocas.  
* **Monitoreo del Aprendizaje:** Durante el ciclo de entrenamiento, la terminal imprimirá en intervalos fijos (ej. cada 100 épocas) la disminución del error cuadrático medio (Loss function). Esto demostrará visualmente el proceso matemático de convergencia o divergencia según los parámetros introducidos.  
* **Métricas de Evaluación Inmediatas:** Una vez concluidas las épocas, el programa desplegará automáticamente la precisión global (Accuracy %) calculada sobre el 20% de datos reservados para validación, junto a la generación de una matriz de confusión textual directamente en la terminal. Esto permitirá comprobar instantáneamente la eficacia de la configuración elegida durante la exposición oral.