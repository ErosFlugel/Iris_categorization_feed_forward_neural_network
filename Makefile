# =============================================================================
# MAKEFILE - Motor de Inferencia Red Neuronal Feed-Forward (Iris Classifier)
#
# Uso:
#   make          -> Compila el proyecto y genera el ejecutable
#   make clean    -> Elimina archivos temporales y el ejecutable
#   make run      -> Compila y ejecuta el programa
#
# Compilador: gcc (C nativo)
# Librerias: stdio.h, stdlib.h, math.h, time.h (solo estandar)
# =============================================================================

# Configuracion de compilador
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LDFLAGS = -lm

# Archivos fuente y objeto
SRCS = main.c network.c
OBJS = main.o network.o
HEADERS = network.h

# Nombre del ejecutable
TARGET = iris_nn

# =============================================================================
# REGLAS DE COMPILACION
# =============================================================================

# Regla principal: compilar el ejecutable
all: $(TARGET)

# Compilar el ejecutable enlazando todos los objetos
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@echo "================================================"
	@echo "  Compilacion exitosa: $(TARGET)"
	@echo "  Ejecute con: ./$(TARGET)"
	@echo "================================================"

# Compilar main.c -> main.o
main.o: main.c $(HEADERS)
	$(CC) $(CFLAGS) -c main.c -o main.o

# Compilar network.c -> network.o
network.o: network.c $(HEADERS)
	$(CC) $(CFLAGS) -c network.c -o network.o

# =============================================================================
# REGLAS DE UTILIDAD
# =============================================================================

# Ejecutar el programa
run: $(TARGET)
	./$(TARGET)

# Limpiar archivos generados
clean:
	rm -f $(OBJS) $(TARGET)
	-del /Q $(OBJS) $(TARGET).exe 2>nul
	@echo "Archivos limpiados."
# =============================================================================
# NOTAS:
# - Se usa -O2 para optimizacion moderada (no afecta la logica)
# - Se usa -lm para enlazar la libreria matematica (math.h)
# - Se usa -Wall -Wextra para advertencias utiles durante desarrollo
# - El ejecutable se llama iris_nn (o iris_nn.exe en Windows)
# =============================================================================
