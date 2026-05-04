# Importamos las herramientas que necesitamos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("Iniciando proyecto de IA...\n")

# --- PASO 1 y 2: Comprensión de los Datos ---
# Cargamos una base de datos que ya viene de ejemplo (Flores Iris)
datos = load_iris()
X = datos.data    # Las características (largo y ancho de pétalos)
y = datos.target  # Las respuestas correctas (qué tipo de flor es)
print("1. Datos cargados exitosamente.")

# --- PASO 3: Preparación de los Datos ---
# Dividimos nuestros datos en dos partes: 
# Un 80% para que la IA estudie (entrenamiento) y un 20% para hacerle un examen (prueba)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
print("2. Datos preparados y divididos para el entrenamiento y la prueba.")

# --- PASO 4: Modelado ---
# Creamos nuestro modelo de IA (un "Árbol de Decisión") y lo ponemos a estudiar (.fit)
modelo_ia = DecisionTreeClassifier()
modelo_ia.fit(X_entrenamiento, y_entrenamiento)
print("3. Modelado completado: ¡La IA ha sido entrenada!")

# --- PASO 5: Evaluación ---
# Le damos a la IA los datos del examen (que nunca ha visto) para que haga sus predicciones
predicciones = modelo_ia.predict(X_prueba)

# Calificamos su examen comparando sus predicciones con las respuestas correctas
calificacion = accuracy_score(y_prueba, predicciones)

print("\n--- RESULTADOS DE LA EVALUACIÓN ---")
print(f"La precisión de nuestra IA es del: {calificacion * 100}%")
if calificacion == 1.0:
    print("¡La IA acertó a todas las flores del examen de prueba!")