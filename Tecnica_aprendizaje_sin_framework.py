import csv
import random
import math

def cargar_dataset(ruta):
    """Lee un CSV y lo regresa como una lista de diccionarios
    
        Args:
            ruta: ruta al archivo .csv, con encabezado en la primera línea.
    
        Returns:
            Lista de dicts, uno por fila del CSV.
        """
    data = []
    
    with open(ruta, "r") as file:
        reader = csv.DictReader(file)
        
        for i in reader:
            data.append(i)
        
    return data

def dividir_dataset(data, proporcion=0.8):
    """Divide el dataset en entrenamiento y prueba de forma aleatoria.
    
        Se usa una semilla fija (seed=42) para que la partición sea reproducible:
        cualquier persona que corra este script obtiene exactamente el mismo
        split de entrenamiento/prueba.
    
        Args:
            data: lista de ejemplos (dicts) a dividir.
            proporcion: fracción de ejemplos que va a entrenamiento (0.8 = 80%).
    
        Returns:
            Tupla (entrenamiento, prueba), ambas listas de dicts.
        """
    datos = data[:]
    
    random.seed(42)
    random.shuffle(datos)
    
    corte = int(len(datos) * proporcion)
    entrenamiento = datos[:corte]
    prueba = datos[corte:]
    
    return entrenamiento, prueba


def calcular_entropia(ejemplos, objetivo):
    """Calcula la entropía de Shannon de un conjunto de ejemplos respecto a
             la variable objetivo
         
             Args:
                 ejemplos: lista de dicts (filas) sobre la que se mide la entropía.
                 objetivo: nombre de la columna que contiene la clase.
         
             Returns:
                 Entropía en bits (float).
    """
    
    total = len(ejemplos)
    
    conteo = {}
    
    for fila in ejemplos:
        clase = fila[objetivo]
        
        if clase in conteo:
            conteo[clase] += 1
        else:
            conteo[clase] = 1
            
    entropia = 0
    
    for cantidad in conteo.values():
        proporcion = cantidad/total
        
        entropia -= proporcion * math.log2(proporcion)
        
    return entropia

def obtener_valores(ejemplos, atributo):
    """Regresa la lista de valores distintos que toma un atributo dentro
        de un conjunto de ejemplos
    """
    valores = []
    
    for fila in ejemplos:
        valor = fila[atributo]
        
        if valor not in valores:
            valores.append(valor)
            
    return valores

def obtener_subconjunto(ejemplos, atributo, valor):
    """Filtra los ejemplos cuyo valor en `atributo` es exactamente `valor`.
    
        """
    subconjinto = []
    
    for fila in ejemplos:
        if fila[atributo] == valor:
            subconjinto.append(fila)
            
    return subconjinto

def calcular_ganancia(ejemplos, atributo, objetivo):
    """Calcula la ganancia de información de dividir `ejemplos` por `atributo`
    
        Args:
            ejemplos: conjunto de datos antes de dividir.
            atributo: atributo candidato para dividir.
            objetivo: columna de la clase.
    
        Returns:
            Ganancia de información (float, en bits).
        """
    entropia_inicial = calcular_entropia(ejemplos, objetivo)
    
    entropia_despues = 0
    
    valores = obtener_valores(ejemplos, atributo)
    
    for valor in valores:
        subconjunto = obtener_subconjunto(ejemplos, atributo, valor)
        
        proporcion = len(subconjunto)/len(ejemplos)
        
        entropia_despues += (proporcion * calcular_entropia(subconjunto, objetivo))
        
    ganancia = entropia_inicial - entropia_despues
    
    return ganancia

def mejor_atributo(ejemplos, atributos, objetivo):
    """Regresa, de una lista de atributos candidatos, el que tiene mayor
        ganancia de información respecto al objetivo. Este es el criterio de
        selección de ID3 en cada nodo del árbol.
        """
    mejor = atributos[0]
    
    mejor_ganancia = calcular_ganancia(ejemplos, mejor, objetivo)
    
    for atributo in atributos:
        ganancia = calcular_ganancia(ejemplos, atributo, objetivo)
        
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor = atributo
            
    return mejor

def misma_clase(ejemplos, objetivo):
    """True si todos los ejemplos pertenecen a la misma clase (nodo puro,
        caso base #1 de la recursión de ID3)."""
    primera_clase = ejemplos[0][objetivo]
    
    for fila in ejemplos:
        if fila[objetivo] != primera_clase:
            return False
        
    return True

def clase_mas_comun(ejemplos, objetivo):
    """Regresa la clase mayoritaria de un conjunto de ejemplos. Se usa como
        valor de una hoja cuando ya no hay atributos para seguir dividiendo, y
        también como "default" de un nodo para valores no vistos en entrenamiento.
        """
    conteo = {}
    
    for fila in ejemplos:
        clase = fila[objetivo]
        
        if clase in conteo:
            conteo[clase] += 1
        else:
            conteo[clase] = 1
            
    return max(conteo, key=lambda clase:conteo[clase])

def id3(ejemplos, objetivo, atributos):
    """Construye recursivamente un árbol de decisión con el algoritmo ID3.
    
    
        Args:
            ejemplos: subconjunto de entrenamiento que llega a este nodo.
            objetivo: columna de la clase.
            atributos: atributos aún disponibles para dividir en este nodo.
    
        Returns:
            Un string (hoja) o un dict (nodo interno), como se describe arriba.
        """
    if misma_clase(ejemplos, objetivo):
        return ejemplos[0][objetivo]
    
    if len(atributos) == 0:
        return clase_mas_comun(ejemplos, objetivo)
    
    mejor = mejor_atributo(ejemplos, atributos, objetivo)
    
    arbol = {
        "atributo": mejor,
        "ramas": {},
        "default": clase_mas_comun(ejemplos, objetivo)
    }
    
    atributos_restantes = []
    
    for atributo in atributos:
        if atributo != mejor:
            atributos_restantes.append(atributo)
    
    valores = obtener_valores(ejemplos, mejor)
    
    for valor in valores:
        subconjunto = obtener_subconjunto(ejemplos, mejor, valor)
        
        if len(subconjunto) == 0:
            arbol["ramas"][valor] = clase_mas_comun(ejemplos, objetivo)
        
        else:
            arbol["ramas"][valor] = id3(subconjunto, objetivo, atributos_restantes)
    
    return arbol   

dataset = cargar_dataset("data/coches_deportivos.csv")

entrenamiento, prueba = dividir_dataset(dataset)

objetivo = "Performance"

atributos = [
    "Motor",
    "Traccion",
    "Peso",
    "Potencia",
    "Transmision",
    "Aerodinamica",
    "Marca"
]

arbol = id3(entrenamiento, objetivo, atributos)
print(arbol)

# Predicciones de prueba
def predecir(ejemplo, arbol):
    """Clasifica un ejemplo nuevo recorriendo el árbol desde la raíz.
    
        Args:
            ejemplo: dict con los atributos del caso a clasificar.
            arbol: árbol (o subárbol) regresado por `id3`.
    
        Returns:
            La clase predicha (string).
        """
    if not isinstance(arbol, dict):
        return arbol
    
    atributo = arbol["atributo"]
    valor = ejemplo.get(atributo)
    
    if valor in arbol["ramas"]:
        return predecir(ejemplo, arbol["ramas"][valor])
    
    return arbol["default"]

# Evaluación completa, matriz de confusión y métricas
def evaluar(arbol, conjunto, objetivo):
    """Clasifica cada ejemplo de `conjunto` con el árbol y regresa una lista
        de tuplas (clase_real, clase_predicha), lista para construir la matriz
        de confusión.
        """
    predicciones = []
    
    for fila in conjunto:
        real = fila[objetivo]
        prediccion = predecir(fila, arbol)
        predicciones.append((real, prediccion))
    
    return predicciones

def matriz_confusion(predicciones, clases):
    """Construye la matriz de confusión a partir de pares (real, predicho).
        Regresa un diccionario anidado matriz[clase_real][clase_predicha] = conteo.
        """
    matriz = {real: {pred: 0 for pred in clases} for real in clases}
    
    for real, prediccion in predicciones: 
        matriz[real][prediccion] += 1
        
    return matriz

def calcular_metricas(matriz, clase_positiva):
    """Calcula Accuracy, Precision, Recall y F1 a partir de la matriz de
        confusión.
    
        Returns:
            Dict con las métricas y los conteos VP/FP/VN/FN usados para calcularlas.
        """
    clases = list(matriz.keys())
    otras = [c for c in clases if c != clase_positiva]
    
    vp = matriz[clase_positiva][clase_positiva]
    fn = sum(matriz[clase_positiva][c] for c in otras)
    fp = sum(matriz[c][clase_positiva] for c in otras)
    vn = sum(matriz[c][c2] for c in otras for c2 in otras)
    
    total = vp + fn + fp + vn
    
    accuracy = (vp + vn)/total if total > 0 else 0
    precision = vp/(vp + fp) if (vp + fp) > 0 else 0
    recall = vp/(vp + fn) if (vp + fn) > 0 else 0
    f1 = (2 * precision * recall/(precision + recall)
        if (precision + recall) > 0 else 0)
    
    return {
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "f1": f1, "Verdaderos positivos": vp, "Falsos positivos": fp,
        "Verdaderos negativos": vn, "Falsos negativos": fn
    }
    
print("=== Tamaños ===")
print(f"Entrenamiento: {len(entrenamiento)} filas | Prueba: {len(prueba)} filas")

print("\n=== Árbol ===")
print(arbol)

predicciones = evaluar(arbol, prueba, objetivo)
clases = ["Alta", "Baja"]
matriz = matriz_confusion(predicciones, clases)

print("\n=== Matriz de confusión (filas=real, columnas=predicho) ===")
print("\t\tAlta\tBaja")
for real in clases:
    print(real, "\t", matriz[real]["Alta"], "\t", matriz[real]["Baja"])

metricas = calcular_metricas(matriz, "Alta")
print("\n=== Métricas (clase positiva = Alta) ===")
for nombre, valor in metricas.items():
    print(nombre, ":", valor)