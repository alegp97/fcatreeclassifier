import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(project_root)


import random
from typing import List, Tuple
import time
import matplotlib.pyplot as plt
from src.FormalContext import FormalContext
import fca_algorithms as fca



def imprimir_diagrama_hasse(conceptos_formales):
    print("\nDiagrama de Hasse: ")
    conceptos_ordenados = sorted(conceptos_formales, key=lambda c: len(c[0]))

    def es_subconjunto(concepto1, concepto2):
        return concepto1[0].issubset(concepto2[0]) and concepto2[1].issubset(concepto1[1])

    def imprimir_concepto(concepto):
        extent_nombres = {str(name) for name in concepto[0]}
        intent_nombres = {str(attr) for attr in concepto[1]}
        return f"({', '.join(extent_nombres)}, {{{', '.join(intent_nombres)}}})"

    for i, concepto in enumerate(conceptos_ordenados):
        enlaces_subconjunto = []
        for j in range(i):
            if es_subconjunto(conceptos_ordenados[j], concepto):
                enlaces_subconjunto.append(imprimir_concepto(conceptos_ordenados[j]))

        cadena_subconjuntos = ' <-- ' + ', '.join(enlaces_subconjunto) if enlaces_subconjunto else ''
        print(f"{imprimir_concepto(concepto)}{cadena_subconjuntos}")




def generate_random_context(num_objetos: int, num_atributos: int) -> Tuple[List[List[int]], List[str], List[str]]:
    """
    Genera un contexto formal aleatorio como una tabla de contingencia con 1s y 0s.

    Args:
    num_objetos (int): Número de objetos en el contexto.
    num_atributos (int): Número de atributos en el contexto.

    Returns:
    Tuple[List[List[int]], List[str], List[str]]: El contexto formal (tabla de contingencia),
                                                  lista de objetos y lista de atributos.
    """
    # Generar la tabla de contingencia aleatoria
    contexto = [[random.randint(0, 1) for _ in range(num_atributos)] for _ in range(num_objetos)]

    # Generar etiquetas para los objetos y los atributos
    objetos = [f"O{i+1}" for i in range(num_objetos)]
    atributos = [f"A{i+1}" for i in range(num_atributos)]

    return contexto, objetos, atributos


def create_custom_context():
    # Definir los nombres de los objetos y atributos
    objects = ['NA', 'NE', 'SE']
    attributes = ['C', 'R', 'A', 'P', 'E']
    
    # Inicializar la matriz del contexto con valores predeterminados, por ejemplo, todos en 0
    context = [
        [0, 1, 1, 0, 1],  # Valores para NA
        [1, 1, 1, 0, 0],  # Valores para NE
        [1, 1, 1, 1, 0]   # Valores para SE
    ]

    return context, objects, attributes

def create_extended_custom_context():
    objects = ['NA', 'NE', 'SE', 'NW', 'SW']
    attributes = ['C', 'R', 'A', 'P', 'E', 'F', 'G']
    
    context = [
        [0, 1, 1, 0, 1, 0, 1],  # Valores para NA
        [1, 1, 1, 0, 0, 1, 0],  # Valores para NE
        [1, 1, 1, 1, 0, 0, 1],  # Valores para SE
        [0, 0, 1, 1, 1, 1, 0],  # Valores para NW
        [1, 0, 0, 1, 1, 1, 1]   # Valores para SW
    ]

    return context, objects, attributes





def try_algorithm(algoritmo, incidence_matrix, objects, attributes):
    start_time = time.time()
    if algoritmo == "inclose":
        
        context = FormalContext(objects=objects, attributes=attributes, incidence_matrix=incidence_matrix)
    
        context.build_lattice()

        print("\nFinalizando Algoritmo:")
        print("Número de objetos:", len(context.objects))
        print("Número de atributos:", len(context.attributes))

    else:
        formal_concepts = fca.nextclosure(incidence_matrix, attributes, objects)
        print("\nFinalizando Algoritmo:")
        print("Número de objetos:", len(objects))
        print("Número de atributos:", len(attributes))
        print("Número de conceptos:", len(formal_concepts))


    time_elapsed = time.time() - start_time

    return time_elapsed


def comparar_algoritmos_Next_Inclose():
    sizes = [10, 20, 30, 40, 50, 70] 
    times_inclose = []
    times_nextclosure = []

    for size in sizes:
        print("\n\nEjecutando ambos con un chunk de :" , size)
        incidence_matrix, objects, attributes = generate_random_context(size, size)
        print("\nObjects:", objects)
        print("\nAttributes:", attributes)
        max_num_concepts = 2 ** min(len(objects), len(attributes))
        print("\nNúmero máximo de conceptos formales (2^max(objects, attributes)):", max_num_concepts)

        times_inclose.append(try_algorithm("inclose", incidence_matrix, objects, attributes))
        times_nextclosure.append(try_algorithm("nextclosure", incidence_matrix, objects, attributes))

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, times_inclose, label='InClose')
    plt.plot(sizes, times_nextclosure, label='NextClosure')
    plt.xlabel('Tamaño del contexto (número de objetos y atributos)')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de tiempos de ejecución de InClose vs NextClosure')
    plt.legend()
    plt.grid(True)
    plt.show()


def comparar_resultados_Inclose():

    sizes = [10, 20, 30, 40, 50, 60, 70, 80] 
    times_inclose = []

    for size in sizes:
        print("\n\nEjecutando Inclose con un chunk de :" , size)
        incidence_matrix, objects, attributes = generate_random_context(size, size)
        #print("\nObjects:", objects)
        #print("\nAttributes:", attributes)
        max_num_concepts = 2 ** min(len(objects), len(attributes))
        print("\nNúmero máximo de conceptos formales (2^max(objects, attributes)):", max_num_concepts)
        tiempo = try_algorithm("inclose", incidence_matrix, objects, attributes)
        times_inclose.append(tiempo)
        print("Tiempo de ejec: ", tiempo)

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, times_inclose, label='InClose')
    plt.xlabel('Tamaño del contexto (número de objetos y atributos)')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de tiempos de ejecución de InClose')
    plt.legend()
    plt.grid(True)
    plt.show()


    


########################################################################################################################################

########################################################################################################################################

########################################################################################################################################

########################################################################################################################################


print("\n######################################################################################################:    ")
print("\n############################################# CONTEXTO DE EJEMPLO ####################################:    ")
print("\n######################################################################################################:    ")
# Definir contexto
incidence_matrix, objects, attributes = create_custom_context()


print("Generated Context:")
for obj, row in zip(objects, incidence_matrix):
    print(f"{obj}: {row}")
print("\nObjects:", objects)
print("\nAttributes:", attributes)




print("\n\nNEXTCLOSURE:     ")
formal_concepts = fca.nextclosure(incidence_matrix, attributes, objects)
print("\nFinalizando Algoritmo:")
print("Número de objetos:", len(objects))
print("Número de atributos:", len(attributes))
print("Conceptos generados:", len(formal_concepts))

for extent, intent in formal_concepts:
    print(f"Extent: {extent}, Intent: {intent}")

print("\n######################################################################################################:    ")

print("\nITERATIVO:     ")
formal_concepts = fca.generar_iterativamente(incidence_matrix, attributes=attributes, objects=objects)
print("\nFinalizando Algoritmo:")
print("Número de objetos:", len(objects))
print("Número de atributos:", len(attributes))
print("Conceptos generados:", len(formal_concepts))

for extent, intent in formal_concepts:
    print(f"Extent: {extent}, Intent: {intent}")


# Imprimir diagrama de Hasse para los retículos de ejemplo
imprimir_diagrama_hasse(formal_concepts)

print("\n######################################################################################################:    ")
print("\n############################################# INCLOSE ################################################:    ")
print("\n######################################################################################################:    ")

context = FormalContext(objects=objects, attributes=attributes, incidence_matrix=incidence_matrix)
context.build_lattice()


print("\nFinalizando Algoritmo:")
print("Número de objetos:", len(objects))
print("Número de atributos:", len(attributes))
print("Número de conceptos formales:", len(context.get_all_concepts_lattice()))

context.print_lattice()
context.print_contingency_table()


print("\n######################################################################################################:    ")
print("\n################################### CONTEXTO DE EJEMPLO EXTENDIDO ####################################:    ")
print("\n######################################################################################################:    ")
# Definir contexto
incidence_matrix, objects, attributes = create_extended_custom_context()


print("Generated Context:")
for obj, row in zip(objects, incidence_matrix):
    print(f"{obj}: {row}")
print("\nObjects:", objects)
print("\nAttributes:", attributes)




print("\n\nNEXTCLOSURE:     ")
formal_concepts = fca.nextclosure(incidence_matrix, attributes, objects)
print("\nFinalizando Algoritmo:")
print("Número de objetos:", len(objects))
print("Número de atributos:", len(attributes))
print("Conceptos generados:", len(formal_concepts))

for extent, intent in formal_concepts:
    print(f"Extent: {extent}, Intent: {intent}")

print("\n######################################################################################################:    ")

print("\nITERATIVO:     ")
formal_concepts = fca.generar_iterativamente(incidence_matrix, attributes=attributes, objects=objects)
print("\nFinalizando Algoritmo:")
print("Número de objetos:", len(objects))
print("Número de atributos:", len(attributes))
print("Conceptos generados:", len(formal_concepts))

for extent, intent in formal_concepts:
    print(f"Extent: {extent}, Intent: {intent}")


# Imprimir diagrama de Hasse para los retículos de ejemplo
imprimir_diagrama_hasse(formal_concepts)

print("\n######################################################################################################:    ")
print("\n############################################# INCLOSE ################################################:    ")
print("\n######################################################################################################:    ")

context = FormalContext(objects=objects, attributes=attributes, incidence_matrix=incidence_matrix)
context.build_lattice()


print("\nFinalizando Algoritmo:")
print("Número de objetos:", len(objects))
print("Número de atributos:", len(attributes))
print("Número de conceptos formales:", len(context.get_all_concepts_lattice()))

context.print_lattice()
context.print_contingency_table()
context.compute_parents_and_childs()
context.display_concept_parents()
print("\n")
context.display_concept_children()




print("\n######################################################################################################:    ")
print("\n############################## EJECUCION DE COMPARACIONES INCLOSE-NEXTCLOSURE ####################################:    ")
print("\n######################################################################################################:    ")
comparar_algoritmos_Next_Inclose()
