
import itertools




#################################################################### NEXTCLOSURE ####################################################################

"""
-G es el conjunto de todos los objetos en el contexto.
-M es el conjunto de todos los atributos en el contexto.
-I es la relación incidencia que muestra qué objetos tienen qué atributos.
-L será el conjunto de todos los conceptos formales (retículo) que se encuentran.
-g es el atributo más grande que no está en A pero que está en G.
-A es el conjunto actual de atributos que se está expandiendo para encontrar nuevos conceptos.
-A'' es el cierre del conjunto A, representando todos los atributos compartidos por los objetos que tienen todos los atributos en A.
-A' es el conjunto de todos los objetos que tienen todos los atributos en A.




1 g := max(G)      - Selecciona el objeto más grande de G, es decir, el último objeto en el orden total de G.
2 L := Ø           - Inicializa el retículo de conceptos L como un conjunto vacío.
3 A := Ø           - Inicializa el conjunto A, que representa el derivacion de la extensión formada A, también como un conjunto vacío.
4 Mientras A ≠ G hacer                          - Mientras el conjunto A sea diferente del conjunto de todos los objetos G, ejecuta el bucle.
5     A := A U {g} \ {h|h ∈ A&g < h}            - Actualiza A añadiendo el objeto g y removiendo cualquier atributo h en A que sea mayor que g para mantener A canónico.
6     si {h|h ∈ A'' \ A&h < g} == Ø entonces    - Realiza el test de canonicidad: si no hay atributos h menores que g en A'' que no estén en A, entonces A es canónico.
7         L := L U {(A'', A')}                  - Si A es canónico, añade el nuevo concepto (A'', A') a L.
8         g := max({h|h ∈ G \ A''})             - Encuentra el objeto más grande que no esté en A'' y actualiza g con este objeto.
9         A := A''                              - Actualiza A para que sea el derivacion de A' es decir A''.
10    fin                                       - Termina el bucle if.
11    en otro caso                                Caso de No Canonicidad: 
12       g := max({h|h ∈ G \ A&h < g'})           - Si el conjunto A no era canónico, entonces encuentra el objeto más grande que es menor que g y no está en A y actualiza g con este objeto.
13    fin                                         - Completa la iteración actual y continúa con la siguiente iteración del bucle comenzando en el paso 4.
14 fin
15 return L;                    El algoritmo se completa cuando todas las combinaciones posibles de atributos han sido exploradas.
                                Devuelve L que contiene todos los conceptos formales encontrados.


"""
def nextclosure(context_matrix, attributes, objects, debug=False):

    # Precomputar índices para objetos y atributos para un acceso rápido
    object_index_map = {obj: idx for idx, obj in enumerate(objects)}
    attribute_index_map = {attr: idx for idx, attr in enumerate(attributes)}


    # Funciones internas
    def es_mayor(h_comparado, h_comparador, objects):
        """
        Determina si el objeto 'h_comparado' es lexicográficamente mayor que el objeto 'h_comparador'
        basándose en sus índices en la lista 'objects'.
        """
        return object_index_map[h_comparado] > object_index_map[h_comparador]

    def elemento_maximal(h_set):
        """
        Encuentra el objeto con el máximo índice en la lista 'objects' dentro del conjunto dado 'h_set'.
        """
        max_element, max_index = None, -1
        for element in h_set:
            # Verifica si el elemento está en el mapa de índices y compara su índice.
            if element in object_index_map and object_index_map[element] > max_index:
                max_index = object_index_map[element]
                max_element = element
        return max_element

    def derivar_extension(extension):
        """Calcula el cierre de intensiones basado en la extensión dada utilizando índices precomputados."""
        intension = set(attributes)
        for obj in extension:
            obj_index = object_index_map[obj]
            obj_attributes = {attr for attr in attributes if context_matrix[obj_index][attribute_index_map[attr]]}
            intension.intersection_update(obj_attributes)
        return intension

    def derivar_intension(intension):
        """Calcula el cierre de extensiones basado en la intension dada utilizando índices precomputados."""
        extension = set(objects)
        for attr in intension:
            attr_index = attribute_index_map[attr]
            objects_with_attr = {obj for obj in objects if context_matrix[object_index_map[obj]][attr_index]}
            extension.intersection_update(objects_with_attr)
        return extension


    # Inicialización de variables
    G = set(objects)
    L = []
    A = set()

    # Manejo del conjunto vacío
    A_cierre = set(attributes)
    A_doble_cierre = set()
    L.append((A_doble_cierre, A_cierre))

    # Inicio del bucle principal
    g = elemento_maximal(G)
    if debug:
        print(f"Inicialización: G = {G}, g = {g}")

    while A != G:
        if debug:
            print(f"\nBucle principal: A = {A}, g = {g}")

        # - Actualiza A añadiendo el objeto g y removiendo cualquier atributo h en A que sea mayor que g para mantener A canónico.
        A = (A.union({g}) - {h for h in A if es_mayor(h, g, objects)}) 


        if debug:
            print(f"Después de actualizar A: A = {A}")


        A_cierre = derivar_extension(A)
        A_doble_cierre = derivar_intension(A_cierre)

        if debug:
            print(f"A_cierre (intensión): {A_cierre}, A_doble_cierre (extensión): {A_doble_cierre}")


        # Realiza el test de canonicidad: si no hay atributos h menores que g en A'' que no estén en A, entonces A es canónico.
        if not {h for h in (A_doble_cierre - A) if es_mayor(g, h, objects)}:
            L.append((A_doble_cierre, A_cierre)) # añade el concepto a la lista

            if debug:
                print(f"Añadido a L: ({A_doble_cierre}, {A_cierre})")

            # Encuentra el objeto más grande que no esté en A'' y actualiza g con este objeto.
            g = elemento_maximal((h for h in G if h not in A_doble_cierre))
            A = A_doble_cierre # Actualiza A para que sea el derivacion de A' es decir A''.
        else:
            # Si el conjunto A no era canónico, entonces encuentra el objeto más grande que es menor que g y no está en A y actualiza g con este objeto.
            g = elemento_maximal((h for h in (G - A) if es_mayor(g, h, objects)))

        if debug:
            print(f"Numero total de conceptos hasta ahora = ", len(L))



    return sorted(L, key=lambda x: len(x[0]))  # Devuelve el retículo formal de conceptos



























#################################################################### ITERATIVO ####################################################################
def generar_iterativamente(context_matrix, objects, attributes):

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1) (2) (3) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
    
    def derivar_extension(extension, context_matrix, objects, attributes):
        """Calcula el cierre de un conjunto de objetos"""
        # Iniciar con todos los atributos y reducir según los objetos presentes en la extensión.
        intension = set(attributes) 
        for obj in extension:
            obj_index = objects.index(obj)
            # Obtener los atributos que el objeto actual posee
            obj_attributes = {attr for attr in attributes if context_matrix[obj_index][attributes.index(attr)]}
            # Realizar la intersección para mantener solo aquellos objetos que cumplen con todos los atributos hasta ahora considerados.
            intension.intersection_update(obj_attributes)

        return intension

    def derivar_intension(intension, context_matrix, objects, attributes):
        """Calcula el cierre de un conjunto de atributos"""
        # Iniciar con todos los objetos y reducir según los atributos presentes en la intensión.
        extension = set(objects)
        for attr in intension:
            attr_index = attributes.index(attr)

            # Obtener los objetos que el atributo actual relaciona
            objects_with_attr = {obj for obj in objects if context_matrix[objects.index(obj)][attr_index]}
            # Realizar la intersección para mantener solo aquellos objetos que cumplen con todos los atributos hasta ahora considerados.
            extension.intersection_update(objects_with_attr)
        
        return extension
    
    def es_canonico(A_pp, A_p, L):
        for ext, inten in L:
            if A_pp == ext and A_p <= inten:
                return False
        return True
    
    """
    1. Inicializar conjunto de conceptos L := 0
    2. Seleccionar A perteneciente P(G)
    3. Derivar A'
    4. Derivar de nuevo A" = (A')'
    5. L := L U {(A'',A')} 
    6. Repetir de (2) a (5) para el siguiente A perteneciente P(G) de los restantes
    7. Devolver L, que es el conjunto de todos los conceptos
    """
    
    P_G = list(powerset(objects))   # Conjunto de todas las partes de objetos
    L = []                          # Lista para almacenar los conceptos formales

    print("\nPOWERSET:")
    print(P_G)
    

    for A in P_G:

        A_p  = derivar_extension(A, context_matrix, objects, attributes)
        A_pp = derivar_intension(A_p, context_matrix, objects, attributes)
        if es_canonico(A_pp, A_p, L):  # Verificación robusta de canonicidad
            L.append((A_pp, A_p))
            """
            print("añadido", (A_pp, A_p))
        else:
            print("redundante", (A_pp, A_p))
           """

    return sorted(L, key=lambda x: len(x[0]))   # devuelve el reticulo lista de conceptos formales