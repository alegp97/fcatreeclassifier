import time
import bisect
import graphviz
import warnings

class Concept:
    def __init__(self, context, extent, intent, inmediate_subconcepts=None):
        """
        Inicializa un concepto dentro de un contexto formal.
        
        :param context: La instancia de Context a la que pertenece este Concepto.
        :param extent: Lista de índices representando los objetos (extent) en este concepto.
        :param intent: Lista de índices representando los atributos (intent) en este concepto.
        :param parents: Lista opcional de conceptos que son padres de este concepto.
        :param inmediate_subconcepts: Lista opcional de conceptos que son hijos de este concepto.
        """
        self.context = context
        self.extent = extent
        self.intent = intent
        self.inmediate_subconcepts = inmediate_subconcepts if inmediate_subconcepts is not None else []
        self.parent = None
        self.label = None
        self.id = None


    def __repr__(self):
        return f"Concept(extent={self.extent}, intent={self.intent})"



class FormalContext:
    def __init__(self, labeled_data=None, target_colum='objective_target', cxt_file_path=None, objects=None, attributes=None, incidence_matrix=None, labels_dict=None):
        if labeled_data is not None:
            self._init__from_pandas_dataset(labeled_data,target_colum)
        elif cxt_file_path is not None:
            self._init_from_cxt(cxt_file_path)
        elif objects is not None and attributes is not None and incidence_matrix is not None:
            self._init_from_matrix(objects, attributes, incidence_matrix, labels_dict=labels_dict)
        else:
            raise ValueError("Debe proporcionar un dataframe de pandas-labeled_data, cxt_file_path, o la combinación de objects, attributes, e incidence_matrix.")

    def _init_from_matrix(self, objects, attributes, incidence_matrix, labels_dict):
        # inicialización directa desde matrices de objetos, atributos e incidencia, y opción de etiqeutas de clase
        self.objects = objects
        self.attributes = attributes
        self.incidence_matrix = incidence_matrix
        self.labels_dict = labels_dict 
        self.lattice = []
        self.processed_intents = set()
        self.num_objects = len(self.objects)
        self.num_properties = len(self.attributes)
        self.max_context_size = self.num_objects * self.num_properties

        if(labels_dict==None):
            warnings.warn("AVISO: no se ha especificado diccionario de etiquetas de clase", UserWarning)

    def _init__from_pandas_dataset(self, labeled_data, target_colum): 
        # inicialización basado en labeled_data
        self.labeled_data = labeled_data

        self.objects = list(labeled_data.index)
        properties = [prop for prop in labeled_data.columns if prop != target_colum]
        self.attributes = properties

        self.labels_dict = {obj: labeled_data.loc[obj, target_colum] for obj in self.objects}

        bool_matrix = labeled_data.drop(columns=[target_colum]).astype(int).values.tolist()
        self.incidence_matrix = bool_matrix

        self.lattice = []
        self.processed_intents = set()

        self.num_objects = len(self.objects)
        self.num_properties = len(self.attributes)
        self.max_context_size = self.num_objects * self.num_properties

    def _init_from_cxt(self, cxt_file_path):
        self.load_from_cxt(cxt_file_path) # Carga del archivo .cxt

    def load_from_cxt(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        assert lines[0].strip() == 'B', "Formato de archivo no reconocido"

        num_objects = int(lines[1].strip())
        num_attributes = int(lines[2].strip())

        # Leer objetos y etiquetas
        i = 3
        objects = []
        self.labels_dict = {}
        while lines[i].strip() != '--':
            line = lines[i].strip()
            obj, label = line.split(' (')
            label = label.rstrip(')')
            objects.append(obj)
            self.labels_dict[obj] = label
            i += 1
        i += 1  # Saltar delimitador

        # Leer atributos
        attributes = []
        while lines[i].strip() != '--':
            attributes.append(lines[i].strip())
            i += 1
        i += 1  # Saltar delimitador

        # Leer matriz de incidencia
        incidence_matrix = []
        for j in range(num_objects):
            row = list(map(int, list(lines[i + j].strip())))
            incidence_matrix.append(row)

        # Asignar valores a la instancia actual
        self.objects = objects
        self.attributes = attributes
        self.incidence_matrix = incidence_matrix
        self.lattice = []
        self.processed_intents = set()

        self.num_objects = len(objects)
        self.num_properties = len(attributes)
        self.max_context_size = self.num_objects * self.num_properties

        # Mensajes de depuración
        print(f"Objetos: {self.objects}")
        print(f"Atributos: {self.attributes}")
        print(f"Matriz de Incidencia: {self.incidence_matrix}")

    def compute_parents_and_childs(self):
        """Asigna a cada concepto su padre inmediato y sus hijos."""
        for concept_i in self.lattice:
            best_parent = None
            for concept_j in self.lattice:
                if concept_i != concept_j:
                    if (set(concept_j.extent).issuperset(set(concept_i.extent)) and
                        set(concept_j.intent).issubset(set(concept_i.intent))):
                        if best_parent is None:
                            best_parent = concept_j
                        elif len(concept_j.extent) < len(best_parent.extent):
                            best_parent = concept_j
            concept_i.parent = best_parent
            if best_parent is not None:
                best_parent.inmediate_subconcepts.append(concept_i)


    def get_concepts_with_attribute(self, attribute):
        """
        Devuelve una lista de conceptos que contienen un atributo específico.

        :param attribute: El atributo a buscar en los conceptos.
        :return: Lista de conceptos que contienen el atributo.
        """
        return [concept for concept in self.lattice if attribute in concept.intent]


    def save_formal_context_to_cxt(self, file_path, target_colum='objective_target'):
        with open(file_path, "w") as file:
            file.write("B\n")
            file.write(f"{len(self.objects)}\n")  # Número de objetos
            file.write(f"{len(self.attributes)}\n")  # Número de atributos

            # Guardar los nombres de los objetos con sus etiquetas de clase
            for obj in self.objects:
                label = self.labels_dict.get(obj, "")
                file.write(f"{obj} ({label})\n")

            file.write("--\n")  # Delimitador entre objetos y atributos

            # Guardar los nombres de los atributos
            for attr in self.attributes:
                file.write(f"{attr}\n")

            file.write("--\n")  # Delimitador entre atributos y relaciones

            # Guardar la matriz de incidencia
            incidence_matrix = [[0] * len(self.attributes) for _ in range(len(self.objects))]
            obj_index_map = {obj: i for i, obj in enumerate(self.objects)}
            attr_index_map = {attr: i for i, attr in enumerate(self.attributes)}

            for concept in self.lattice:
                for obj in concept.extent:
                    if obj not in obj_index_map:
                        print(f"Objeto {obj} no encontrado en obj_index_map")
                        continue
                    for attr_index in concept.intent:
                        if attr_index < 0 or attr_index >= len(self.attributes):
                            print(f"Índice de atributo {attr_index} fuera de rango")
                            continue
                        attr = self.attributes[attr_index]
                        if attr not in attr_index_map:
                            print(f"Atributo {attr} no encontrado en attr_index_map")
                            continue
                        incidence_matrix[obj_index_map[obj]][attr_index_map[attr]] = 1

            for row in incidence_matrix:
                file.write("".join(map(str, row)) + "\n")



    def add_concepts(self, new_concepts):
        """Añade nuevos conceptos al retículo"""
        self.lattice.extend(new_concepts)

    def is_immediate_successor(self, lower_concept, higher_concept):
            """
            Verifica si lower_concept es un sucesor inmediato de higher_concept.

            Parámetros:
            - lower_concept: El concepto que se está evaluando como subconcepto.
            - higher_concept: El concepto que se está evaluando como superconcepto.

            Retorna:
            - True si lower_concept es un sucesor inmediato de higher_concept, False en caso contrario.
            """
            lower_intent = set(lower_concept.intent)
            higher_intent = set(higher_concept.intent)

            if not lower_intent.issubset(higher_intent):
                return False

            # Verifica si hay algún concepto intermedio entre lower_concept y higher_concept
            for intermediate_concept in self.lattice:
                if intermediate_concept != lower_concept and intermediate_concept != higher_concept:
                    intermediate_intent = set(intermediate_concept.intent)
                    if lower_intent.issubset(intermediate_intent) and intermediate_intent.issubset(higher_intent):
                        return False

            return True

    def format_concept(self, concept):
        """ Formatea un concepto para impresión sin 'Extensión', 'Intensión' """
        extent = ', '.join(str(self.objects[e]) for e in sorted(concept.extent)) or '∅'
        extent = str(extent)

        intent = ', '.join(str(self.objects[e]) for e in sorted(concept.intent)) or '∅'
        intent = str(intent) 

        return f"{extent} ({intent})"
    
    def set_concept_lattice_labels_ids(self, method='max_extent'):
        """Asigna las etiquetas de clase a los conceptos de retículo para problemas de clasificación"""
        if (method == 'max_extent'):
            for index, concept in enumerate(self.lattice):
                concept.label = self.get_class_label_max_extent(concept)
                concept.id = index
    
    def get_class_label_max_extent(self, concept):
        """Saca la etiqueta de clase mayoritaria en la extensión"""
        class_counts = {}
        
        for object in concept.extent:
            label = self.get_object_label(index=object)
            if label:
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
        
        if not class_counts:
            return None
        
        return max(class_counts, key=class_counts.get)
    
    def get_labels_dict(self):
        return self.labels_dict
    
    def get_object_label(self, index):
        return self.labels_dict[self.objects[index]]
        
    def get_objects(self):                 
        return self.objects

    def get_attributes(self):
        return self.attributes
    
    def get_all_concepts_lattice(self):
        return self.lattice
    




    #################################################################### REPRESENTACIÓN ####################################################################
    
    def display_concept_children(self):
        for i, concept in enumerate(self.lattice):
            concept_number = i + 1
            # Encontrar los números de los conceptos que son hijos (basado en su posición en la lista)
            subconcepts_numbers = [self.lattice.index(child) + 1 for child in concept.inmediate_subconcepts]
            if subconcepts_numbers:
                print(f"Concepto {concept_number} tiene como hijos a los Conceptos {subconcepts_numbers}")
            else:
                print(f"Concepto {concept_number} no tiene hijos")

    def display_concept_parents(self):
        for i, concept in enumerate(self.lattice):
            concept_number = i + 1
            if concept.parent is not None:
                # Encontrar el número de concepto del padre (basado en su posición en la lista)
                parent_number = self.lattice.index(concept.parent) + 1
                print(f"Concepto {concept_number} tiene como padre al Concepto {parent_number}")
            else:
                print(f"Concepto {concept_number} no tiene padre")
    
    def print_contingency_table(self):
        # Imprime los nombres de los atributos como encabezado de la tabla
        header = "Objetos/Atributos\t" + "\t".join(self.attributes)
        print(header)
        print("-" * len(header) * 2)  # Imprimir una línea divisoria

        # Imprimir cada fila de la matriz de incidencia
        for obj_index, obj in enumerate(self.objects):
            row = f"{obj}\t\t\t" + "\t".join(str(value) for value in self.incidence_matrix[obj_index])
            print(row)
        

    def print_lattice(self):
        if not self.lattice:
            print("El retículo está vacío.")
            return

        print("\nRetículo de Conceptos Formales:")
        for i, concept in enumerate(self.lattice, start=1):
            # Crea la representación de la extensión, si está vacía utiliza el símbolo de conjunto vacío
            if concept.extent:
                extent = ', '.join(str(self.objects[e]) for e in sorted(concept.extent))
                extent = str(extent)
            else:
                extent = '∅'  # Símbolo de conjunto vacío
            
            # Crea la representación de la intención, si está vacía utiliza el símbolo de conjunto vacío
            if concept.intent:
                intent = ', '.join(str(self.attributes[e]) for e in sorted(concept.intent))
                intent = str(intent)
            else:
                intent = '∅'  # Símbolo de conjunto vacío
            
            print(f"\nConcepto {i}:")
            print(f"  Extensión: {extent}")
            print(f"  Intensión: {intent}")
        print("\n")


    def create_hasse_diagram(self, output_filename):
        dot = graphviz.Digraph(comment='Hasse Diagram')
        
        # Crear nodos para cada concepto
        for concept in self.lattice:
            extent_str = ','.join(str(self.objects[e]) for e in concept.extent)
            intent_str = ','.join(str(self.attributes[i]) for i in concept.intent)
            node_id = extent_str  # Usar la extensión como ID, convertido a cadena
            dot.node(node_id, f'Ext: {{{extent_str}}}\nInt: {{{intent_str}}}')
        
        # Crear aristas entre conceptos y sus subconceptos inmediatos
        for concept in self.lattice:
            parent_id = ','.join(str(self.objects[e]) for e in concept.extent)
            for subconcept in concept.inmediate_subconcepts:
                subconcept_id = ','.join(str(self.objects[e]) for e in subconcept.extent)
                dot.edge(parent_id, subconcept_id)
        dot.render(output_filename, format='png')

    def __repr__(self):
        return (
            f"FormalContext(\n"
            f"  num_objects={self.num_objects},\n"
            f"  num_attributes={self.num_properties},\n"
            f"  context_size={self.max_context_size},\n"
            f"  objects={self.objects},\n"
            f"  attributes={self.attributes},\n"
            f"  incidence_matrix={self.incidence_matrix},\n"
            f"  lattice={self.lattice},\n"
            f"  processed_intents={self.processed_intents}\n"
            f")"
        )
    

    #################################################################### INCLOSE ####################################################################

    """
    Algoritmo 4: Pseudocódigo del algoritmo INCLOSE(r, y). 
    FUENTE: TFG - ESTUDIO EXPERIMENTAL DE ALGORITMOS DE CÁLCULO DE RETÍCULOS EN ANÁLISIS FORMAL DE CONCEPTOS, por Miguel Ángel Cantarero López

    Variación del algoritmo Inclose de la librería fca_algorithms-0.2.4

    S. Andrews. In-close, a fast algorithm for computing formal concepts. In Inter-
    national Conference on Conceptual Structures (ICCS), January 2009. Final version of
    paper accepted (via peer review) for the International Conference on Conceptual
    Structures (ICCS) 2009, Moscow.



    Entrada: lista de extensiones A0, ..., Ar
            lista de intensiones B0, ..., Br
            número de concepto actual r 
            número del atributo actual y
            un contexto formal (G, M, I)
    Salida: Modifica los parámetros de entrada:
            lista de extensiones A0, ..., Arnew
            lista de intensiones B0, ..., Brnew
            número del siguiente concepto rnew > r
            número del siguiente atributo ynew > y

    1  rnew ← rnew + 1                       // Incrementa el contador de nuevos conceptos
    2  para j = y ... |M| hacer              // Itera sobre todos los atributos desde el atributo actual y
    3     A[rnew] ← Ø;                       // Inicializa la nueva extensión como un conjunto vacío
    4     para i ∈ A[r] hacer                // Itera sobre cada objeto en la extensión del concepto actual
    5         si I[i][j] entonces            // Si el objeto i tiene el atributo j en la relación de incidencia
    6             A[rnew] ← A[rnew] U {i}    // Añade el objeto i a la nueva extensión
    7         fin
    8     fin
    9     si |A[rnew]| > 0 entonces          // Si la nueva extensión no es vacía
    10        si |A[rnew]| = |A[r]| entonces // Si la nueva extensión es igual a la extensión actual
    11            B[r] ← B[r] U {j}          // Añade el atributo j a la intensión del concepto actual
    12        fin
    13    fin
    14    en otro caso                       // Si la nueva extensión es vacía o difiere de la extensión actual
    15        si IsCanonical(r, j - 1) entonces // Comprueba si la extensión actual es canónica
    16            B[rnew] ← B[r] U {j}       // Si es canónica, añade el atributo j a la nueva intensión
    17            InClose(rnew, j + 1)       // Llama recursivamente a InClose con el nuevo concepto e índice de atributo
    18        fin
    19    fin
    fin                                    // Finaliza el algoritmo
    """
    def build_lattice(self, compute_parents_childs=False, debug=False):
        """
        Inicia la construcción del retículo desde el concepto más general posible mediante el alg. Inclose.
        """
        if debug:
            start_time = time.time()

        initial_concept = Concept(self, [i for i in range(len(self.objects))], [])
        self.lattice = [initial_concept]
        self.newest_relation = 0
        self.inclose(0, 0)

        # Eliminar el último concepto si su intensión es vacía
        if len(self.lattice[-1].intent) == 0:
            self.lattice.pop()

        if not compute_parents_childs:
            # Asegurarse de que el concepto con Extensión: ∅ solo se añada si no hay conceptos con extensión vacía
            if not any(len(concept.extent) == 0 for concept in self.lattice):
                self.infimum_concept = Concept(self, [], [i for i in range(len(self.attributes))])
                self.lattice.append(self.infimum_concept)
        else:
            self.compute_parents_and_childs() # omite el concepto con Extensión: ∅ para problemas de clasificación
            if not any(len(concept.extent) == 0 for concept in self.lattice):
                self.infimum_concept = Concept(self, [], [i for i in range(len(self.attributes))])
        

        if debug:  # Opcional: Imprimir el tiempo de ejecución
            end_time = time.time()
            print(f"Tiempo de ejecución para el retículo: {end_time - start_time:.2f} segundos")



    def contains_element(self, element, sorted_list):
        """
        Verifica si un elemento está presente en una lista ordenada y devuelve su posición.
        Utiliza la función `bisect_left` del módulo `bisect`, que implementa una búsqueda binaria 
        """
        # Encuentra el índice donde el elemento debería estar si existe
        index = bisect.bisect_left(sorted_list, element)
        
        # Comprueba si el elemento existe en ese índice y si es igual al elemento buscado
        if index != len(sorted_list) and sorted_list[index] == element:
            return True, index  # Elemento encontrado, retorna True y el índice
        return False, None  # Elemento no encontrado, retorna False y None
    


    def is_canonical(self, r, y):
        """
        VERIFICACIÓN DE CANONICIDAD
        
        Comprueba si la extensión actual es canónica antes de añadir un nuevo atributo.
        El chequeo de canonicidad es crucial para evitar la generación de conceptos redundantes
        y asegurar que el espacio de conceptos se explore eficientemente.
        
        Parámetros:
            r: Índice del concepto actual.
            y: Índice del último atributo añadido a la intención del concepto actual.
        
        Retorna:
            bool: True si la extensión es canónica, False en caso contrario.
        """
        # Acceso a la extensión actual y la matriz de incidencia.
        current_extent = self.lattice[self.newest_relation].extent
        incidence_matrix = self.incidence_matrix

        # Revisa todos los atributos en la intención desde y hasta el principio para asegurar que añadir cualquier
        # atributo menor no genere un concepto ya existente.
        for k in reversed(range(len(self.lattice[r].intent))):
            # El atributo en la intención desde el cual empezar la comprobación hacia atrás
            attribute_k = self.lattice[r].intent[k]
            for j in range(y, attribute_k, -1):
                # Si todos los objetos en la extensión actual tienen el atributo j, no es canónico.
                if all(incidence_matrix[obj][j] for obj in current_extent):
                    return False
            # Actualiza y para la próxima iteración en el bucle más externo.
            y = attribute_k - 1

        # Finalmente, verifica cualquier atributo menor que el más pequeño en la intención actual.
        for j in range(y, -1, -1):
            if all(incidence_matrix[obj][j] for obj in current_extent):
                return False

        return True

    def inclose(self, r, y):
        """Implementación del algoritmo recursivo de InClose para generar conceptos formales.
    
        Construye el retículo generan nuevos conceptos a partir de un concepto actual mediante la exploración de atributos 
        y la aplicación de operaciones de cierre. Utiliza la canonicidad para optimizar la generación de conceptos.
        
        Parámetros:
            r: Índice del concepto actual.
            y: Índice del atributo desde el cual comenzar la exploración.
        """
        # 1 Incrementa el contador de nuevos conceptos
        self.newest_relation += 1  

        # 2 Itera sobre todos los atributos desde el atributo actual y
        for j in range(y, len(self.attributes)):  

            # 3 Inicializa la nueva extensión como un conjunto vacío
            if len(self.lattice) <= self.newest_relation:
                self.lattice.append(Concept(self, [], []))
            else:
                self.lattice[self.newest_relation].extent = []

            # 4 y 5 Itera sobre cada objeto en la extensión del concepto actual y verifica la incidencia
            for i in self.lattice[r].extent:
                if self.incidence_matrix[i][j]:  # 5 Si el objeto i tiene el atributo j
                    if ( self.contains_element(i, self.lattice[self.newest_relation].extent) ): # 6 Añade el objeto i a la nueva extensión
                        bisect.insort_left(self.lattice[self.newest_relation].extent, i) # (Utiliza bisect_left para encontrar el índice adecuado y luego inserta el elemento en ese índice)

            # 9 Si la nueva extensión no es vacía
            if self.lattice[self.newest_relation].extent:

                # 10 Si la nueva extensión es igual a la extensión actual
                if self.lattice[self.newest_relation].extent == self.lattice[r].extent:
                    if ( self.contains_element(j, self.lattice[r].intent) ): # 11 Añade el atributo j a la intensión del concepto actual
                        bisect.insort_left(self.lattice[r].intent, j)

                # 14 y 15 Comprueba si la extensión actual es canónica
                elif self.is_canonical(r, j - 1):  
                    # 16 Si es canónica, añade el atributo j a la nueva intensión
                    new_attributes = self.lattice[r].intent.copy()
                    if ( self.contains_element(j, new_attributes) ):
                        bisect.insort_left(new_attributes, j)

                    self.lattice[self.newest_relation].intent = new_attributes
                    self.inclose(self.newest_relation, j + 1) # 17 Llama recursivamente a InClose con el nuevo concepto e índice de atributo





