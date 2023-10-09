def inclose(context, c, Y, concepts, C=[]):
    """
    Algoritmo InClose.
    - context: Lista de listas binarias que representa el contexto.
    - c: Concepto actual.
    - Y: Atributos a considerar.
    - concepts: Lista donde se guardan los conceptos encontrados.
    - C: Cierre del conjunto de objetos c respecto al contexto.
    """
    if not C:
        C = closure(context, c)
    if C not in [concept[1] for concept in concepts]:
        concepts.append((c, C))
        for y in range(Y, len(context[0])):
            if y not in C:
                B = closure(context, c + [y])
                if B not in [concept[1] for concept in concepts]:
                    inclose(context, c + [y], y+1, concepts, B)



def closure(context, c):
    """
    Función para calcular el cierre de un conjunto de objetos c respecto al contexto.
    """
    if not c:
        return set(range(len(context[0])))
    else:
        C = set(range(len(context[0])))
        for i in c:
            C &= set([y for y, val in enumerate(context[i]) if val])
        return C