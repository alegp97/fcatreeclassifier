

def incremental_FCA_Norris(g, added, c, l):


    g_prime = c.object_prime(g)


    for f in l.formal_concepts:
        if is_subset(g_prime, f.second):
            l.replace(f, (f.first + g, f.second))
            f.first = f.first + g
        else:
            D = set(f.second).intersection(g_prime)
            h_vec = set(c.objects) - set(f.first)
            empty = True

            for h in h_vec:
                h_prime = c.object_prime([h])
                f2 = (h, h_prime)

                if h in added and is_subset(h_prime, D):
                    empty = False
                    break

            if empty:
                l.add((f.first + g, list(D)))

    empty2 = True
    for h in c.objects:
        h_prime2 = c.object_prime([h])

        if h in added and is_subset(h_prime2, g_prime):
            empty2 = False
            break

    if empty2:
        l.add((g, g_prime))
















    """
    def incremental_FCA(context, new_objects, new_attributes, new_relations)
    An incremental Formal Concept Analysis (FCA) algorithm builds a formal context incrementally, 
    by adding new objects or attributes to the existing context, rather than starting from scratch with a new set of objects and attributes.

    Perform incremental formal context analysis on a given formal context, by adding new objects,
    attributes and relations.
    Returns the updated formal context.

    This implementation takes in an existing formal context, represented as a dictionary, as well as lists of new objects, new attributes, 
    and new relations, and updates the context by adding the new elements to it. First, the new objects are added to the context's object list, 
    and to the G-I matrix by appending new rows with 1 if this new object is related to an existing attribute or 0 otherwise. After that, 
    the new attributes are added to the context's attributes list and to the G-I matrix by appending new columns with 1 if an existing object 
    is related to this new attribute or 0 otherwise. Finally, the new relations list is added to the context's relations list.
    
    context: the formal context to be updated, represented as a dictionary containing
             the objects, attributes, relations and the G-I matrix
    new_objects: list of new objects to be added
    new_attributes: list of new attributes to be added
    new_relations: list of tuples (object, attribute) representing the new relations to be added
    
    # add new objects to the context
    for o in new_objects:
        context['objects'].append(o)
        for a in context['attributes']:
            if (o, a) in new_relations:
                new_relations.remove((o, a))
        new_row = [1 if (o, a) in new_relations else 0 for a in context['attributes']]
        context['g_i_matrix'].append(new_row)
    
    # add new attributes to the context
    for a in new_attributes:
        context['attributes'].append(a)
        for o in context['objects']:
            if (o, a) in new_relations:
                new_relations.remove((o, a))
        new_col = [1 if (o, a) in new_relations else 0 for o in context['objects']]
        for i, row in enumerate(context['g_i_matrix']):
            row.append(new_col[i])

    context['relations'].extend(new_relations)
    return context

    """

