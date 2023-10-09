from fcapy.lattice import Lattice, ConceptLattice, FormalConcept
from fcapy.visualizer import ConceptLatticeVisualizer
from fcapy.ml import DecisionLattice, DecisionBasedModel


class Lattice:
    def __init__(self):
        self.lattice = None

    def from_context(self, context):
        """Construye el retículo a partir de un contexto formal."""
        self.lattice = ConceptLattice.from_context(context)

    def from_concepts(self, concepts):
        """Construye el retículo a partir de una lista de conceptos."""
        self.lattice = ConceptLattice(concepts)

    def from_json(self, filepath):
        """Carga el retículo desde un archivo JSON."""
        self.lattice = ConceptLattice.from_json(filepath)

    def save_to_json(self, filepath):
        """Guarda el retículo en un archivo JSON."""
        self.lattice.to_json(filepath)

    def visualize(self, filepath="lattice.html"):
        """Visualiza el retículo en un archivo HTML."""
        visualizer = ConceptLatticeVisualizer(self.lattice)
        visualizer.save(filepath)

    def get_concepts(self):
        """Devuelve todos los conceptos del retículo."""
        return self.lattice.concepts

    def get_extent(self, concept):
        """Devuelve el extent de un concepto."""
        return concept.extent

    def get_intent(self, concept):
        """Devuelve el intent de un concepto."""
        return concept.intent

    def get_subconcepts(self, concept):
        """Devuelve todos los subconceptos de un concepto."""
        return concept.subconcepts

    def get_superconcepts(self, concept):
        """Devuelve todos los superconceptos de un concepto."""
        return concept.superconcepts

    def is_subconcept(self, concept1, concept2):
        """Devuelve True si concept1 es un subconcepto de concept2, False en caso contrario."""
        return concept1.is_subconcept(concept2)

    def is_superconcept(self, concept1, concept2):
        """Devuelve True si concept1 es un superconcepto de concept2, False en caso contrario."""
        return concept1.is_superconcept(concept2)

    def find_concept(self, extent):
        """Encuentra el concepto con el extent especificado."""
        return self.lattice.find_concept(extent)

    def find_concepts_with_intent(self, intent):
        """Encuentra todos los conceptos con el intent especificado."""
        return self.lattice.find_concepts_with_intent(intent)