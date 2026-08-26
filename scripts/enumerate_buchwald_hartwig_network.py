from pathlib import Path
import json
from itertools import product
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdChemReactions


def enumerate_reaction_network(
        reactants: list[str],
        reactions: list[str],
        max_depth: int = 10
) -> nx.DiGraph:
    """
    Enumerates a reaction network given a set of reactants and reactions.

    Args:
        reactants (list[str]): A list of SMILES strings representing the starting reactants.
        reactions (list[str]): A list of SMIRKS strings representing the reactions to apply.
        max_depth (int): The maximum depth of the reaction network to explore.

    Returns:
        nx.DiGraph: A directed graph representing the reaction network, where nodes are compounds (SMILES strings)
                    and edges represent reactions.
    """
    graph = nx.MultiDiGraph()

    for smiles in reactants:
        mol = Chem.MolFromSmiles(smiles)
        graph.add_node(Chem.MolToSmiles(mol), mol=mol)

    reaction_objects = {smirks: rdChemReactions.ReactionFromSmarts(smirks) for smirks in reactions}

    num_nodes, num_edges, iteration = 0, 0, 0

    while iteration < max_depth:
        num_nodes, num_edges = graph.number_of_nodes(), graph.number_of_edges()
        all_reactants = {smiles: graph.nodes[smiles]["mol"] for smiles in graph.nodes}

        for smirks, reaction in reaction_objects.items():

            # Identify all possible reactants (per reactant template) that are present in the graph
            reactant_count = reaction.GetNumReactantTemplates()
            possible_reactants = []  # list of length `reactant_count`
            for i in range(reactant_count):
                possible_reactants_i = []
                reactant_template = reaction.GetReactantTemplate(i)
                for mol_smiles, mol in all_reactants.items():
                    if mol.HasSubstructMatch(reactant_template):
                        possible_reactants_i.append(mol_smiles)
                possible_reactants.append(possible_reactants_i)

            # Generate all combinations of reactants, enumerate all product sets, and add them to the graph (nodes +
            # + edges) if they are not already present
            for reactant_combination in product(*possible_reactants):
                reactant_mols = [graph.nodes[smiles]["mol"] for smiles in reactant_combination]
                product_sets = reaction.RunReactants(reactant_mols)

                for product_set in product_sets:
                    for product_mol in product_set:
                        Chem.SanitizeMol(product_mol)
                        product_smiles = Chem.MolToSmiles(product_mol)
                        if product_smiles not in graph:
                            graph.add_node(product_smiles, mol=product_mol)

                        edge_id = f"{smirks}_{'_'.join(reactant_combination)}"

                        for reactant_smiles in reactant_combination:
                            if not graph.has_edge(reactant_smiles, product_smiles, key=edge_id):
                                graph.add_edge(reactant_smiles, product_smiles, key=edge_id, smirks=smirks)

        if graph.number_of_nodes() == num_nodes and graph.number_of_edges() == num_edges:
            break

        iteration += 1

    return graph


if __name__ == "__main__":

    data_dir = Path.cwd() / "results" / "buchwald_hartwig_experiment"

    reaction_types = [
        "[c:1]!@[N&H,N&H2:2].[c:3]!@[Cl,Br,I]>>[c:1][N:2][c:3]",  # Buchwald-Hartwig amination
        "[*:1][O-:2].[c:3]!@[Cl,Br,I]>>[*:1][O+0:2][c:3]",  # Buchwald-Hartwig etherification
        "[c:1]!@[Cl,Br,I]>>[c:1]",  # protodehalogenation
        "[c:1]!@[Cl,Br,I].[c:2]!@[Cl,Br,I]>>[c:1][c:2]",  # Reductive homocoupling
        "[c:1]!@[N&H,N&H2:2].[c:3]!@[N&H,N&H2:4]>>[c:1][N:2][N:4][c:3]",  # Oxidative homocoupling
    ]

    molecules = [
        "BrC1=CC2=CC=C(Br)C=C2C=C1",  # Dibromonaphthalene
        "CNC1=CC=CC=C1",  # Methylaniline,
        "[Na+].[O-]C(C)(C)C",  # Sodium tert-butoxide
    ]

    reaction_network = enumerate_reaction_network(molecules, reaction_types, max_depth=2)

    molecules = [node for node in reaction_network.nodes]
    with open(data_dir / "buchwald_hartwig_reaction_network.json", "w") as f:
        json.dump(molecules, f, indent=4)
