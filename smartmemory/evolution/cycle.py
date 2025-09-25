from smartmemory.plugins.evolvers.episodic_decay import EpisodicDecayEvolver
from smartmemory.plugins.evolvers.episodic_to_semantic import EpisodicToSemanticEvolver
from smartmemory.plugins.evolvers.episodic_to_zettel import EpisodicToZettelEvolver
from smartmemory.plugins.evolvers.semantic_decay import SemanticDecayEvolver
from smartmemory.plugins.evolvers.working_to_episodic import WorkingToEpisodicEvolver
from smartmemory.plugins.evolvers.working_to_procedural import WorkingToProceduralEvolver
from smartmemory.plugins.evolvers.zettel_prune import ZettelPruneEvolver

ALL_EVOLVERS = [
    WorkingToEpisodicEvolver,
    WorkingToProceduralEvolver,
    EpisodicToSemanticEvolver,
    EpisodicDecayEvolver,
    SemanticDecayEvolver,
    EpisodicToZettelEvolver,
    ZettelPruneEvolver,
]


def run_evolution_cycle(memory, config=None, logger=None):
    """
    Runs all evolvers in sequence. Config can override thresholds per evolver.
    """
    for EvolverClass in ALL_EVOLVERS:
        evolver = EvolverClass(config=config or {})
        evolver.evolve(memory, logger=logger)
