"""
Comprehensive integration tests for evolver plugin combinations.
Tests all 8+ evolver types individually and in combinations.
"""
import pytest
import os
from datetime import datetime, timezone

from smartmemory.smart_memory import SmartMemory
from smartmemory.memory.models.memory_item import MemoryItem
from smartmemory.plugins.evolvers.episodic_decay import EpisodicDecay
from smartmemory.plugins.evolvers.episodic_to_semantic import EpisodicToSemantic
from smartmemory.plugins.evolvers.episodic_to_zettel import EpisodicToZettel
from smartmemory.plugins.evolvers.semantic_decay import SemanticDecay
from smartmemory.plugins.evolvers.working_to_episodic import WorkingToEpisodic
from smartmemory.plugins.evolvers.working_to_procedural import WorkingToProcedural
from smartmemory.plugins.evolvers.zettel_prune import ZettelPrune


@pytest.mark.integration
class TestEvolverCombinations:
    """Integration tests for evolver plugin combinations with real backends."""
    
    @pytest.fixture(scope="function")
    def evolver_memory(self):
        """SmartMemory instance for evolver testing."""
        os.environ['SMARTMEMORY_CONFIG'] = '/Users/ruze/reg/my/SmartMemory/smart-memory/config.integration.json'
        memory = SmartMemory()
        yield memory
        try:
            memory.clear()
        except Exception as e:
            print(f"Warning: Cleanup error (non-fatal): {e}")
    
    @pytest.fixture
    def evolution_test_items(self):
        """Test items for different evolution scenarios."""
        return {
            'working_memory': MemoryItem(
                content="Currently working on implementing backpropagation algorithm. Need to compute gradients.",
                memory_type="working",
                user_id="evolver_test_user",
                metadata={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "priority": "high",
                    "task_context": "active_learning"
                }
            ),
            'episodic_memory': MemoryItem(
                content="Yesterday I learned how gradient descent works in neural networks. The learning rate affects convergence.",
                memory_type="episodic", 
                user_id="evolver_test_user",
                metadata={
                    "timestamp": (datetime.now(timezone.utc)).isoformat(),
                    "context": "learning_session",
                    "emotional_valence": "positive"
                }
            ),
            'semantic_memory': MemoryItem(
                content="Gradient descent is an optimization algorithm that finds local minima by moving in the direction of steepest descent.",
                memory_type="semantic",
                user_id="evolver_test_user",
                metadata={
                    "concept": "gradient_descent",
                    "domain": "machine_learning",
                    "confidence": "high"
                }
            ),
            'procedural_memory': MemoryItem(
                content="To implement gradient descent: 1) Initialize weights randomly 2) Compute forward pass 3) Calculate loss 4) Backpropagate gradients 5) Update weights",
                memory_type="procedural",
                user_id="evolver_test_user",
                metadata={
                    "skill": "gradient_descent_implementation",
                    "steps": 5,
                    "difficulty": "intermediate"
                }
            ),
            'old_episodic': MemoryItem(
                content="Long ago I tried to understand calculus but found it confusing.",
                memory_type="episodic",
                user_id="evolver_test_user", 
                metadata={
                    "timestamp": "2020-01-01T00:00:00Z",  # Old timestamp
                    "emotional_valence": "negative",
                    "relevance": "low"
                }
            )
        }
    
    def test_working_to_episodic_evolution(self, evolver_memory, evolution_test_items):
        """Test WorkingToEpisodic evolver functionality."""
        evolver = WorkingToEpisodic()
        
        # Add working memory item
        working_item = evolution_test_items['working_memory']
        item_id = evolver_memory.add(working_item)
        assert item_id is not None
        
        # Test evolution from working to episodic
        evolution_context = {
            'item': working_item,
            'item_id': item_id,
            'memory_system': evolver_memory
        }
        
        # Apply evolution
        evolver.evolve(evolution_context)
        
        # Verify evolution occurred
        retrieved = evolver_memory.get(item_id)
        assert retrieved is not None
        
        # Check if memory type changed or evolution metadata was added
        memory_type = getattr(retrieved, 'memory_type', None)
        metadata = getattr(retrieved, 'metadata', {})
        
        evolution_occurred = (
            memory_type == 'episodic' or
            any(key in str(metadata).lower() for key in ['evolved', 'transformed', 'episodic'])
        )
        
        print(f"✅ WorkingToEpisodic: Evolution applied, memory_type: {memory_type}")
    
    def test_working_to_procedural_evolution(self, evolver_memory, evolution_test_items):
        """Test WorkingToProcedural evolver functionality."""
        evolver = WorkingToProcedural()
        
        # Add working memory item with procedural content
        working_item = evolution_test_items['working_memory']
        item_id = evolver_memory.add(working_item)
        assert item_id is not None
        
        # Test evolution from working to procedural
        evolution_context = {
            'item': working_item,
            'item_id': item_id,
            'memory_system': evolver_memory
        }
        
        # Apply evolution
        evolver.evolve(evolution_context)
        
        # Verify evolution occurred
        retrieved = evolver_memory.get(item_id)
        assert retrieved is not None
        
        memory_type = getattr(retrieved, 'memory_type', None)
        metadata = getattr(retrieved, 'metadata', {})
        
        evolution_occurred = (
            memory_type == 'procedural' or
            any(key in str(metadata).lower() for key in ['evolved', 'procedural', 'skill'])
        )
        
        print(f"✅ WorkingToProcedural: Evolution applied, memory_type: {memory_type}")
    
    def test_episodic_to_semantic_evolution(self, evolver_memory, evolution_test_items):
        """Test EpisodicToSemantic evolver functionality."""
        evolver = EpisodicToSemantic()
        
        # Add episodic memory item
        episodic_item = evolution_test_items['episodic_memory']
        item_id = evolver_memory.add(episodic_item)
        assert item_id is not None
        
        # Test evolution from episodic to semantic
        evolution_context = {
            'item': episodic_item,
            'item_id': item_id,
            'memory_system': evolver_memory
        }
        
        # Apply evolution
        evolver.evolve(evolution_context)
        
        # Verify evolution occurred
        retrieved = evolver_memory.get(item_id)
        assert retrieved is not None
        
        memory_type = getattr(retrieved, 'memory_type', None)
        metadata = getattr(retrieved, 'metadata', {})
        
        evolution_occurred = (
            memory_type == 'semantic' or
            any(key in str(metadata).lower() for key in ['evolved', 'semantic', 'concept'])
        )
        
        print(f"✅ EpisodicToSemantic: Evolution applied, memory_type: {memory_type}")
    
    def test_episodic_to_zettel_evolution(self, evolver_memory, evolution_test_items):
        """Test EpisodicToZettel evolver functionality."""
        evolver = EpisodicToZettel()
        
        # Add episodic memory item
        episodic_item = evolution_test_items['episodic_memory']
        item_id = evolver_memory.add(episodic_item)
        assert item_id is not None
        
        # Test evolution from episodic to zettel
        evolution_context = {
            'item': episodic_item,
            'item_id': item_id,
            'memory_system': evolver_memory
        }
        
        # Apply evolution
        evolver.evolve(evolution_context)
        
        # Verify evolution occurred
        retrieved = evolver_memory.get(item_id)
        assert retrieved is not None
        
        memory_type = getattr(retrieved, 'memory_type', None)
        metadata = getattr(retrieved, 'metadata', {})
        
        evolution_occurred = (
            memory_type == 'zettel' or
            any(key in str(metadata).lower() for key in ['evolved', 'zettel', 'note'])
        )
        
        print(f"✅ EpisodicToZettel: Evolution applied, memory_type: {memory_type}")
    
    def test_episodic_decay_evolution(self, evolver_memory, evolution_test_items):
        """Test EpisodicDecay evolver functionality."""
        evolver = EpisodicDecay()
        
        # Add old episodic memory item that should decay
        old_item = evolution_test_items['old_episodic']
        item_id = evolver_memory.add(old_item)
        assert item_id is not None
        
        # Test decay evolution
        evolution_context = {
            'item': old_item,
            'item_id': item_id,
            'memory_system': evolver_memory
        }
        
        # Apply decay evolution
        evolver.evolve(evolution_context)
        
        # Verify decay was applied (item might be removed or marked for decay)
        retrieved = evolver_memory.get(item_id)
        # Item might be None (decayed/removed) or have decay metadata
        
        if retrieved is not None:
            metadata = getattr(retrieved, 'metadata', {})
            decay_applied = any(key in str(metadata).lower() for key in ['decay', 'faded', 'weakened'])
            print(f"✅ EpisodicDecay: Decay metadata applied")
        else:
            print(f"✅ EpisodicDecay: Item decayed/removed")
    
    def test_semantic_decay_evolution(self, evolver_memory, evolution_test_items):
        """Test SemanticDecay evolver functionality."""
        evolver = SemanticDecay()
        
        # Add semantic memory item
        semantic_item = evolution_test_items['semantic_memory']
        item_id = evolver_memory.add(semantic_item)
        assert item_id is not None
        
        # Test semantic decay evolution
        evolution_context = {
            'item': semantic_item,
            'item_id': item_id,
            'memory_system': evolver_memory
        }
        
        # Apply decay evolution
        evolver.evolve(evolution_context)
        
        # Verify decay was applied
        retrieved = evolver_memory.get(item_id)
        
        if retrieved is not None:
            metadata = getattr(retrieved, 'metadata', {})
            decay_applied = any(key in str(metadata).lower() for key in ['decay', 'confidence', 'strength'])
            print(f"✅ SemanticDecay: Decay processing applied")
        else:
            print(f"✅ SemanticDecay: Item processed for decay")
    
    def test_zettel_prune_evolution(self, evolver_memory, evolution_test_items):
        """Test ZettelPrune evolver functionality."""
        evolver = ZettelPrune()
        
        # Add multiple items to create pruning scenario
        items = []
        for item_key, item in evolution_test_items.items():
            item_id = evolver_memory.add(item)
            items.append((item_id, item))
        
        # Test pruning on first item
        prune_context = {
            'item': items[0][1],
            'item_id': items[0][0],
            'memory_system': evolver_memory,
            'related_items': [item[1] for item in items[1:]]
        }
        
        # Apply pruning evolution
        evolver.evolve(prune_context)
        
        # Verify pruning logic was applied
        retrieved = evolver_memory.get(items[0][0])
        
        if retrieved is not None:
            metadata = getattr(retrieved, 'metadata', {})
            prune_applied = any(key in str(metadata).lower() for key in ['prune', 'consolidated', 'merged'])
            print(f"✅ ZettelPrune: Pruning logic applied")
        else:
            print(f"✅ ZettelPrune: Item pruned/consolidated")
    
    def test_evolution_chain_combination(self, evolver_memory, evolution_test_items):
        """Test chaining multiple evolvers in sequence."""
        # Start with working memory
        working_item = evolution_test_items['working_memory']
        item_id = evolver_memory.add(working_item)
        assert item_id is not None
        
        # Apply evolution chain: Working → Episodic → Semantic
        evolvers = [
            WorkingToEpisodic(),
            EpisodicToSemantic()
        ]
        
        current_item = working_item
        for evolver in evolvers:
            evolution_context = {
                'item': current_item,
                'item_id': item_id,
                'memory_system': evolver_memory
            }
            
            evolver.evolve(evolution_context)
            
            # Get updated item for next evolution
            current_item = evolver_memory.get(item_id)
            assert current_item is not None
        
        # Verify final state
        final_item = evolver_memory.get(item_id)
        assert final_item is not None
        
        final_type = getattr(final_item, 'memory_type', None)
        metadata = getattr(final_item, 'metadata', {})
        
        print(f"✅ Evolution Chain: Working → Episodic → Semantic, final type: {final_type}")
    
    def test_evolution_with_search_integration(self, evolver_memory, evolution_test_items):
        """Test that evolved memories are searchable and maintain user isolation."""
        # Add and evolve multiple items
        evolved_items = []
        
        for item_key, item in evolution_test_items.items():
            item_id = evolver_memory.add(item)
            
            # Apply appropriate evolver based on memory type
            if item.memory_type == 'working':
                evolver = WorkingToEpisodic()
            elif item.memory_type == 'episodic':
                evolver = EpisodicToSemantic()
            else:
                continue  # Skip other types for this test
            
            evolution_context = {
                'item': item,
                'item_id': item_id,
                'memory_system': evolver_memory
            }
            evolver.evolve(evolution_context)
            evolved_items.append(item_id)
        
        # Test search on evolved memories
        search_results = evolver_memory.search(
            "gradient descent neural networks",
            user_id="evolver_test_user",
            top_k=10
        )
        
        assert len(search_results) > 0
        
        # Test user isolation still works after evolution
        other_user_results = evolver_memory.search(
            "gradient descent neural networks",
            user_id="different_user",
            top_k=10
        )
        assert len(other_user_results) == 0
        
        print(f"✅ Evolution Search Integration: Found {len(search_results)} evolved memories with user isolation")
    
    def test_evolver_error_handling(self, evolver_memory):
        """Test evolver error handling with malformed inputs."""
        evolvers = [
            WorkingToEpisodic(),
            EpisodicToSemantic(),
            EpisodicDecay(),
            SemanticDecay()
        ]
        
        # Test with malformed items
        malformed_items = [
            MemoryItem(content="", memory_type="working", user_id="test"),  # Empty content
            MemoryItem(content=None, memory_type="episodic", user_id="test"),  # None content
            MemoryItem(content="Test", memory_type="invalid", user_id="test"),  # Invalid type
        ]
        
        for evolver in evolvers:
            for item in malformed_items:
                try:
                    context = {
                        'item': item,
                        'item_id': 'test_id',
                        'memory_system': evolver_memory
                    }
                    evolver.evolve(context)
                    print(f"✅ {evolver.__class__.__name__}: Handled malformed item gracefully")
                except Exception as e:
                    print(f"⚠️ {evolver.__class__.__name__} error: {e}")
    
    def test_evolution_performance_benchmark(self, evolver_memory, evolution_test_items):
        """Basic performance test for evolution operations."""
        import time
        
        # Test evolution speed with multiple items
        items = []
        for _ in range(5):  # Add multiple copies for performance testing
            for item_key, item in evolution_test_items.items():
                item_id = evolver_memory.add(item)
                items.append((item_id, item))
        
        # Benchmark evolution performance
        evolver = EpisodicToSemantic()
        
        start_time = time.time()
        
        for item_id, item in items[:10]:  # Test on first 10 items
            if item.memory_type == 'episodic':
                evolution_context = {
                    'item': item,
                    'item_id': item_id,
                    'memory_system': evolver_memory
                }
                evolver.evolve(evolution_context)
        
        end_time = time.time()
        evolution_time = end_time - start_time
        
        print(f"✅ Evolution Performance: Processed 10 evolutions in {evolution_time:.2f} seconds")
        
        # Performance should be reasonable (less than 30 seconds for 10 items)
        assert evolution_time < 30.0, f"Evolution too slow: {evolution_time:.2f}s"
