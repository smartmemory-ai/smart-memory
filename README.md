# SmartMemory - Multi-Layered AI Memory System

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

SmartMemory is a comprehensive AI memory system that provides persistent, multi-layered memory storage and retrieval for AI applications. It combines graph databases, vector stores, and intelligent processing pipelines to create a unified memory architecture.

## Architecture Overview

SmartMemory implements a multi-layered memory architecture with the following components:

### Core Components

- **SmartMemory**: Main unified memory interface (`smartmemory.smart_memory.SmartMemory`)
- **SmartGraph**: Graph database backend using FalkorDB for relationship storage
- **Memory Types**: Specialized memory stores for different data types
- **Pipeline Stages**: Processing stages for ingestion, enrichment, and evolution
- **Plugin System**: Extensible architecture for custom evolvers and enrichers

### Memory Types

- **Working Memory**: Short-term context buffer (in-memory, capacity=10)
- **Semantic Memory**: Facts and concepts with vector embeddings
- **Episodic Memory**: Personal experiences and learning history
- **Procedural Memory**: Skills, strategies, and learned patterns

### Storage Backend

- **FalkorDB**: Graph database for relationships and structured data
- **ChromaDB**: Vector database for semantic similarity search
- **Redis**: Caching layer for performance optimization

### Processing Pipeline

The memory ingestion flow processes data through several stages:

1. **Input Adaptation**: Convert input data to MemoryItem format
2. **Classification**: Determine appropriate memory type
3. **Extraction**: Extract entities and relationships
4. **Storage**: Persist to appropriate memory stores
5. **Linking**: Create connections between related memories
6. **Enrichment**: Enhance memories with additional context
7. **Evolution**: Transform memories based on configured rules

## Key Features

- **Multi-Type Memory System**: Working, Semantic, Episodic, and Procedural memory types
- **Graph-Based Storage**: FalkorDB backend for complex relationship modeling
- **Vector Similarity**: ChromaDB integration for semantic search capabilities
- **Extensible Pipeline**: Modular processing stages for ingestion and evolution
- **Plugin Architecture**: Custom evolvers, enrichers, and grounders
- **Multi-User Support**: User and group isolation for enterprise applications
- **Caching Layer**: Redis-based performance optimization
- **Configuration Management**: Flexible configuration with environment variable support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/smartmemory-ai/smart-memory.git
cd smart-memory

# Install dependencies
pip install -r requirements.txt

# Install spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from smartmemory.smart_memory import SmartMemory
from smartmemory.models.memory_item import MemoryItem
from datetime import datetime

# Initialize SmartMemory
memory = SmartMemory()

# Create a memory item
item = MemoryItem(
    content="User prefers Python for data analysis tasks",
    memory_type="semantic",
    user_id="user123",
    metadata={'topic': 'preferences', 'domain': 'programming'}
)

# Add to memory
memory.add(item)

# Search memories
results = memory.search("Python programming", top_k=5)
for result in results:
    print(f"Content: {result.content}")
    print(f"Type: {result.memory_type}")
    print(f"Metadata: {result.metadata}")

# Get memory summary
summary = memory.summary()
print(f"Total memories: {summary}")
```

### Using Specific Memory Types

```python
from smartmemory.memory.types.working_memory import WorkingMemory
from smartmemory.memory.types.semantic_memory import SemanticMemory

# Working memory for short-term context
working = WorkingMemory(capacity=10)
working.add(MemoryItem(content="Current conversation context"))

# Semantic memory for facts and concepts
semantic = SemanticMemory()
semantic.add(MemoryItem(
    content="Python is a high-level programming language",
    memory_type="semantic"
))
```

## Use Cases

### Conversational AI Systems
- Maintain context across multiple conversation sessions
- Learn user preferences and adapt responses
- Build comprehensive user profiles over time

### Educational Applications
- Track learning progress and adapt teaching strategies
- Remember previous topics and build upon them
- Personalize content based on individual learning patterns

### Knowledge Management
- Store and retrieve complex information relationships
- Connect related concepts across different domains
- Evolve understanding through continuous learning

### Personal AI Assistants
- Remember user preferences and past interactions
- Provide contextually relevant recommendations
- Learn from user feedback to improve responses

## Examples

The `examples/` directory contains several demonstration scripts:

- `memory_system_usage_example.py`: Basic memory operations (add, search, delete)
- `conversational_assistant_example.py`: Conversational AI with memory
- `advanced_programming_tutor.py`: Educational application example
- `working_holistic_example.py`: Comprehensive multi-session demo
- `background_processing_demo.py`: Asynchronous processing example

## Configuration

SmartMemory uses a configuration system that supports both file-based and environment variable configuration:

```python
from smartmemory.configuration.manager import ConfigManager

# Load configuration
config = ConfigManager(config_path="config.json")

# Access configuration values
graph_config = config.get("graph_db")
vector_config = config.get("vector_store")
```

### Environment Variables

Key environment variables:
- `FALKORDB_HOST`: FalkorDB server host
- `FALKORDB_PORT`: FalkorDB server port
- `REDIS_HOST`: Redis server host
- `REDIS_PORT`: Redis server port
- `OPENAI_API_KEY`: OpenAI API key for embeddings

## Memory Evolution

SmartMemory includes an evolution system that can transform memories based on configured rules:

```python
from smartmemory.evolution.flow import EvolutionFlow, EvolutionNode

# Create evolution flow
flow = EvolutionFlow()

# Add evolution nodes
node = EvolutionNode(
    node_id="enrich_semantic",
    evolver_path="smartmemory.plugins.evolvers.semantic_enricher",
    params={"threshold": 0.8}
)
flow.add_node(node)

# Execute evolution
flow.execute(memory, context={})
```

### Plugin System

SmartMemory supports custom plugins for extending functionality:

- **Evolvers**: Transform memories based on rules or conditions
- **Enrichers**: Add additional context or metadata to memories
- **Grounders**: Connect memories to external knowledge sources

Plugins are located in the `smartmemory/plugins/` directory and follow a standard interface.

## Testing

Run the test suite:

```bash
# Run all tests
PYTHONPATH=. pytest -v tests/

# Run specific test categories
PYTHONPATH=. pytest tests/unit/
PYTHONPATH=. pytest tests/integration/
PYTHONPATH=. pytest tests/e2e/

# Run examples
PYTHONPATH=. python examples/memory_system_usage_example.py
PYTHONPATH=. python examples/conversational_assistant_example.py
```

## API Reference

### SmartMemory Class

Main interface for memory operations:

```python
class SmartMemory:
    def add(self, item: MemoryItem) -> Optional[MemoryItem]
    def get(self, item_id: str) -> Optional[MemoryItem]
    def search(self, query: str, top_k: int = 10) -> List[MemoryItem]
    def delete(self, item_id: str) -> bool
    def clear(self) -> None
    def summary(self) -> Dict[str, Any]
    def ingest(self, content: str, **kwargs) -> MemoryItem
```

### MemoryItem Class

Core data structure for memory storage:

```python
@dataclass
class MemoryItem:
    content: str
    memory_type: str = 'semantic'
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    valid_start_time: Optional[datetime] = None
    valid_end_time: Optional[datetime] = None
    transaction_time: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    entities: Optional[list] = None
    relations: Optional[list] = None
    metadata: dict = field(default_factory=dict)
```

## Dependencies

SmartMemory requires the following key dependencies:

- `falkordb`: Graph database backend
- `chromadb`: Vector database for embeddings
- `spacy`: Natural language processing and entity extraction
- `litellm`: LLM integration layer
- `openai`: OpenAI API client
- `scikit-learn`: Machine learning utilities
- `redis`: Caching layer (via redis-py)
- `boto3`: AWS integration (optional)

See `requirements.txt` for the complete list of dependencies.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

## License

SmartMemory is dual-licensed to provide flexibility for both open-source and commercial use:
[LICENSE](LICENSE)

**Get started with SmartMemory by exploring the examples and documentation!**
[Full docs](https://docs.smartmemory.ai)
