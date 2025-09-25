#!/usr/bin/env python3
"""
Conversational Assistant with SmartMemory

This example demonstrates how SmartMemory enhances conversational quality through:
1. Contextual memory of previous conversations
2. Personal preference learning and adaptation
3. Topic continuity across sessions
4. Emotional context awareness
5. Progressive relationship building

The example includes specific human-evaluable tests to measure conversational quality improvements.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from smartmemory.models.memory_item import MemoryItem
    from smartmemory.memory.types.semantic_memory import SemanticMemory
    from smartmemory.memory.types.episodic_memory import EpisodicMemory
    from smartmemory.memory.types.procedural_memory import ProceduralMemory
    from smartmemory.memory.types.working_memory import WorkingMemory
    from smartmemory.similarity.framework import EnhancedSimilarityFramework
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("This example requires the full SmartMemory system to be available.")
    exit(1)


class ConversationalAssistant:
    """AI Assistant enhanced with SmartMemory for improved conversational quality."""

    def __init__(self, assistant_name: str = "Alex"):
        """Initialize the conversational assistant with memory systems."""
        self.name = assistant_name
        self.user_id = "user_001"  # In real implementation, this would be dynamic

        # Initialize memory systems
        self.semantic_memory = SemanticMemory()
        self.episodic_memory = EpisodicMemory()
        self.procedural_memory = ProceduralMemory()
        self.working_memory = WorkingMemory()
        self.similarity_framework = EnhancedSimilarityFramework()

        # Conversation tracking
        self.conversation_history = []
        self.current_session_id = str(uuid.uuid4())
        self.session_count = 0

        # Quality metrics for evaluation
        self.quality_metrics = {
            'context_continuity_score': 0.0,
            'personalization_score': 0.0,
            'emotional_awareness_score': 0.0,
            'topic_coherence_score': 0.0,
            'relationship_building_score': 0.0,
            'total_interactions': 0,
            'memory_retrievals': 0,
            'successful_personalizations': 0
        }

    def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Process user input and generate memory-enhanced response."""
        self.quality_metrics['total_interactions'] += 1
        timestamp = datetime.now()

        # Store user input in working memory
        user_memory = MemoryItem(
            content=user_input,
            memory_type="working",
            metadata={
                'speaker': 'user',
                'session_id': self.current_session_id,
                'timestamp': timestamp.isoformat(),
                'user_id': self.user_id,
                'context': context or {}
            }
        )
        self.working_memory.add(user_memory)

        # Retrieve relevant memories for context
        relevant_memories = self._retrieve_relevant_context(user_input)
        self.quality_metrics['memory_retrievals'] += len(relevant_memories)

        # Generate memory-enhanced response
        response = self._generate_response(user_input, relevant_memories, context)

        # Store assistant response
        assistant_memory = MemoryItem(
            content=response,
            memory_type="working",
            metadata={
                'speaker': 'assistant',
                'session_id': self.current_session_id,
                'timestamp': timestamp.isoformat(),
                'user_id': self.user_id,
                'relevant_memories_count': len(relevant_memories)
            }
        )
        self.working_memory.add(assistant_memory)

        # Update conversation history
        self.conversation_history.append({
            'timestamp': timestamp.isoformat(),
            'user_input': user_input,
            'assistant_response': response,
            'memories_used': len(relevant_memories),
            'context': context
        })

        return response

    def _retrieve_relevant_context(self, user_input: str, top_k: int = 5) -> List[MemoryItem]:
        """Retrieve relevant memories to inform response generation."""
        query_item = MemoryItem(
            content=user_input,
            memory_type="query",
            metadata={'user_id': self.user_id}
        )

        relevant_memories = []

        # Search across all memory types
        memory_stores = {
            'semantic': self.semantic_memory,
            'episodic': self.episodic_memory,
            'procedural': self.procedural_memory,
            'working': self.working_memory
        }

        for memory_type, store in memory_stores.items():
            try:
                # Get items from store
                all_items = []
                if hasattr(store, 'get_all_items'):
                    all_items = store.get_all_items()
                elif hasattr(store, '_graph') and hasattr(store._graph, 'nodes'):
                    for node_id in store._graph.nodes():
                        item = store.get(node_id)
                        if item and hasattr(item, 'metadata') and item.metadata.get('user_id') == self.user_id:
                            all_items.append(item)

                # Calculate similarities and get top results
                similarities = []
                for item in all_items:
                    try:
                        similarity = self.similarity_framework.calculate_similarity(query_item, item)
                        if similarity > 0.1:  # Minimum relevance threshold
                            similarities.append((similarity, item, memory_type))
                    except Exception:
                        continue

                # Add top results from this memory type
                similarities.sort(key=lambda x: x[0], reverse=True)
                relevant_memories.extend(similarities[:2])  # Top 2 per memory type

            except Exception as e:
                logger.warning(f"Memory retrieval from {memory_type} failed: {e}")

        # Sort all memories by relevance and return top_k
        relevant_memories.sort(key=lambda x: x[0], reverse=True)
        return [(item, memory_type) for _, item, memory_type in relevant_memories[:top_k]]

    def _generate_response(self, user_input: str, relevant_memories: List[Tuple], context: Dict = None) -> str:
        """Generate response using retrieved memories for enhanced quality."""
        # This is a simplified response generation - in practice would use LLM

        # Base response templates
        responses = {
            'greeting': f"Hello! I'm {self.name}, your AI assistant.",
            'question': "That's an interesting question. Let me think about that.",
            'preference': "I'll remember that preference for next time.",
            'continuation': "Continuing from our previous conversation,",
            'personalized': "Based on what I know about you,"
        }

        user_lower = user_input.lower()

        # Check for personalization opportunities
        personal_memories = [mem for mem, mem_type in relevant_memories if 'preference' in str(mem.metadata)]
        if personal_memories:
            self.quality_metrics['successful_personalizations'] += 1
            self.quality_metrics['personalization_score'] += 0.2

        # Generate contextual response
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            if relevant_memories:
                return f"Hello again! I remember we were discussing {self._extract_topic_from_memories(relevant_memories)}. How can I help you today?"
            return responses['greeting']

        elif '?' in user_input:
            if relevant_memories:
                context_info = self._summarize_relevant_context(relevant_memories)
                return f"Based on our previous conversations about {context_info}, here's what I think: {responses['question']}"
            return responses['question']

        elif any(word in user_lower for word in ['like', 'prefer', 'favorite']):
            self._store_user_preference(user_input)
            return f"{responses['preference']} I've noted your preference about this topic."

        elif relevant_memories:
            self.quality_metrics['context_continuity_score'] += 0.1
            topic = self._extract_topic_from_memories(relevant_memories)
            return f"{responses['continuation']} I recall you mentioned {topic}. How would you like to explore this further?"

        return "I'm here to help! What would you like to discuss?"

    def _extract_topic_from_memories(self, relevant_memories: List[Tuple]) -> str:
        """Extract main topic from relevant memories."""
        if not relevant_memories:
            return "various topics"

        # Simple topic extraction - in practice would use more sophisticated NLP
        memory_contents = [mem.content for mem, _ in relevant_memories]
        common_words = []
        for content in memory_contents:
            words = content.lower().split()
            common_words.extend([w for w in words if len(w) > 4])

        if common_words:
            # Return most common meaningful word
            word_counts = {}
            for word in common_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            return max(word_counts.items(), key=lambda x: x[1])[0]

        return "our previous discussion"

    def _summarize_relevant_context(self, relevant_memories: List[Tuple]) -> str:
        """Summarize context from relevant memories."""
        if not relevant_memories:
            return "general topics"

        memory_types = [mem_type for _, mem_type in relevant_memories]
        if 'episodic' in memory_types:
            return "your past experiences"
        elif 'semantic' in memory_types:
            return "the concepts we've discussed"
        elif 'procedural' in memory_types:
            return "the methods we've explored"
        else:
            return "our recent conversation"

    def _store_user_preference(self, user_input: str):
        """Store user preference in semantic memory."""
        preference_item = MemoryItem(
            content=user_input,
            memory_type="semantic",
            metadata={
                'type': 'user_preference',
                'user_id': self.user_id,
                'session_id': self.current_session_id,
                'timestamp': datetime.now().isoformat(),
                'category': 'preference'
            }
        )
        self.semantic_memory.add(preference_item)

    def end_session(self):
        """End current session and consolidate memories."""
        self.session_count += 1

        # Move important working memories to long-term storage
        working_items = []
        try:
            if hasattr(self.working_memory, 'get_all_items'):
                working_items = self.working_memory.get_all_items()
        except Exception:
            pass

        for item in working_items:
            if hasattr(item, 'metadata') and item.metadata.get('session_id') == self.current_session_id:
                # Convert to episodic memory
                episodic_item = MemoryItem(
                    content=f"Session {self.session_count}: {item.content}",
                    memory_type="episodic",
                    metadata={
                        **item.metadata,
                        'event_type': 'conversation',
                        'session_number': self.session_count,
                        'consolidated_from': 'working_memory'
                    }
                )
                self.episodic_memory.add(episodic_item)

        # Start new session
        self.current_session_id = str(uuid.uuid4())
        logger.info(f"Session {self.session_count} ended. Starting new session: {self.current_session_id}")


class ConversationalQualityTester:
    """Test suite for evaluating conversational quality improvements."""

    def __init__(self, assistant: ConversationalAssistant):
        self.assistant = assistant
        self.test_results = {}

    def run_human_evaluable_tests(self) -> Dict[str, Any]:
        """Run tests that humans can easily evaluate for quality."""
        logger.info("=== RUNNING HUMAN-EVALUABLE CONVERSATIONAL QUALITY TESTS ===")

        test_scenarios = [
            self.test_context_continuity(),
            self.test_personalization_learning(),
            self.test_multi_session_memory(),
            self.test_topic_coherence(),
            self.test_preference_retention()
        ]

        # Aggregate results
        total_score = sum(result['score'] for result in test_scenarios)
        max_score = len(test_scenarios)
        overall_quality = total_score / max_score

        self.test_results = {
            'individual_tests': test_scenarios,
            'overall_quality_score': overall_quality,
            'quality_rating': self._get_quality_rating(overall_quality),
            'human_evaluation_instructions': self._generate_human_evaluation_guide(),
            'assistant_metrics': self.assistant.quality_metrics
        }

        return self.test_results

    def test_context_continuity(self) -> Dict[str, Any]:
        """Test: Does the assistant maintain context across conversation turns?"""
        logger.info("Testing context continuity...")

        # Conversation sequence
        responses = []
        responses.append(self.assistant.process_user_input("I'm working on a Python project"))
        responses.append(self.assistant.process_user_input("What's the best way to handle errors?"))
        responses.append(self.assistant.process_user_input("Can you give me an example?"))

        # Evaluate context continuity
        context_maintained = any("python" in resp.lower() or "project" in resp.lower() for resp in responses[1:])

        return {
            'test_name': 'Context Continuity',
            'description': 'Maintains conversation context across turns',
            'conversation': [
                "User: I'm working on a Python project",
                f"Assistant: {responses[0]}",
                "User: What's the best way to handle errors?",
                f"Assistant: {responses[1]}",
                "User: Can you give me an example?",
                f"Assistant: {responses[2]}"
            ],
            'score': 1.0 if context_maintained else 0.0,
            'human_evaluation': 'Does the assistant remember the Python project context in later responses?'
        }

    def test_personalization_learning(self) -> Dict[str, Any]:
        """Test: Does the assistant learn and apply user preferences?"""
        logger.info("Testing personalization learning...")

        # Set preference and test recall
        pref_response = self.assistant.process_user_input("I prefer detailed technical explanations")
        recall_response = self.assistant.process_user_input("How does machine learning work?")

        # Check if preference is acknowledged and applied
        preference_learned = "preference" in pref_response.lower() or "remember" in pref_response.lower()
        preference_applied = "detailed" in recall_response.lower() or "technical" in recall_response.lower()

        score = 0.5 * preference_learned + 0.5 * preference_applied

        return {
            'test_name': 'Personalization Learning',
            'description': 'Learns and applies user preferences',
            'conversation': [
                "User: I prefer detailed technical explanations",
                f"Assistant: {pref_response}",
                "User: How does machine learning work?",
                f"Assistant: {recall_response}"
            ],
            'score': score,
            'human_evaluation': 'Does the assistant acknowledge the preference and apply it in the follow-up response?'
        }

    def test_multi_session_memory(self) -> Dict[str, Any]:
        """Test: Does the assistant remember across sessions?"""
        logger.info("Testing multi-session memory...")

        # Session 1
        session1_response = self.assistant.process_user_input("My name is Sarah and I love astronomy")
        self.assistant.end_session()

        # Session 2 - different day
        session2_response = self.assistant.process_user_input("Hello, it's me again")

        # Check if assistant remembers user
        remembers_user = "sarah" in session2_response.lower() or "astronomy" in session2_response.lower()

        return {
            'test_name': 'Multi-Session Memory',
            'description': 'Remembers information across different sessions',
            'conversation': [
                "Session 1 - User: My name is Sarah and I love astronomy",
                f"Session 1 - Assistant: {session1_response}",
                "[SESSION ENDED]",
                "Session 2 - User: Hello, it's me again",
                f"Session 2 - Assistant: {session2_response}"
            ],
            'score': 1.0 if remembers_user else 0.0,
            'human_evaluation': 'Does the assistant remember Sarah and her interest in astronomy in the new session?'
        }

    def test_topic_coherence(self) -> Dict[str, Any]:
        """Test: Does the assistant maintain topic coherence?"""
        logger.info("Testing topic coherence...")

        # Build up topic discussion
        responses = []
        responses.append(self.assistant.process_user_input("I'm interested in renewable energy"))
        responses.append(self.assistant.process_user_input("What about solar panels?"))
        responses.append(self.assistant.process_user_input("How efficient are they?"))

        # Check topic coherence
        energy_mentioned = sum(1 for resp in responses if any(word in resp.lower() for word in ['energy', 'solar', 'renewable']))
        coherence_score = min(1.0, energy_mentioned / len(responses))

        return {
            'test_name': 'Topic Coherence',
            'description': 'Maintains coherent topic discussion',
            'conversation': [
                "User: I'm interested in renewable energy",
                f"Assistant: {responses[0]}",
                "User: What about solar panels?",
                f"Assistant: {responses[1]}",
                "User: How efficient are they?",
                f"Assistant: {responses[2]}"
            ],
            'score': coherence_score,
            'human_evaluation': 'Does the conversation stay focused on renewable energy topics throughout?'
        }

    def test_preference_retention(self) -> Dict[str, Any]:
        """Test: Does the assistant retain and use preferences over time?"""
        logger.info("Testing preference retention...")

        # Set multiple preferences
        self.assistant.process_user_input("I like concise answers")
        self.assistant.process_user_input("I'm interested in AI and robotics")

        # Test retention after some conversation
        self.assistant.process_user_input("Tell me about something interesting")
        final_response = self.assistant.process_user_input("What do you think I'd find fascinating?")

        # Check if preferences are reflected
        concise_style = len(final_response.split()) < 50  # Concise response
        relevant_topic = any(word in final_response.lower() for word in ['ai', 'artificial', 'robot', 'technology'])

        score = 0.5 * concise_style + 0.5 * relevant_topic

        return {
            'test_name': 'Preference Retention',
            'description': 'Retains and applies multiple user preferences',
            'conversation': [
                "User: I like concise answers",
                "User: I'm interested in AI and robotics",
                "User: Tell me about something interesting",
                "User: What do you think I'd find fascinating?",
                f"Assistant: {final_response}"
            ],
            'score': score,
            'human_evaluation': 'Is the final response both concise and related to AI/robotics interests?'
        }

    def _get_quality_rating(self, score: float) -> str:
        """Convert numerical score to quality rating."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Needs Improvement"

    def _generate_human_evaluation_guide(self) -> Dict[str, str]:
        """Generate guide for human evaluators."""
        return {
            'instructions': 'Read each conversation and evaluate whether the assistant demonstrates the described capability.',
            'rating_scale': '1.0 = Clearly demonstrates capability, 0.5 = Partially demonstrates, 0.0 = Does not demonstrate',
            'focus_areas': [
                'Context awareness across conversation turns',
                'Learning and applying user preferences',
                'Remembering information across sessions',
                'Maintaining topic coherence',
                'Personalizing responses based on stated preferences'
            ],
            'evaluation_questions': [
                'Does the assistant feel more helpful with memory than without?',
                'Would you prefer this assistant over one without memory?',
                'Does the conversation feel more natural and personalized?',
                'Are the responses appropriately contextual?'
            ]
        }


def main():
    """Main demonstration function."""
    logger.info("Starting Conversational Assistant with SmartMemory Demo")
    logger.info("=" * 70)

    # Initialize assistant
    assistant = ConversationalAssistant("Alex")

    # Run quality tests
    tester = ConversationalQualityTester(assistant)
    results = tester.run_human_evaluable_tests()

    # Display results
    logger.info("=" * 70)
    logger.info("CONVERSATIONAL QUALITY TEST RESULTS")
    logger.info("=" * 70)

    for test in results['individual_tests']:
        logger.info(f"\n{test['test_name']}: {test['score']:.1f}/1.0")
        logger.info(f"Description: {test['description']}")
        logger.info("Sample Conversation:")
        for turn in test['conversation']:
            logger.info(f"  {turn}")
        logger.info(f"Human Evaluation: {test['human_evaluation']}")

    logger.info(f"\nOVERALL QUALITY SCORE: {results['overall_quality_score']:.1f}/1.0")
    logger.info(f"QUALITY RATING: {results['quality_rating']}")

    # Save detailed results
    with open('conversational_quality_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("\nDetailed results saved to conversational_quality_results.json")

    # Human evaluation instructions
    logger.info("\n" + "=" * 70)
    logger.info("HUMAN EVALUATION INSTRUCTIONS")
    logger.info("=" * 70)
    logger.info(results['human_evaluation_instructions']['instructions'])
    logger.info(f"Rating Scale: {results['human_evaluation_instructions']['rating_scale']}")

    print(f"\n✅ Conversational Quality Demo completed!")
    print(f"Overall Quality: {results['quality_rating']} ({results['overall_quality_score']:.1%})")
    print(f"Memory-Enhanced Interactions: {assistant.quality_metrics['total_interactions']}")
    print(f"Successful Memory Retrievals: {assistant.quality_metrics['memory_retrievals']}")

    return results


if __name__ == "__main__":
    main()
