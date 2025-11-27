"""
Core Logic: MemoryController

Implements Async Non-Blocking I/O using `run_in_executor` and Batch-Saving strategy.
"""

import asyncio
from typing import List
from ..storage.engine import StorageManager
from ..utils.llm import LLMService
from ..models.note import AtomicNote, NoteInput, SearchResult

class MemoryController:
    def __init__(self):
        self.storage = StorageManager()
        self.llm = LLMService()

    async def create_note(self, input_data: NoteInput) -> str:
        """
        Phase 1: Creation. 
        Critical I/O operations are offloaded to threads.
        """
        loop = asyncio.get_running_loop()

        # 1. CPU-Bound / Network Ops (LLM)
        # LLM calls sind meist intern async oder schnell genug, hier sync wrapper
        metadata = await loop.run_in_executor(None, self.llm.extract_metadata, input_data.content)
        
        # 2. Objekt Erstellung
        note = AtomicNote(
            content=input_data.content,
            contextual_summary=metadata.get("summary", ""),
            keywords=metadata.get("keywords", []),
            tags=metadata.get("tags", [])
        )
        
        # 3. Embedding calculation (Paper Section 3.1, Formula 3):
        # ei = fenc[concat(ci, Ki, Gi, Xi)]
        # Concatenation of all text components for complete semantic representation
        text_for_embedding = f"{note.content} {note.contextual_summary} {' '.join(note.keywords)} {' '.join(note.tags)}"
        embedding = await loop.run_in_executor(None, self.llm.get_embedding, text_for_embedding)
        
        # 4. Blocking I/O Offloading (Storage)
        await loop.run_in_executor(None, self.storage.vector.add, note, embedding)
        await loop.run_in_executor(None, self.storage.graph.add_node, note)
        
        # Explicit snapshot save after adding
        await loop.run_in_executor(None, self.storage.graph.save_snapshot)
        
        # 5. Background Evolution
        asyncio.create_task(self._evolve_memory(note, embedding))
        
        return note.id

    async def _evolve_memory(self, new_note: AtomicNote, embedding: List[float]):
        """
        Phase 2: Asynchronous Knowledge Evolution.
        Batch-Update strategy for the graph.
        """
        loop = asyncio.get_running_loop()
        print(f"🔄 Evolving memory for note {new_note.id}...")
        
        try:
            # 1. Candidate search (I/O in thread)
            candidate_ids, distances = await loop.run_in_executor(
                None, self.storage.vector.query, embedding, 5
            )
            
            links_found = 0
            evolutions_found = 0
            candidate_notes = []
            
            # 2. Linking logic + Memory Evolution
            for c_id, dist in zip(candidate_ids, distances):
                if c_id == new_note.id: continue
                
                candidate_note = self.storage.get_note(c_id)
                if not candidate_note: continue
                
                candidate_notes.append(candidate_note)

                # LLM Check (Network I/O)
                # In Production sollte check_link auch async sein, hier wrapper
                is_related, relation = await loop.run_in_executor(
                    None, self.llm.check_link, new_note, candidate_note
                )
                
                if is_related and relation:
                    print(f"🔗 Linking {new_note.id} -> {c_id} ({relation.relation_type})")
                    # In-Memory Update (fast)
                    self.storage.graph.add_edge(relation)
                    links_found += 1
            
            # 3. Memory Evolution (Paper Section 3.3)
            # Check if existing memories should be updated
            for candidate_note in candidate_notes:
                evolved_note = await loop.run_in_executor(
                    None, self.llm.evolve_memory, new_note, candidate_note
                )
                
                if evolved_note:
                    print(f"🧠 Evolving memory {candidate_note.id} based on new information")
                    
                    # Calculate new embedding (Paper Section 3.1, Formula 3):
                    # ei = fenc[concat(ci, Ki, Gi, Xi)]
                    # Concatenation of all text components including tags
                    evolved_text = f"{evolved_note.content} {evolved_note.contextual_summary} {' '.join(evolved_note.keywords)} {' '.join(evolved_note.tags)}"
                    new_embedding = await loop.run_in_executor(
                        None, self.llm.get_embedding, evolved_text
                    )
                    
                    # Update in VectorStore
                    await loop.run_in_executor(
                        None, self.storage.vector.update, 
                        candidate_note.id, evolved_note, new_embedding
                    )
                    
                    # Update in GraphStore
                    await loop.run_in_executor(
                        None, self.storage.graph.update_node, evolved_note
                    )
                    
                    evolutions_found += 1
            
            # 4. Batch Save (Single write to disk)
            if links_found > 0 or evolutions_found > 0:
                await loop.run_in_executor(None, self.storage.graph.save_snapshot)
                print(f"✅ Evolution finished. {links_found} links, {evolutions_found} memory updates saved.")
            else:
                print("✅ Evolution finished. No new links or updates.")

        except Exception as e:
            print(f"❌ Evolution failed for {new_note.id}: {e}")

    async def retrieve(self, query: str) -> List[SearchResult]:
        loop = asyncio.get_running_loop()
        
        # Embedding calculation
        q_embedding = await loop.run_in_executor(None, self.llm.get_embedding, query)
        
        # Vector Query
        ids, scores = await loop.run_in_executor(None, self.storage.vector.query, q_embedding, 5)
        
        results = []
        for n_id, score in zip(ids, scores):
            note = self.storage.get_note(n_id)
            if not note: continue
            
            # Graph Traversal (In-Memory, fast enough for Main Thread)
            neighbors_data = self.storage.graph.get_neighbors(n_id)
            related_notes = []
            for n in neighbors_data:
                # Validate and filter invalid nodes
                if not n or not isinstance(n, dict):
                    continue
                # Check if content is present (required field)
                if "content" not in n or not n.get("content"):
                    continue
                try:
                    related_note = AtomicNote(**n)
                    related_notes.append(related_note)
                except Exception as e:
                    # Skip invalid nodes (e.g., corrupted by evolution)
                    print(f"Warning: Skipping invalid neighbor node: {e}")
                    continue
            
            results.append(SearchResult(
                note=note,
                score=score,
                related_notes=related_notes
            ))
            
        return results
    
    async def delete_note(self, note_id: str) -> bool:
        """Deletes a note from Graph and Vector Store."""
        loop = asyncio.get_running_loop()
        
        # Check if note exists in graph (directly, not via get_note)
        note_exists = await loop.run_in_executor(
            None, lambda: note_id in self.storage.graph.graph
        )
        if not note_exists:
            return False
        
        # Delete from both stores (in thread)
        success = await loop.run_in_executor(
            None, self.storage.delete_note, note_id
        )
        
        if success:
            # Save graph snapshot after deletion
            await loop.run_in_executor(None, self.storage.graph.save_snapshot)
        
        return success
    
    async def reset_memory(self) -> bool:
        """Resets the complete memory system (Graph + Vector Store)."""
        loop = asyncio.get_running_loop()
        
        try:
            # Reset in Thread (blocking I/O)
            await loop.run_in_executor(None, self.storage.reset)
            return True
        except Exception as e:
            print(f"Error resetting memory: {e}")
            return False

