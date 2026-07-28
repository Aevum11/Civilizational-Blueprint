"""
Exception Theory Batch 3 Classes
Distributed Consciousness (The Code of Connection)

Implements distributed systems patterns using Exception Theory:
- SwarmConsensus: Byzantine consensus via variance minimization
- PrecognitiveCache: Trajectory extrapolation for negative latency
- ImmortalSupervisor: Homeostatic crash recovery
- SemanticManifold: Meaning as geometric proximity
- VarianceLimiter: Entropy-based adaptive rate limiting
- ProofOfTraversal: Anti-spam hashcash protocol
- EphemeralVault: Perfect forward secrecy encryption
- ConsistentHashingRing: Sharded DHT topology
- TimeTraveler: Event sourcing with undo/redo
- FractalReality: Procedural world generation

From: "For every exception there is an exception, except the exception."

Author: Derived from M.J.M.'s Exception Theory
"""

import hashlib
import time
import random
import threading
import os
import math
from typing import List, Optional, Dict, Any, Callable, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..core.constants import (
    DEFAULT_SWARM_COHERENCE,
    DEFAULT_SWARM_ALIGNMENT_BONUS,
    DEFAULT_SWARM_STABILITY_BONUS,
    PRECOG_HISTORY_SIZE,
    PRECOG_PROBABILITY_THRESHOLD,
    DEFAULT_VARIANCE_CAPACITY,
    DEFAULT_VARIANCE_REFILL_RATE,
    DEFAULT_POT_DIFFICULTY,
    DEFAULT_HASH_RING_REPLICAS,
    FRACTAL_DEFAULT_OCTAVES,
    FRACTAL_DEFAULT_PERSISTENCE,
)
from ..core.mathematics import ETMathV2


# ============================================================================
# Eq 21: SwarmConsensus (The Gravity Protocol)
# ============================================================================

class SwarmConsensus:
    """
    Batch 3, Eq 21: Byzantine consensus without voting.
    
    ET Math: S_truth = argmin_S(Σ Variance(P_i, S))
             Weight(S) = 1/V_global
    
    Nodes drift toward Maximum Coherence (heaviest Descriptor density)
    via variance minimization - data gravity pulling cluster to truth.
    """
    
    def __init__(self, node_id: str, initial_data: Any):
        """
        Initialize a swarm node.
        
        Args:
            node_id: Unique identifier for this node
            initial_data: Initial state data
        """
        self.node_id = node_id
        self.data = initial_data
        self.coherence_score = DEFAULT_SWARM_COHERENCE
        self._descriptor_hash = self._compute_hash(initial_data)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute descriptor hash of data."""
        return hashlib.sha256(str(data).encode()).hexdigest()
    
    def gossip(self, neighbors: List['SwarmConsensus']) -> Dict[str, Any]:
        """
        Execute one round of gossip protocol.
        
        ET Math: Listen to neighbors, drift toward heaviest Descriptor.
        
        Args:
            neighbors: List of peer nodes to gossip with
        
        Returns:
            Dict with alignment result
        """
        # Calculate local descriptor
        my_d = self._descriptor_hash
        
        # Survey the manifold (listen to neighbors)
        votes = Counter()
        votes[my_d] += self.coherence_score
        
        for neighbor in neighbors:
            n_d = neighbor._descriptor_hash
            votes[n_d] += neighbor.coherence_score
        
        # Variance minimization (gravity)
        if votes:
            consensus_d, weight = votes.most_common(1)[0]
            
            if my_d != consensus_d:
                # Detect incoherence and align
                # Find the data corresponding to consensus descriptor
                for neighbor in neighbors:
                    if neighbor._descriptor_hash == consensus_d:
                        self.data = neighbor.data
                        self._descriptor_hash = consensus_d
                        self.coherence_score += DEFAULT_SWARM_ALIGNMENT_BONUS
                        return {
                            'aligned': True,
                            'to_hash': consensus_d[:8],
                            'weight': weight,
                            'new_coherence': self.coherence_score
                        }
            else:
                # Reinforce stability
                self.coherence_score += DEFAULT_SWARM_STABILITY_BONUS
        
        return {
            'aligned': False,
            'coherence': self.coherence_score
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current node state."""
        return {
            'node_id': self.node_id,
            'data': self.data,
            'hash': self._descriptor_hash,
            'coherence': self.coherence_score
        }


# ============================================================================
# Eq 22: PrecognitiveCache (Trajectory Extrapolation)
# ============================================================================

class PrecognitiveCache:
    """
    Batch 3, Eq 22: Teleological caching via T-momentum.
    
    ET Math: P_next ≈ P_current + v_T·Δt + ½a_T·Δt²
    
    Predicts next required Point before request via velocity/acceleration
    calculation, achieving negative latency.
    """
    
    def __init__(self, max_history: int = PRECOG_HISTORY_SIZE):
        """
        Initialize precognitive cache.
        
        Args:
            max_history: Maximum history size for trajectory calculation
        """
        self.history = []
        self.cache = {}
        self.max_history = max_history
        self._hits = 0
        self._misses = 0
        self._predictions = 0
    
    def access(self, resource_id: Any, fetch_func: Optional[Callable] = None) -> Any:
        """
        Access a resource with trajectory prediction.
        
        Args:
            resource_id: Resource identifier
            fetch_func: Function to fetch if not cached
        
        Returns:
            Resource data
        """
        # Update history
        self.history.append(resource_id)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Predict future accesses
        self._predict_and_prefetch(fetch_func)
        
        # Check cache
        if resource_id in self.cache:
            self._hits += 1
            return self.cache[resource_id]
        else:
            self._misses += 1
            if fetch_func:
                data = fetch_func(resource_id)
                self.cache[resource_id] = data
                return data
            return None
    
    def _predict_and_prefetch(self, fetch_func: Optional[Callable] = None):
        """
        Predict next resources and prefetch.
        
        ET Math: Uses trajectory_extrapolate from ETMathV2
        """
        if len(self.history) < 2:
            return
        
        # Use ET's trajectory extrapolation
        predicted = ETMathV2.trajectory_extrapolate(self.history, delta_t=1.0)
        
        # Convert to integer if dealing with numeric IDs
        if all(isinstance(x, (int, float)) for x in self.history):
            predicted = int(round(predicted))
        
        # Prefetch predicted resource
        if predicted not in self.cache and fetch_func:
            self._predictions += 1
            self.cache[predicted] = fetch_func(predicted)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'predictions': self._predictions,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


# ============================================================================
# Eq 23: ImmortalSupervisor (Homeostatic Restoration)
# ============================================================================

class ImmortalSupervisor:
    """
    Batch 3, Eq 23: Supervisor tree for infinite uptime.
    
    ET Math: S_worker ∈ I ⟹ Kill(S_worker) ∧ Spawn(P_template)
             Uptime → ∞
    
    When worker becomes Incoherent (crashes), supervisor kills and spawns
    fresh grounded instance, maintaining system homeostasis.
    """
    
    def __init__(self, 
                 name: str,
                 target_func: Callable,
                 args: Tuple = (),
                 max_restarts: int = -1,
                 cooldown: float = 1.0):
        """
        Initialize immortal supervisor.
        
        Args:
            name: Supervisor identifier
            target_func: Worker function to supervise
            args: Arguments for worker function
            max_restarts: Maximum restart count (-1 for infinite)
            cooldown: Seconds to wait between restarts
        """
        self.name = name
        self.target_func = target_func
        self.args = args
        self.max_restarts = max_restarts
        self.cooldown = cooldown
        
        self._restarts = 0
        self._worker_thread = None
        self._running = False
        self._lock = threading.Lock()
    
    def start(self):
        """Start supervising the worker."""
        self._running = True
        self._spawn_worker()
    
    def stop(self):
        """Stop the supervisor."""
        self._running = False
    
    def _spawn_worker(self):
        """Spawn a new worker thread."""
        def worker_wrapper():
            try:
                self.target_func(*self.args)
            except Exception as e:
                # Worker became Incoherent
                with self._lock:
                    if self._running:
                        if self.max_restarts == -1 or self._restarts < self.max_restarts:
                            time.sleep(self.cooldown)
                            self._restarts += 1
                            # Spawn fresh grounded instance
                            self._spawn_worker()
        
        self._worker_thread = threading.Thread(target=worker_wrapper, daemon=True)
        self._worker_thread.start()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get supervisor metrics."""
        return {
            'name': self.name,
            'restarts': self._restarts,
            'running': self._running,
            'alive': self._worker_thread.is_alive() if self._worker_thread else False
        }


# ============================================================================
# Eq 24: SemanticManifold (The Meaning Manifold)
# ============================================================================

class SemanticManifold:
    """
    Batch 3, Eq 24: Meaning as geometric proximity.
    
    ET Math: θ = arccos((D_A·D_B)/(|D_A||D_B|))
             Similarity = 1 - θ/π
    
    Words are coordinates, not strings. Cosine similarity measures
    geodesic distance between concepts in Descriptor Space.
    """
    
    def __init__(self, name: str):
        """
        Initialize semantic manifold.
        
        Args:
            name: Manifold identifier
        """
        self.name = name
        self.embeddings: Dict[str, List[float]] = {}
    
    def bind(self, word: str, vector: List[float]):
        """
        Bind a word to its semantic vector.
        
        Args:
            word: Word or concept
            vector: Descriptor vector (embedding)
        """
        self.embeddings[word] = vector
    
    def bind_batch(self, word_vectors: Dict[str, List[float]]):
        """
        Bind multiple words at once.
        
        Args:
            word_vectors: Dict mapping words to vectors
        """
        self.embeddings.update(word_vectors)
    
    def search(self, query_word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar words to query.
        
        Args:
            query_word: Word to search for
            top_k: Number of results to return
        
        Returns:
            List of (word, similarity) tuples, sorted by similarity
        """
        if query_word not in self.embeddings:
            return []
        
        query_vec = self.embeddings[query_word]
        similarities = []
        
        for word, vec in self.embeddings.items():
            if word != query_word:
                sim = ETMathV2.cosine_similarity(query_vec, vec)
                similarities.append((word, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analogy(self, word_a: str, word_b: str, word_c: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Solve analogy: A:B :: C:?
        
        ET Math: D_? ≈ D_B - D_A + D_C
        
        Args:
            word_a: First word in analogy
            word_b: Second word in analogy
            word_c: Third word in analogy
            top_k: Number of results
        
        Returns:
            List of candidate words with similarities
        """
        if not all(w in self.embeddings for w in [word_a, word_b, word_c]):
            return []
        
        vec_a = self.embeddings[word_a]
        vec_b = self.embeddings[word_b]
        vec_c = self.embeddings[word_c]
        
        # Calculate target vector: B - A + C
        target = [b - a + c for a, b, c in zip(vec_a, vec_b, vec_c)]
        
        # Find closest matches
        similarities = []
        for word, vec in self.embeddings.items():
            if word not in [word_a, word_b, word_c]:
                sim = ETMathV2.cosine_similarity(target, vec)
                similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ============================================================================
# Eq 25: VarianceLimiter (The Variance Cost)
# ============================================================================

class VarianceLimiter:
    """
    Batch 3, Eq 25: Entropy-based adaptive rate limiting.
    
    ET Math: V_cost(Req) = Complexity(D_req)^1.5
    
    Complex queries cost more substrate than simple pings.
    Users have Variance Budget depleted by operation complexity.
    """
    
    def __init__(self, 
                 capacity: float = DEFAULT_VARIANCE_CAPACITY,
                 refill_rate: float = DEFAULT_VARIANCE_REFILL_RATE):
        """
        Initialize variance limiter.
        
        Args:
            capacity: Maximum variance budget
            refill_rate: Tokens per second refill rate
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.current = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def request(self, complexity: float) -> bool:
        """
        Request permission to perform operation.
        
        Args:
            complexity: Operation complexity measure
        
        Returns:
            True if permitted, False if budget exceeded
        """
        with self._lock:
            self._refill()
            
            cost = ETMathV2.variance_cost(complexity)
            
            if self.current >= cost:
                self.current -= cost
                return True
            else:
                return False
    
    def _refill(self):
        """Refill variance budget based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        
        self.current = min(self.capacity, self.current + refill_amount)
        self.last_refill = now
    
    def get_remaining(self) -> float:
        """Get remaining variance budget."""
        with self._lock:
            self._refill()
            return self.current


# ============================================================================
# Eq 26: ProofOfTraversal (Anti-Spam)
# ============================================================================

class ProofOfTraversal:
    """
    Batch 3, Eq 26: Hashcash-style proof of work.
    
    ET Math: Find n s.t. Hash(D_msg + n) < Target_difficulty
    
    Sender must prove T-Traversal (CPU work) by finding nonce that
    binds message to hash target. Makes spam computationally expensive.
    """
    
    def __init__(self, name: str, difficulty: int = DEFAULT_POT_DIFFICULTY):
        """
        Initialize proof of traversal validator.
        
        Args:
            name: Validator identifier
            difficulty: Number of leading zeros required in hash
        """
        self.name = name
        self.difficulty = difficulty
    
    def mint_stamp(self, message: str) -> Tuple[int, str]:
        """
        Mine a proof of traversal stamp.
        
        Args:
            message: Message to stamp
        
        Returns:
            Tuple of (nonce, hash)
        """
        nonce = 0
        target = ETMathV2.proof_of_traversal_target(self.difficulty)
        
        while True:
            candidate = f"{message}:{nonce}"
            hash_val = hashlib.sha256(candidate.encode()).hexdigest()
            
            if hash_val.startswith(target):
                return (nonce, hash_val)
            
            nonce += 1
    
    def verify(self, message: str, nonce: int) -> bool:
        """
        Verify a proof of traversal stamp.
        
        Args:
            message: Original message
            nonce: Claimed nonce
        
        Returns:
            True if proof is valid
        """
        return ETMathV2.verify_traversal_proof(message, nonce, self.difficulty)


# ============================================================================
# Eq 27: EphemeralVault (Perfect Forward Secrecy)
# ============================================================================

class EphemeralVault:
    """
    Batch 3, Eq 27: Perfect forward secrecy encryption.
    
    ET Math: P_encrypted = P_clear ⊕ K_session
             Destroy(K_session) after retrieval
    
    One-time pad encryption where keys are destroyed after use,
    ensuring perfect forward secrecy.
    """
    
    def __init__(self, name: str):
        """
        Initialize ephemeral vault.
        
        Args:
            name: Vault identifier
        """
        self.name = name
        self._vault: Dict[str, bytes] = {}
    
    def store(self, key: str, secret: str) -> bytes:
        """
        Store a secret with ephemeral encryption.
        
        Args:
            key: Storage key
            secret: Secret to store
        
        Returns:
            One-time pad (save this separately!)
        """
        secret_bytes = secret.encode('utf-8')
        
        # Generate one-time pad
        pad = os.urandom(len(secret_bytes))
        
        # XOR encryption
        encrypted = ETMathV2.ephemeral_bind(secret_bytes, pad)
        
        # Store encrypted data
        self._vault[key] = encrypted
        
        # Return pad (user must save this!)
        return pad
    
    def retrieve(self, key: str, pad: bytes) -> Optional[str]:
        """
        Retrieve and destroy a secret.
        
        Args:
            key: Storage key
            pad: One-time pad from storage
        
        Returns:
            Decrypted secret (or None if not found)
        """
        if key not in self._vault:
            return None
        
        encrypted = self._vault[key]
        
        # XOR decryption
        decrypted_bytes = ETMathV2.ephemeral_bind(encrypted, pad)
        
        # DESTROY the stored data (perfect forward secrecy)
        del self._vault[key]
        
        return decrypted_bytes.decode('utf-8')
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the vault."""
        return key in self._vault


# ============================================================================
# Eq 28: ConsistentHashingRing (Sharded DHT)
# ============================================================================

class ConsistentHashingRing:
    """
    Batch 3, Eq 28: Consistent hashing for distributed hash tables.
    
    ET Math: Node(P) = Hash(P) mod N_nodes
    
    Maps keys to nodes in a way that minimizes redistribution when
    nodes are added/removed. Uses virtual nodes for load balancing.
    """
    
    def __init__(self, name: str, replicas: int = DEFAULT_HASH_RING_REPLICAS):
        """
        Initialize consistent hashing ring.
        
        Args:
            name: Ring identifier
            replicas: Number of virtual nodes per physical node
        """
        self.name = name
        self.replicas = replicas
        self.ring: Dict[int, str] = {}  # hash -> node_id
        self.nodes: Set[str] = set()
        self._sorted_hashes: List[int] = []
    
    def add_node(self, node_id: str):
        """
        Add a node to the ring.
        
        Args:
            node_id: Node identifier
        """
        self.nodes.add(node_id)
        
        # Add virtual nodes
        for i in range(self.replicas):
            virtual_key = f"{node_id}:{i}"
            hash_val = ETMathV2.consistent_hash(virtual_key)
            self.ring[hash_val] = node_id
        
        # Update sorted hash list
        self._sorted_hashes = sorted(self.ring.keys())
    
    def remove_node(self, node_id: str):
        """
        Remove a node from the ring.
        
        Args:
            node_id: Node identifier
        """
        if node_id not in self.nodes:
            return
        
        self.nodes.remove(node_id)
        
        # Remove virtual nodes
        hashes_to_remove = [h for h, n in self.ring.items() if n == node_id]
        for h in hashes_to_remove:
            del self.ring[h]
        
        # Update sorted hash list
        self._sorted_hashes = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """
        Find which node should handle this key.
        
        Args:
            key: Key to route
        
        Returns:
            Node ID that should handle this key
        """
        if not self.ring:
            return None
        
        hash_val = ETMathV2.consistent_hash(key)
        
        # Find first node >= hash_val (clockwise on ring)
        for h in self._sorted_hashes:
            if h >= hash_val:
                return self.ring[h]
        
        # Wrap around to first node
        return self.ring[self._sorted_hashes[0]]


# ============================================================================
# Eq 29: TimeTraveler (Event Sourcing)
# ============================================================================

class TimeTraveler:
    """
    Batch 3, Eq 29: Event sourcing with undo/redo.
    
    ET Math: D_delta = (key, old, new)
             S_t = S_{t-1} ∘ D_t
    
    Every state change is a Delta. Time travel = apply/reverse deltas.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize time traveler.
        
        Args:
            name: Time traveler identifier
        """
        self.name = name
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.future: List[Dict[str, Any]] = []  # For redo
    
    def commit(self, key: str, new_value: Any):
        """
        Commit a state change.
        
        Args:
            key: State key
            new_value: New value
        """
        old_value = self.state.get(key)
        
        # Create delta
        delta = ETMathV2.event_delta(old_value, new_value, key)
        
        # Apply delta
        self.state = ETMathV2.apply_delta(self.state, delta, reverse=False)
        
        # Record in history
        self.history.append(delta)
        
        # Clear future (can't redo after new commit)
        self.future.clear()
    
    def undo(self) -> bool:
        """
        Undo the last change.
        
        Returns:
            True if undo succeeded, False if no history
        """
        if not self.history:
            return False
        
        # Pop last delta
        delta = self.history.pop()
        
        # Apply reverse delta
        self.state = ETMathV2.apply_delta(self.state, delta, reverse=True)
        
        # Save to future for redo
        self.future.append(delta)
        
        return True
    
    def redo(self) -> bool:
        """
        Redo the last undone change.
        
        Returns:
            True if redo succeeded, False if no future
        """
        if not self.future:
            return False
        
        # Pop from future
        delta = self.future.pop()
        
        # Apply forward delta
        self.state = ETMathV2.apply_delta(self.state, delta, reverse=False)
        
        # Restore to history
        self.history.append(delta)
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return self.state.copy()


# ============================================================================
# Eq 30: FractalReality (Procedural Generation)
# ============================================================================

class FractalReality:
    """
    Batch 3, Eq 30: Deterministic procedural world generation.
    
    ET Math: P(x, y) = Σ (1/i) · sin(i · D_seed · x)
    
    Infinite explorable worlds via fractal noise. Seed is the Descriptor
    that constrains all Points - same seed always generates same terrain.
    """
    
    def __init__(self, 
                 name: str,
                 seed: int,
                 octaves: int = FRACTAL_DEFAULT_OCTAVES,
                 persistence: float = FRACTAL_DEFAULT_PERSISTENCE):
        """
        Initialize fractal reality.
        
        Args:
            name: World identifier
            seed: World seed (determines entire topology)
            octaves: Number of noise octaves
            persistence: Amplitude decay per octave
        """
        self.name = name
        self.seed = seed
        self.octaves = octaves
        self.persistence = persistence
    
    def get_elevation(self, x: float, y: float) -> float:
        """
        Get elevation at coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            Elevation value (-1 to 1)
        """
        return ETMathV2.fractal_noise(x, y, self.seed, self.octaves, self.persistence)
    
    def get_elevation_int(self, x: float, y: float, scale: int = 100) -> int:
        """
        Get integer elevation (for discrete terrain).
        
        Args:
            x: X coordinate
            y: Y coordinate
            scale: Scaling factor
        
        Returns:
            Integer elevation
        """
        raw = self.get_elevation(x, y)
        # Map [-1, 1] to [0, scale]
        return int((raw + 1.0) * scale / 2.0)
    
    def render_chunk(self, start_x: int, start_y: int, size: int = 10) -> List[List[int]]:
        """
        Render a chunk of terrain.
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            size: Chunk size
        
        Returns:
            2D array of elevation values
        """
        chunk = []
        for y in range(start_y, start_y + size):
            row = []
            for x in range(start_x, start_x + size):
                elevation = self.get_elevation_int(x, y, scale=10)
                row.append(elevation)
            chunk.append(row)
        return chunk
    
    def render_chunk_string(self, start_x: int, start_y: int, size: int = 10) -> str:
        """
        Render a chunk as ASCII art.
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            size: Chunk size
        
        Returns:
            ASCII representation of terrain
        """
        chunk = self.render_chunk(start_x, start_y, size)
        
        # Map elevations to characters
        chars = " .,:;~#^"
        lines = []
        for row in chunk:
            line = ""
            for elevation in row:
                idx = min(elevation * len(chars) // 11, len(chars) - 1)
                line += chars[idx]
            lines.append(line)
        
        return "\n".join(lines)


__all__ = [
    'SwarmConsensus',
    'PrecognitiveCache',
    'ImmortalSupervisor',
    'SemanticManifold',
    'VarianceLimiter',
    'ProofOfTraversal',
    'EphemeralVault',
    'ConsistentHashingRing',
    'TimeTraveler',
    'FractalReality',
]
