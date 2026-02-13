"""
Exception Theory Batch 1 Classes
Computational Exception Theory (The Code of Reality)

Implements fundamental computational patterns derived from Exception Theory:
- TraverserEntropy: True entropy from T-singularities
- TrinaryState: Superposition computing with bias propagation
- ChameleonObject: Polymorphic contextual binding
- TraverserMonitor: Halting heuristic via state recurrence
- RealityGrounding: Exception handler for system coherence
- TemporalCoherenceFilter: Kalman filtering for variance minimization
- EvolutionarySolver: Genetic algorithms via manifold optimization
- PNumber: Infinite precision arithmetic

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
"""

import threading
import time
import math
import hashlib
import sys
import copy
import random
import logging
from typing import List, Optional, Dict, Any, Callable, Tuple
from decimal import Decimal, getcontext
from collections import deque

from ..core.constants import T_SINGULARITY_THRESHOLD, COHERENCE_VARIANCE_FLOOR
from ..core.mathematics import ETMathV2

# Configure decimal precision
getcontext().prec = 100

class TraverserEntropy:
    """
    Batch 1, Eq 1: The Indeterminacy Generator (True Entropy from T-Singularities)
    
    Implements ET Rule 5: T is Indeterminate ([0/0]).
    Harvests entropy from the 'singularity' between thread execution
    and clock precision (The T-Gap).
    
    Standard PRNGs are deterministic (D-bound). True randomness requires
    accessing T (the Indeterminate). T manifests at singularities of choice
    where CPU state is effectively "undetermined" relative to the clock.
    
    ET Math: T_val = lim(Î”tâ†’0) Î”State/Î”Clock = [0/0]
    """
    
    def __init__(self, pool_size=1000):
        """
        Initialize the entropy generator.
        
        Args:
            pool_size: Maximum size of timing sample pool
        """
        self._pool = []
        self._pool_size = pool_size
        self._lock = threading.Lock()
        self._traversing = False
        self._harvest_count = 0
    
    def _t_navigator(self):
        """
        T traverses P (memory) without D (synchronization)
        creating intentional indeterminacy (race condition).
        """
        while self._traversing:
            self._pool.append(time.time_ns())
            if len(self._pool) > self._pool_size:
                self._pool.pop(0)
    
    def substantiate(self, length=32):
        """
        Bind the indeterminate state to a finite Descriptor (Hash).
        
        Args:
            length: Length of output entropy string (hex chars)
        
        Returns:
            Hex string of substantiated entropy
        """
        self._traversing = True
        t_thread = threading.Thread(target=self._t_navigator)
        t_thread.start()
        
        capture = []
        start_t = time.time_ns()
        traversal_duration = 1000000
        
        while time.time_ns() - start_t < traversal_duration:
            with self._lock:
                if self._pool:
                    capture.append(sum(self._pool[-10:]))
        
        self._traversing = False
        t_thread.join()
        
        self._harvest_count += 1
        
        raw_data = str(capture).encode('utf-8')
        full_hash = hashlib.sha256(raw_data).hexdigest()
        
        return full_hash[:length]
    
    def substantiate_bytes(self, length=16):
        """
        Generate raw entropy bytes.
        
        Args:
            length: Number of bytes to generate
        
        Returns:
            bytes object of specified length
        """
        hex_str = self.substantiate(length * 2)
        return bytes.fromhex(hex_str)
    
    def substantiate_int(self, bits=64):
        """
        Generate random integer with specified bit length.
        
        Args:
            bits: Number of bits (default 64)
        
        Returns:
            Random integer
        """
        byte_length = (bits + 7) // 8
        entropy_bytes = self.substantiate_bytes(byte_length)
        result = int.from_bytes(entropy_bytes, byteorder='big')
        return result & ((1 << bits) - 1)
    
    def substantiate_float(self):
        """
        Generate random float in [0, 1).
        
        Returns:
            Random float
        """
        return self.substantiate_int(53) / (2**53)
    
    def get_metrics(self):
        """Get entropy generation metrics."""
        return {
            'harvest_count': self._harvest_count,
            'pool_size': len(self._pool),
            'max_pool_size': self._pool_size
        }
    
    def analyze_pool(self):
        """
        Analyze current timing pool for T-signatures.
        
        Returns:
            Dict with indeterminacy analysis
        """
        with self._lock:
            if len(self._pool) < 2:
                return {'status': 'INSUFFICIENT_DATA', 'samples': len(self._pool)}
            
            return ETMathV2.compute_indeterminacy_signature(self._pool.copy())




class TrinaryState:
    """
    Batch 1, Eq 2: Trinary Logic Gate (Superposition Computing)
    
    ENHANCED from v2.0 - Now includes bias propagation through operations.
    
    States: 0 (False/Dâ‚€), 1 (True/Dâ‚), 2 (Superposition/Unsubstantiated)
    Implements P, D, T logic: Point, Descriptor, Traverser
    
    ET Math:
        S = (P âˆ˜ (Dâ‚€, Dâ‚)) âŸ¹ State 2 (Superposition)
        Observe(S) â†’[T] {0, 1}
    """
    
    FALSE = 0
    TRUE = 1
    POTENTIAL = 2
    UNSUBSTANTIATED = 2
    
    def __init__(self, state=2, bias=0.5):
        """
        Initialize trinary state.
        
        Args:
            state: Initial state (0, 1, or 2)
            bias: Probability weight toward TRUE when collapsing (0.0-1.0)
        """
        if state not in [0, 1, 2]:
            raise ValueError("State must be 0, 1, or 2")
        self._state = state
        self._bias = max(0.0, min(1.0, bias))
    
    def collapse(self, observer_bias=None):
        """
        Collapse superposition to 0 or 1 based on observer.
        T (Traverser) makes the choice.
        
        Args:
            observer_bias: Override bias for this collapse (optional)
        
        Returns:
            Final state (0 or 1)
        """
        if self._state == self.POTENTIAL:
            effective_bias = observer_bias if observer_bias is not None else self._bias
            self._state = 1 if random.random() < effective_bias else 0
        return self._state
    
    def substantiate(self):
        """Alias for collapse() - ET terminology."""
        return self.collapse()
    
    def measure(self):
        """Measure without collapse (peek at state)."""
        return self._state
    
    def is_superposed(self):
        """Check if in superposition."""
        return self._state == self.POTENTIAL
    
    def get_bias(self):
        """Get current bias value."""
        return self._bias
    
    def set_bias(self, new_bias):
        """Set new bias value."""
        self._bias = max(0.0, min(1.0, new_bias))
    
    def __bool__(self):
        """Boolean conversion collapses superposition."""
        if self._state == self.POTENTIAL:
            self.collapse()
        return self._state == 1
    
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, TrinaryState):
            return self._state == other._state
        elif isinstance(other, int):
            return self._state == other
        return False
    
    def __repr__(self):
        state_names = {0: "FALSE", 1: "TRUE", 2: "SUPERPOSITION"}
        bias_str = f", bias={self._bias:.2f}" if self._state == 2 else ""
        return f"TrinaryState({state_names[self._state]}{bias_str})"
    
    def __and__(self, other):
        """
        Trinary AND with bias propagation.
        
        Logic in Superposition:
        - If either is FALSE (grounded), result is FALSE
        - If both TRUE, result is TRUE
        - Unresolved superposition creates compounded potential
        """
        if isinstance(other, TrinaryState):
            other_val = other._state
            other_bias = other._bias
        else:
            other_val = 1 if other else 0
            other_bias = 1.0 if other else 0.0
        
        if self._state == 0 or other_val == 0:
            return TrinaryState(0)
        if self._state == 1 and other_val == 1:
            return TrinaryState(1)
        
        new_bias = self._bias * other_bias
        return TrinaryState(self.POTENTIAL, bias=new_bias)
    
    def __or__(self, other):
        """
        Trinary OR with bias propagation.
        """
        if isinstance(other, TrinaryState):
            other_val = other._state
            other_bias = other._bias
        else:
            other_val = 1 if other else 0
            other_bias = 1.0 if other else 0.0
        
        if self._state == 1 or other_val == 1:
            return TrinaryState(1)
        if self._state == 0 and other_val == 0:
            return TrinaryState(0)
        
        new_bias = self._bias + other_bias - (self._bias * other_bias)
        return TrinaryState(self.POTENTIAL, bias=new_bias)
    
    def __invert__(self):
        """Trinary NOT."""
        if self._state == 0:
            return TrinaryState(1)
        elif self._state == 1:
            return TrinaryState(0)
        else:
            return TrinaryState(self.POTENTIAL, bias=1.0 - self._bias)
    
    def AND(self, other):
        """Trinary AND (legacy interface)."""
        return self.__and__(other)
    
    def OR(self, other):
        """Trinary OR (legacy interface)."""
        return self.__or__(other)
    
    def NOT(self):
        """Trinary NOT (legacy interface)."""
        return self.__invert__()


# ============================================================================
# NEW IN v2.2 - Batch 2 Classes
# ============================================================================



class ChameleonObject:
    """
    Batch 1, Eq 7: Polymorphic "Chameleon" Class (Pure Relativism)
    
    Implements ET Rule 9: "Everything is relational. There is only pure relativism."
    A Descriptor's meaning depends on the observer (T).
    
    This class changes its behavior (Methods/Descriptors) based on who calls it
    (the calling frame), implementing Contextual Binding.
    
    ET Math: D_observed = f(T_observer)
    """
    
    def __init__(self, **default_attributes):
        """
        Initialize Chameleon with default attributes.
        
        Args:
            **default_attributes: Default attribute values
        """
        self._default_attrs = default_attributes
        self._context_bindings = {}
        self._access_log = []
    
    def bind_context(self, caller_name, **attributes):
        """
        Bind specific attribute values for a given caller context.
        
        Args:
            caller_name: Name of the calling function
            **attributes: Attribute values for this context
        """
        self._context_bindings[caller_name] = attributes
    
    def __getattribute__(self, name):
        """
        Get attribute based on calling context (T observer).
        """
        if name.startswith('_') or name in ['bind_context', 'get_access_log', 'get_contexts']:
            return object.__getattribute__(self, name)
        
        stack = inspect.stack()
        caller_name = stack[1].function if len(stack) > 1 else '__main__'
        
        access_log = object.__getattribute__(self, '_access_log')
        access_log.append({
            'attribute': name,
            'caller': caller_name,
            'timestamp': time.time()
        })
        
        context_bindings = object.__getattribute__(self, '_context_bindings')
        if caller_name in context_bindings and name in context_bindings[caller_name]:
            return context_bindings[caller_name][name]
        
        default_attrs = object.__getattribute__(self, '_default_attrs')
        if name in default_attrs:
            return default_attrs[name]
        
        raise AttributeError(f"Attribute '{name}' not bound for context '{caller_name}'")
    
    def get_access_log(self):
        """Get log of all attribute accesses."""
        return self._access_log.copy()
    
    def get_contexts(self):
        """Get all registered context bindings."""
        return {k: dict(v) for k, v in self._context_bindings.items()}




class TraverserMonitor:
    """
    Batch 1, Eq 8: The "Halting" Heuristic (Traversal Cycle Detection)
    
    While the Halting Problem is generally unsolvable, ET re-interprets
    infinite loops as Incoherent Traversal Cycles (visiting the same
    P âˆ˜ D state with identical memory).
    
    We detect Recurrent Variance (V=0 change over time).
    
    ET Math: If (P âˆ˜ D)_t = (P âˆ˜ D)_{t-k} âŸ¹ Loop (Infinite T-Trap)
    """
    
    def __init__(self, max_history=10000):
        """
        Initialize the traverser monitor.
        
        Args:
            max_history: Maximum state history to track
        """
        self._history = set()
        self._max_history = max_history
        self._detection_count = 0
        self._enabled = True
    
    def trace(self, frame, event, arg):
        """
        Trace function for sys.settrace().
        
        Monitors execution for state recurrence.
        """
        if not self._enabled:
            return self.trace
        
        if event == 'line':
            try:
                locals_repr = tuple(sorted(
                    (k, repr(v)[:100]) for k, v in frame.f_locals.items()
                    if not k.startswith('_')
                ))
                state_descriptor = (frame.f_lineno, locals_repr)
                state_hash = hash(state_descriptor)
                
                if state_hash in self._history:
                    self._detection_count += 1
                    raise RuntimeError(
                        f"(!) Infinite Traversal Loop Detected: State Recurrence at line {frame.f_lineno}"
                    )
                
                self._history.add(state_hash)
                
                if len(self._history) > self._max_history:
                    self._history = set(list(self._history)[self._max_history // 2:])
                    
            except (TypeError, ValueError):
                pass
        
        return self.trace
    
    def reset(self):
        """Clear state history."""
        self._history.clear()
    
    def enable(self):
        """Enable monitoring."""
        self._enabled = True
    
    def disable(self):
        """Disable monitoring."""
        self._enabled = False
    
    def get_metrics(self):
        """Get monitoring metrics."""
        return {
            'history_size': len(self._history),
            'max_history': self._max_history,
            'detections': self._detection_count,
            'enabled': self._enabled
        }
    
    def check_state(self, state):
        """
        Manually check a state for recurrence.
        
        Args:
            state: Hashable state descriptor
        
        Returns:
            True if state is a recurrence, False otherwise
        """
        state_hash = hash(str(state))
        is_recurrence = state_hash in self._history
        
        if not is_recurrence:
            self._history.add(state_hash)
        else:
            self._detection_count += 1
        
        return is_recurrence




class RealityGrounding:
    """
    Batch 1, Eq 4: The "Exception" Error Handler (Grounding Incoherence)
    
    PRESERVED FROM v2.0.
    
    In ET, an error is Incoherence (I). The Exception Axiom states that
    "For every exception there is an exception." This handler catches
    "Incoherent" states (bugs) and forces Grounding (E) to a safe state.
    
    ET Math:
        S_state âˆˆ I âŸ¹ ForceTraverse(S_state â†’ E)
        V(E) = 0 (Variance is zeroed)
    """
    
    def __init__(self, safe_state_callback):
        """
        Initialize with callback to safe state (E).
        
        Args:
            safe_state_callback: Function that restores zero-variance state
        """
        self.safe_state = safe_state_callback
        self.grounding_history = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type:
            timestamp = time.time()
            self.grounding_history.append({
                'timestamp': timestamp,
                'exception_type': exc_type.__name__,
                'exception_value': str(exc_value),
                'traceback': tb
            })
            
            logger.info(f"[!] INCOHERENCE DETECTED: {exc_value}")
            logger.info("[!] Initiating T-Traversal to Grounded Exception (E)...")
            
            try:
                self.safe_state()
                logger.info("[+] Reality Grounded. System Stability Restored.")
                return True
            except Exception as grounding_error:
                logger.error(f"[!] CRITICAL: Grounding Failed. Total Incoherence: {grounding_error}")
                return False
    
    def get_grounding_history(self):
        """Return history of groundings."""
        return self.grounding_history




class TemporalCoherenceFilter:
    """
    Batch 2, Eq 15 / Batch 1: Temporal Coherence Filter (The Kalman Stabilizer)
    
    PRESERVED FROM v2.0.
    
    Filters noisy Traverser (T) data to find true Point (P).
    1D Kalman filter as variance minimization.
    
    ET Math:
        D_est = D_pred + K Â· (T_obs - D_pred)
        K = V_err / (V_err + V_noise)
    """
    
    def __init__(self, process_variance=0.01, measurement_variance=0.1, initial_estimate=0.0):
        """
        Initialize Kalman filter parameters.
        
        Args:
            process_variance: How much system changes naturally (D_process)
            measurement_variance: Sensor noise level (T_noise)
            initial_estimate: Starting point (P_initial)
        """
        self.Q = process_variance
        self.R = measurement_variance
        self.x = initial_estimate
        self.P = 1.0
    
    def update(self, measurement):
        """
        Update filter with new measurement.
        Returns: filtered value (true P estimate)
        """
        self.P += self.Q
        
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        
        return self.x
    
    def get_variance(self):
        """Return current estimate variance."""
        return self.P
    
    def get_metrics(self):
        """Get filter metrics."""
        return {
            'estimate': self.x,
            'variance': self.P,
            'process_noise': self.Q,
            'measurement_noise': self.R,
            'kalman_gain': self.P / (self.P + self.R)
        }




class EvolutionarySolver:
    """
    Batch 2, Eq 17: The Evolutionary Descriptor (Genetic Solver)
    
    PRESERVED FROM v2.0.
    
    When exact formula D unknown, evolve it by spawning multiple
    configurations P and selecting those with lowest Variance.
    
    ET Math:
        P_next = Select(P_pop) + Mutate(T)
        min Variance(P_pop â†’ Goal)
    """
    
    def __init__(self, fitness_function, population_size=50, mutation_rate=0.1):
        """
        Initialize evolutionary solver.
        
        Args:
            fitness_function: Function(individual) -> variance (lower is better)
            population_size: Number of configurations (P)
            mutation_rate: Mutation probability (T-indeterminacy)
        """
        self.fitness_func = fitness_function
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        self.best_ever = None
        self.best_fitness = float('inf')
    
    def initialize_population(self, generator_func):
        """
        Generate initial population using descriptor generator.
        
        Args:
            generator_func: Function() -> individual
        """
        self.population = [generator_func() for _ in range(self.pop_size)]
        self.generation = 0
    
    def evolve(self, generations=100):
        """
        Evolve population for N generations.
        Returns: best individual found
        """
        for gen in range(generations):
            self.generation += 1
            
            fitness_scores = [(ind, self.fitness_func(ind)) for ind in self.population]
            fitness_scores.sort(key=lambda x: x[1])
            
            if fitness_scores[0][1] < self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_ever = copy.deepcopy(fitness_scores[0][0])
            
            survivors = [ind for ind, score in fitness_scores[:self.pop_size // 2]]
            
            offspring = []
            while len(offspring) < self.pop_size - len(survivors):
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child = self._crossover(parent1, parent2)
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                offspring.append(child)
            
            self.population = survivors + offspring
        
        return self.best_ever
    
    def _crossover(self, parent1, parent2):
        """Crossover two parents."""
        if isinstance(parent1, (list, tuple)):
            child = []
            for g1, g2 in zip(parent1, parent2):
                child.append(g1 if random.random() < 0.5 else g2)
            return type(parent1)(child)
        elif isinstance(parent1, dict):
            child = {}
            for key in parent1.keys():
                child[key] = parent1[key] if random.random() < 0.5 else parent2.get(key, parent1[key])
            return child
        else:
            return (parent1 + parent2) / 2
    
    def _mutate(self, individual):
        """Mutate individual."""
        if isinstance(individual, (list, tuple)):
            mutated = list(individual)
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] += random.gauss(0, 0.1)
            return type(individual)(mutated)
        elif isinstance(individual, dict):
            mutated = individual.copy()
            key = random.choice(list(mutated.keys()))
            mutated[key] += random.gauss(0, 0.1)
            return mutated
        else:
            return individual + random.gauss(0, 0.1)
    
    def get_metrics(self):
        """Get solver metrics."""
        return {
            'generation': self.generation,
            'population_size': self.pop_size,
            'mutation_rate': self.mutation_rate,
            'best_fitness': self.best_fitness,
            'best_individual': self.best_ever
        }




class PNumber:
    """
    Batch 1, Eq 5: Infinite Precision Number (The P-Type)
    
    PRESERVED FROM v2.0.
    
    Stores generating descriptor (algorithm) rather than value,
    allowing Traverser to navigate to any precision on demand.
    
    ET Math:
        N_P = P âˆ˜ D_algo
        Value(N_P, n) = T(N_P) â†’ Precision_n
    """
    
    def __init__(self, generator_func, *args):
        """
        Initialize P-Number with generator function.
        
        Args:
            generator_func: Function that generates value at precision
            *args: Arguments for generator function
        """
        self._generator = generator_func
        self._args = args
        self._cache = {}
    
    def substantiate(self, precision):
        """
        T traverses P-structure to depth 'precision'.
        Caches results for efficiency.
        """
        if precision in self._cache:
            return self._cache[precision]
        
        decimal.getcontext().prec = precision
        value = self._generator(*self._args, precision=precision)
        self._cache[precision] = value
        return value
    
    def __add__(self, other):
        """Adding two P-Numbers creates composite Descriptor."""
        if isinstance(other, PNumber):
            def new_d(precision=50):
                return self.substantiate(precision) + other.substantiate(precision)
            return PNumber(new_d)
        else:
            def new_d(precision=50):
                return self.substantiate(precision) + decimal.Decimal(str(other))
            return PNumber(new_d)
    
    def __mul__(self, other):
        """Multiplying P-Numbers."""
        if isinstance(other, PNumber):
            def new_d(precision=50):
                return self.substantiate(precision) * other.substantiate(precision)
            return PNumber(new_d)
        else:
            def new_d(precision=50):
                return self.substantiate(precision) * decimal.Decimal(str(other))
            return PNumber(new_d)
    
    def __repr__(self):
        return f"PNumber({self._generator.__name__}, precision=âˆž)"
    
    @staticmethod
    def pi(precision=50):
        """Generate Ï€ to arbitrary precision using BBP formula."""
        decimal.getcontext().prec = precision + 10
        
        pi_sum = decimal.Decimal(0)
        for k in range(precision):
            ak = decimal.Decimal(1) / (16 ** k)
            bk = (decimal.Decimal(4) / (8*k + 1) -
                  decimal.Decimal(2) / (8*k + 4) -
                  decimal.Decimal(1) / (8*k + 5) -
                  decimal.Decimal(1) / (8*k + 6))
            pi_sum += ak * bk
        
        return pi_sum
    
    @staticmethod
    def e(precision=50):
        """Generate e to arbitrary precision."""
        decimal.getcontext().prec = precision + 10
        
        e_sum = decimal.Decimal(1)
        factorial = 1
        for n in range(1, precision):
            factorial *= n
            e_sum += decimal.Decimal(1) / factorial
        
        return e_sum
    
    @staticmethod
    def phi(precision=50):
        """Generate Ï† (golden ratio) to arbitrary precision."""
        decimal.getcontext().prec = precision + 10
        
        five = decimal.Decimal(5)
        sqrt5 = five.sqrt()
        return (decimal.Decimal(1) + sqrt5) / decimal.Decimal(2)





__all__ = [
    'TraverserEntropy',
    'TrinaryState',
    'ChameleonObject',
    'TraverserMonitor',
    'RealityGrounding',
    'TemporalCoherenceFilter',
    'EvolutionarySolver',
    'PNumber',
]
