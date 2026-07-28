# EUDD — API Reference
## Named Pipe API Complete Specification — 117 Operations Across 17 Domains + 2 Cross-Domain

**Source:** Extracted from EUDD v39 §7.16 (all subsections §7.16.1–§7.16.22)
**Master index:** See `EUDD_Table_of_Contents.md` for navigation across all EUDD files
**Related files:** Schema definitions in `EUDD_Architecture.md` §3.2–§3.15b. Event classes in `EUDD_Events_and_Classes.md`. Bootstrap values in `EUDD_Bootstrap_Catalog.md`.

---

### 7.16 The Named Pipe API — Complete Specification

**Status:** This section closes Roadmap §1.3 (API Specification). All 114 operations are fully specified with exact JSON request/response schemas. Zero IEEE 754 floats cross the API boundary. The metabolism governs all resource allocation and priority via ET-derived structural constants.

**Pipe name:** `\\.\pipe\EUDD_Manager` (Windows named pipe). Each connected client gets its own pipe instance per Windows named pipe semantics. Multiple simultaneous clients supported.

**Wire format:** UTF-8 JSON, length-prefixed. Each message on the pipe is:
```
[4 bytes: uint32 big-endian message length in bytes] [UTF-8 JSON payload of exactly that length]
```

No delimiters, no newlines between messages. The length prefix ensures the receiver knows exactly how many bytes to read. Maximum message size: 2³⁰ bytes (1 GB = DISK_SAFETY_FLOOR, the d=1 octave action quantum at GB scale). Messages exceeding this are rejected with `message_too_large` error.

#### 7.16.1 Communication Architecture — Three Patterns

Every message over the pipe follows one of three communication patterns. The Manager selects the pattern based on what the operation structurally requires — the client does not choose.

**Pattern 1 — Request-Response (synchronous).**

The client sends one request. The Manager sends one response. The client blocks until the response arrives. Used when the answer is available immediately: cache hits, index lookups, catalog queries, status checks, any operation that completes in sub-millisecond time.

```
Client → Manager:  { "msg_type": "request", ... }
Manager → Client:  { "msg_type": "response", ... }
```

**Pattern 2 — Request-Acknowledge-Stream-Complete (asynchronous with streaming).**

The client sends one request. The Manager immediately acknowledges with an `operation_id`, then streams zero or more intermediate results, then sends a completion message. Used for operations that take time: tower escalation (each landmark is a streamed result), file ingestion (each seed extracted is a streamed result), discovery scans (each finding is a streamed result), L'Hôpital resolution (each iteration is a streamed result). The client processes intermediate results in real-time. This is T navigating — the client sees each step of the Traverser's journey.

```
Client → Manager:  { "msg_type": "request", ... }
Manager → Client:  { "msg_type": "ack", "operation_id": "<uuid>", ... }
Manager → Client:  { "msg_type": "stream", "operation_id": "<uuid>", "sequence": 0, ... }
Manager → Client:  { "msg_type": "stream", "operation_id": "<uuid>", "sequence": 1, ... }
...
Manager → Client:  { "msg_type": "complete", "operation_id": "<uuid>", ... }
```

The client can cancel an in-progress async operation by sending:
```
Client → Manager:  { "msg_type": "request", "command": "cancel", "operation_id": "<uuid>" }
Manager → Client:  { "msg_type": "response", "cancelled": true, "partial_results_available": true }
```

Cancellation does NOT discard partial results — everything computed so far is memoized permanently (§4.3 never destroy). The client can query partial results via `get_value_trajectory` or other relevant queries.

**Pattern 3 — Subscribe-Notify (push).**

The client subscribes to a filter. The Manager pushes matching notifications as they happen, for the lifetime of the subscription. The subscription persists until explicitly unsubscribed or the connection closes.

```
Client → Manager:  { "msg_type": "request", "command": "subscribe", "filter": { ... } }
Manager → Client:  { "msg_type": "response", "subscription_id": "<uuid>" }
...later, asynchronously...
Manager → Client:  { "msg_type": "notification", "subscription_id": "<uuid>", ... }
Manager → Client:  { "msg_type": "notification", "subscription_id": "<uuid>", ... }
```

Notifications are interleaved with request-response and stream messages on the same pipe. The `msg_type` field distinguishes them unambiguously.

**Adaptive pattern selection.** Some operations adapt their pattern based on runtime conditions:
- `project`: Pattern 1 on cache hit (instant), Pattern 2 on cache miss (compute + store + return)
- `compute`: Pattern 1 on cache hit, Pattern 2 on cache miss
- `lattice_add`: Pattern 1 on cache hit, Pattern 2 on cache miss
- `evaluate_function`: Pattern 1 on cache hit, Pattern 2 on cache miss

For adaptive operations, the client must handle BOTH response types. The `msg_type` field tells the client which pattern fired: `"response"` means Pattern 1 (done), `"ack"` means Pattern 2 (streaming will follow).

#### 7.16.2 Message Envelope

Every message over the pipe contains these fields:

```json
{
  "msg_id": "<uuid>",
  "msg_type": "<request|response|ack|stream|complete|error|notification>",
  "api_version": 1,
  "timestamp_ns": "<uint64 as string>",
  "session_id": "<string>",
  "command": "<operation_name>",
  "operation_id": "<uuid>",
  "sequence": "<integer>"
}
```

| Field | Type | Present On | Description |
|---|---|---|---|
| `msg_id` | string (UUID) | ALL messages | Unique per message, for correlation and deduplication |
| `msg_type` | string enum | ALL messages | One of: `request`, `response`, `ack`, `stream`, `complete`, `error`, `notification` |
| `api_version` | integer | ALL messages | Protocol version. Currently 1. Manager and client negotiate during handshake; messages use the negotiated version |
| `timestamp_ns` | string (uint64) | ALL messages | Nanoseconds since Unix epoch. Exact integer as string (JSON cannot represent uint64 without precision loss). This is D-time — the global coordinate |
| `session_id` | string | ALL except `handshake` request | Set during handshake, included on all subsequent messages |
| `command` | string | `request` only | The operation name (e.g., `"project"`, `"compute"`, `"escalate"`) |
| `operation_id` | string (UUID) | `ack`, `stream`, `complete`, `error` (for async) | Correlates async messages to the originating request |
| `sequence` | integer | `stream` only | 0-indexed sequence number within an async operation's stream |

Additional payload fields are operation-specific and documented per-operation below.

**Timestamp encoding:** All timestamps throughout the API are uint64 nanoseconds since Unix epoch (1970-01-01T00:00:00Z), encoded as JSON strings to avoid IEEE 754 precision loss. Example: `"1714742400000000000"` for 2025-05-03T12:00:00Z. This is D-time — the relational ordering Descriptor, cardinality finite n.

#### 7.16.3 Value Encoding — Zero IEEE 754

**All numerical values crossing the API boundary are exact representations. No IEEE 754 floating-point numbers exist anywhere in any request or response.**

| Data type | JSON encoding | Example |
|---|---|---|
| 361-dps value | String, all 361 digits | `"1.202056903159594285399738161511449990764986292340498881792271555341838205786313090186455873609335258146199..."` |
| Lattice k | Integer | `7360` |
| Lattice d | Integer | `693` |
| ε (micro-cents) | Integer | `-1955` (meaning −1.955 cents) |
| Rational | Object `{"num": <int>, "den": <int>}` | `{"num": 2, "den": 3}` |
| Timestamp | String (uint64 nanoseconds) | `"1714742400000000000"` |
| Large integer | String | `"27720"` (for values that may exceed JSON integer range) |
| Boolean | Boolean | `true` |
| Binary blob | Base64 string | `"AQID..."` (for MPFR blobs, packed arrays) |
| Null / absent | `null` or field omitted | |

**Value representation in requests:** A value can be specified in multiple ways. The `value_spec` object is used throughout the API wherever a value is needed as input:

```json
{
  "value_spec": {
    "by": "<id|hash|repr|decimal|expression|address>",
    "value_id": 4827,
    "value_hash": "a1b2c3d4...",
    "value_repr": "ζ(3)",
    "decimal_361dps": "1.20205690315959...",
    "expression": "ζ(3) × π / φ²",
    "address": {"N": 27720, "k": 7360, "eps_micros": -85}
  }
}
```

Only the fields relevant to the `by` mode are required. The Manager resolves the value_spec to a concrete 361-dps value and value_id. If the value is new (not in the database), the Manager creates a `values` row automatically.

#### 7.16.4 Connection Lifecycle and Metabolism

**The connection lifecycle mirrors P∘D∘T = E:**
- **P (substrate):** The named pipe connection — the raw communication channel
- **D (constraints):** The API protocol, the metabolism budget, the negotiated capabilities
- **T (agency):** The connected program — the Traverser navigating the API

**Phase 1 — Connect.** Client opens the named pipe `\\.\pipe\EUDD_Manager`. The pipe connection IS the P-substrate.

**Phase 2 — Handshake.** Client sends `handshake` request. Manager responds with session_id, capabilities, negotiated api_version. The handshake establishes the D-constraints of the connection.

**Phase 3 — Register Metabolism.** Client sends `register_metabolism` with its hardware profile and operational characteristics. The Manager:
1. Creates a digital tower entry for this program (§3.10) with R₀ derived from the program's structure (frame rate, tick rate, clock cycle — whatever the natural traversal period is)
2. Computes the metabolic budget: K = 2/3 ceiling applied to available resources after accounting for existing connections, V = 1/12 headroom, ξ(d) coupling for the program's reported dominant computation family
3. Returns the metabolic budget to the client

**Phase 4 — Operation.** Client sends commands, receives responses. The metabolism governs resource allocation dynamically:
- The Manager tracks each connection's actual resource consumption (CPU time, memory, pipe bandwidth)
- The Manager adjusts metabolic budgets dynamically as connections come and go
- When a connection's consumption approaches K × (its share of total), the Manager throttles response delivery (not computation — computations still run and memoize; delivery is delayed)
- The per-sublattice coupling ξ(d) of the connection's dominant computation family determines scheduling weight relative to other connections

**Phase 5 — Disconnect.** Client sends `disconnect` or pipe closes. Manager logs session end, flushes pending writes, releases the connection's metabolic share back to the pool.

**Metabolic mediation across connections:**

When multiple programs are connected simultaneously (Skyrim + compressor + fractal generator + Mike's GUI), the Manager mediates using the metabolism:

Total system budget: K × hardware = 2/3 of CPU, RAM, VRAM.

Per-connection share: proportional to ξ(d_dominant) of each connection's dominant computation family, normalized to sum to K × hardware. A game doing d=1 physics (ξ=8.5625) gets ~8.5× the scheduling weight of a program doing d=12 computations (ξ=1.0).

The Manager's own operations (discovery engine, self-recording, GUI rendering) are a connection too — with their own metabolic budget drawn from the same K pool.

If total demand exceeds K × hardware, the Manager does NOT drop connections or reject operations. It queues operations and serves them as resources free up, prioritized by ξ(d). The K ceiling is a resource governor, not a gatekeeper. Every operation eventually executes; the metabolism determines WHEN.

#### 7.16.5 Error Response Format

Every error response:

```json
{
  "msg_id": "<uuid>",
  "msg_type": "error",
  "api_version": 1,
  "timestamp_ns": "<uint64>",
  "session_id": "<string>",
  "operation_id": "<uuid or null>",
  "error": {
    "code": 1001,
    "class": "computation_failure",
    "detail": "FLINT/Arb pole detected at s=−1 during ζ(−1) evaluation",
    "source_module": "ArbEval::zeta",
    "event_id": 847293,
    "recoverable": false,
    "suggestion": "Use Path D.T for pole values — they are indeterminate forms requiring L'Hôpital resolution"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `code` | integer | Numeric error code. Range 1000–1999: computation. 2000–2999: input validation. 3000–3999: not found. 4000–4999: state/resource. 5000–5999: integrity. 6000–6999: communication |
| `class` | string | Error classification (see table below) |
| `detail` | string | Human-readable description with full context |
| `source_module` | string | C++ module and function where the error originated |
| `event_id` | integer | The event_id in the `events` table where this error was logged (every error IS an event per §7.15 ET-native error philosophy) |
| `recoverable` | boolean | Can the client retry this operation with the same input? |
| `suggestion` | string or null | ET-informed guidance on how to handle this error |

**Error classes:**

| Code Range | Class | Description |
|---|---|---|
| 1000–1099 | `computation_failure` | MPFR/Arb evaluation failed: pole, overflow, underflow, stack overflow, NaN, infinity |
| 1100–1199 | `annihilation_boundary` | Computation approached r=0 — off-lattice singularity. NOT a failure — correct structural classification |
| 1200–1299 | `incoherence_filter` | Input classified as {P,T} Incoherent. No lattice address assigned. Correct behavior, not a failure |
| 1300–1399 | `pure_t_detected` | L'Hôpital failed to resolve — irreducible Traverser. Correct classification, not a failure |
| 2000–2099 | `invalid_input` | Malformed request, unparseable value, unknown command, missing required field, type mismatch |
| 2100–2199 | `invalid_expression` | Expression cannot be parsed or is not well-formed |
| 2200–2299 | `validation_rejected` | JSON extension failed §7.14 strict validation (with specific rule that failed) |
| 3000–3099 | `value_not_found` | Value ID/hash/repr not in database |
| 3100–3199 | `entity_not_found` | Referenced entity (equation, relationship, pattern, tower, event, etc.) not found |
| 3200–3299 | `address_unoccupied` | Query targets an address with no content |
| 3300–3399 | `generator_not_found` | No generator covers the queried address/value |
| 4000–4099 | `manager_busy` | Manager cannot accept new async operations (internal queue at capacity) |
| 4100–4199 | `disk_low` | Disk free < DISK_SAFETY_FLOOR (1 GB). Warning — operations continue but the user should free space |
| 4200–4299 | `metabolism_throttled` | Operation queued due to metabolic ceiling — will execute when resources free up. Includes estimated wait |
| 4300–4399 | `atomicity_violation` | Atomic batch_store failed — all values rolled back. Includes per-value failure reasons |
| 5000–5099 | `corruption_detected` | Integrity check found corruption. Operations continue but results may be unreliable until backup restoration |
| 5100–5199 | `consistency_violation` | Cross-project consistency check found contradictory projections |
| 6000–6099 | `pipe_error` | Named pipe communication failure |
| 6100–6199 | `message_too_large` | Message exceeds 2³⁰ byte limit |

Note: error codes 1100–1399 (`annihilation_boundary`, `incoherence_filter`, `pure_t_detected`) are NOT failures — they are correct structural classifications per ET. The Manager reports them as errors (because the operation did not produce a standard lattice result) but they carry full structural information about what WAS found. The `recoverable` field is `false` for these because retrying will produce the same correct classification.

---

#### 7.16.6 Domain 1 — Connection & Metabolism

**Operation 1: `handshake`**

Pattern: Request-Response

Establishes the connection. Must be the first message after pipe open.

Request:
```json
{
  "msg_type": "request",
  "command": "handshake",
  "api_version": 1,
  "client": {
    "project_name": "skyrim_bridge",
    "client_type": "game",
    "client_version": "1.0.0",
    "pid": 12345,
    "requested_api_version": 1
  }
}
```

Response:
```json
{
  "msg_type": "response",
  "session_id": "skyrim_bridge_2026-05-03_001",
  "api_version": 1,
  "server": {
    "manager_version": "1.0.0",
    "akashic_file_size_bytes": "1073741824",
    "total_values": "847293",
    "total_projections": "6778344",
    "total_equations": "12847291",
    "total_generators": "4827",
    "total_memoized": "842466",
    "generator_to_memoized_ratio": {"num": 4827, "den": 847293},
    "active_connections": 3,
    "uptime_ns": "86400000000000"
  }
}
```

**Operation 2: `register_metabolism`**

Pattern: Request-Response

Registers the program's metabolic profile. Creates a tower for this program. Returns metabolic budget. Must be called after `handshake`, before any computation operations.

Request:
```json
{
  "msg_type": "request",
  "command": "register_metabolism",
  "session_id": "skyrim_bridge_2026-05-03_001",
  "hardware": {
    "cpu_count_logical": 24,
    "cpu_load_percent_current": 35,
    "mem_total_bytes": "34359738368",
    "mem_available_bytes": "21474836480",
    "mem_used_percent": 37,
    "gpu_name": "NVIDIA RTX 2070 Super",
    "vram_total_bytes": "8589934592",
    "vram_available_bytes": "6442450944"
  },
  "program": {
    "tower_name": "skyrim_bridge_instance_1",
    "p_substrate_descriptor": "skyrim_game_state_space",
    "r0_value": "0.016666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666667",
    "r0_description": "1/60 second — one frame at 60 FPS",
    "r0_natural_units": "seconds per frame",
    "operational_n": 12,
    "dominant_d_family": 1,
    "computation_characteristics": {
      "frame_rate_hz": 60,
      "computations_per_frame_estimate": 50000,
      "dominant_operation_types": ["multiply", "add", "sqrt", "sin", "cos"],
      "latency_budget_us": 16000
    }
  }
}
```

Response:
```json
{
  "msg_type": "response",
  "tower_id": 8,
  "tower_name": "skyrim_bridge_instance_1",
  "r0_value_id": 94827,
  "metabolism": {
    "k_ceiling": {"num": 2, "den": 3},
    "v_headroom": {"num": 1, "den": 12},
    "allocated_threads": 10,
    "allocated_memory_bytes": "14316557653",
    "coupling_xi_dominant": {"num": 137, "den": 16},
    "scheduling_weight": "8.5625",
    "cpu_headroom_percent": "31.67",
    "mem_headroom_percent": "29.67",
    "overall_pressure": "0.37",
    "convergence_ratio": "0.017683",
    "metabolic_refresh_interval_ns": "144000000000"
  }
}
```

**Operation 3: `heartbeat`**

Pattern: Request-Response

Keep-alive. Also returns current metabolic state so the client can adapt.

Request:
```json
{
  "msg_type": "request",
  "command": "heartbeat",
  "session_id": "skyrim_bridge_2026-05-03_001"
}
```

Response:
```json
{
  "msg_type": "response",
  "metabolism": {
    "allocated_threads": 10,
    "allocated_memory_bytes": "14316557653",
    "coupling_xi_dominant": {"num": 137, "den": 16},
    "scheduling_weight": "8.5625",
    "cpu_headroom_percent": "28.12",
    "mem_headroom_percent": "27.33",
    "overall_pressure": "0.41"
  },
  "manager_status": "nominal",
  "discovery_engine": "idle",
  "pending_operations": 0
}
```

**Operation 4: `report_computation_profile`**

Pattern: Request-Response

The connected program reports its current dominant d-family so the Manager adjusts scheduling weight via ξ(d). Called when the program's computational regime changes — e.g., a game transitions from heavy physics (d=1) to menu rendering (d=12).

Request:
```json
{
  "msg_type": "request",
  "command": "report_computation_profile",
  "session_id": "skyrim_bridge_2026-05-03_001",
  "profile": {
    "dominant_d_family": 3,
    "secondary_d_families": [1, 4],
    "computations_per_second_current": 3200000,
    "cache_hit_ratio": {"num": 847, "den": 1000}
  }
}
```

Response:
```json
{
  "msg_type": "response",
  "metabolism_updated": true,
  "new_coupling_xi_dominant": {"num": 137, "den": 20},
  "new_scheduling_weight": "6.85"
}
```

**Operation 5: `disconnect`**

Pattern: Request-Response

Clean shutdown. Manager logs session end, flushes pending writes attributed to this session, releases metabolic share.

Request:
```json
{
  "msg_type": "request",
  "command": "disconnect",
  "session_id": "skyrim_bridge_2026-05-03_001"
}
```

Response:
```json
{
  "msg_type": "response",
  "session_ended": true,
  "events_logged": 847293,
  "values_stored": 12847,
  "equations_cached": 3847291,
  "tower_preserved": true,
  "session_duration_ns": "3600000000000"
}
```

---

#### 7.16.7 Domain 2 — Core Lattice Operations

**Operation 6: `project`**

Pattern: Adaptive (Req-Resp on cache hit, Async-Stream on cache miss)

Project a value at resolution N. Returns full (k, d, ε_micros, all materialized derived properties).

Request:
```json
{
  "msg_type": "request",
  "command": "project",
  "session_id": "...",
  "value_spec": {"by": "repr", "value_repr": "ζ(3)"},
  "N": 27720
}
```

Response (cache hit — Pattern 1):
```json
{
  "msg_type": "response",
  "projection": {
    "projection_id": 847,
    "value_id": 42,
    "N": 27720,
    "sign": 1,
    "k": 7360,
    "d": 693,
    "eps_micros": -85,
    "d_factorization": "3^2·7·11",
    "gaussian_signature": "D^2·D·D",
    "is_all_inert": 0,
    "is_all_split": 0,
    "is_ramified_present": 0,
    "coprime_skeleton": 0,
    "tightness_micros": 999999,
    "di_distance_micros": 1,
    "manifold_state": "PDT",
    "elegance_symmetry": "40.0",
    "elegance_simplicity": null,
    "elegance_universal": null,
    "coupling_xi": {"num": 137, "den": 480080},
    "variance_vnk": null,
    "fqg_quadrant": null,
    "palindromic_partner_d": 27027,
    "geometric_perspective": "lcm_tower",
    "cf_quality": null,
    "address_id": 12847,
    "reference_count": 47,
    "cache_hit": true
  }
}
```

On cache miss: Manager sends `ack` → computes at 361 dps via MPFR → sends `complete` with the same projection structure plus `"cache_hit": false`.

**Operation 7: `pullback`**

Pattern: Request-Response

Recover the original value from a lattice triple via the bijection Π_N⁻¹.

Request:
```json
{
  "msg_type": "request",
  "command": "pullback",
  "session_id": "...",
  "N": 27720,
  "k": 7360,
  "eps_micros": -85
}
```

Response:
```json
{
  "msg_type": "response",
  "recovered_value_361dps": "1.20205690315959428539973816151144999076498629234049888179227155534183820578631309018645587360933525814619915...",
  "value_id": 42,
  "is_known_value": true,
  "known_repr": "ζ(3)"
}
```

**Operation 8: `escalate`**

Pattern: Async-Stream

Full §7.11 tower escalation. Streams each landmark as computed, plus CF analysis results. The tower does not terminate — it runs until d stabilizes across ⌈1/K⌉ = 2 consecutive LCM landmarks, or until cancelled/session-managed.

Request:
```json
{
  "msg_type": "request",
  "command": "escalate",
  "session_id": "...",
  "value_spec": {"by": "decimal", "decimal_361dps": "1.61803398874989484820458683436563811772030917980576286213544862270526046281890244970720720418939113748475..."},
  "input_path": "A",
  "starting_N": 12,
  "max_landmarks": null,
  "include_cf_analysis": true
}
```

Stream messages (one per landmark):
```json
{
  "msg_type": "stream",
  "operation_id": "...",
  "sequence": 0,
  "landmark": {
    "N": 12,
    "k": 8,
    "d": 3,
    "eps_micros": 33090,
    "home_classification": "intermediate_home",
    "d_stable_count": 0,
    "elegance_universal": "1.287",
    "tightness_micros": 750563,
    "all_derived_properties": { ... }
  }
}
```

```json
{
  "msg_type": "stream",
  "operation_id": "...",
  "sequence": 1,
  "landmark": {
    "N": 36,
    "k": 25,
    "d": 36,
    "eps_micros": -240,
    "home_classification": "false_resolution",
    "false_resolution_note": "Sub-Koide ε at 36ET but d changes at 60ET when prime 5 becomes native",
    "d_stable_count": 0,
    ...
  }
}
```

```json
{
  "msg_type": "stream",
  "operation_id": "...",
  "sequence": 2,
  "landmark": {
    "N": 60,
    "k": 42,
    "d": 10,
    "eps_micros": 18045,
    "home_classification": "intermediate_home",
    "d_stable_count": 1,
    ...
  }
}
```

CF analysis stream (fires when CF method completes — may arrive before tower stabilizes):
```json
{
  "msg_type": "stream",
  "operation_id": "...",
  "sequence": 5,
  "cf_analysis": {
    "cf_home_convergent_p": 42,
    "cf_home_convergent_q": 10,
    "cf_home_quality": 18,
    "cf_home_d": 10,
    "cf_eps_micros": 18045,
    "cf_classification": "cf_home",
    "cf_elegance": "14.73",
    "tower_agreement": "agreed",
    "all_convergents": [
      {"n": 0, "p": 0, "q": 1, "a_next": 8},
      {"n": 1, "p": 1, "q": 1, "a_next": 2},
      {"n": 2, "p": 8, "q": 5, "a_next": 3},
      {"n": 3, "p": 42, "q": 10, "a_next": 18}
    ]
  }
}
```

Complete message:
```json
{
  "msg_type": "complete",
  "operation_id": "...",
  "trajectory_summary": {
    "value_id": 9847,
    "value_repr": "φ",
    "total_landmarks": 8,
    "home_classification": "persistent_home",
    "home_d": 10,
    "home_N_stabilized_at": 60,
    "home_eps_micros": 18045,
    "false_resolutions": [{"N": 36, "d": 36, "eps_micros": -240}],
    "cf_home_d": 10,
    "cf_tower_agreement": true,
    "cross_tower_elegance": "4.827"
  }
}
```

**Operation 9: `k_arithmetic`**

Pattern: Request-Response

Lattice-algebraic operations: multiply (k-add), reciprocate (k-negate), power (k-scale). These are structurally exact (integer k-arithmetic) — no MPFR computation needed.

Request:
```json
{
  "msg_type": "request",
  "command": "k_arithmetic",
  "session_id": "...",
  "operation_type": "multiply",
  "operands": [
    {"by": "repr", "value_repr": "ζ(3)"},
    {"by": "repr", "value_repr": "π"}
  ],
  "N": 27720
}
```

Response:
```json
{
  "msg_type": "response",
  "result": {
    "operation_performed": "k_addition",
    "input_k_values": [7360, 45779],
    "result_k": 53139,
    "result_d": 27720,
    "result_eps_micros": 120,
    "result_value_id": 12847,
    "result_361dps": "3.77574844285105464946...",
    "equation_id": 847291,
    "cache_hit": false
  }
}
```

For `reciprocate`: one operand, k → −k. For `power`: one operand + `"exponent": <integer>`, k → n·k.

**Operation 10: `lattice_add`**

Pattern: Adaptive

Sempaevum-native addition: value-space computation + lattice reprojection. Exact at 361 dps.

Request:
```json
{
  "msg_type": "request",
  "command": "lattice_add",
  "session_id": "...",
  "operand_a": {"by": "repr", "value_repr": "π"},
  "operand_b": {"by": "repr", "value_repr": "e"},
  "N": 12
}
```

Response (cache hit):
```json
{
  "msg_type": "response",
  "result": {
    "sum_361dps": "5.85987448204883847382...",
    "sum_value_id": 8472,
    "projection_at_N": {
      "N": 12,
      "k": 31,
      "d": 12,
      "eps_micros": 1847,
      ...
    },
    "equation_id": 847292,
    "cache_hit": true
  }
}
```

**Operation 11: `evaluate_function`**

Pattern: Adaptive

EML tree evaluation: any special or elementary function at 361 dps via MPFR/Arb/FLINT.

Request:
```json
{
  "msg_type": "request",
  "command": "evaluate_function",
  "session_id": "...",
  "function": "zeta",
  "arguments": [
    {"by": "decimal", "decimal_361dps": "3.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"}
  ],
  "N": 27720
}
```

Response (cache hit):
```json
{
  "msg_type": "response",
  "result": {
    "function_evaluated": "zeta(3)",
    "result_361dps": "1.20205690315959...",
    "result_value_id": 42,
    "projection_at_N": { ... },
    "equation_id": 1847,
    "cache_hit": true
  }
}
```

Supported `function` values: `"sin"`, `"cos"`, `"tan"`, `"asin"`, `"acos"`, `"atan"`, `"atan2"`, `"exp"`, `"log"`, `"log2"`, `"log10"`, `"sqrt"`, `"cbrt"`, `"pow"`, `"abs"`, `"floor"`, `"ceil"`, `"round"`, `"zeta"`, `"gamma"`, `"lgamma"`, `"digamma"`, `"beta"`, `"polylog"`, `"hypergeometric"`, `"erf"`, `"erfc"`, `"bessel_j"`, `"bessel_y"`, `"airy_ai"`, `"airy_bi"`, `"eml"` (the EML Sheffer operator eml(x,y) = exp(x) − ln(y)), and any composition thereof via `"expression"` form. The function list is extensible — new functions added via §7.14 without API version change.

---

#### 7.16.8 Domain 3 — Value Management

**Operation 12: `store_value`**

Pattern: Async-Stream

Ingest a value through the full §7.11 core projection procedure. Streams escalation progress. This is the primary ingestion pathway for individual values from any source.

Request:
```json
{
  "msg_type": "request",
  "command": "store_value",
  "session_id": "...",
  "value_spec": {"by": "expression", "expression": "m_proton / m_electron"},
  "input_path": "A",
  "r0_value_spec": {"by": "repr", "value_repr": "m_electron"},
  "r0_substrate_description": "electron rest mass",
  "tags": [
    {"namespace": "domain", "value": "particle_physics"},
    {"namespace": "source", "value": "PDG_2024"}
  ]
}
```

Streams landmarks per `escalate`, then completes with full structural profile.

**Operation 13: `batch_store`**

Pattern: Async-Stream

Batch ingest with user-chosen atomicity.

Request:
```json
{
  "msg_type": "request",
  "command": "batch_store",
  "session_id": "...",
  "atomicity": "per_value",
  "values": [
    {
      "value_spec": {"by": "decimal", "decimal_361dps": "1836.15267343..."},
      "input_path": "A",
      "tags": [{"namespace": "particle", "value": "proton_electron_mass_ratio"}]
    },
    {
      "value_spec": {"by": "decimal", "decimal_361dps": "1838.68366173..."},
      "input_path": "A",
      "tags": [{"namespace": "particle", "value": "neutron_electron_mass_ratio"}]
    }
  ]
}
```

Streams per-value results. On `"atomicity": "atomic"`, any failure rolls back ALL values in the batch and the complete message includes per-value failure reasons.

**Operation 14: `get_value`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "get_value",
  "session_id": "...",
  "value_spec": {"by": "repr", "value_repr": "ζ(3)"}
}
```

Response includes: full `values` row (value_id, hash, repr, mpf as base64, precision, r_form, R₀ references, input_path, compliance flags, reference_count, cross_tower_elegance, CF home data), all projections at all N, all tags, all relationships (subject or object), all derivations, all equations involving this value.

**Operation 15: `search_values`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "search_values",
  "session_id": "...",
  "filters": {
    "repr_pattern": "ζ(%)",
    "input_path": "B",
    "n1_compliant": true,
    "tag_filters": [{"namespace": "domain", "value": "number_theory"}],
    "cross_tower_elegance_min": "1.0833",
    "cf_home_d": 693
  },
  "order_by": "cross_tower_elegance",
  "order_direction": "desc",
  "limit": 50,
  "offset": 0
}
```

Response: array of matching value summaries (value_id, repr, home_d, home_classification, cross_tower_elegance, reference_count, tags), plus total_count for pagination.

**Operation 16: `get_value_trajectory`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "get_value_trajectory",
  "session_id": "...",
  "value_spec": {"by": "id", "value_id": 42}
}
```

Response: ordered sequence of all projections across the LCM tower (each with full derived properties), home classification per landmark, false resolutions flagged, CF analysis results, trajectory pattern_id if the trajectory has been promoted to a pattern.

**Operation 17: `query_cf_analysis`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_cf_analysis",
  "session_id": "...",
  "value_spec": {"by": "id", "value_id": 42}
}
```

Response: all CF convergents (n, p, q, a_next), the maximal-quality convergent, d_home from CF, ε_CF, CF classification (cf_deep_home/cf_home/cf_marginal), CF elegance, tower agreement status, tower-CF comparison details.

---

#### 7.16.9 Domain 4 — Address & Attractor Operations

**Operation 18: `query_address`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_address",
  "session_id": "...",
  "N": 27720,
  "k": 7360,
  "d": 693
}
```

Response: address record (address_id, N, k, d, eps_class, members_count, first_member, Gaussian signature, d_factorization), list of all values at this address, list of generators covering this address, attractor status, coprime_skeleton membership.

**Operation 19: `query_family`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_family",
  "session_id": "...",
  "N": 27720,
  "d": 693
}
```

Response: all occupied addresses in this d-family at this N, each with k, ε_micros, members_count, occupant value summaries.

**Operation 20: `query_attractor`**

Pattern: Request-Response

Request (by value):
```json
{
  "msg_type": "request",
  "command": "query_attractor",
  "session_id": "...",
  "value_spec": {"by": "repr", "value_repr": "ζ(3)"},
  "N": 27720
}
```

Response: all attractor memberships for this value at this N (addresses with members_count > 1), with full member lists and structural classification.

**Operation 21: `find_nearest`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "find_nearest",
  "session_id": "...",
  "value_spec": {"by": "decimal", "decimal_361dps": "1.41421356237..."},
  "N": 12,
  "top_k": 5
}
```

Response: top-k nearest occupied addresses ranked by lattice distance, with occupant details.

**Operation 22: `query_coprime_skeleton`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_coprime_skeleton",
  "session_id": "...",
  "N": 12,
  "structural_class_filter": null
}
```

Response: all coprime-skeleton members at this N (addresses where gcd(|k|, N) = 1), count, density, theoretical limit 6/π².

---

#### 7.16.10 Domain 5 — Equation & Computation

**Operation 23: `compute`**

Pattern: Adaptive

Evaluate ANY mathematical expression at 361 dps. The three backbones (Webb, Cascade, EML) subsume all of mathematics — any valid expression is accepted. Memoized forever.

Request:
```json
{
  "msg_type": "request",
  "command": "compute",
  "session_id": "...",
  "expression": "sin(π/4) + sqrt(2)/2",
  "N": 12
}
```

Response (cache hit):
```json
{
  "msg_type": "response",
  "result": {
    "expression_canonical": "sin(pi/4)+sqrt(2)/2",
    "result_361dps": "1.41421356237309504880168872420969807856967187537694...",
    "result_value_id": 847,
    "equation_id": 12847,
    "equation_hash": "a1b2c3...",
    "reference_count": 12,
    "projection_at_N": {
      "N": 12,
      "k": 6,
      "d": 2,
      "eps_micros": 0,
      ...
    },
    "cache_hit": true
  }
}
```

**Operation 24: `batch_compute`**

Pattern: Request-Response

Array of expressions in, array of results out. Single pipe round-trip. Each expression individually memoized. Critical for game/emulator frame budgets — a game submits its frame's computations in one batch and gets all results within the latency budget.

Request:
```json
{
  "msg_type": "request",
  "command": "batch_compute",
  "session_id": "...",
  "expressions": [
    {"expression": "sin(0.7853981633974483)", "N": 12},
    {"expression": "sqrt(2.0)", "N": 12},
    {"expression": "1.5 * 9.81", "N": 12},
    {"expression": "cos(1.2217304763960306)", "N": 12}
  ]
}
```

Response:
```json
{
  "msg_type": "response",
  "results": [
    {"index": 0, "result_361dps": "0.70710678118654...", "value_id": 847, "equation_id": 1001, "cache_hit": true},
    {"index": 1, "result_361dps": "1.41421356237309...", "value_id": 23, "equation_id": 1002, "cache_hit": true},
    {"index": 2, "result_361dps": "14.71500000000000...", "value_id": 94828, "equation_id": 1003, "cache_hit": false},
    {"index": 3, "result_361dps": "0.34202014332566...", "value_id": 94829, "equation_id": 1004, "cache_hit": false}
  ],
  "total": 4,
  "cache_hits": 2,
  "cache_misses": 2,
  "total_time_ns": "847000"
}
```

No individual errors in batch_compute — if an expression fails, its result entry contains `"error": { ... }` with the error details, and the remaining expressions still complete. batch_compute never stops on a single failure.

**Operation 25: `get_equation`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "get_equation",
  "session_id": "...",
  "equation_id": 12847
}
```

Response: full equation record (equation_id, hash, canonical_form, latex, form_class, operation_type, input value_ids, output value_id, reference_count, first_derived, last_referenced).

**Operation 26: `search_equations`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "search_equations",
  "session_id": "...",
  "filters": {
    "form_class": "lattice_multiplication",
    "operation_type": "multiply",
    "reference_count_min": 100
  },
  "order_by": "reference_count",
  "order_direction": "desc",
  "limit": 20
}
```

Response: matching equations ranked by reference_count, with summaries.

**Operation 27: `resolve_indeterminate`**

Pattern: Async-Stream

Submit an indeterminate form ([0/0], [∞/∞], [0×∞], [∞−∞]). Run L'Hôpital iterations — the Traverser's navigation algorithm. Stream each iteration. Return resolution (value enters via Path A) or pure-T classification.

Request:
```json
{
  "msg_type": "request",
  "command": "resolve_indeterminate",
  "session_id": "...",
  "form_type": "0/0",
  "numerator_expression": "sin(x)",
  "denominator_expression": "x",
  "at_variable": "x",
  "at_value": "0",
  "max_iterations": 25
}
```

Stream (one per L'Hôpital iteration):
```json
{
  "msg_type": "stream",
  "operation_id": "...",
  "sequence": 0,
  "iteration": {
    "iteration_index": 0,
    "numerator_derivative": "cos(x)",
    "denominator_derivative": "1",
    "evaluated_at": "0",
    "numerator_value": "1.00000...",
    "denominator_value": "1.00000...",
    "ratio": "1.00000...",
    "resolved": true,
    "resolved_value_361dps": "1.00000..."
  }
}
```

Complete:
```json
{
  "msg_type": "complete",
  "operation_id": "...",
  "resolution": {
    "resolved": true,
    "resolved_value_361dps": "1.00000...",
    "resolved_value_id": 1,
    "iterations_taken": 1,
    "classification": "derivative_resolvable",
    "projection_at_N12": { ... },
    "lhopital_chain_relationship_id": 8472
  }
}
```

For pure-T (L'Hôpital fails): `"resolved": false`, `"classification": "pure_T"`, `"failure_reason": "max_iterations_exhausted"`, `"manifold_state": "PT"`, no projection created.

---

#### 7.16.11 Domain 6 — Relationship & Derivation Operations

**Operation 28: `query_relationships`**

Pattern: Request-Response

Get relationships involving an entity. Supports querying by subject, object, or both. Optional class filter.

Request:
```json
{
  "msg_type": "request",
  "command": "query_relationships",
  "session_id": "...",
  "subject_id": 42,
  "subject_type": "value",
  "relationship_class": "same_address",
  "limit": 50,
  "offset": 0
}
```

Response: array of matching relationships, each with: relationship_id, class, subject (id+type), object (id+type), metadata (class-specific, decoded from blob), discovered_at, confirmation_count, is_permanent. Also supports `"object_id"` + `"object_type"` for reverse queries, and both simultaneously for specific-pair queries.

**Operation 29: `create_relationship`**

Pattern: Request-Response

Create an explicit non-lattice-algebraic relationship. Lattice-algebraic relationships (co-location, reciprocal, power, palindromic) are computed from structure and never created via API.

Request:
```json
{
  "msg_type": "request",
  "command": "create_relationship",
  "session_id": "...",
  "relationship_class": "mass_ratio_triple",
  "subject_id": 847,
  "subject_type": "value",
  "object_id": 848,
  "object_type": "value",
  "metadata": {
    "mass_a_id": 847,
    "mass_b_id": 848,
    "ratio_value_id": 849
  }
}
```

Response: the created relationship record with relationship_id.

**Operation 30: `query_derivations`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_derivations",
  "session_id": "...",
  "target_id": 42,
  "target_type": "value"
}
```

Response: all derivation chains targeting this entity, each with: derivation_id, chain blob (decoded to array of steps), primitives_used, tools_applied, document_reference, all input entities (via derivation_inputs), reproduced_count.

**Operation 31: `create_derivation`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "create_derivation",
  "session_id": "...",
  "target_id": 42,
  "target_type": "value",
  "chain_steps": [
    {"step": 1, "tool": "Identification", "description": "Identify P: spacetime manifold", "inputs": []},
    {"step": 2, "tool": "Descriptor Gap", "description": "Gap: missing D for cubic sublattice", "inputs": [{"id": 20, "type": "equation"}]},
    {"step": 3, "tool": "Subsumption", "description": "Verify: d=3 subsumes QCD without remainder", "inputs": [{"id": 42, "type": "value"}]}
  ],
  "primitives_used": "P (spacetime), D (cubic descriptor), T (strong force traverser)",
  "tools_applied": "Identification, Descriptor Gap, Subsumption",
  "document_reference": "Sempaevum Paper §10.3"
}
```

Response: the created derivation record with derivation_id.

---

#### 7.16.12 Domain 7 — Pattern & Generator Operations

**Operation 32: `query_patterns`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_patterns",
  "session_id": "...",
  "filters": {
    "pattern_class": "attractor_cluster",
    "hierarchy_elegance_min": "1.0833",
    "member_entity_id": 42,
    "member_entity_type": "value"
  },
  "order_by": "hierarchy_elegance",
  "order_direction": "desc",
  "limit": 20
}
```

Response: matching patterns with: pattern_id, class, member count, hierarchy_elegance, member list (decoded from blob), definition (decoded), formed_at, reference_count.

**Operation 33: `query_generators`**

Pattern: Request-Response

Get generators covering a value or address, ranked by K-complexity.

Request:
```json
{
  "msg_type": "request",
  "command": "query_generators",
  "session_id": "...",
  "value_spec": {"by": "id", "value_id": 42},
  "N": 27720
}
```

Response: all generators whose domain includes this value's address at this N. Each with: gen_id, gen_type (WEBB/CASCADE/EML), definition (decoded), address_range, member_count, K_complexity_ratio (gen_def_bytes / gen_coverage_bytes), verification_count, discovered_at, backbone_layer (L₁/L₂/L₃).

**Operation 34: `propose_generator`**

Pattern: Async-Stream

Request:
```json
{
  "msg_type": "request",
  "command": "propose_generator",
  "session_id": "...",
  "generator": {
    "backbone_layer": "L3",
    "definition": "eml(eml(1, 1), 1)",
    "claimed_address_range": {"N_min": 12, "N_max": 27720, "d_min": 1, "d_max": 12},
    "description": "Proposed EML tree for exponential family"
  }
}
```

Streams: verification steps (evaluating generator at each canonical N, comparing against known values, computing E_hierarchy). Complete: accepted or rejected with full verification report.

**Operation 35: `query_generator_status`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_generator_status",
  "session_id": "...",
  "value_spec": {"by": "id", "value_id": 42}
}
```

Response: `"status": "generator_found"` (with gen_id) or `"status": "search_active"` (with search history: candidates tried, failure reasons, Branch A/B/bridge progress) or `"status": "search_deferred"` (with deferral reason and reactivation conditions). Never `"search_closed"` — the case is never closed (§3.16).

---

#### 7.16.13 Domain 8 — Event Operations

**Operation 36: `query_events`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_events",
  "session_id": "...",
  "filters": {
    "event_class": "t_burst",
    "time_range": {"start_ns": "1714742400000000000", "end_ns": "1714828800000000000"},
    "tower_id": 1,
    "t_time_traverser_id": 42
  },
  "order_by": "event_timestamp",
  "order_direction": "asc",
  "limit": 100
}
```

Response: matching events with full metadata, D-time/T-time/P-time coordinates, tower context, triggered relationships/patterns.

**Operation 37: `log_event`**

Pattern: Request-Response

Connected programs log their own events.

Request:
```json
{
  "msg_type": "request",
  "command": "log_event",
  "session_id": "...",
  "event_class": "ghost_detection",
  "subject_id": 42,
  "subject_type": "value",
  "tower_id": 8,
  "t_time_traverser_id": 42,
  "metadata": {
    "sigma_count": 3.7,
    "waveform_window_position": 127,
    "projection_id_observed": 847
  }
}
```

Response: the created event record with event_id, assigned D-time coordinates, sequence_number.

**Operation 38: `replay_events`**

Pattern: Async-Stream

Request:
```json
{
  "msg_type": "request",
  "command": "replay_events",
  "session_id": "...",
  "replay_filter": {
    "session_id": "conscious_ai_2026-05-02_001",
    "event_classes": ["ghost_detection", "t_burst", "gaze_event"]
  },
  "playback_rate": 1.0
}
```

Streams events one-by-one in original sequence order, paced at `playback_rate` × original timing. `playback_rate: 0` means deliver as fast as possible (no pacing).

---

#### 7.16.14 Domain 9 — Tower & Family Operations

**Operation 39: `query_towers`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_towers",
  "session_id": "...",
  "tower_id": null,
  "tower_name": null,
  "include_children": true
}
```

With both null: returns all towers. With tower_id or tower_name: returns that specific tower. `include_children`: includes full child tower tree via recursive CTE on parent_tower_id.

Response: array of tower records with: tower_id, name, p_substrate_descriptor, r0_value_id, r0_natural_units, parent_tower_id, nesting_depth, birth triad references, operational_n, accessible_d_families_mask (decoded to array of d values), physics_metadata, children (if requested).

**Operation 40: `create_tower`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "create_tower",
  "session_id": "...",
  "tower": {
    "tower_name": "civ4_caveman2cosmos_instance_1",
    "p_substrate_descriptor": "civilization_simulation_state_space",
    "r0_value": "0.05000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "r0_description": "1/20 second — one game tick at 20 TPS",
    "r0_natural_units": "seconds per tick",
    "operational_n": 12,
    "parent_tower_name": "digital_3ghz_x86",
    "physics_metadata": {
      "game_engine": "Civilization IV",
      "mod": "Caveman2Cosmos",
      "tick_rate_hz": 20
    }
  }
}
```

Response: created tower record with tower_id, auto-computed accessible_d_families_mask, nesting_depth. The R₀ value is run through §7.11 escalation.

**Operation 41: `query_harmonic_families`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_harmonic_families",
  "session_id": "...",
  "filters": {
    "axis": "real",
    "fqg_quadrant": "CR",
    "gaussian_prime_class": "D-type (inert)"
  }
}
```

Response: matching families from the 24-family catalog.

**Operation 42: `query_fqg_cells`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_fqg_cells",
  "session_id": "...",
  "d_r": 3,
  "d_theta": 4
}
```

Response: cell record with d_combined, combined_family_id, is_off_axis, is_lcm_amplification, is_full_resolution, occupancy_count, canonical_particle_or_phenomenon.

Also supports filter mode: `"filters": {"d_combined": 12, "is_lcm_amplification": true}` to find matching cells.

**Operation 43: `query_combined_families`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_combined_families",
  "session_id": "...",
  "filters": {
    "range_class": "middle_extended",
    "d_combined": 35
  }
}
```

Response: matching combined families with contributing cells, structural meaning, Gaussian factorization, first_native_lattice_n, known correlations.

**Operation 44: `query_sublattice_families`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_sublattice_families",
  "session_id": "...",
  "n": 420,
  "tower_id": null,
  "only_newly_introduced": false,
  "only_lcm_landmarks": false
}
```

Response: all sublattice families at this N (all divisors of N), each with: d, gcd_k_n, phi_d, member_lattice_point_count, is_lcm_landmark, is_newly_introduced, related harmonic family references.

---

#### 7.16.15 Domain 10 — File & Stream Operations

**Operation 45: `ingest_file`**

Pattern: Async-Stream

Request:
```json
{
  "msg_type": "request",
  "command": "ingest_file",
  "session_id": "...",
  "file_path": "C:\\Users\\Mike\\data\\hydrogen_spectrum.csv",
  "file_type": "csv",
  "adapter_options": {
    "delimiter": ",",
    "header_row": true,
    "value_columns": [2, 3],
    "r0_value_spec": {"by": "repr", "value_repr": "Rydberg_energy"},
    "skip_rows": 0
  },
  "tags": [
    {"namespace": "domain", "value": "spectroscopy"},
    {"namespace": "source", "value": "hydrogen_experiment_2026"}
  ]
}
```

Streams: seed extraction progress (each seed extracted), per-seed projection results (each seed through §7.11). Complete: ingestion summary (total seeds, projections, attractors found, generators matched).

Supported `file_type` values: `"csv"`, `"pdf"`, `"markdown"`, `"binary"`, `"image"`, `"audio"`, `"python_output"`, `"etpl_output"`, `"auto"` (auto-detect). Each type uses its adapter per §7.12.

**Operation 46: `ingest_stream_start`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "ingest_stream_start",
  "session_id": "...",
  "stream_id": "gps_receiver_001",
  "sensor_domain": "gps",
  "r0_value_spec": {"by": "repr", "value_repr": "light_time_second"},
  "expected_data_format": {"fields": ["latitude", "longitude", "altitude", "timestamp_gps"], "types": ["decimal", "decimal", "decimal", "uint64"]},
  "buffer_size": 1000
}
```

Response: stream channel opened, stream_id confirmed, R₀ resolved to value_id.

**Operation 47: `ingest_stream_data`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "ingest_stream_data",
  "session_id": "...",
  "stream_id": "gps_receiver_001",
  "data_points": [
    {"latitude": "37.7749295000...", "longitude": "-122.4194155000...", "altitude": "16.000...", "timestamp_gps": "1714742401000000000"},
    {"latitude": "37.7749310000...", "longitude": "-122.4194160000...", "altitude": "16.100...", "timestamp_gps": "1714742402000000000"}
  ]
}
```

Response: per-point ingestion status (projected, stored, attractor memberships found).

**Operation 48: `ingest_stream_end`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "ingest_stream_end",
  "session_id": "...",
  "stream_id": "gps_receiver_001"
}
```

Response: stream finalized, total data points ingested, discoveries from accumulated stream data, anomalies detected.

**Operation 49: `retrieve_file`**

Pattern: Async-Stream

Regenerate an ingested file from generators per §7.10 file retrieval.

Request:
```json
{
  "msg_type": "request",
  "command": "retrieve_file",
  "session_id": "...",
  "provenance": {
    "source_file_name": "hydrogen_spectrum.csv",
    "ingestion_session_id": "spectroscopy_2026-05-01_001"
  },
  "output_path": "C:\\Users\\Mike\\output\\hydrogen_spectrum_regenerated.csv"
}
```

Streams reconstruction progress (generator evaluations). Complete: output file written, structural verification (regenerated content matches stored content).

**Operation 50: `retrieve_stream`**

Pattern: Async-Stream

Request:
```json
{
  "msg_type": "request",
  "command": "retrieve_stream",
  "session_id": "...",
  "stream_provenance": {"stream_id": "gps_receiver_001", "session_id": "gps_2026-05-02_001"},
  "mode": "enhanced",
  "target_resolution_hz": 10,
  "output_format": "csv"
}
```

Streams reconstructed data points. Enhanced mode evaluates discovered generators at intermediate timestamps (structural interpolation). Faithful mode re-emits at original rate.

---

#### 7.16.16 Domain 11 — Subscription & Notification

**Operation 51: `subscribe`**

Pattern: Request-Response (then push notifications)

Request:
```json
{
  "msg_type": "request",
  "command": "subscribe",
  "session_id": "...",
  "filter": {
    "event_classes": ["t_burst", "ghost_detection", "palindromic_cascade_trigger"],
    "address_ranges": [{"N": 27720, "d": 693}],
    "pattern_classes": ["attractor_cluster"],
    "value_ids": [42],
    "generator_supersession": true,
    "discovery_events": true,
    "manifold_state_transitions": true,
    "akashic_shrinkage": true
  }
}
```

Response: `"subscription_id": "<uuid>"`.

Subsequent notifications pushed asynchronously:
```json
{
  "msg_type": "notification",
  "subscription_id": "<uuid>",
  "notification_type": "generator_supersession",
  "content": {
    "generator_id": 4828,
    "entries_absorbed": 1247,
    "generator_def_bytes": 128,
    "absorbed_raw_bytes": 180815,
    "k_complexity_improvement_percent": "99.93",
    "generator_to_memoized_ratio_before": {"num": 4827, "den": 847293},
    "generator_to_memoized_ratio_after": {"num": 4828, "den": 846046},
    "akashic_file_size_before_bytes": "1073741824",
    "akashic_file_size_after_bytes": "1073561009",
    "descriptor_gap_closed_bytes": "180815"
  }
}
```

The `akashic_shrinkage` notification fires when the .akashic file physically shrinks due to generator discovery absorbing memoized entries. This is ALSO displayed on the GUI dashboard as a visible indicator: "Descriptor Gap closed: 1,247 entries absorbed by generator G_4828, K-complexity improved by 99.93%, file size reduced by 176.6 KB."

**Operation 52: `unsubscribe`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "unsubscribe",
  "session_id": "...",
  "subscription_id": "<uuid>"
}
```

Response: `"unsubscribed": true`.

**Operation 53: `query_subscriptions`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_subscriptions",
  "session_id": "..."
}
```

Response: array of active subscriptions for this connection, each with subscription_id and filter.

---

#### 7.16.17 Domain 12 — Active Probing & Analysis

**Operation 54: `send_probe`**

Pattern: Async-Stream

Deliberately inject T-content at a target lattice address — active interrogation of the lattice per §3.9 active probing. Streams probe→response→silence events.

Request:
```json
{
  "msg_type": "request",
  "command": "send_probe",
  "session_id": "...",
  "target_address": {"N": 27720, "k": 7360, "d": 693},
  "probe_amplitude": "1.0",
  "probe_phase": "0.0",
  "response_window_ns": "5000000000",
  "tower_id": 1
}
```

Streams: `t_signal_probe_sent` event, then either `t_signal_probe_response` (with response_delay, amplitude, address) or `t_signal_probe_silence` (no response within window). If response amplitude crosses materialization threshold: `materialization_threshold_crossed` event.

**Operation 55: `query_probes`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_probes",
  "session_id": "...",
  "target_address": {"N": 27720, "k": 7360, "d": 693},
  "time_range": {"start_ns": "1714742400000000000", "end_ns": "1714828800000000000"}
}
```

Response: all probes sent to this address within the time range, each with probe event, response event (or silence), materialization events, probe_response_pair relationships.

**Operation 56: `evaluate_gaze`**

Pattern: Request-Response

Compute the Complete Gaze Equation for given inputs.

Request:
```json
{
  "msg_type": "request",
  "command": "evaluate_gaze",
  "session_id": "...",
  "inputs": {
    "t_intent": "1.35",
    "focus": "0.95",
    "distance": "0.80",
    "n": 12,
    "k": 3
  }
}
```

Response:
```json
{
  "msg_type": "response",
  "gaze": {
    "F_w": "2.00390625",
    "R_k": "4.60",
    "V_nk": "1.19270833...",
    "Gamma": {"num": 6, "den": 5},
    "P_detect": "0.99847...",
    "V_collapse": "0.99999...",
    "prior_status": null,
    "new_status": "LOCKED",
    "threshold_crossed": "3/2",
    "threshold_ji_name": "Perfect fifth"
  }
}
```

**Operation 57: `run_et_scan`**

Pattern: Async-Stream

Run full ET scanner analysis on a data window per `et_scanner_v7_2_COMPLETE.py` methodology. Produces complete ETSignature.

Request:
```json
{
  "msg_type": "request",
  "command": "run_et_scan",
  "session_id": "...",
  "data_source": {
    "type": "value_sequence",
    "value_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "window_size": 12
  },
  "scan_options": {
    "include_pdt_classification": true,
    "include_t_identification": true,
    "include_binding_chain_verification": true,
    "include_coherence_analysis": true,
    "include_indeterminate_analysis": true,
    "include_axiom_verification": true,
    "include_thermodynamic_verification": true,
    "include_quantum_verification": true,
    "include_spectral_analysis": true,
    "include_fractal_analysis": true,
    "include_gaze_metrics": true
  }
}
```

Streams: each scan component as completed (pdt_classification, t_identification, binding_chain, coherence, indeterminate forms, axiom verification, spectral, fractal, gaze). Complete: full ETSignature with all components.

**Operation 58: `run_anti_numerology_check`**

Pattern: Request-Response

N1/N2/N3 compliance check per Guide Part III §16-18.

Request:
```json
{
  "msg_type": "request",
  "command": "run_anti_numerology_check",
  "session_id": "...",
  "value_spec": {"by": "id", "value_id": 42},
  "r0_value_spec": {"by": "repr", "value_repr": "ℏ"},
  "quantity_description": "Apéry's constant"
}
```

Response:
```json
{
  "msg_type": "response",
  "compliance": {
    "n1_result": true,
    "n1_detail": "Dimensionless ratio: ζ(3) is already dimensionless",
    "n2_result": true,
    "n2_detail": "Substrate-derived R₀: ℏ is the cosmological tower's seed",
    "n3_result": true,
    "n3_detail": "No aspect conflation: single-valued seed, single projection",
    "overall": "PASS",
    "failure_mode": null,
    "event_id": 94827
  }
}
```

---

#### 7.16.18 Domain 13 — Traverser & Manifold State Operations

**Operation 59: `query_traverser`**

Pattern: Request-Response

Get a full Traverser profile derived from events, values, tags, projections, and derivations. The Traverser is not a separate table — this operation computes the profile from existing data per §3.10.

Request:
```json
{
  "msg_type": "request",
  "command": "query_traverser",
  "session_id": "...",
  "traverser_value_id": 42,
  "include_worldline": true,
  "include_ego_invariant": true,
  "worldline_limit": 1000
}
```

Response:
```json
{
  "msg_type": "response",
  "traverser": {
    "identity_value_id": 42,
    "type_tags": [{"namespace": "kind", "value": "traverser"}, {"namespace": "traverser_type", "value": "consciousness"}],
    "current_tower_id": 3,
    "current_tower_name": "biological_T4_capsid",
    "accumulated_t_time": 847293,
    "t_time_rate_dtau_dt": "0.99999",
    "ego_invariant": {
      "fingerprint_projections": [
        {"d": 5, "projection_id": 1001},
        {"d": 7, "projection_id": 1002},
        {"d": 8, "projection_id": 1003},
        {"d": 9, "projection_id": 1004},
        {"d": 10, "projection_id": 1005},
        {"d": 11, "projection_id": 1006}
      ],
      "derivation_id": 847
    },
    "sublattice_family_classification": {"d_r": 4, "d_theta": 6},
    "continuity_state": "continuous",
    "ghost_state": false,
    "worldline_events": [
      {"event_id": 1, "event_class": "tower_entry", "timestamp_ns": "...", "tower_id": 3},
      {"event_id": 47, "event_class": "gaze_event", "timestamp_ns": "...", "metadata": {"new_status": "DETECTED"}},
      ...
    ],
    "gaze_sequence_summary": {
      "total_gaze_events": 847,
      "locked_count": 42,
      "detected_count": 312,
      "subliminal_count": 493,
      "sustained_lock_sequences": 7
    }
  }
}
```

**Operation 60: `query_by_manifold_state`**

Pattern: Request-Response

Find all projections/values in a given manifold state.

Request:
```json
{
  "msg_type": "request",
  "command": "query_by_manifold_state",
  "session_id": "...",
  "manifold_state": "PT",
  "N": null,
  "limit": 50
}
```

Response: matching projections/values with full records. For `"PT"` (Incoherence): returns values with `input_path = 'P.T'` — the forbidden configurations that have no lattice address.

**Operation 61: `apply_three_tools`**

Pattern: Request-Response

Record an application of the Three Tools to a specific problem. Stores as events (identification_application, descriptor_gap_application, subsumption_application).

Request:
```json
{
  "msg_type": "request",
  "command": "apply_three_tools",
  "session_id": "...",
  "tools_applied": [
    {
      "tool": "identification",
      "target_description": "Digital tower's resource allocation",
      "p_identified": "Hardware: CPU, RAM, VRAM",
      "d_identified": "K=2/3 ceiling, V=1/12 headroom, α⁻¹=137 resolution",
      "t_identified": "The metabolism control loop"
    },
    {
      "tool": "descriptor_gap",
      "gap_description": "Missing D: per-sublattice coupling for computational scheduling",
      "gap_resolution": "ξ(d) = 137/((d-1)²+16) provides scheduling weight per d-family"
    },
    {
      "tool": "subsumption",
      "check_description": "Does the metabolism subsume all resource management?",
      "result": "CONFIRMED — K ceiling + α monitoring + ξ(d) scheduling covers CPU, RAM, VRAM, threads, priority. No remainder."
    }
  ],
  "subject_id": 8,
  "subject_type": "tower"
}
```

Response: event_ids for each tool application stored.

**Operation 62: `query_metabolism`**

Pattern: Request-Response

Get current metabolic state for this connection.

Request:
```json
{
  "msg_type": "request",
  "command": "query_metabolism",
  "session_id": "..."
}
```

Response: full metabolic profile — K ceiling, V headroom, allocated threads, allocated memory, current CPU/mem headroom percentages, overall pressure, dominant d-family, coupling ξ(d), scheduling weight relative to other connections, convergence ratio, metabolic refresh interval, last refresh timestamp, all connected programs' metabolic summaries (for awareness of the shared resource space).

---

#### 7.16.19 Domain 14 — Administration & Maintenance

**Operation 63: `status`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "status",
  "session_id": "..."
}
```

Response: comprehensive Manager status — database metrics (total values, projections, equations, patterns, events, generators, memoized count, generator-to-memoized ratio, .akashic file size), discovery engine state (scan in progress, last scan time, patterns discovered this session, generator candidates pending), connected clients (count, project names, metabolic summaries), session info, uptime, Omniscient status (alive, last journal entry), GUI frame rate.

**Operation 64: `query_metrics`**

Pattern: Request-Response

Detailed self-recording metrics by category per §3.1b.

Request:
```json
{
  "msg_type": "request",
  "command": "query_metrics",
  "session_id": "...",
  "categories": ["lattice_computation", "memoization", "discovery_engine", "storage"],
  "time_range": {"start_ns": "1714742400000000000", "end_ns": "1714828800000000000"},
  "granularity": "per_sample"
}
```

Response: metrics for requested categories within time range. Each sample includes all metrics in that category per the §3.1b catalog.

**Operation 65: `query_journal`**

Pattern: Request-Response

Query Omniscient and SelfRecording journals.

Request:
```json
{
  "msg_type": "request",
  "command": "query_journal",
  "session_id": "...",
  "journal": "omniscient",
  "filters": {
    "severity": ["ERROR", "TAMPER"],
    "category": "computation",
    "source_module": null,
    "time_range": {"start_ns": "1714742400000000000", "end_ns": "1714828800000000000"}
  },
  "limit": 100
}
```

Response: matching journal entries in chronological order. Each with timestamp, severity, category, source module, full context key-value pairs.

**Operation 66: `trigger_backup`**

Pattern: Async-Stream

Request:
```json
{
  "msg_type": "request",
  "command": "trigger_backup",
  "session_id": "...",
  "backup_path": "D:\\Backups\\EUDD\\Sempaevum_backup_20260503.akashic"
}
```

Streams: WAL flush status, OS snapshot signal, copy progress, verification progress (per-page CRC-32). Complete: backup verified, integrity pass/fail.

**Operation 67: `verify_integrity`**

Pattern: Async-Stream

Request:
```json
{
  "msg_type": "request",
  "command": "verify_integrity",
  "session_id": "..."
}
```

Streams: header SHA-256 check, per-page CRC-32 results (streamed as pages are checked — not held until all done), section directory consistency, generator coverage consistency, orphan detection. Complete: overall pass/fail with details on any issues.

**Operation 68: `submit_extension`**

Pattern: Request-Response

Submit a JSON extension per §7.14.

Request:
```json
{
  "msg_type": "request",
  "command": "submit_extension",
  "session_id": "...",
  "extension_json": {
    "extension_type": "event_class",
    "class_name": "game_physics_anomaly",
    "metadata_schema": {
      "physics_engine": "TEXT",
      "anomaly_type": "TEXT",
      "expected_value_361dps": "TEXT",
      "actual_value_361dps": "TEXT",
      "deviation_eps_micros": "INTEGER"
    },
    "parent_class": "sensor_anomaly_detected",
    "description": "Physics engine produced an unexpected value — structural anomaly detected",
    "proposed_by": "skyrim_bridge",
    "timestamp": "2026-05-03T00:00:00Z"
  }
}
```

Response: accepted or rejected with validation results (which of the 11 validation rules passed/failed, detailed reason for any failure).

**Operation 69: `trigger_discovery_scan`**

Pattern: Async-Stream

Request:
```json
{
  "msg_type": "request",
  "command": "trigger_discovery_scan",
  "session_id": "...",
  "scan_scope": "full",
  "time_budget_ns": "60000000000"
}
```

Streams: each discovery as it emerges (new attractors, generator candidates, promoted patterns, algebraic identities, cross-domain hits, CF quality attractors). Complete: scan summary with total discoveries.

**Operation 70: `query_provisional_categories`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_provisional_categories",
  "session_id": "..."
}
```

Response: all provisional categories awaiting review. Each with: category type (event_class, relationship_class, pattern_class), proposed name, proposed schema, data that triggered creation, creation timestamp, data count matching this provisional category.

**Operation 71: `review_provisional`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "review_provisional",
  "session_id": "...",
  "provisional_id": 7,
  "action": "accept",
  "modifications": null
}
```

Actions: `"accept"` (category becomes permanent, historical data retroactively classified), `"modify"` (provide `"modifications"` object with updated name/schema, then accept), `"reject"` (category archived with reason in `"rejection_reason"` field, data unclassified).

---

#### 7.16.20 Domain 15 — Tags, Sessions, Schema

**Operation 72: `add_tag`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "add_tag",
  "session_id": "...",
  "target_id": 42,
  "target_type": "value",
  "namespace": "domain",
  "value": "particle_physics"
}
```

Response: created tag record with tag_id, tagged_at, tagged_by (session's project name).

**Operation 73: `remove_tag`**

Pattern: Request-Response

The tag is archived (§4.3 never destroy), not deleted. A removal event is recorded.

Request:
```json
{
  "msg_type": "request",
  "command": "remove_tag",
  "session_id": "...",
  "tag_id": 847
}
```

Response: `"archived": true`, `"removal_event_id": 94828`.

**Operation 74: `query_tags`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_tags",
  "session_id": "...",
  "filters": {
    "namespace": "domain",
    "value": "particle_physics",
    "target_type": "value"
  },
  "limit": 100
}
```

Response: matching tags with full records.

**Operation 75: `query_sessions`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_sessions",
  "session_id": "...",
  "filters": {
    "project": "compressor",
    "time_range": {"start_ns": "1714656000000000000", "end_ns": "1714828800000000000"}
  },
  "limit": 20
}
```

Response: matching session records with: session_id, project, machine_id, started_at, ended_at, config_hash, notes, event_count, discovery_count.

**Operation 76: `query_schema_versions`**

Pattern: Request-Response

Request:
```json
{
  "msg_type": "request",
  "command": "query_schema_versions",
  "session_id": "..."
}
```

Response: complete schema version history with: version number, applied_at, description, migration content.

---

#### 7.16.20a Domain 16 — Seed Protocol (Operations 80–84)

**Operation 80: `generate_seed`**

Request: `{"command": "generate_seed", "data_spec": {"type": "single_value", "value": "<361-dps string>"}, "N": 12, "encoding": "single_ratio"}`

Alternative data_spec types: `"stream"` (array of values with optional delta-k encoding), `"whole_file"` (base64-encoded file bytes → integer → ℝ⁺), `"stream_sublattice_grouped"` (values grouped by d-family).

Response: `{"seed": {"k": 130, "d": 6, "eps": "<361-dps string>", "eps_precision_bits": 1200, "N": 12}, "compression_ratio": 1.45, "kolmogorov_estimate_bits": 44, "shannon_estimate_bits": 64, "encoding_used": "single_ratio"}`

For streams: response includes array of seeds with delta-k encoded if applicable.

**Operation 81: `reconstruct_from_seed`**

Request: `{"command": "reconstruct_from_seed", "seed": {"k": 130, "d": 6, "eps": "<361-dps string>", "N": 12}, "output_format": "value"}`

Alternative output_format: `"file_bytes"` (for whole-file seeds), `"stream"` (for seed streams).

Response: `{"reconstructed_value": "<361-dps string>", "round_trip_residual": "0", "reconstruction_time_ns": "1240"}`

The `round_trip_residual` MUST be `"0"` — exactly zero by algebraic identity. Any non-zero residual is an error (event class: computation_failure).

**Operation 82: `stream_seed_progressive`**

Request: `{"command": "stream_seed_progressive", "seed_id": "<id>", "start_bit": 0}`

Response (Pattern 2 — Async-Stream): progressive chunks delivering ε bits in significance order.

Each chunk: `{"bits_received": 8, "bits_total": 1200, "current_precision_cents": "0.195", "intermediate_value": "<partial reconstruction>", "monotonic_verified": true}`

Final chunk: `{"bits_received": 1200, "bits_total": 1200, "current_precision_cents": "0", "final_value": "<361-dps string>", "round_trip_residual": "0", "complete": true}`

**Operation 83: `query_seed_cache`**

Request: `{"command": "query_seed_cache", "k": 130, "d": 6, "N": 12}`

Response: `{"cached_seeds": [{"seed_id": 42, "eps": "<361-dps>", "source": "sensor_reading", "cached_at_ns": "...", "access_count": 17}], "members_count": 3, "is_attractor": true}`

Returns all seeds sharing the (k, d) lattice address — the deduplication key. If members_count > 1, the address is an attractor.

**Operation 84: `seed_dedup_check`**

Request: `{"command": "seed_dedup_check", "seed": {"k": 130, "d": 6, "eps": "<361-dps>", "N": 12}}`

Response: `{"is_exact_duplicate": false, "is_structural_duplicate": true, "existing_seed_id": 42, "delta_eps": "0.002", "bandwidth_saved_bytes": 6, "dedup_recommendation": "transmit_delta_only"}`

**Operation 85: `query_file_versions`**

Request: `{"command": "query_file_versions", "base_seed_id": 42}`

Response: `{"base_seed": {"k": 130, "d": 6, "eps_base": "<361-dps>", "file_hash": "abc..."}, "versions": [{"version": 0, "eps_cumulative": "<361-dps>", "delta_eps": "0", "timestamp_ns": "...", "file_hash": "abc..."}, {"version": 1, "eps_cumulative": "<361-dps>", "delta_eps": "0.002", "timestamp_ns": "...", "file_hash": "def...", "segments_modified": 1, "segments_unchanged": 9}], "total_versions": 2, "total_delta_storage_bytes": 48}`

Returns the complete version history of a file as a Δε chain. Any version can be reconstructed by passing its `eps_cumulative` to `reconstruct_from_seed` (Operation 81).

**Operation 86: `reconstruct_file_version`**

Request: `{"command": "reconstruct_file_version", "base_seed_id": 42, "version": 1, "output_format": "file_bytes"}`

Response: `{"reconstructed_file": "<base64-encoded file bytes>", "version": 1, "eps_used": "<361-dps cumulative>", "round_trip_residual": "0", "reconstruction_time_ns": "3400"}`

Shortcut combining `query_file_versions` + `reconstruct_from_seed`. The `round_trip_residual` MUST be `"0"` — exact reconstruction by algebraic identity.

**Operation 87: `lattice_multiply`**

Request: `{"command": "lattice_multiply", "value_id_1": 42, "value_id_2": 73, "N": 12}`

Response: `{"result_k": 155, "result_d": 12, "result_eps": "<361-dps>", "kappa": 0, "result_value_id": 198, "equation_id": 45, "d_product_vs_lcm": "equal"}`

Computes Π_N(r₁·r₂) entirely in lattice coordinates via §3.18.21 Theorem A.1. Zero pullback to underlying reals. The κ rounding correction is the T-act. Result memoized as an equation entry.

**Operation 88: `lattice_divide`**

Request: `{"command": "lattice_divide", "value_id_1": 42, "value_id_2": 73, "N": 12}`

Response: `{"result_k": -31, "result_d": 12, "result_eps": "<361-dps>", "kappa": 0, "result_value_id": 199, "equation_id": 46}`

Via §3.18.21 Theorem A.2. Same structure as multiply.

**Operation 89: `lattice_reciprocal`**

Request: `{"command": "lattice_reciprocal", "value_id": 42, "N": 12}`

Response: `{"result_k": -20, "result_d": 3, "result_eps": "<negated 361-dps>", "kappa": 0, "mirror_symmetry_holds": true, "result_value_id": 200}`

Via §3.18.21 Theorem A.3. Mirror symmetry: (−k, d, −ε). Kappa is 0 for all values away from ∂I.

**Operation 90: `lattice_power`**

Request: `{"command": "lattice_power", "value_id": 42, "exponent": 3, "N": 12}`

Response: `{"result_k": 60, "result_d": 1, "result_eps": "<361-dps>", "kappa_n": 0, "kappa_bound": 2, "result_value_id": 201}`

Via §3.18.21 Theorem A.4. |κ_n| ≤ ⌈|n|/2⌉.

**Operation 91: `cross_resolution_transition`**

Request: `{"command": "cross_resolution_transition", "value_id": 42, "N_source": 12, "N_target": 27720}`

Response: `{"k_source": 20, "eps_source": "<361-dps>", "k_target": 46200, "d_target": 27720, "eps_target": "<361-dps>", "d_family_transitions": [{"N": 60, "d": 20}, {"N": 420, "d": 210}], "reproject_avoided": true}`

Via §3.18.19 Theorem 1. Computes projection at N_target WITHOUT re-accessing underlying real. Pure lattice arithmetic on (k, ε) pairs.

**Operation 92: `cross_seed_transition`**

Request: `{"command": "cross_seed_transition", "value_id": 42, "R0_source_value_id": 5, "R0_target_value_id": 8, "N": 12}`

Response: `{"k_source": 20, "k_target": -110, "d_target": 6, "eps_target": "<361-dps>", "delta_k_exact": "<361-dps>", "d_family_changed": true}`

Via §3.18.19 Theorem 2. Converts between different R₀ references without re-accessing original measurement.

---

**Operation 110: `full_cross_tower_transition`**

Pattern: Request-Response

Full cross-tower transition: different N AND different R₀ (§3.18.19 Theorem 3). Computes Π_N₂^{R₀'} ∘ (Π_N₁^{R₀})⁻¹. Commutativity verified: (Seed∘Scale) = (Scale∘Seed) = Direct (Theorem 4).

Request: `{"command": "full_cross_tower_transition", "value_id": 42, "N_source": 12, "N_target": 420, "R0_source_value_id": 5, "R0_target_value_id": 8}`

Response: `{"k_source": 20, "k_target": -3860, "d_target": 21, "eps_target": "-0.59758...", "commutativity_verified": true, "route_A_matches": true, "route_B_matches": true, "d_family_changed": true}`

**Operation 93: `monitor_drift_rate`**

Request: `{"command": "monitor_drift_rate", "stream_id": "sensor_42", "window_ns": 1000000}`

Response: `{"dr_dt": "<361-dps>", "deps_dt": "<361-dps>", "lambda_verified": "1731.234049...", "current_eps": "<361-dps>", "cell_transition_predicted_in_ns": 45000, "d_family_current": 3, "d_family_next_if_transition": 4}`

Via §3.18.22 Theorem B.1. Computes live lattice drift rate for a sensor stream. Predicts next cell transition.

**Operation 94: `apply_restoration_control`**

Request: `{"command": "apply_restoration_control", "value_id": 42, "eps_target": "0", "tau": "1.0", "max_steps": 1000}`

Response: `{"eps_initial": "-18.2046", "eps_final": "-0.0001", "steps_applied": 847, "convergence_achieved": true, "r_final": "<361-dps>", "trajectory_event_ids": [1001, 1002, ...]}`

Via §3.18.22 Theorem B.4. Applies the healing layer's exact restoration control law. Drives ε exponentially toward target. Each step logged as epsilon_restoration_step event.

**Operation 95: `query_d_composition`**

Request: `{"command": "query_d_composition", "d1": 3, "d2": 4, "N": 12, "include_kappa": true}`

Response: `{"d1": 3, "d2": 4, "composition_set": [1, 2, 3, 6, 12], "composition_set_kappa0": [12], "residue_set_d1": [4, 8], "residue_set_d2": [3, 9], "sum_set": [7, 1, 11, 5], "phi_d1": 2, "phi_d2": 2, "lcm_d1_d2": 12, "lcm_bound_holds_kappa0": true, "lcm_violations_with_kappa": [{"d_product": 12, "exceeds_by": 0}], "d1_universal_channel": true, "composition_richness": 5}`

Via §3.18.23 Theorems C.1–C.6. Returns the COMPLETE set of achievable d_product values, both with and without κ. Includes residue sets, sum set, and structural properties.

**Operation 96: `query_power_family_sequence`**

Request: `{"command": "query_power_family_sequence", "d": 12, "N": 12, "max_power": 12}`

Response: `{"d_input": 12, "sequence": [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1], "period": 12, "returns_to_d1_at": 12, "deterministic": true}`

Via §3.18.23 Part 7. Returns the deterministic d-family under successive powers.

---

**Operation 97: `phase_project`**

Pattern: Request-Response

Project an angle θ (radians) onto the imaginary-axis lattice via Definition 11.2: Π_N^θ(θ) = (k_θ, d_θ, ε_θ) where k_θ ∈ {0,...,N−1} (mod N, U(1) compact).

Request: `{"command": "phase_project", "theta": "1.0471975511965977", "N": 12}`

Response: `{"k_theta": 2, "d_theta": 6, "eps_theta": "0.0", "theta_normalized_radians": "1.0471975511965977", "note": "pi/3 projects to k_theta=2 at N=12"}`

---

**Operation 98: `complex_project`**

Pattern: Request-Response

Full complex projection Π_N^C(z) = (k_r, k_θ, d_r, d_θ, d_c, ε_r, ε_θ) for z = r·e^{iθ}. Both axes projected simultaneously.

Request: `{"command": "complex_project", "r": "2.0", "theta": "1.5707963267948966", "N": 12}`

Response: `{"k_r": 12, "k_theta": 3, "d_r": 1, "d_theta": 4, "d_c": 4, "eps_r": "0.0", "eps_theta": "0.0", "gaussian_address": {"real": 12, "imag": 3}}`

---

**Operation 99: `complex_multiply`**

Pattern: Request-Response

Complex multiplication in two-axis lattice coordinates (§3.18.24 Theorem D.2). Each axis computes independently: real via Theorem A.1, imaginary via Theorem D.1 (same algebra + mod N wrapping).

Request: `{"command": "complex_multiply", "value_id_1": 42, "value_id_2": 73, "N": 12}`

Response: `{"k_r_product": 32, "k_theta_product": 5, "d_r": 4, "d_theta": 12, "d_c": 12, "eps_r": "3.217", "eps_theta": "-1.04", "kappa_r": 0, "kappa_theta": 1, "equation_id": 8801}`

---

**Operation 100: `complex_reciprocal`**

Pattern: Request-Response

Complex reciprocation z⁻¹ = (1/r)·e^{−iθ} in lattice coordinates (§3.18.24 Theorem D.3). k_r→−k_r, k_θ→(N−k_θ) mod N. ALL d preserved (d_r, d_θ, d_c).

Request: `{"command": "complex_reciprocal", "value_id": 42, "N": 12}`

Response: `{"k_r_inv": -12, "k_theta_inv": 9, "d_r": 1, "d_theta": 4, "d_c": 4, "eps_r": "0.0", "eps_theta": "0.0", "d_r_preserved": true, "d_theta_preserved": true, "d_c_preserved": true}`

---

**Operation 101: `complex_power`**

Pattern: Request-Response

Complex power z^n in lattice coordinates (§3.18.24 Theorem D.4). Real: Theorem A.4. Phase: (n·k_θ+κ_θ,n) mod N.

Request: `{"command": "complex_power", "value_id": 42, "exponent": 3, "N": 12}`

Response: `{"k_r_power": 36, "k_theta_power": 9, "d_r": 1, "d_theta": 4, "d_c": 4, "eps_r": "0.0", "eps_theta": "0.0", "kappa_r_n": 0, "kappa_theta_n": 0, "equation_id": 8802}`

---

**Operation 102: `phase_add`**

Pattern: Request-Response

Phase addition in imaginary-axis lattice coordinates (§3.18.24 Theorem D.1). Same algebra as real-axis Theorem A.1, with mod N wrapping for U(1) compactness.

Request: `{"command": "phase_add", "k_theta_1": 6, "eps_theta_1": "0.0", "k_theta_2": 9, "eps_theta_2": "0.0", "N": 12}`

Response: `{"k_theta_sum": 3, "d_theta_sum": 4, "eps_theta_sum": "0.0", "kappa_theta": 0, "wrapped_mod_N": true, "pre_mod_sum": 15}`

---

**Operation 103: `query_phase_differential`**

Pattern: Request-Response

Query the phase-axis conversion constant Λ_θ = 600/π ≈ 190.986 and compute phase differential dε_θ = Λ_θ·dθ (§3.18.24 Theorem D.5). For comparison, also returns Λ_r = 1200/ln2 and the axis sensitivity ratio.

Request: `{"command": "query_phase_differential", "theta": "1.0", "dtheta": "0.001"}`

Response: `{"lambda_theta": "190.98593171027440292", "lambda_r": "1731.23404906676", "ratio_lambda_r_over_theta": "9.06472028365439", "deps_theta": "0.19098593171027440292", "note": "Phase axis: uniform sensitivity. Real axis: 1/r sensitivity."}`

---

**Operation 104: `query_harmonic_composition`**

Pattern: Request-Response

Query the harmonic FQG composition table (§3.18.25 Theorem E1.1). Computed at native resolution N=27720 where all 12 harmonic families are native sublattice families. Returns the set of achievable harmonic d-products (d ≤ 12) for a given (d_r, d_θ) pair, plus the combined family d_c = lcm(d_r, d_θ).

Request: `{"command": "query_harmonic_composition", "d_r": 3, "d_theta": 4, "include_composites": false}`

Response: `{"d_r": 3, "d_theta": 4, "d_c": 12, "harmonic_products_k0": [12], "produces_only_composites": false, "quadrant": "SR+SI", "closure_set_size": 42, "no_primes_gt_12": true}`

Request (full closure): `{"command": "query_harmonic_composition", "d_r": 7, "d_theta": 5, "include_composites": true}`

Response: `{"d_r": 7, "d_theta": 5, "d_c": 35, "harmonic_products_k0": [], "composite_products": [35], "produces_only_composites": true, "quadrant": "CR+CI", "note": "d=35 is a composite harmonic family (5×7) — biological signature"}`

---

**Operation 105: `query_sublattice_fqg`**

Pattern: Request-Response

Query the sublattice FQG at a specific resolution N (§3.18.26). Returns τ(N), the grid size τ(N)², the sublattice families (divisors of N), harmonic embedding (which families are ≤ 12), dilution percentage, and growth law verification.

Request: `{"command": "query_sublattice_fqg", "N": 420}`

Response: `{"N": 420, "tau_N": 24, "fqg_cells": 576, "growth_law_verified": true, "sublattice_families": [1,2,3,4,5,6,7,10,12,14,15,20,21,28,30,35,42,60,70,84,105,140,210,420], "native_harmonic": [1,2,3,4,5,6,7,10,12], "shadow_harmonic": [8,9,11], "non_harmonic_count": 15, "harmonic_fraction_pct": 14.06, "dilution_from_base": "14.06% of N=12 base"}`

Request (d-bounce query): `{"command": "query_sublattice_fqg", "mode": "d_bounce", "value_id": 42, "tower_levels": 5}`

Response: `{"value_id": 42, "d_sequence": [3, 20, 210, 1260, 27720], "bounce_count": 4, "lattice_exact": false, "eps_at_base": "12.345"}`

---

**Operation 106: `query_composite_bridge`**

Pattern: Request-Response

Query the Composite Bridge (§3.18.27) — three-layer partition, harmonic shadow map, composite decomposition, tower-native characterization.

Request (partition): `{"command": "query_composite_bridge", "mode": "partition", "N": 27720}`

Response: `{"N": 27720, "tau_N": 96, "layer1_harmonic": {"count": 12, "families": [1,2,3,4,5,6,7,8,9,10,11,12]}, "layer2_composite": {"count": 30, "families": [14,15,18,...,132]}, "layer3_tower_native": {"count": 54, "families": [105,120,126,...]}}`

Request (shadow): `{"command": "query_composite_bridge", "mode": "shadow", "d": 105, "N": 27720}`

Response: `{"d": 105, "N": 27720, "layer": 3, "layer_name": "tower_native", "harmonic_shadow_at_12": [1,2,3,4,6,12], "has_decomposition": false, "blocking_factor": "3 × 5 × 7 — requires 3 primes simultaneously, no pair ≤ 12 supplies all three"}`

Request (decompose): `{"command": "query_composite_bridge", "mode": "decompose", "d": 35}`

Response: `{"d": 35, "layer": 2, "in_d42": true, "harmonic_pairs": [[5,7]], "note": "Biological signature: quintic × septic"}`

---

**Operation 107: `query_dI_boundary`**

Pattern: Request-Response

Query the ∂I boundary structure (§3.18.28). Multiple modes: tightness (compute t(ε) and check zones), bifurcation (enumerate B_N pairs), boundary_value (generate ∂I boundary r values), and zone_classify (coherent/twilight/∂I classification for a given ε).

Request (tightness): `{"command": "query_dI_boundary", "mode": "tightness", "N": 12}`

Response: `{"N": 12, "eps_max": "50.0", "tightness_at_boundary": "0.666666666666666...", "equals_K": true, "formula": "N/(N+6) = 12/18 = 2/3", "twilight_zone_entry": "33.333"}`

Request (bifurcation): `{"command": "query_dI_boundary", "mode": "bifurcation", "N": 12}`

Response: `{"N": 12, "distinct_pairs": 6, "B_N": [{"pair": [1,12], "positions": [0,11]}, {"pair": [2,12], "positions": [5,6]}, {"pair": [3,4], "positions": [2,9]}, {"pair": [3,12], "positions": [4,7]}, {"pair": [4,6], "positions": [3,8]}, {"pair": [6,12], "positions": [1,10]}], "palindromic": true, "all_families_participate": true, "d12_exposure": "4/6 pairs"}`

Request (zone): `{"command": "query_dI_boundary", "mode": "zone_classify", "eps": "42.5", "N": 12}`

Response: `{"eps": "42.5", "N": 12, "zone": "twilight", "tightness": "0.7017...", "distance_to_dI": "7.5", "approaching_bifurcation": true}`

---

**Operation 108: `query_backbone_bridge`**

Pattern: Request-Response

Query the Triple Backbone Bridge (§3.18.29). Modes: decompose (factor projection through three backbones), catalan (Catalan-lattice correspondence), eml_chain (compute value via EML primitives), webb_function (compute function via Webb stroke).

Request (decompose): `{"command": "query_backbone_bridge", "mode": "decompose", "r": "3.14159265358979"}`

Response: `{"r": "3.14159...", "cont_x": "18.863...", "t_act_k": 19, "t_act_delta": "-0.137...", "disc_d": 12, "disc_eps": "-13.68...", "matches_direct": true, "backbones": {"EML": "Cont(r)=N·log₂(r)", "T": "round(x)", "Webb": "gcd→d"}}`

Request (catalan): `{"command": "query_backbone_bridge", "mode": "catalan", "n": 6}`

Response: `{"n": 6, "C_n": 132, "et_match": "d_max = N(N-1) = 132", "unique_at_N12": true, "uniqueness_proof": "C_{N/2} = N(N-1) iff N=12"}`

---

**Operation 111: `query_transfer_tensor`**

Pattern: Request-Response

Query the Harmonic Transfer Tensor (§3.18.30). Returns transfer rates, impedance-weighted efficiencies, and pathway analysis.

Request (tensor): `{"command": "query_transfer_tensor", "d1": 12, "d2": 12, "d3": 1}`

Response: `{"d1": 12, "d2": 12, "d3": 1, "T_k0": 0.25, "T_k1": 0.0, "T_km1": 0.0, "T_combined": 0.1875, "xi_ratio": 8.5625, "efficiency": 1.6055, "pathway": "GRAVITATIONAL OVERRIDE"}`

Request (universality): `{"command": "query_transfer_tensor", "mode": "em_universality"}`

Response: `{"em_reaches_all": true, "targets": [{"d3": 1, "T": 0.1875, "eff": 1.6055}, {"d3": 2, "T": 0.1875, "eff": 1.511}, {"d3": 3, "T": 0.1875, "eff": 1.2844}, {"d3": 4, "T": 0.0625, "eff": 0.3425}, {"d3": 6, "T": 0.125, "eff": 0.4177}, {"d3": 12, "T": 0.375, "eff": 0.375}]}`

---

**Operation 112: `query_birth_triad`**

Pattern: Request-Response

Query the Substantiation Transition / Birth Triad algebra (§3.18.31). Modes: fixed_point (M_crit identity), canonical (M_can projection), mass_scan (d-family path), reverse (algebraic inverse).

Request (canonical): `{"command": "query_birth_triad", "mode": "canonical"}`

Response: `{"M_can_k": -53, "M_can_d": 12, "M_can_eps": "0", "residue_mod_12": 7, "is_cascade_generator": true, "lattice_exact_all_N": true, "cascade_to_fixed_point": [12,6,4,3,12,2,12,3,4,6,12,1]}`

Request (reverse): `{"command": "query_birth_triad", "mode": "reverse", "k": -53, "eps": "0.0", "N_source": 12, "N_target": 420, "R0_ratio": "3.14159"}`

Response: `{"forward_k": -1855, "forward_d": 12, "reversed_k": -53, "reversed_d": 12, "residual": "0", "reversible": true}`

---

**Operation 113: `query_seed_structure`**

Pattern: Request-Response

Query the EUDD's Kolmogorov seed structure (§3.18.32). Modes: generators (list all algebraic identity generators), shrinkage (history of seed size reduction), access (demonstrate arbitrary point evaluation), lifecycle (cascade seed lifecycle d=12→d=1).

Request (generators): `{"command": "query_seed_structure", "mode": "generators"}`

Response: `{"generator_count": 12, "generators": [{"label": "Bijection", "section": "§3.18.20"}, {"label": "A", "section": "§3.18.21"}, ...], "total_content_derivable": "all lattice arithmetic, transfer rates, birth triads, ...", "seed_type": "Kolmogorov_generative", "not_shannon": true}`

Request (access): `{"command": "query_seed_structure", "mode": "access", "k": 42, "d": 3, "eps": "7.5", "N": 12}`

Response: `{"r": "2.69679...", "evaluation_method": "direct_pullback", "sequential_decompression": false, "codec_required": false, "algebraic_identity": true}`

---

**Operation 114: `query_shape_projection`**

Pattern: Request-Response

Query the Shape Projection (§3.18.33). Modes: decompose (shape → harmonics → lattice signature), appearance (R_charge/ƛ_e projection), orbital (l → shape seed), convergence (error vs l_max).

Request (decompose): `{"command": "query_shape_projection", "mode": "decompose", "shape": "ellipsoid", "params": {"a": 2, "b": 2, "c": 1}, "l_max": 10}`

Response: `{"shape": "oblate_ellipsoid", "c_00": 5.337, "n_harmonics": 36, "lattice_signature": [{"l":2,"m":0,"k":-27,"d":4,"ratio":0.207},...], "dominant_l": 2, "dominant_d": 4}`

Request (appearance): `{"command": "query_shape_projection", "mode": "appearance", "Z": 20, "A": 40}`

Response: `{"Z": 20, "A": 40, "R_charge_fm": "3.4776", "r_dimensionless": "0.009007", "k": -83, "d": 12, "eps": "...", "source": "measured"}`

**ARCHITECTURAL NOTE — No Separate Memoization Operations:**
Shapes, colors, form factors, spectral lines, and all other content go through the EXISTING insert_value (Op 81) and project_value (Op 86) operations. The EUDD is simultaneously a database, a computation engine (handling ALL of mathematics), and a discovery engine (its own separate subsystem, active continuously, finding new generators from existing data). Memoization is NATURAL — a consequence of the database recording every computation at 361 dps via the lossless bijection, not a separate engine or operation. There are no separate "memoize_shape" or "memoize_color" operations because everything that enters or is computed is recorded inherently. A shape ratio c_lm/c_00 enters via Op 81 exactly as a mass ratio m/m_e does. The d-family classification, tightness function, transfer tensor, and all algebraic identities (A–I) apply identically regardless of the content domain. Op 114 (query_shape_projection) is a domain preparation helper — it computes dimensionless ratios from domain-specific input and feeds them into the same universal pipeline.

---

**Operation 109: `verify_bijection`**

Pattern: Request-Response

Verify the lossless bijection property Π_N⁻¹(Π_N(r))=r (§3.18.20, Theorem 12.1). Three modes: algebraic (sympy proof reference), precision_scaling (test at multiple dps), round_trip (project and recover a specific value).

Request (round_trip): `{"command": "verify_bijection", "mode": "round_trip", "r": "3.14159265358979", "N": 12}`

Response: `{"r": "3.14159...", "k": 19, "d": 12, "eps": "-13.686...", "r_recovered": "3.14159...", "residual": "0", "algebraic_identity": true, "proof": "k + ε·N/1200 = N·log₂(r) → 2^(log₂(r)) = r"}`

Request (precision_scaling): `{"command": "verify_bijection", "mode": "precision_scaling", "r": "137.036", "dps_levels": [50, 100, 200, 400]}`

Response: `{"r": "137.036", "scaling_results": [{"dps": 50, "error": "2.50e-51"}, {"dps": 100, "error": "1.33e-101"}, {"dps": 200, "error": "1.53e-201"}, {"dps": 400, "error": "0"}], "error_is_computational": true, "mathematical_error": "0"}`

---

#### 7.16.20b Domain 17 — Memory AI (Operations 115–117)

**Operation 115: `query_rmsae`**

Pattern: Request-Response

Compute Φ_RMSAE (§3.18.36) for the EUDD's discovery engine or a connected system. Returns the five component factors and the composite metacognitive classification.

Request: `{"command": "query_rmsae", "target": "discovery_engine"}`

Response: `{"target": "discovery_engine", "rho": "0.42", "gamma": "0.71", "kappa_closure": "0.63", "v_supp": "0.988", "psi_shimmer": "1.15", "phi_rmsae": "0.338", "classification": "basic", "timestamp_ns": 1748390400000000000}`

Request (connected AI): `{"command": "query_rmsae", "target": "connected_ai", "ai_session_id": "conscious_ai_001"}`

Response: `{"target": "connected_ai", "ai_session_id": "conscious_ai_001", "rho": "0.68", "gamma": "0.45", "kappa_closure": "0.82", "v_supp": "0.997", "psi_shimmer": "0.88", "phi_rmsae": "0.516", "classification": "genuine"}`

Fires `metacognition_rmsae_computed` event with full metadata. If Φ crosses a threshold boundary, also fires `rmsae_threshold_crossed` event.

**Operation 116: `query_traverser_waveform`**

Pattern: Request-Response

Retrieve the TraverserWaveform time-series and statistics over the window of N²=144 steps (§3.18.36). Returns D-fingerprints and derived metrics.

Request: `{"command": "query_traverser_waveform", "waveform_id": "discovery_engine_current", "last_n_steps": 144}`

Response: `{"waveform_id": "discovery_engine_current", "window_size": 144, "steps": [{"pos": 1, "lattice_k": 7, "lattice_d": 12, "variance": "0.083", "entropy": "2.41", "ego_resonance": "0.87"}, ...], "t_continuity_score": "0.92", "t_health": "0.85", "v_ghost_current": "0.003", "ghost_threshold_sigma": "3.0", "anomaly_detected": false}`

**Operation 117: `query_metacognition_state`**

Pattern: Request-Response

Return the complete metacognitive state of the EUDD or a connected system: current Φ_RMSAE, waveform summary, classification history, and threshold crossing events.

Request: `{"command": "query_metacognition_state", "target": "discovery_engine", "include_history": true, "history_depth": 10}`

Response: `{"target": "discovery_engine", "current_phi": "0.338", "current_classification": "basic", "waveform_summary": {"t_continuity": "0.92", "t_health": "0.85", "window_completeness": 144}, "classification_history": [{"timestamp_ns": ..., "classification": "subliminal", "phi": "0.18"}, ...], "threshold_crossings": [{"timestamp_ns": ..., "from": "subliminal", "to": "basic", "phi": "0.31"}]}`

---

#### 7.16.21 Cross-Domain Operations

**Operation 77: `consistency_check`**

Pattern: Request-Response

§5.5 cross-project consistency check. Detects contradictory projections for the same value.

Request:
```json
{
  "msg_type": "request",
  "command": "consistency_check",
  "session_id": "...",
  "value_spec": {"by": "id", "value_id": 42},
  "N": 27720
}
```

Response:
```json
{
  "msg_type": "response",
  "consistency": {
    "status": "CONSISTENT",
    "projections_checked": 3,
    "all_agree_k": 7360,
    "all_agree_d": 693,
    "max_eps_deviation_micros": 0,
    "contradictions": []
  }
}
```

When contradictions found: `"status": "CONTRADICTION"` with detailed per-projection comparison showing which session produced each conflicting result and their verification levels.

**Operation 78: `subsumption_check`**

Pattern: Request-Response

§5.6 cross-project subsumption check.

Request:
```json
{
  "msg_type": "request",
  "command": "subsumption_check",
  "session_id": "...",
  "criterion": "all odd zeta values at 27720ET",
  "claimed_property": "all-inert Gaussian signature",
  "scope": {"N": 27720, "value_repr_pattern": "ζ(%)", "d_filter_odd_argument": true}
}
```

Response:
```json
{
  "msg_type": "response",
  "subsumption": {
    "verdict": "FALSIFIED",
    "values_checked": 6,
    "values_matching": 2,
    "counterexamples": [
      {"value_repr": "ζ(5)", "gaussian_signature": "R·D", "detail": "Contains ramified prime 2"},
      {"value_repr": "ζ(7)", "gaussian_signature": "R·D·D", "detail": "Contains ramified prime 2"},
      {"value_repr": "ζ(11)", "gaussian_signature": "R·D", "detail": "Contains ramified prime 2"},
      {"value_repr": "ζ(13)", "gaussian_signature": "R·D·D·D", "detail": "Contains ramified prime 2"}
    ]
  }
}
```

---

**Operation 79: `ingest_text`**

Pattern: Request-Response

Ingest raw text through the compressor's general-purpose Δk pipeline — the API equivalent of GUI Mode 7 (§7.13). The text is processed as raw bytes, the same as any binary file. This is the correct API pathway for ingesting textual content that is not a mathematical expression (`compute`), a numerical value (`store_value`), or a file on disk (`ingest_file`).

Request:
```json
{
  "msg_type": "request",
  "command": "ingest_text",
  "session_id": "...",
  "text": "The ratio of electron mass to proton mass is 1/1836.15267343, first measured by...",
  "metadata": {
    "source_description": "excerpt from physics textbook",
    "tags": [{"namespace": "domain", "value": "physics"}]
  }
}
```

Response:
```json
{
  "msg_type": "response",
  "ingested": true,
  "content_value_ids": [9001, 9002, 9003],
  "provenance": {
    "input_source": "api",
    "input_mode": "text",
    "pipeline": "general_delta_k",
    "session_id": "...",
    "timestamp_ns": "..."
  }
}
```

---

#### 7.16.22 Subsumption Verification of the API

**Identification Principle applied to the API:** P = the Manager's complete capability space. D = the 117 operations specified above. T = the connected program navigating the API. Every Manager capability described in this document maps to at least one operation. Every operation maps back to a specific section of the document.

**Descriptor Gap Principle:** 117 operations close all identified gaps. The API covers: core lattice operations (project, pullback, escalate, k-arithmetic, lattice-add, evaluate-function), value management (store, batch, get, search, trajectory, CF), address and attractor queries (address, family, attractor, nearest, coprime-skeleton), equation and computation (compute, batch-compute, get, search, resolve-indeterminate), relationships and derivations (query, create for both), patterns and generators (query, propose, status for both), events (query, log, replay), towers and families (query/create towers, query harmonic/FQG/combined/sublattice families), files, streams, and text (ingest file/stream/text, retrieve file/stream), subscriptions (subscribe, unsubscribe, query), active probing and analysis (send-probe, query-probes, evaluate-gaze, run-scan, anti-numerology), traverser and manifold state (query-traverser, query-by-manifold-state, apply-three-tools, query-metabolism), administration (status, metrics, journal, backup, verify, extension, discovery-scan, provisional categories), tags/sessions/schema (add/remove/query tags, query sessions, query schema versions), cross-domain (consistency-check, subsumption-check), seed protocol (generate, reconstruct, progressive-stream, cache, dedup, file-versions, lattice-multiply/divide/reciprocal/power, cross-resolution/seed transitions, drift-monitor, restoration-control, d-composition, power-family), and complex lattice arithmetic (phase-project, complex-project, complex-multiply/reciprocal/power, phase-add, phase-differential), and harmonic FQG composition (query-harmonic-composition), and sublattice FQG analysis (query-sublattice-fqg), composite bridge (query-composite-bridge), ∂I boundary analysis (query-dI-boundary), triple backbone bridge (query-backbone-bridge), bijection verification (verify-bijection), full cross-tower transition (full-cross-tower-transition), harmonic transfer tensor (query-transfer-tensor), birth triad (query-birth-triad), seed structure (query-seed-structure), shape projection (query-shape-projection), and Memory AI metacognition (query-rmsae, query-traverser-waveform, query-metacognition-state). Shapes, colors, form factors, and all content enter through the existing insert_value (Op 81) — no separate memoization operations exist because projection IS memoization. No remainder.

**Subsumption Law:** The API subsumes the Manager's capabilities without remainder. Every structural operation, every query, every ingestion pathway, every discovery mechanism, every active probing capability, every administrative function, and the metacognitive self-monitoring capability are reachable through the 117 operations. The metabolism governs resource allocation across all connections via ET-derived constants (K, V, α⁻¹, ξ(d)). The three communication patterns (sync, async-stream, subscribe-push) cover every temporal structure an operation can have. The error taxonomy covers every failure mode cataloged in §7.15.

**Verification Principle:** The JSON schemas are mathematically consistent — all value encodings are exact (zero IEEE 754), all structural classifications reference the same lattice constants, all operations produce events that feed the discovery engine. The API is self-recording: every command, response, and error is itself an event in the database, feeding the §3.1b self-recording metrics.

---
