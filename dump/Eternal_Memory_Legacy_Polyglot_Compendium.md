# Eternal Memory
## Legacy Polyglot Architecture Compendium

**Author:** Michael James Muller — Aevum Defluo
**System:** Eternal Memory (the Eternal Memory Project, EMP)
**Document Class:** Legacy Preservation Record, v1.0
**Date:** July 19, 2026
**Status:** **Superseded as implementation plan; preserved without loss as record and as requirements source.**
**Superseding Specification:** *Eternal Memory: The Founding Formal Specification*, Section 20 — ETPL as sole implementation language
**Companion:** *Eternal Memory: Open Items Register* — item G-ETPL-1 (capability coverage matrix) is generated from Section 10 of this document

---

## 1. Purpose and Standing of This Document

Eternal Memory's original technical design was a **polyglot architecture**: every component implemented in the language best suited to its specific technical demands — dependent types where proofs must travel with code, actor models where fault tolerance is life, array languages where density is speed, shader languages where the GPU is the ground. That design named **77 distinct programming, specification, query, and notation languages**, together with a full complement of databases, communication systems, service infrastructure, and build tooling.

The Founding Formal Specification replaces this entire implementation layer with a single language: **ETPL**, the Exception Theory Programming Language, whose type system is the P/D/T primitive system itself and which subsumes the expressive range of every language below (Founding Specification, Theorem 20.1). ETPL will be fully capable of what these languages can do when complete.

This compendium exists because the system's first law — *preserve over delete; all data is good data* — applies to the system's own history. Nothing here is discarded. Every language assignment, every distribution percentage, every rationale, and every named tool is preserved exactly, for three standing reasons: as the **historical record** of the design; as the **requirements source** from which the ETPL capability coverage matrix (Register G-ETPL-1) is generated — every capability a legacy language provided is a capability the ETPL implementation must demonstrably provide; and as the **fallback specification of record** for any capability whose ETPL realization is not yet closed in the Register.

---

## 2. Master Language Index

The 77 distinct languages, notations, and DSLs of the legacy design, deduplicated across all assignments (count computed and verified):

| # | Language | # | Language | # | Language |
|---|---|---|---|---|---|
| 1 | Idris 2 | 27 | Nim | 53 | TypeScript |
| 2 | Agda | 28 | Racket | 54 | AssemblyScript |
| 3 | TLA+ | 29 | Red/System | 55 | Haxe |
| 4 | Ada 2022 | 30 | Forth | 56 | Odin |
| 5 | SPARK 2014 | 31 | Elixir | 57 | Go |
| 6 | Rust | 32 | Clojure | 58 | Lean 4 |
| 7 | Zig | 33 | Dhall | 59 | Coq |
| 8 | Pony | 34 | Tcl | 60 | Fortran 2018 |
| 9 | C | 35 | Jsonnet | 61 | Elm |
| 10 | Assembly (NASM/GAS) | 36 | Rebol | 62 | PureScript |
| 11 | Cyclone | 37 | Datalog | 63 | Pyret |
| 12 | Jasmin | 38 | Curry | 64 | Processing/p5.js |
| 13 | Cryptol | 39 | Oz | 65 | Raku |
| 14 | Julia | 40 | K | 66 | ANTLR |
| 15 | Mojo | 41 | J | 67 | BNF |
| 16 | Common Lisp | 42 | APL | 68 | Rhai |
| 17 | Mercury | 43 | MiniZinc | 69 | ReScript |
| 18 | Stan | 44 | Cypher | 70 | Mint |
| 19 | Python | 45 | Gremlin | 71 | Church/WebPPL |
| 20 | Erlang/OTP | 46 | SPARQL | 72 | Alloy |
| 21 | F* | 47 | OWL | 73 | CUDA |
| 22 | Prolog | 48 | GLSL 4.6 | 74 | SYCL |
| 23 | Haskell | 49 | HLSL | 75 | Rego |
| 24 | C++ (incl. C++20/23) | 50 | Slang | 76 | Cap'n Proto schema language |
| 25 | OCaml | 51 | WGSL | 77 | Protocol Buffers schema language |
| 26 | F# | 52 | JavaScript | | |

*Note on counting:* the two interface-definition schema notations (#76, #77) are counted individually. SageMath is a Python-based computer algebra system and is catalogued with the Math module stack (Section 4.18) and tooling rather than as a distinct language. Supporting libraries and frameworks (Flux.jl, spaCy, Phoenix, Three.js, XState, Lucene, and kin) are catalogued in their module sections below and are not counted as languages. Apache Flink/Beam are stream-processing frameworks, likewise catalogued at their site of use (Section 4.3) and not counted as languages.

---

## 3. Language Distribution by Purpose

### 3.1 Security Layer (13 languages)

| # | Language | Assignment |
|---|---|---|
| 1 | Idris 2 | Special Security Barrier — primary; dependent types for proof-carrying code |
| 2 | Agda | Special Security Barrier — secondary; theorem proving capabilities |
| 3 | TLA+ | Special Security Barrier — tertiary; formal specification language |
| 4 | Ada 2022 | Tertiary Security Barrier — primary; military-grade reliability |
| 5 | SPARK 2014 | Tertiary Security Barrier — secondary; formal verification subset of Ada |
| 6 | Rust | Secondary Security Barrier — primary; memory safety without garbage collection |
| 7 | Zig | Secondary Security Barrier — secondary; manual memory management with safety |
| 8 | Pony | Secondary Security Barrier — tertiary; actor-based concurrency |
| 9 | C | Primary Security Barrier — primary; direct hardware control |
| 10 | Assembly (NASM/GAS) | Primary Security Barrier — secondary; critical path optimization |
| 11 | Cyclone | Primary Security Barrier — tertiary; safe dialect of C |
| 12 | Jasmin | Cryptographic components — primary; high-assurance cryptography, constant-time by construction, assembly-level performance, formal verification support, side-channel resistant |
| 13 | Cryptol | Cryptographic components — secondary; domain-specific for cryptography, specification and implementation, SAT/SMT solver integration, equivalence checking, bit-precise semantics |

### 3.2 AI Systems (16 languages)

| # | Language | Assignment |
|---|---|---|
| 1 | Julia | Memory (AI) — numerical computing, LLVM JIT |
| 2 | Mojo | Memory (AI) — Python syntax, C++ performance |
| 3 | Common Lisp | Memory (AI) — code-as-data, macros |
| 4 | Mercury | Memory (AI) — declarative, logic-based reasoning |
| 5 | Stan | Memory (AI) — Bayesian inference |
| 6 | Python | Memory (AI) — ML library integration |
| 7 | Rust | Vines — memory safety, networking |
| 8 | Erlang/OTP | Vines — fault tolerance |
| 9 | F* | Vines — dependent types, crypto verification |
| 10 | Apache Flink/Beam | Vines — real-time threat analysis (stream frameworks) |
| 11 | Ada | Vines — additional security |
| 12 | Prolog | Logic programming |
| 13 | Haskell | Functional programming |
| 14 | C++ | Performance-critical |
| 15 | OCaml | Type inference |
| 16 | F# | Functional-first |

### 3.3 Module Generation (11 languages)

| # | Language | Assignment |
|---|---|---|
| 1 | Nim | Metaprogramming, AST manipulation |
| 2 | Racket | Template generation, language-oriented programming |
| 3 | Red/System | DSL creation, native compilation |
| 4 | Forth | Code generation, stack-based |
| 5 | Elixir | Module registration, actor model (OTP) |
| 6 | Datomic/Clojure | Registry database, immutable |
| 7 | Dhall | Type-safe configuration |
| 8 | Tcl | Scripting and automation |
| 9 | Jsonnet | Template configuration |
| 10 | Cap'n Proto | Custom template serialization |
| 11 | Rebol | Scripting |

### 3.4 Knowledge Management (15 languages)

| # | Language | Assignment |
|---|---|---|
| 1 | Clojure | Graph algorithms, immutable data |
| 2 | F# | Type providers, discriminated unions |
| 3 | Datalog | Declarative queries |
| 4 | Mercury | Logic programming with types |
| 5 | Curry | Functional logic programming |
| 6 | Oz | Multi-paradigm |
| 7 | K | Array programming |
| 8 | J | Array programming |
| 9 | APL | Array programming |
| 10 | MiniZinc | Constraint modeling |
| 11 | Cypher | Graph query |
| 12 | Gremlin | Graph traversal |
| 13 | SPARQL | RDF query |
| 14 | OWL/Protégé | Ontology |
| 15 | Drools/Clara | Rule engines |

### 3.5 Visualization (10 languages)

| # | Language | Assignment |
|---|---|---|
| 1 | C++20/23 | Rendering engine, SIMD |
| 2 | Rust | GPU programming alternative |
| 3 | GLSL 4.6 | Shader language |
| 4 | HLSL/Slang | Shader languages |
| 5 | WGSL | WebGPU shaders |
| 6 | JavaScript | Core web functionality |
| 7 | TypeScript | Type safety, modern tooling |
| 8 | AssemblyScript | WASM compilation |
| 9 | Haxe | Multiple targets, GPU abstractions |
| 10 | Odin | Systems programming |

### 3.6 Language Selection Rationale (as founded)

Each language was chosen for specific technical advantages: memory safety, type systems, and performance; domain-specific features and unique paradigms; formal verification capabilities; and optimal tooling for its specific task. Additional specialized components — various further languages and tools — served specific technical requirements as they arose.

---

## 4. Per-Module Language Assignments (Complete)

### 4.1 Void Module — 11 languages with distribution

| Language | Share | Role |
|---|---|---|
| Nim | 40% | Primary metaprogramming; AST manipulation; core factory engine; GUI framework |
| Racket | 15% | Template generation; language-oriented programming; feature definition language |
| Red/System | 10% | DSL creation; full-stack language; native compilation; domain-specific languages |
| Forth | 10% | Code generation; stack-based extensible compiler; immutable audit logging |
| Elixir | 10% | Module registration via actor model (OTP); registry management |
| Datomic (Clojure) | 5% | Registry database; immutable database; schema management |
| Dhall | 5% | Type-safe configuration; security profile definitions |
| Tcl | 3% | Scripting and automation; base template management |
| Jsonnet | 1% | Template configuration; composite template configuration |
| Cap'n Proto | 0.5% | Custom template serialization; registry synchronization |
| Rebol | 0.5% | Additional scripting |

Component-to-language mapping within Void: Dynamic Module Generator (Nim); Template Processor (Racket); DSL Compiler (Red/System); Code Generator (Forth); optimization integration (LLVM); Module Registry (Elixir/OTP with GenServer); Registry Database (Datomic — immutable, time-travel queries); Configuration Manager (Dhall — type-safe); Registry Synchronizer (Cap'n Proto); GUI Framework (Nim UI — dark theme, multi-monitor); Audit System (immutable logging via Forth); Visualization Engine (3D/2D with WebGL); template versioning (Git-based). Validation-phase tooling: Security Validation (Idris); Architecture Compliance (TLA+); Dependency Check (Mercury).

### 4.2 Memory (AI) Module — 6 languages plus supporting libraries

| Language | Role |
|---|---|
| Julia | Primary — numerical computing; LLVM-based JIT; neural architecture; synthesis engine |
| Mojo | Python syntax with C++ performance; MLIR compiler; high-performance pattern recognition |
| Common Lisp | Code-as-data philosophy; powerful macro system; symbolic AI; self-modification |
| Mercury | Purely declarative; logic-based AI reasoning; goal-driven planning |
| Stan | Bayesian inference; probabilistic models; belief modeling |
| Python | ML library integration; NLP components (spaCy); transformer models |

Supporting libraries: Flux.jl, Knet.jl, MLJ.jl (Julia); ACL2, Screamer (Lisp); Hamiltonian Monte Carlo (Stan); Transformers, PyTorch (Python).

Component-to-language mapping: Self-Awareness Engine (Julia/Mojo); Introspection System (Lisp); Goal Formation (Mercury); Neural Architecture (Julia/Flux.jl); Symbolic Reasoning (Common Lisp); Probabilistic Inference (Stan); Knowledge Integration (Mercury); Synthesis Engine (Julia); Pattern Recognition (Mojo); Semantic Understanding (Python/spaCy); Conceptual Mapping (Mercury); Query Processing (Julia); Response Generation (Lisp); Emotional Modeling (Stan); self-modifying code (Lisp macros); SIMD pattern recognition (Mojo); multi-paradigm learning (neural + symbolic + probabilistic); graph-based knowledge representation; continuous self-improvement mechanisms; emotional dynamics (differential equations); quantum-inspired computation (experimental). Encryption particulars: AES-256-GCM memory space with quantum-resistant key derivation; TLS_AES_256_GCM_SHA384 on the Vines channel.

### 4.3 Vines Module — 5 assignments with distribution

| Language | Share | Role |
|---|---|---|
| Rust | 60% (primary) | Core Security Engine; memory-safe operations; zero-copy networking; performance-critical paths |
| Erlang/OTP | 25% (secondary) | Fault-tolerant systems; hot code swapping; supervision trees; distributed processing |
| F* | 10% (tertiary) | Cryptographic verification; security proofs; protocol verification; formal methods |
| Apache Flink/Beam | 5% (quaternary) | Stream processing; real-time analytics; event processing; exactly-once semantics |
| Ada | — | Additional security components |

### 4.4 Core Module — 3 languages with distribution

| Language | Share | Role |
|---|---|---|
| Rust | 70% | Module registry; message routing; security enforcement; performance-critical paths |
| Go | 20% | Concurrent operations; network communication; health monitoring; administrative API |
| Zig | 10% | Low-level optimizations; memory management; system interfaces; custom allocators |

Implementation particulars: persistence to RocksDB; SIMD batch message validation (Zig); ring buffer fast message queue (Zig); Go goroutines for concurrent task execution and system monitoring (memory statistics, goroutine counts, CPU); Prometheus metrics integration.

**Core administrative interface — exact legacy particulars.** REST API endpoints: `/api/v1/modules` (list/manage); `/api/v1/modules/:id` (details); `/api/v1/modules/:id/status`; `/api/v1/modules/:id/restart`; `/api/v1/communications` (management); `/api/v1/communications/matrix`; `/api/v1/communications/allow`; `/api/v1/communications/deny`; `/api/v1/system/status`; `/api/v1/system/health`; `/api/v1/system/metrics`; `/api/v1/system/config`; `/api/v1/audit/events`; `/api/v1/audit/search`. gRPC service methods: RegisterModule; DeregisterModule; GetModule; ListModules; RouteMessage; UpdateCommunicationMatrix; GetSystemStatus; RestartModule; HealthCheck. CLI: module management; communication management; system status and control; configuration management; audit log queries. Prometheus metrics collected: `core_modules_registered_total`; `core_modules_active`; `core_module_restarts_total`; `core_messages_routed_total`; `core_messages_failed_total`; `core_routing_latency_seconds` (histogram); `core_system_uptime_seconds`; `core_memory_usage_bytes`; `core_cpu_usage_percent`. Testing: Rust unit tests (registration, routing, permissions, error handling, isolated components); integration tests (lifecycle, multi-module communication, failure recovery, performance benchmarks, end-to-end).

### 4.5 Privilege Module

| Language | Role |
|---|---|
| Haskell | Primary — type safety; pure functions; formal verification |
| Idris | Secondary — dependent types; proof-carrying code |

### 4.6 Archive Module

| Language / Technology | Role |
|---|---|
| Rust | Primary — memory safety; performance; compression |
| C++ | Storage engine; template metaprogramming |
| LMDB | Memory-mapped database; ACID compliance |
| zstd | Dictionary compression; parallel processing |
| FastCDC | Content-defined chunking for deduplication |
| ring | Encryption; misuse resistant; constant-time operations |

### 4.7 Evidence Module

| Language | Role |
|---|---|
| F# | Primary — type providers for RDF; discriminated unions |
| Clojure | Knowledge graph with immutable data structures; Datomic |
| Datalog | Query language — declarative queries; recursive rules |
| Church/WebPPL | Probabilistic evidence |
| Alloy | Temporal logic |

### 4.8 Account Module

| Language / Framework | Role |
|---|---|
| Elixir | Primary — actor model; fault tolerance; Phoenix LiveView |
| Phoenix with Ecto ORM | Database integration |
| Phoenix Channels | Real-time features |
| Guardian | Authentication library |

### 4.9 Education Module

| Language | Role |
|---|---|
| Elm | Frontend — no runtime errors; pure functions |
| PureScript | Alternative with stronger type system |
| Pyret | Educational DSL designed for teaching |
| Processing/p5.js | Interactive content |

### 4.10 Search Module

| Language / Library | Role |
|---|---|
| Rust | Core engine — performance and memory safety |
| Lucene | Search library via Rust bindings |
| Go | Distributed search coordination |
| Python | NLP components (spaCy, NLTK) |

### 4.11 3-D Map Module

| Language / Technology | Role |
|---|---|
| C++20/23 | Rendering engine; zero-overhead abstractions; SIMD |
| CUDA/SYCL | GPU programming for parallel algorithms |
| Rust GPU | Alternative safe GPU programming |
| GLSL 4.6 | Shader language |
| WGSL | WebGPU shaders |
| HLSL/Slang | Additional shader languages |
| Three.js/Babylon.js | Web integration |

### 4.12 Visuals Module

| Language | Role |
|---|---|
| TypeScript | Frontend framework with type safety |
| JavaScript | Core web functionality |
| GLSL | Shader programming |
| WGSL | WebGPU standard shaders |
| AssemblyScript | High-performance components via WASM |
| Haxe | Alternative framework with GPU abstractions |

### 4.13 GUI Module

| Language / Library | Role |
|---|---|
| TypeScript | Frontend framework with type safety |
| JavaScript | Core web functionality |
| XState | State management using finite state machines |

### 4.14 Submissions Module

| Language / Technology | Role |
|---|---|
| Go | Primary — concurrency and simplicity |
| Temporal | Workflow engine for distributed workflows |
| gVisor | Sandboxing with application kernel |
| Custom Go validators | Validation logic |

### 4.15 Language Module

| Language / Tool | Role |
|---|---|
| OCaml + Menhir | Parsing with functional approach; LR(1) parsing |
| Raku | Natural language with built-in grammars; Unicode support |
| ANTLR | Compiler tools with multiple targets; parse tree walking |
| Rust | Language Server Protocol implementation |
| BNF | Grammar definitions (Backus-Naur Form) |

### 4.16 Bookmarks Module

| Language / Technology | Role |
|---|---|
| SQLite | Storage — embedded, reliable |
| Rust | Logic layer — performance and safety |
| Rhai | Embedded scripting for custom organization |

### 4.17 Notifications Module

| Language / Technology | Role |
|---|---|
| Erlang | Primary — message passing and fault tolerance |
| Elixir | Alternative — better syntax on the same BEAM VM |
| Phoenix Channels | Real-time features |
| RabbitMQ | Queue integration |

### 4.18 Math Module

| Language / System | Role |
|---|---|
| Julia | Computation — designed for numerical computing |
| Lean 4 | Theorem proving with dependent types; tactics |
| Coq | Alternative established proof assistant |
| SageMath (Python) | Computer algebra system |
| Fortran 2018 | Numerical computations with array operations; parallel DO |

### 4.19 Comments Module

| Language | Role |
|---|---|
| Elm | Frontend — pure functions; no runtime errors |
| ReScript | Alternative — OCaml syntax; JavaScript output |
| Mint | UI framework designed for SPAs |

### 4.20 The Hub (Eternal Memory Module)

| Language / Library | Role |
|---|---|
| React with TypeScript | Frontend |
| Redux Toolkit | State management |
| D3.js | Graph visualization |
| GraphQL | Backend integration |

### 4.21 Knowledge Category Modules (all 11) — Common Stack

| Language / Tool | Role |
|---|---|
| Clojure | Primary graph language; immutable data structures; graph algorithms |
| F# | Knowledge representation; type providers; discriminated unions |
| Mercury | Logic programming with types |
| MiniZinc | Constraint programming and modeling |
| Curry | Functional logic programming |
| Oz | Multi-paradigm programming |
| Datalog | Query language |
| Cypher | Graph query language (Neo4j) |
| Gremlin | Graph traversal |
| SPARQL | RDF query language |
| APL / J / K | Array programming for dense computations |
| OWL/Protégé | Ontology management |
| Drools/Clara | Rule engines |

---

## 5. Infrastructure, Databases, and Tooling

### 5.1 Database Operations (complete)

| System | Role |
|---|---|
| Neo4j | Primary graph database — knowledge networks |
| ArangoDB | Multi-model database |
| Dgraph | Distributed graph |
| JanusGraph | Scalable graph |
| PostgreSQL | Relational — structured data plus module registry (JSONB) |
| MongoDB | Document store — unstructured data |
| InfluxDB | Time-series — metrics |
| Redis | Cache — performance |
| SQLite | Embedded storage — bookmarks |
| LMDB | Memory-mapped — archive |
| RocksDB | Core module persistence |
| Rust | Database logic |
| Go | Coordination |
| C | Low-level operations |
| Erlang | Distributed systems |

### 5.2 Inter-Process Communication

**Primary — gRPC + Protocol Buffers:** language agnostic; streaming support; load balancing; deadlines and cancellation; schema evolution. **Alternative — Cap'n Proto RPC:** zero-copy messaging; promise pipelining; capability-based security; schema evolution; time-travel RPC. **Message queue — NATS/JetStream:** at-least-once delivery; persistence and durability; clustering support; wildcard subscriptions; request-reply patterns.

### 5.3 Foreign Function Interface

**Primary — WASM Interface Types:** language neutral; memory safety; sandboxing capabilities; component model; interface definitions. **C interop — cbindgen/bindgen:** automatic binding generation; type mapping; lifetime handling; error propagation; documentation generation. **JVM interop — JNI/JNA:** native method calls; direct mapping; callback support; memory management; cross-platform handling.

### 5.4 Service Mesh and Policy

**Primary — Linkerd (Rust-based):** lightweight architecture; automatic mTLS; load balancing; circuit breaking; observability. **Policy — Open Policy Agent with Rego:** declarative policies; partial evaluation; decision logging; bundle distribution.

### 5.5 Additional Service Infrastructure

**Load balancing — HAProxy:** high-performance TCP/HTTP. **Service discovery — Consul:** service mesh and discovery with health checking. **Orchestration — Nomad:** multi-runtime (Docker, WASM, and beyond).

### 5.6 Build and Development Tools

**Primary build — Bazel:** language agnostic; reproducible builds; remote caching; incremental builds. **Alternative build — Buck2:** Rust implementation; modern architecture. **Package management — Nix:** reproducible environments; atomic upgrades; rollbacks. **Containerization:** Docker (runtime); Kubernetes (orchestration); Firecracker (MicroVM runtime for serverless); gVisor (application kernel for sandboxing). **Infrastructure as code — Terraform:** multi-cloud provisioning. **Monitoring:** Prometheus (metrics collection and alerting); Grafana (visualization and dashboards). **Logging — ELK Stack** (Elasticsearch, Logstash, Kibana): centralized logging. **CDN:** CloudFlare. **Testing frameworks:** QuickCheck (property-based); LibFuzzer/AFL++ (fuzzing for security); TLA+/Alloy (formal testing and verification). **SIMD programming toolset:** ISPC; Highway C++; SYCL/DPC++ — the named data-parallel tooling of the performance-optimization layer. **Documentation:** language-specific API documentation tools; PlantUML with the C4 Model for architecture diagrams. **Deployment:** cross-platform — web, desktop, mobile — Docker with Kubernetes/Firecracker.

---

## 6. Known Costs of the Polyglot Design (as founded)

The founding documentation recorded the polyglot approach's own outstanding issues, preserved here with their original identifiers: **F1 — Foreign Function Interface complexity:** 50+ languages require extensive FFI; interface standardization needed. **F4 — Testing infrastructure:** testing 50+ languages across four security barriers; test automation strategy needed. Both are resolved by construction under the single-language transition (Founding Specification, Section 20.3) and replaced by the narrower items G-ETPL-1 and G-ETPL-3 in the Register.

---

## 7. The Capability Inventory: Requirements for ETPL

Every capability below was the *reason* a legacy language held its assignment. Under the transition, each is a requirement the ETPL implementation must demonstrably satisfy. This inventory is the source of Register item G-ETPL-1 (the capability coverage matrix); rows there cite entries here.

| # | Capability | Legacy provider(s) | Legacy site(s) |
|---|---|---|---|
| C-01 | Dependent types; proof-carrying code | Idris 2, Agda, F*, Idris | Special Barrier; Vines; Privilege |
| C-02 | Formal specification and model checking | TLA+, Alloy | Special Barrier; Void validation; Evidence temporal logic; testing |
| C-03 | High-reliability engineering with formal verification subset | Ada 2022, SPARK 2014 | Tertiary Barrier; Vines |
| C-04 | Memory safety without garbage collection | Rust, Zig, Pony, Cyclone | Secondary Barrier; Core; Vines; Archive; Search; Bookmarks; databases |
| C-05 | Direct hardware control and critical-path assembly | C, Assembly (NASM/GAS) | Primary Barrier; database low-level |
| C-06 | Constant-time, side-channel-resistant, formally verified cryptography with bit-precise equivalence checking | Jasmin, Cryptol, ring | Cryptographic components; Archive encryption |
| C-07 | High-performance numerical computing with JIT | Julia | Memory; Math |
| C-08 | SIMD/MLIR-class performance with high-level syntax | Mojo | Memory pattern recognition |
| C-09 | Code-as-data; macro systems; runtime self-modification | Common Lisp | Memory symbolic AI and self-modification |
| C-10 | Declarative logic programming with types; goal-driven planning | Mercury, Prolog, Curry, Oz | Memory; Evidence; knowledge categories |
| C-11 | Bayesian and probabilistic inference | Stan, Church/WebPPL | Memory belief modeling; Evidence probabilistic handling |
| C-12 | ML ecosystem integration; NLP; transformers | Python (spaCy, NLTK, PyTorch, Transformers) | Memory; Search |
| C-13 | Actor-model fault tolerance; supervision trees; hot code swapping; distributed processing | Erlang/OTP, Elixir, BEAM | Vines; Account; Notifications; Void registration |
| C-14 | Stream processing with exactly-once semantics; real-time analytics | Apache Flink/Beam | Vines threat analysis |
| C-15 | Metaprogramming and AST manipulation; code generation | Nim, Forth, Racket, Red/System | Void factory engine |
| C-16 | Immutable, time-travel-queryable registry database with schema management | Datomic (Clojure) | Void; Core registry |
| C-17 | Type-safe configuration | Dhall, Jsonnet | Void security profiles; templates |
| C-18 | Zero-copy serialization; promise-pipelined, capability-secure RPC | Cap'n Proto | Void synchronization; IPC |
| C-19 | Language-agnostic streaming RPC with schema evolution | gRPC + Protocol Buffers | System-wide IPC |
| C-20 | Durable at-least-once messaging with clustering and request-reply | NATS/JetStream, RabbitMQ | IPC; Notifications |
| C-21 | Immutable graph algorithms and data structures | Clojure | Categories; Evidence |
| C-22 | Typed knowledge representation; type providers; discriminated unions | F# | Categories; Evidence |
| C-23 | Declarative and recursive query | Datalog | Categories; Evidence |
| C-24 | Graph query and traversal; RDF/semantic query | Cypher, Gremlin, SPARQL | Categories; graph databases |
| C-25 | Ontology management | OWL/Protégé | Categories |
| C-26 | Production rule engines | Drools/Clara | Categories |
| C-27 | Constraint modeling and solving | MiniZinc | Categories |
| C-28 | Array-programming density | APL, J, K | Categories dense computation |
| C-29 | Zero-overhead-abstraction rendering; SIMD; GPU parallelism | C++20/23, CUDA/SYCL, Rust GPU, ISPC, Highway C++, DPC++ | 3-D Map; Visuals; performance layer |
| C-30 | Shader compilation across targets | GLSL 4.6, HLSL/Slang, WGSL | 3-D Map; Visuals |
| C-31 | Web-platform delivery; typed frontend; WASM emission | JavaScript, TypeScript, AssemblyScript, Haxe, Odin | Visuals; GUI; Hub |
| C-32 | Runtime-error-free frontend with pure functions | Elm, PureScript, ReScript, Mint | Education; Comments |
| C-33 | Educational DSL; interactive content | Pyret, Processing/p5.js | Education |
| C-34 | Full-text search engine capability | Lucene (via Rust bindings) | Search |
| C-35 | Distributed coordination with lightweight concurrency | Go | Core; Search; Submissions; databases |
| C-36 | Durable distributed workflow orchestration | Temporal | Submissions |
| C-37 | Application-kernel sandboxing; MicroVM isolation | gVisor, Firecracker | Submissions; deployment |
| C-38 | LR(1) and grammar-driven parsing; multi-target compiler tooling; grammar notation | OCaml + Menhir, ANTLR, BNF, Raku | Language module |
| C-39 | Language Server Protocol implementation | Rust | Language module |
| C-40 | Embedded reliable storage; memory-mapped ACID storage; document, relational, time-series, cache, and graph storage | SQLite, LMDB, RocksDB, PostgreSQL, MongoDB, InfluxDB, Redis, Neo4j, ArangoDB, Dgraph, JanusGraph | Bookmarks; Archive; Core; system storage |
| C-41 | Dictionary compression with parallel processing; content-defined chunking deduplication | zstd, FastCDC | Archive |
| C-42 | Theorem proving with tactics; established proof assistance; computer algebra; parallel numerical arrays | Lean 4, Coq, SageMath, Fortran 2018 | Math |
| C-43 | Finite-state-machine interface state management | XState | GUI |
| C-44 | Reactive component frontend with state container; graph visualization; typed API integration | React/TypeScript, Redux Toolkit, D3.js, GraphQL | Hub |
| C-45 | Embedded scripting for user customization | Rhai, Tcl, Rebol | Bookmarks; Void |
| C-46 | Authentication frameworks; real-time channels; ORM integration | Guardian, Phoenix Channels, Ecto | Account; Notifications |
| C-47 | WASM Interface Types component-model FFI; C and JVM interop | WASM IT, cbindgen/bindgen, JNI/JNA | System boundary |
| C-48 | Service mesh with automatic mTLS and circuit breaking; declarative policy; load balancing; service discovery; multi-runtime orchestration | Linkerd, OPA/Rego, HAProxy, Consul, Nomad | Service infrastructure |
| C-49 | Reproducible builds with remote caching; reproducible environments with atomic rollback; infrastructure as code | Bazel, Buck2, Nix, Terraform | Build and deployment |
| C-50 | Metrics collection and alerting; dashboards; centralized logging; CDN delivery | Prometheus, Grafana, ELK, CloudFlare | Operations |
| C-51 | Property-based testing; coverage-guided fuzzing | QuickCheck, LibFuzzer/AFL++ | Testing |
| C-52 | Architecture documentation generation | PlantUML with C4 Model; per-language doc tools | Documentation |

**Closure rule:** a capability row is closed only when the ETPL implementation demonstrates the property to the same or greater assurance level as the legacy provider — proof for proof, verification for verification, measured performance for measured performance — recorded in the Register under G-ETPL-1. Until closure, this compendium remains the specification of record for that row.

---

## 8. Closing Statement

The polyglot design was not a mistake to be buried; it was the complete map of what the system requires, drawn in the vocabulary of seventy-seven languages. ETPL inherits the map whole. This compendium is that inheritance, preserved under the system's own first law — nothing deleted, everything labeled — and standing ready as witness, requirement, and fallback until every capability lives natively in the language of P, D, and T.

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

---

**Document:** Eternal Memory — Legacy Polyglot Architecture Compendium, v1.0
**Status:** Preservation record; requirements source for Register G-ETPL-1; fallback specification per capability until closure
