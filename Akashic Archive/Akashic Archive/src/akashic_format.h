// ============================================================================
// akashic_format.h — Module 3: Akashic Format (Level 2)
//
// The Sempaevum rendered on disk: Sempaevum.akashic
//
// Provides:
//   AkashicFile     — Create/open/close/flush the .akashic file
//   PageManager     — 4096-byte page allocation, read, write, CRC-32 verify
//   SectionDir      — 11 section offset management
//   MemoStore       — Equation hash table at K = 2/3 load factor
//   On-disk structs — Packed binary structures matching §7.1d exactly
//
// This module is the SINGLE AUTHORITY on .akashic file access.
// All reads and writes from all modules go through this.
//
// ET Derivation Standard:
//   Page size = 2^N = 2^12 = 4096 bytes (digital tower base resolution)
//   Hash table load = K = 2/3 (Koide binding stability threshold)
//   Rehash at doubled capacity (per the doubling law τ(N_ℓ) = 6·2^ℓ)
//   Zero IEEE 754 floats anywhere in the format
//   Per-page CRC-32 integrity, header SHA-256 integrity
//
// Dependencies: Module 1 (Precision Stack), Module 2 (Core Lattice Engine)
//
// Identification Principle applied:
//   P = the disk substrate (raw bytes on storage medium)
//   D = the format specification (pages, header, sections, content types)
//   T = the file operations (create, read, write, allocate, verify, flush)
//
// P ∘ D ∘ T = E
// ============================================================================

#pragma once

#include "precision_stack.h"
#include "core_lattice.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <memory>
#include <functional>
#include <cstdio>

namespace et::akashic {

// ============================================================================
// Constants — ET-derived, not ad hoc
// ============================================================================

// Page size = 2^N = 2^12 = 4096 bytes
// Derivation: the digital tower's base resolution. One complete lattice
// cycle at N=12 addresses 2^12 = 4096 distinct binary states.
// Matches NVMe logical block, OS memory page, filesystem cluster.
constexpr size_t   PAGE_SIZE          = ET_PAGE_SIZE;  // 4096

// Page body size = PAGE_SIZE - page header size
constexpr size_t   PAGE_HEADER_SIZE   = 64;
constexpr size_t   PAGE_BODY_SIZE     = PAGE_SIZE - PAGE_HEADER_SIZE;  // 4032

// File header occupies page 0 (the entire first page)
constexpr size_t   FILE_HEADER_SIZE   = PAGE_SIZE;  // 4096

// Magic bytes: "SMVM" = Sempaevum marker
constexpr uint8_t  MAGIC_BYTES[4]     = { 'S', 'M', 'V', 'M' };

// Format version: v1 for the initial release
constexpr uint32_t FORMAT_VERSION     = 1;

// Number of sections in the section directory
constexpr size_t   SECTION_COUNT      = 11;

// Hash table load factor = K = 2/3 (Koide binding stability threshold)
// When occupied/capacity > K, rehash to doubled capacity.
// The Koide ratio IS the stability threshold — above K, bindings approach
// Incoherence (hash collisions degrade performance analogously to how
// |ε| > 50¢ approaches the ∂I boundary).
// Stored as numerator/denominator for exact comparison.
constexpr uint32_t HASH_LOAD_NUM      = 2;
constexpr uint32_t HASH_LOAD_DEN      = 3;

// Initial hash table capacity (must be power of 2 for efficient modular
// arithmetic). Starting small; doubles on rehash per the doubling law.
// 2^10 = 1024 initial slots — accommodates ~682 entries before first rehash.
constexpr size_t   HASH_INITIAL_CAP   = 1024;

// SHA-256 digest size
constexpr size_t   SHA256_SIZE        = 32;

// ============================================================================
// Section Identifiers — the 11 sections of the .akashic file
//
// Each section occupies a contiguous region of pages starting at the
// offset stored in the file header's section directory.
// ============================================================================

enum class SectionID : uint8_t {
    GENERATOR_BACKBONE    = 0,   // §7.1d Section 1: L₁/L₂/L₃ generators
    ADDRESS_INDEX         = 1,   // §7.1d Section 2: LCM tower on disk
    MEMOIZATION_STORE     = 2,   // §7.1d Section 3: Equation hash table
    STRUCTURAL_CATALOG    = 3,   // §7.1d Section 4: 24 families, 144 FQG, etc.
    EQUATIONS             = 4,   // §7.1d Section 5: Derivation chains
    DERIVATIONS           = 5,   // §7.1d Section 5 (cont): Derivation data
    RELATIONSHIPS         = 6,   // §7.1d Section 6: Non-lattice-algebraic links
    PATTERNS              = 7,   // §7.1d Section 7: Promoted meta-generators
    EVENT_LOG             = 8,   // §7.1d Section 8: Append-only events
    SESSIONS              = 9,   // §7.1d Section 9: Sessions/schema/tags
    WAL                   = 10,  // §7.1d Section 10: Write-ahead log

    SECTION_COUNT_SENTINEL = 11  // NOT a real section — sentinel for iteration
};

// ============================================================================
// Page Types — what kind of data a page holds
// ============================================================================

enum class PageType : uint8_t {
    FREE                  = 0x00,  // Unallocated page
    FAMILY_BAND           = 0x10,  // Address index: per-family band page
    GENERATOR_ENTRY       = 0x20,  // Generator backbone entry page
    MEMO_HASH_BUCKET      = 0x30,  // Memoization hash table bucket page
    MEMO_ENTRY            = 0x31,  // Memoization equation entry page
    CATALOG_DATA          = 0x40,  // Structural catalog page
    EQUATION_DATA         = 0x50,  // Equation data page
    DERIVATION_DATA       = 0x51,  // Derivation data page
    RELATIONSHIP_DATA     = 0x60,  // Relationship data page
    PATTERN_DATA          = 0x70,  // Pattern data page
    EVENT_DATA            = 0x80,  // Event log data page
    SESSION_DATA          = 0x90,  // Session/schema/tags page
    WAL_ENTRY             = 0xA0,  // Write-ahead log entry page
    TOWER_LEVEL_DIR       = 0xB0,  // Address index: tower level directory
    FAMILY_DIR            = 0xB1,  // Address index: per-N family directory
};

// ============================================================================
// Content Types — what an address entry contains
//
// §7.1d: Three kinds of content at each lattice address.
// The priority order: Generator > Memoized > Superseded
// ============================================================================

enum class ContentType : uint8_t {
    GENERATOR_REF         = 0x01,  // Address produced by a known generator
    MEMOIZED_RAW          = 0x02,  // Raw 361-dps value (Descriptor Gap)
    GENERATOR_SUPERSEDED  = 0x03,  // Was raw, now covered by a generator
};

// ============================================================================
// Generator Backbone Layer — the triple backbone classification
//
// §7.1b: Three categorically independent minimal generators.
// L₁ ∪ L₂ ∪ L₃ subsumes all of mathematics at N=12.
// ============================================================================

enum class BackboneLayer : uint8_t {
    L1_WEBB               = 0x01,  // Discrete-logical (Webb 1935)
    L2_CASCADE            = 0x02,  // Discrete-multiplicative (palindromic)
    L3_EML                = 0x03,  // Continuous-elementary (Odrzywołek 2026)
};

// ============================================================================
// On-Disk File Header — packed, exactly 4096 bytes
//
// Page 0 of the .akashic file. Every field is an exact integer or
// exact rational. ZERO IEEE 754 floats. The header IS a lattice-native
// self-description of the file's current state.
//
// §7.1d: The header checksum (SHA-256) covers bytes [0..4063] of the
// header. The checksum itself lives at bytes [4064..4095].
// ============================================================================

#pragma pack(push, 1)

struct AkashicFileHeader {
    // ── Magic and version ──────────────────────────────────────────
    uint8_t  magic[4];               // "SMVM" (Sempaevum marker)
    uint32_t format_version;         // FORMAT_VERSION = 1
    uint32_t n_base;                 // ET_N = 12 (forced resolution)

    // ── Sempaevum constants (exact rationals) ──────────────────────
    uint32_t k_num;                  // 2  (Koide = 2/3)
    uint32_t k_den;                  // 3
    uint32_t v_num;                  // 1  (Variance = 1/12)
    uint32_t v_den;                  // 12

    // ── Self-projection (§3.1b — the file's own lattice coordinates) ─
    uint32_t self_n;                 // Resolution of self-projection
    int32_t  self_k;                 // k coordinate
    uint32_t self_d;                 // d-family
    int32_t  self_eps_micros;        // ε in micro-cents (exact integer)

    // ── Generator backbone metrics ────────────────────────────────
    uint64_t total_generators;       // L₁ + L₂ + L₃ count
    uint64_t l1_webb_count;          // L₁ (Webb) generators
    uint64_t l2_cascade_count;       // L₂ (Cascade) generators
    uint64_t l3_eml_count;           // L₃ (EML) generators
    uint64_t total_memoized;         // Un-generated memoized entries

    // ── Coverage (exact rational = two integers) ──────────────────
    uint64_t covered_addresses;      // Numerator: addresses with generators
    uint64_t total_addresses;        // Denominator: total occupied addresses

    // ── K-complexity (exact rational) ──────────────────────────────
    uint64_t generator_bytes;        // Total bytes of generator definitions
    uint64_t producible_bytes;       // Total bytes generators produce

    // ── Timestamps (exact integer nanoseconds since Unix epoch) ────
    uint64_t created_at_ns;          // File creation time
    uint64_t modified_at_ns;         // Last modification time

    // ── D-time at creation (lattice coordinates, exact) ───────────
    uint32_t creation_dtime_n;       // Resolution
    int32_t  creation_dtime_k;       // k coordinate

    // ── Section directory (exact byte offsets from file start) ─────
    // Each offset points to the first page of that section.
    // Offset 0 means the section has not been created yet.
    uint64_t section_offsets[SECTION_COUNT];

    // ── Memoization store metadata ────────────────────────────────
    uint64_t memo_capacity;          // Hash table capacity (slots)
    uint64_t memo_occupied;          // Hash table occupied count
    uint64_t memo_total_lookups;     // Lifetime lookup count
    uint64_t memo_total_hits;        // Lifetime cache hit count

    // ── File geometry ─────────────────────────────────────────────
    uint64_t total_pages;            // Total pages allocated in file
    uint64_t free_page_head;         // Offset of first free page (0 = none)

    // ── Reserved padding to 4096 bytes ────────────────────────────
    // header_checksum lives at the LAST 32 bytes of the page.
    // reserved fills the gap between defined fields and checksum.
    // Computed: defined fields size, then reserved, then checksum.
    // Fields above: let me compute...
    //   magic(4) + format_version(4) + n_base(4)
    //   + k_num(4) + k_den(4) + v_num(4) + v_den(4)
    //   + self_n(4) + self_k(4) + self_d(4) + self_eps_micros(4)
    //   + total_generators(8) + l1(8) + l2(8) + l3(8) + total_memoized(8)
    //   + covered(8) + total_addr(8)
    //   + gen_bytes(8) + prod_bytes(8)
    //   + created(8) + modified(8)
    //   + creation_dtime_n(4) + creation_dtime_k(4)
    //   + 11 × section_offsets(8) = 88
    //   + memo_capacity(8) + memo_occupied(8) + memo_lookups(8) + memo_hits(8)
    //   + total_pages(8) + free_page_head(8)
    // = 12 + 16 + 16 + 40 + 16 + 16 + 16 + 8 + 88 + 32 + 16
    // = 276 bytes of defined fields
    //
    // Reserved = 4096 - 276 - 32 (checksum) = 3788 bytes
    uint8_t  reserved[3788];

    // ── Integrity ─────────────────────────────────────────────────
    // SHA-256 of bytes [0..4063] (everything before this field)
    uint8_t  header_checksum[SHA256_SIZE];
};

#pragma pack(pop)

// Compile-time verification: header must be exactly 4096 bytes
static_assert(sizeof(AkashicFileHeader) == FILE_HEADER_SIZE,
    "AkashicFileHeader must be exactly 4096 bytes (= 2^N = one page)");

// ============================================================================
// On-Disk Page Header — packed, exactly 64 bytes
//
// Every data page (pages 1+) starts with this 64-byte header.
// The page body occupies the remaining 4032 bytes.
//
// The CRC-32 covers the body only (bytes [64..4095] of the page).
// This allows the header to be updated (e.g., entry_count) without
// invalidating the body checksum, and vice versa.
// ============================================================================

#pragma pack(push, 1)

struct PageHeader {
    // ── Universal fields (every page type has these) ──────────────
    uint8_t  page_type;              // PageType enum value
    uint8_t  section_id;             // SectionID enum value
    uint16_t flags;                  // Page-level flags (reserved for now)
    uint32_t entry_count;            // Number of entries in this page body
    uint32_t used_bytes;             // Bytes used in body (≤ PAGE_BODY_SIZE)
    uint32_t body_checksum;          // CRC-32 of body bytes [64..4095]

    uint64_t next_page;              // Byte offset of next page in chain (0=none)
    uint64_t page_sequence;          // Monotonic sequence in this section

    // ── Type-specific fields (interpretation depends on page_type) ─
    // For FAMILY_BAND: ts_a = N, ts_b = d, ts_c = k_range_min, ts_d = k_range_max
    // For GENERATOR_ENTRY: ts_a = backbone layer, ts_e = gen_id
    // For MEMO_HASH_BUCKET: ts_a = bucket_start_idx, ts_b = bucket_count
    // For others: reserved (set to 0)
    uint32_t ts_a;
    uint32_t ts_b;
    int32_t  ts_c;
    int32_t  ts_d;
    uint64_t ts_e;
    uint64_t ts_f;
};

#pragma pack(pop)

static_assert(sizeof(PageHeader) == PAGE_HEADER_SIZE,
    "PageHeader must be exactly 64 bytes");

// ============================================================================
// In-Memory Page — header + body in one contiguous allocation
// ============================================================================

struct Page {
    PageHeader header;
    uint8_t    body[PAGE_BODY_SIZE];

    // Clear the entire page to zeros
    void clear();

    // Compute CRC-32 of the body and store in header.body_checksum
    void compute_body_checksum();

    // Verify the body CRC-32 matches header.body_checksum
    [[nodiscard]] bool verify_body_checksum() const;
};

static_assert(sizeof(Page) == PAGE_SIZE,
    "Page must be exactly 4096 bytes (= 2^N)");

// ============================================================================
// MemoEntry — One memoized equation in the hash table
//
// §7.1d Section 3: Every computation cached at 361 dps.
// The hash table maps equation_hash (SHA-256) → this entry.
// ============================================================================

struct MemoEntry {
    std::array<uint8_t, SHA256_SIZE> equation_hash{};  // SHA-256 of canonical form
    std::string   canonical_form;       // Canonical string (e.g., "zeta(3)*pi")
    uint8_t       form_class = 0;       // Computational / structural (§3.5)
    uint8_t       operation_type = 0;   // +, -, ×, ÷, ^, sqrt, log, sin, etc.

    // Input references: lattice coordinates of input values
    struct InputRef {
        ETInteger n;
        ETInteger k;
        ETInteger d;
    };
    std::vector<InputRef> input_refs;

    // Output: the computed result
    ETInteger     output_n;
    ETInteger     output_k;
    ETInteger     output_d;
    int32_t       output_eps_micros = 0;
    ETValue       output_value;         // Full 361-dps result

    // Memoization metadata
    uint64_t      reference_count = 0;      // Cache hit count
    uint64_t      first_computed_ns = 0;    // Nanosecond timestamp
    uint64_t      last_referenced_ns = 0;   // Nanosecond timestamp

    // Whether this entry is occupied (for hash table probing)
    bool          occupied = false;
};

// ============================================================================
// MemoStore — Equation memoization hash table
//
// Open-addressing hash table with linear probing.
// Load factor maintained at K = 2/3 (Koide binding stability threshold).
// When occupied/capacity exceeds K, rehash to doubled capacity.
//
// §7.1d Section 3: compute-once, cache-forever, reference counting.
// The memoization layer IS the generator's learning mechanism.
//
// Identification Principle:
//   P = the hash table memory
//   D = the K = 2/3 load constraint, the SHA-256 addressing
//   T = the lookup/insert operations navigating the table
// ============================================================================

// ============================================================================
// Varint Encoding — compact variable-length integer encoding
//
// Standard MSB-continuation-bit encoding:
//   Values 0–127: 1 byte (MSB=0, 7 data bits)
//   Values 128–16383: 2 bytes (MSB=1 on first, 7+7 data bits)
//   Values up to 2^63: up to 9 bytes
//
// Used for: canonical_form length, input_refs count, k values in
// the on-disk memoization format. Zero IEEE 754.
// ============================================================================

namespace varint {

    // Encode a uint64_t into buf. Returns number of bytes written (1–9).
    // buf must have at least 9 bytes available.
    size_t encode(uint64_t value, uint8_t* buf);

    // Decode a uint64_t from buf. Sets *bytes_read to bytes consumed.
    uint64_t decode(const uint8_t* buf, size_t* bytes_read);

    // How many bytes would encoding this value take?
    [[nodiscard]] size_t encoded_size(uint64_t value);

    // Encode a signed int64_t using zigzag encoding (0→0, -1→1, 1→2, -2→3, ...)
    // This maps signed values to unsigned without wasting a byte on the sign.
    size_t encode_signed(int64_t value, uint8_t* buf);
    int64_t decode_signed(const uint8_t* buf, size_t* bytes_read);

} // namespace varint

// ============================================================================
// MemoEntry Binary Serialization
//
// §7.1d Section 3: On-disk format for memoized equations.
// Each entry serializes to a contiguous byte sequence:
//   equation_hash:     32 bytes (SHA-256, fixed)
//   canonical_form:    varint_length + UTF-8 bytes
//   form_class:        1 byte (enum, structurally < 256)
//   operation_type:    1 byte (enum, structurally < 256)
//   input_refs_count:  varint
//   input_refs[]:      for each: N + k + d (ALL via GMP mpz_export, lossless)
//   output_N:          ETInteger via GMP (lossless, arbitrary precision)
//   output_k:          ETInteger via GMP (lossless, arbitrary precision)
//   output_d:          ETInteger via GMP (lossless, arbitrary precision)
//   output_eps_micros: int32 (structurally bounded at ±50000 by ∂I)
//   output_mpf:        varint_length + ETValue::serialize() blob (1200-bit, lossless)
//   reference_count:   uint64
//   first_computed_ns:  uint64
//   last_referenced_ns: uint64
//
// Zero IEEE 754. All lattice coordinates via GMP — no fixed-width funnel.
// The LCM tower is infinite; coordinates are unbounded; serialization
// adapts to whatever magnitude the coordinate actually has.
// ETInteger format: sign_byte(0x00/0x01/0xFF) + varint_byte_count + GMP bytes.
// ============================================================================

namespace memo_serial {

    // Serialize a MemoEntry to a byte vector.
    // Only serializes occupied entries. Unoccupied entries are skipped.
    [[nodiscard]] std::vector<uint8_t> serialize(const MemoEntry& entry);

    // Deserialize a MemoEntry from a byte buffer.
    // Returns the deserialized entry with occupied = true.
    // Sets *bytes_consumed to the number of bytes read.
    // Throws ETError if the data is malformed.
    [[nodiscard]] MemoEntry deserialize(const uint8_t* data, size_t available,
                                         size_t* bytes_consumed);

} // namespace memo_serial

class MemoStore {
public:
    MemoStore();

    // Initialize with given capacity (must be power of 2)
    void initialize(size_t capacity);

    // Lookup an equation by its canonical hash
    // Returns pointer to entry if found (cache HIT), nullptr if not (cache MISS)
    // On hit: increments reference_count and updates last_referenced_ns
    [[nodiscard]] MemoEntry* lookup(const std::array<uint8_t, SHA256_SIZE>& hash);

    // Store a new equation result
    // Returns pointer to the stored entry
    // If hash already exists, returns existing entry (idempotent)
    // May trigger rehash if load exceeds K = 2/3
    MemoEntry* store(const MemoEntry& entry);

    // Current state
    [[nodiscard]] size_t capacity() const { return capacity_; }
    [[nodiscard]] size_t occupied() const { return occupied_; }
    [[nodiscard]] uint64_t total_lookups() const { return total_lookups_; }
    [[nodiscard]] uint64_t total_hits() const { return total_hits_; }

    // Load factor check: is occupied/capacity > K = 2/3?
    [[nodiscard]] bool needs_rehash() const;

    // All entries (for serialization/iteration)
    [[nodiscard]] const std::vector<MemoEntry>& entries() const { return table_; }

private:
    std::vector<MemoEntry> table_;
    size_t   capacity_;
    size_t   occupied_;
    uint64_t total_lookups_;
    uint64_t total_hits_;

    // Find the slot for a given hash (linear probing)
    [[nodiscard]] size_t probe(const std::array<uint8_t, SHA256_SIZE>& hash) const;

    // Rehash to doubled capacity (per the doubling law)
    void rehash();
};

// ============================================================================
// AkashicFile — The complete .akashic file manager
//
// Creates, opens, reads, writes, and verifies Sempaevum.akashic files.
// This is the SINGLE AUTHORITY on file access. All modules interact
// with the .akashic file exclusively through this class.
//
// The file is organized as:
//   Page 0: AkashicFileHeader (4096 bytes)
//   Page 1+: Data pages, each with PageHeader (64 bytes) + body (4032 bytes)
//
// Sections are contiguous regions of pages. The section directory in
// the header stores the byte offset of each section's first page.
//
// Identification Principle:
//   P = the file on disk (substrate)
//   D = the header, section directory, page structure (constraints)
//   T = the create/open/read/write/close operations (agency)
// ============================================================================

class AkashicFile {
public:
    AkashicFile();
    ~AkashicFile();

    // ── File lifecycle ────────────────────────────────────────────

    // Create a new .akashic file at the given path.
    // Writes the initial header with all ET constants and section directory.
    // The file is ready for use after this call.
    // Throws ETError if the file already exists or cannot be created.
    void create(const std::string& path);

    // Open an existing .akashic file.
    // Reads and verifies the header (magic, version, SHA-256 checksum).
    // Loads the memoization store into memory.
    // Throws ETError if the file doesn't exist, is corrupt, or has
    // invalid magic/version/checksum.
    void open(const std::string& path);

    // Flush all pending writes and close the file.
    // Updates the header (modified_at_ns, counts, checksum).
    // After close(), the file is in a verified-consistent state.
    void close();

    // Is the file currently open?
    [[nodiscard]] bool is_open() const { return file_ != nullptr; }

    // The file path
    [[nodiscard]] const std::string& path() const { return path_; }

    // ── Header access ─────────────────────────────────────────────

    // Read the current header (in-memory copy, always consistent)
    [[nodiscard]] const AkashicFileHeader& header() const { return header_; }

    // Update header fields (in memory — written to disk on flush/close)
    AkashicFileHeader& header_mut() { return header_; }

    // Recompute and write the header to disk with updated SHA-256
    void flush_header();

    // Verify the on-disk header's SHA-256 checksum
    [[nodiscard]] bool verify_header_checksum();

    // ── Page management ───────────────────────────────────────────

    // Allocate a new page. Returns the byte offset of the new page.
    // The page is initialized to all zeros with the given type and section.
    [[nodiscard]] uint64_t allocate_page(PageType type, SectionID section);

    // Read a page from disk at the given byte offset
    void read_page(uint64_t offset, Page& page) const;

    // Write a page to disk at the given byte offset
    // Computes and stores the body CRC-32 before writing.
    void write_page(uint64_t offset, Page& page);

    // Verify a page's body CRC-32 at the given offset
    [[nodiscard]] bool verify_page(uint64_t offset) const;

    // ── Section directory ─────────────────────────────────────────

    // Get the byte offset of a section's first page (0 = not yet created)
    [[nodiscard]] uint64_t section_offset(SectionID section) const;

    // Set the byte offset of a section's first page
    void set_section_offset(SectionID section, uint64_t offset);

    // Initialize a section: allocate its first page and set the offset
    // Returns the byte offset of the new section's first page
    uint64_t initialize_section(SectionID section, PageType initial_page_type);

    // ── Memoization store access ──────────────────────────────────

    // Get the memoization store (in-memory hash table)
    [[nodiscard]] MemoStore& memo_store() { return memo_store_; }
    [[nodiscard]] const MemoStore& memo_store() const { return memo_store_; }

    // ── Persistent memoization (Stage 3b) ─────────────────────────

    // Write all memoization entries to the MEMOIZATION_STORE section.
    // Creates the section if it doesn't exist. Overwrites all pages
    // in the section with the current in-memory entries.
    // Called by close() before header flush.
    void flush_memo_store();

    // Load memoization entries from the MEMOIZATION_STORE section
    // into the in-memory hash table. Rebuilds the hash table from
    // the on-disk entries. Called by open() after header verification.
    void load_memo_store();

    // ── Integrity ─────────────────────────────────────────────────

    // Verify the entire file: header SHA-256 + every page CRC-32
    // Returns true if all checks pass. On failure, reports the first
    // corrupted page offset via the output parameter.
    [[nodiscard]] bool verify_full_integrity(uint64_t* first_corrupt_page = nullptr) const;

    // ── Utility ───────────────────────────────────────────────────

    // Get current nanosecond timestamp (for timestamps in the file)
    static uint64_t now_ns();

    // Compute SHA-256 of the header (bytes [0..4063])
    static std::array<uint8_t, SHA256_SIZE> compute_header_checksum(
        const AkashicFileHeader& hdr);

private:
    std::string          path_;
    FILE*                file_;
    AkashicFileHeader    header_;
    MemoStore            memo_store_;
    bool                 header_dirty_;   // Header modified since last flush

    // Initialize the header with ET constants and default values
    void init_header();

    // Write raw bytes at a file offset
    void write_at(uint64_t offset, const void* data, size_t size);

    // Read raw bytes from a file offset
    void read_at(uint64_t offset, void* data, size_t size) const;

    // Extend the file to accommodate new pages
    void extend_file(uint64_t new_size);

    // Get file size
    [[nodiscard]] uint64_t file_size() const;
};

// ============================================================================
// Operation type encoding — for equation memoization
//
// Each Sempaevum-native operation has a unique byte code.
// The Sempaevum IS Σ — every mathematical operation is native.
// ============================================================================

namespace OpType {
    constexpr uint8_t ADD        = 0x01;  // Value-space addition + reprojection
    constexpr uint8_t SUB        = 0x02;  // Value-space subtraction + reprojection
    constexpr uint8_t MUL        = 0x03;  // k-addition (lattice multiplication)
    constexpr uint8_t DIV        = 0x04;  // k-subtraction (lattice division)
    constexpr uint8_t POW        = 0x05;  // k-scaling (lattice power)
    constexpr uint8_t SQRT       = 0x06;  // k-scaling by 1/2
    constexpr uint8_t CBRT       = 0x07;  // k-scaling by 1/3
    constexpr uint8_t NEG        = 0x08;  // Unary negation
    constexpr uint8_t RECIP      = 0x09;  // k-negation (reciprocation)
    constexpr uint8_t ABS        = 0x0A;  // Absolute value
    constexpr uint8_t LOG        = 0x10;  // Natural logarithm
    constexpr uint8_t LOG2       = 0x11;  // Base-2 log (fundamental to projection)
    constexpr uint8_t LOG10      = 0x12;  // Base-10 log
    constexpr uint8_t EXP        = 0x13;  // Exponential
    constexpr uint8_t EXP2       = 0x14;  // 2^x (bijection pullback)
    constexpr uint8_t SIN        = 0x20;  // Trigonometric
    constexpr uint8_t COS        = 0x21;
    constexpr uint8_t TAN        = 0x22;
    constexpr uint8_t ASIN       = 0x23;
    constexpr uint8_t ACOS       = 0x24;
    constexpr uint8_t ATAN       = 0x25;
    constexpr uint8_t ATAN2      = 0x26;
    constexpr uint8_t SINH       = 0x30;  // Hyperbolic
    constexpr uint8_t COSH       = 0x31;
    constexpr uint8_t TANH       = 0x32;
    constexpr uint8_t ZETA       = 0x40;  // Special functions (FLINT/Arb)
    constexpr uint8_t GAMMA      = 0x41;
    constexpr uint8_t LGAMMA     = 0x42;
    constexpr uint8_t DIGAMMA    = 0x43;
    constexpr uint8_t BETA       = 0x44;
    constexpr uint8_t POLYLOG    = 0x45;
    constexpr uint8_t ERF        = 0x46;
    constexpr uint8_t ERFC       = 0x47;
    constexpr uint8_t BERNOULLI  = 0x48;
    constexpr uint8_t EML        = 0x50;  // EML Sheffer: eml(x,y) = exp(x) - ln(y)
    constexpr uint8_t FLOOR      = 0x60;  // Rounding
    constexpr uint8_t CEIL       = 0x61;
    constexpr uint8_t ROUND      = 0x62;
    constexpr uint8_t TRUNC      = 0x63;
    constexpr uint8_t FRAC       = 0x64;
    constexpr uint8_t IDENTITY   = 0xFE;  // Structural identity (no operation)
    constexpr uint8_t UNKNOWN    = 0xFF;  // Unknown/other
} // namespace OpType

// ============================================================================
// Equation form class encoding — for equation memoization
//
// §3.5: Computational classes (lattice computing answers) vs
// Structural classes (lattice declaring identities)
// ============================================================================

namespace FormClass {
    // Computational classes
    constexpr uint8_t ARITHMETIC_COMPUTATION     = 0x01;
    constexpr uint8_t LATTICE_MULTIPLICATION     = 0x02;
    constexpr uint8_t LATTICE_RECIPROCATION      = 0x03;
    constexpr uint8_t LATTICE_POWER              = 0x04;
    constexpr uint8_t LATTICE_ADDITION           = 0x05;
    constexpr uint8_t FUNCTION_EVALUATION        = 0x06;
    constexpr uint8_t ALGEBRAIC_SIMPLIFICATION   = 0x07;
    constexpr uint8_t SERIES_EVALUATION          = 0x08;

    // Structural classes
    constexpr uint8_t MASTER_EQUATION_INSTANCE   = 0x80;
    constexpr uint8_t DERIVATION_FORMULA         = 0x81;
    constexpr uint8_t STRUCTURAL_IDENTITY        = 0x82;
    constexpr uint8_t SUBSUMPTION_RELATIONSHIP   = 0x83;
    constexpr uint8_t PROJECTION_FORMULA         = 0x84;
    constexpr uint8_t RECURRENCE                 = 0x85;
    constexpr uint8_t SERIES_DEFINITION          = 0x86;
    constexpr uint8_t ALGEBRAIC_RELATION         = 0x87;
    constexpr uint8_t PREDICTION                 = 0x88;
} // namespace FormClass

} // namespace et::akashic