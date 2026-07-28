// ============================================================================
// akashic_format.cpp — Module 3: Akashic Format (Level 2)
//
// Implementation of the Sempaevum.akashic file format.
//
// Every operation is precise. Nothing silently fails. All integrity
// checks are mandatory, not optional. The file IS the Sempaevum on disk.
//
// P ∘ D ∘ T = E
// ============================================================================

#include "akashic_format.h"

#include <cstring>
#include <ctime>
#include <algorithm>
#include <stdexcept>

// Platform-specific high-resolution timestamp
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX   // Prevent windows.h from defining min/max macros that break std::min/std::max
#endif
#include <windows.h>
#else
#include <time.h>
#endif

namespace et::akashic {

// ============================================================================
// Page Implementation
// ============================================================================

void Page::clear() {
    std::memset(this, 0, sizeof(Page));
}

void Page::compute_body_checksum() {
    CRC32 crc;
    crc.update(body, PAGE_BODY_SIZE);
    header.body_checksum = crc.finalize();
}

bool Page::verify_body_checksum() const {
    CRC32 crc;
    crc.update(body, PAGE_BODY_SIZE);
    return crc.finalize() == header.body_checksum;
}

// ============================================================================
// MemoStore Implementation
//
// Open-addressing hash table with linear probing.
// Load factor maintained at K = 2/3. Rehash at doubled capacity.
//
// The hash function uses the first 8 bytes of the SHA-256 equation_hash
// as a uint64_t index (modular arithmetic with capacity). SHA-256 has
// excellent distribution — the first 64 bits provide uniform hashing.
// ============================================================================

MemoStore::MemoStore()
    : capacity_(0)
    , occupied_(0)
    , total_lookups_(0)
    , total_hits_(0)
{
}

void MemoStore::initialize(size_t capacity) {
    // Capacity must be power of 2 for efficient modular arithmetic.
    // Verify: capacity & (capacity - 1) == 0
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
        throw ETError(ETError::Code::INVALID_INPUT, "MemoStore::initialize",
            "Capacity must be a power of 2",
            "got: " + std::to_string(capacity));
    }

    capacity_ = capacity;
    occupied_ = 0;
    total_lookups_ = 0;
    total_hits_ = 0;

    // Resize table to capacity, all entries unoccupied
    table_.clear();
    table_.resize(capacity_);
    for (auto& entry : table_) {
        entry.occupied = false;
    }
}

size_t MemoStore::probe(const std::array<uint8_t, SHA256_SIZE>& hash) const {
    // Extract first 8 bytes of SHA-256 as uint64_t for initial index.
    // Little-endian interpretation (matching x86/x64 platform).
    uint64_t h = 0;
    for (int i = 7; i >= 0; --i) {
        h = (h << 8) | hash[static_cast<size_t>(i)];
    }

    // Modular reduction via bitmask (capacity is power of 2)
    size_t mask = capacity_ - 1;
    size_t idx = static_cast<size_t>(h & mask);

    // Linear probing: advance until we find the hash or an empty slot
    size_t probes = 0;
    while (probes < capacity_) {
        const auto& entry = table_[idx];
        if (!entry.occupied) {
            return idx;  // Empty slot — hash not found, insert here
        }
        if (entry.equation_hash == hash) {
            return idx;  // Found the hash
        }
        idx = (idx + 1) & mask;
        probes++;
    }

    // Should never reach here if load factor < 1
    // This IS a Descriptor Gap — the table is full (structurally impossible
    // if we rehash at K = 2/3, but we check anyway).
    throw ETError(ETError::Code::STACK_OVERFLOW, "MemoStore::probe",
        "Hash table full — this should be structurally impossible at K=2/3 load");
}

bool MemoStore::needs_rehash() const {
    if (capacity_ == 0) return true;
    // Check: occupied * HASH_LOAD_DEN > capacity * HASH_LOAD_NUM
    // i.e., occupied/capacity > 2/3
    // Using integer arithmetic to avoid IEEE 754:
    return (occupied_ * HASH_LOAD_DEN) > (capacity_ * HASH_LOAD_NUM);
}

MemoEntry* MemoStore::lookup(const std::array<uint8_t, SHA256_SIZE>& hash) {
    if (capacity_ == 0) {
        total_lookups_++;
        return nullptr;
    }

    total_lookups_++;

    size_t idx = probe(hash);
    auto& entry = table_[idx];

    if (!entry.occupied) {
        return nullptr;  // Cache MISS
    }

    if (entry.equation_hash == hash) {
        // Cache HIT — increment reference count and update timestamp
        total_hits_++;
        entry.reference_count++;
        entry.last_referenced_ns = AkashicFile::now_ns();
        return &entry;
    }

    return nullptr;  // Not found (probe returned empty slot)
}

MemoEntry* MemoStore::store(const MemoEntry& entry) {
    // Ensure the table is initialized
    if (capacity_ == 0) {
        initialize(HASH_INITIAL_CAP);
    }

    // Check if inserting one more entry would exceed K = 2/3
    // Proactive: rehash BEFORE the insert so the load factor invariant
    // (occupied/capacity ≤ K) is maintained at all times, including
    // immediately after this insertion completes.
    // (occupied + 1) * HASH_LOAD_DEN > capacity * HASH_LOAD_NUM
    if ((occupied_ + 1) * HASH_LOAD_DEN > capacity_ * HASH_LOAD_NUM) {
        rehash();
    }

    size_t idx = probe(entry.equation_hash);
    auto& slot = table_[idx];

    if (slot.occupied && slot.equation_hash == entry.equation_hash) {
        // Already exists — idempotent. Increment reference count.
        slot.reference_count++;
        slot.last_referenced_ns = AkashicFile::now_ns();
        return &slot;
    }

    // Insert into empty slot
    slot = entry;
    slot.occupied = true;
    occupied_++;

    return &slot;
}

void MemoStore::rehash() {
    // Double capacity per the doubling law τ(N_ℓ) = 6·2^ℓ
    size_t new_cap = (capacity_ == 0) ? HASH_INITIAL_CAP : capacity_ * 2;

    // Save old entries
    std::vector<MemoEntry> old_table = std::move(table_);

    // Reinitialize with new capacity
    capacity_ = new_cap;
    occupied_ = 0;
    table_.clear();
    table_.resize(new_cap);
    for (auto& e : table_) {
        e.occupied = false;
    }

    // Re-insert all occupied entries
    for (auto& old_entry : old_table) {
        if (old_entry.occupied) {
            size_t idx = probe(old_entry.equation_hash);
            table_[idx] = std::move(old_entry);
            table_[idx].occupied = true;
            occupied_++;
        }
    }
}

// ============================================================================
// Varint Encoding Implementation
// ============================================================================

namespace varint {

size_t encode(uint64_t value, uint8_t* buf) {
    size_t i = 0;
    while (value >= 0x80) {
        buf[i++] = static_cast<uint8_t>((value & 0x7F) | 0x80);
        value >>= 7;
    }
    buf[i++] = static_cast<uint8_t>(value & 0x7F);
    return i;
}

uint64_t decode(const uint8_t* buf, size_t* bytes_read) {
    uint64_t result = 0;
    size_t shift = 0;
    size_t i = 0;
    while (true) {
        uint8_t byte = buf[i];
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        i++;
        if ((byte & 0x80) == 0) break;
        shift += 7;
        if (shift >= 63) break;  // Prevent overflow — max 9 bytes for uint64
    }
    if (bytes_read) *bytes_read = i;
    return result;
}

size_t encoded_size(uint64_t value) {
    size_t size = 1;
    while (value >= 0x80) {
        value >>= 7;
        size++;
    }
    return size;
}

size_t encode_signed(int64_t value, uint8_t* buf) {
    // Zigzag encoding: (value << 1) ^ (value >> 63)
    // Maps: 0→0, -1→1, 1→2, -2→3, 2→4, ...
    auto zigzag = static_cast<uint64_t>((value << 1) ^ (value >> 63));
    return encode(zigzag, buf);
}

int64_t decode_signed(const uint8_t* buf, size_t* bytes_read) {
    uint64_t zigzag = decode(buf, bytes_read);
    // Reverse zigzag: (zigzag >> 1) ^ -(zigzag & 1)
    return static_cast<int64_t>((zigzag >> 1) ^ (~(zigzag & 1) + 1));
}

} // namespace varint

// ============================================================================
// MemoEntry Serialization Implementation
//
// Binary format per §7.1d Section 3. Zero IEEE 754.
// ALL lattice coordinates (N, k, d) serialize via GMP mpz_export/import —
// arbitrary precision, lossless at any magnitude. The LCM tower is infinite;
// lattice coordinates are unbounded. No uint32/int64 funnel exists anywhere.
//
// Fixed-width types used ONLY where structurally bounded:
//   eps_micros: int32 — bounded at ±50000 by the ∂I boundary (structural)
//   timestamps: uint64 — nanoseconds since Unix epoch (sufficient for ~584 years)
//   reference_count: uint64 — cache hits (sufficient for ~10^19 lookups)
//   form_class, operation_type: uint8 — enum values (structurally < 256)
// ============================================================================

namespace memo_serial {

// ── ETInteger binary serialization — lossless, arbitrary precision ─────
//
// Format:
//   sign_byte: uint8 (0x00 = zero, 0x01 = positive, 0xFF = negative)
//   if sign_byte != 0x00:
//     byte_count: varint (number of bytes in GMP export)
//     gmp_bytes: byte_count bytes (mpz_export, big-endian, unsigned magnitude)
//
// This is the Precision Stack's algebraic identity property extended to
// disk: ETInteger → GMP bytes → disk → GMP bytes → ETInteger, lossless.

static void serialize_etinteger(std::vector<uint8_t>& out, const ETInteger& val) {
    int sgn = mpz_sgn(val.raw());
    if (sgn == 0) {
        out.push_back(0x00);  // Zero: single byte
        return;
    }
    out.push_back(sgn > 0 ? 0x01 : 0xFF);  // Sign byte

    // Export absolute value to temporary buffer via GMP
    size_t count_estimate = (mpz_sizeinbase(val.raw(), 2) + 7) / 8;
    std::vector<uint8_t> tmp(count_estimate + 1);  // +1 for safety
    size_t actual_count = 0;
    mpz_export(tmp.data(), &actual_count, 1, 1, 1, 0, val.raw());
    // order=1 (MSB first), size=1 (bytes), endian=1 (big-endian), nails=0

    // Write actual byte count as varint
    uint8_t vbuf[9];
    size_t vlen = varint::encode(actual_count, vbuf);
    out.insert(out.end(), vbuf, vbuf + vlen);

    // Write the GMP bytes
    out.insert(out.end(), tmp.data(), tmp.data() + actual_count);
}

static ETInteger deserialize_etinteger(const uint8_t* data, size_t available,
                                        size_t* consumed) {
    size_t pos = 0;

    if (pos >= available) {
        throw ETError(ETError::Code::BLOB_CORRUPT, "deserialize_etinteger",
            "Unexpected end of data at sign byte");
    }
    uint8_t sign_byte = data[pos++];

    if (sign_byte == 0x00) {
        // Zero
        if (consumed) *consumed = pos;
        return ETInteger(int64_t(0));
    }

    // Read byte count
    size_t vread = 0;
    if (pos >= available) {
        throw ETError(ETError::Code::BLOB_CORRUPT, "deserialize_etinteger",
            "Unexpected end of data at byte count");
    }
    uint64_t byte_count = varint::decode(data + pos, &vread);
    pos += vread;

    if (pos + byte_count > available) {
        throw ETError(ETError::Code::BLOB_CORRUPT, "deserialize_etinteger",
            "GMP data extends beyond buffer",
            "need " + std::to_string(byte_count) + " bytes at offset " +
            std::to_string(pos) + ", available " + std::to_string(available));
    }

    // Import via GMP — lossless
    ETInteger result;
    mpz_import(result.raw(), static_cast<size_t>(byte_count),
               1, 1, 1, 0, data + pos);
    pos += static_cast<size_t>(byte_count);

    // Apply sign
    if (sign_byte == 0xFF) {
        mpz_neg(result.raw(), result.raw());
    }

    if (consumed) *consumed = pos;
    return result;
}

// ── Fixed-width helpers — ONLY for structurally bounded fields ────────

// eps_micros: int32, bounded at ±50000 by ∂I (structural invariant)
static void write_i32(uint8_t* buf, int32_t v) {
    auto u = static_cast<uint32_t>(v);
    buf[0] = static_cast<uint8_t>(u & 0xFF);
    buf[1] = static_cast<uint8_t>((u >> 8) & 0xFF);
    buf[2] = static_cast<uint8_t>((u >> 16) & 0xFF);
    buf[3] = static_cast<uint8_t>((u >> 24) & 0xFF);
}

static int32_t read_i32(const uint8_t* buf) {
    uint32_t u = static_cast<uint32_t>(buf[0])
               | (static_cast<uint32_t>(buf[1]) << 8)
               | (static_cast<uint32_t>(buf[2]) << 16)
               | (static_cast<uint32_t>(buf[3]) << 24);
    return static_cast<int32_t>(u);
}

// timestamps and reference_count: uint64, structurally bounded
static void write_u64(uint8_t* buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFF);
    }
}

static uint64_t read_u64(const uint8_t* buf) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(buf[i]) << (i * 8);
    }
    return v;
}

// ── MemoEntry serialization ───────────────────────────────────────────

std::vector<uint8_t> serialize(const MemoEntry& entry) {
    std::vector<uint8_t> out;
    out.reserve(512);

    // 1. equation_hash: 32 bytes (SHA-256, fixed)
    out.insert(out.end(), entry.equation_hash.begin(), entry.equation_hash.end());

    // 2. canonical_form: varint_length + UTF-8 bytes
    uint8_t vbuf[9];
    size_t vlen = varint::encode(entry.canonical_form.size(), vbuf);
    out.insert(out.end(), vbuf, vbuf + vlen);
    out.insert(out.end(), entry.canonical_form.begin(), entry.canonical_form.end());

    // 3. form_class: uint8 (enum, structurally < 256)
    out.push_back(entry.form_class);

    // 4. operation_type: uint8 (enum, structurally < 256)
    out.push_back(entry.operation_type);

    // 5. input_refs: varint_count + array of (N, k, d) — ALL via GMP, lossless
    vlen = varint::encode(entry.input_refs.size(), vbuf);
    out.insert(out.end(), vbuf, vbuf + vlen);
    for (const auto& ref : entry.input_refs) {
        serialize_etinteger(out, ref.n);  // Arbitrary-precision N
        serialize_etinteger(out, ref.k);  // Arbitrary-precision k
        serialize_etinteger(out, ref.d);  // Arbitrary-precision d
    }

    // 6. output_N: ETInteger via GMP — lossless
    serialize_etinteger(out, entry.output_n);

    // 7. output_k: ETInteger via GMP — lossless
    serialize_etinteger(out, entry.output_k);

    // 8. output_d: ETInteger via GMP — lossless
    serialize_etinteger(out, entry.output_d);

    // 9. output_eps_micros: int32 — structurally bounded at ±50000 by ∂I
    {
        uint8_t i32buf[4];
        write_i32(i32buf, entry.output_eps_micros);
        out.insert(out.end(), i32buf, i32buf + 4);
    }

    // 10. output_mpf: ETValue serialized blob (1200-bit MPFR, lossless)
    {
        auto blob = entry.output_value.serialize();
        vlen = varint::encode(blob.size(), vbuf);
        out.insert(out.end(), vbuf, vbuf + vlen);
        out.insert(out.end(), blob.begin(), blob.end());
    }

    // 11. reference_count: uint64 (structurally bounded — cache hit counter)
    {
        uint8_t u64buf[8];
        write_u64(u64buf, entry.reference_count);
        out.insert(out.end(), u64buf, u64buf + 8);
    }

    // 12. first_computed_ns: uint64 (nanosecond timestamp)
    {
        uint8_t u64buf[8];
        write_u64(u64buf, entry.first_computed_ns);
        out.insert(out.end(), u64buf, u64buf + 8);
    }

    // 13. last_referenced_ns: uint64 (nanosecond timestamp)
    {
        uint8_t u64buf[8];
        write_u64(u64buf, entry.last_referenced_ns);
        out.insert(out.end(), u64buf, u64buf + 8);
    }

    return out;
}

MemoEntry deserialize(const uint8_t* data, size_t available, size_t* bytes_consumed) {
    MemoEntry entry;
    size_t pos = 0;

    auto check_avail = [&](size_t need) {
        if (pos + need > available) {
            throw ETError(ETError::Code::BLOB_CORRUPT, "memo_serial::deserialize",
                "Unexpected end of data",
                "at offset " + std::to_string(pos) + ", need " +
                std::to_string(need) + ", available " + std::to_string(available));
        }
    };

    // 1. equation_hash: 32 bytes
    check_avail(SHA256_SIZE);
    std::memcpy(entry.equation_hash.data(), data + pos, SHA256_SIZE);
    pos += SHA256_SIZE;

    // 2. canonical_form: varint_length + UTF-8
    size_t vread = 0;
    check_avail(1);
    uint64_t str_len = varint::decode(data + pos, &vread);
    pos += vread;
    check_avail(static_cast<size_t>(str_len));
    entry.canonical_form.assign(reinterpret_cast<const char*>(data + pos),
                                 static_cast<size_t>(str_len));
    pos += static_cast<size_t>(str_len);

    // 3. form_class: uint8
    check_avail(1);
    entry.form_class = data[pos++];

    // 4. operation_type: uint8
    check_avail(1);
    entry.operation_type = data[pos++];

    // 5. input_refs: varint_count + array of (N, k, d) via GMP
    check_avail(1);
    uint64_t ref_count = varint::decode(data + pos, &vread);
    pos += vread;
    entry.input_refs.resize(static_cast<size_t>(ref_count));
    for (size_t i = 0; i < static_cast<size_t>(ref_count); ++i) {
        size_t consumed = 0;
        entry.input_refs[i].n = deserialize_etinteger(data + pos, available - pos, &consumed);
        pos += consumed;
        entry.input_refs[i].k = deserialize_etinteger(data + pos, available - pos, &consumed);
        pos += consumed;
        entry.input_refs[i].d = deserialize_etinteger(data + pos, available - pos, &consumed);
        pos += consumed;
    }

    // 6. output_N: ETInteger via GMP
    {
        size_t consumed = 0;
        entry.output_n = deserialize_etinteger(data + pos, available - pos, &consumed);
        pos += consumed;
    }

    // 7. output_k: ETInteger via GMP
    {
        size_t consumed = 0;
        entry.output_k = deserialize_etinteger(data + pos, available - pos, &consumed);
        pos += consumed;
    }

    // 8. output_d: ETInteger via GMP
    {
        size_t consumed = 0;
        entry.output_d = deserialize_etinteger(data + pos, available - pos, &consumed);
        pos += consumed;
    }

    // 9. output_eps_micros: int32 (structurally bounded by ∂I)
    check_avail(4);
    entry.output_eps_micros = read_i32(data + pos);
    pos += 4;

    // 10. output_mpf: varint_length + ETValue blob (1200-bit, lossless)
    check_avail(1);
    uint64_t blob_len = varint::decode(data + pos, &vread);
    pos += vread;
    check_avail(static_cast<size_t>(blob_len));
    entry.output_value = ETValue::deserialize(data + pos, static_cast<size_t>(blob_len));
    pos += static_cast<size_t>(blob_len);

    // 11. reference_count: uint64
    check_avail(8);
    entry.reference_count = read_u64(data + pos);
    pos += 8;

    // 12. first_computed_ns: uint64
    check_avail(8);
    entry.first_computed_ns = read_u64(data + pos);
    pos += 8;

    // 13. last_referenced_ns: uint64
    check_avail(8);
    entry.last_referenced_ns = read_u64(data + pos);
    pos += 8;

    entry.occupied = true;

    if (bytes_consumed) *bytes_consumed = pos;
    return entry;
}

} // namespace memo_serial

// ============================================================================
// AkashicFile Implementation
// ============================================================================

AkashicFile::AkashicFile()
    : file_(nullptr)
    , header_{}
    , header_dirty_(false)
{
}

AkashicFile::~AkashicFile() {
    if (file_) {
        try {
            close();
        } catch (...) {
            // Destructor must not throw. If close fails, the file
            // may be in an inconsistent state — Omniscient (Module 26)
            // will detect this via header hash mismatch.
            if (file_) {
                std::fclose(file_);
                file_ = nullptr;
            }
        }
    }
}

uint64_t AkashicFile::now_ns() {
#ifdef _WIN32
    // Windows: GetSystemTimePreciseAsFileTime for nanosecond wall-clock time.
    // FILETIME gives 100-nanosecond intervals since 1601-01-01.
    // Convert to nanoseconds since Unix epoch (1970-01-01).
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    // Unix epoch offset: 11644473600 seconds = 116444736000000000 × 100ns
    constexpr uint64_t EPOCH_OFFSET_100NS = 116444736000000000ULL;
    uint64_t unix_100ns = uli.QuadPart - EPOCH_OFFSET_100NS;
    return unix_100ns * 100;  // Convert 100ns intervals to nanoseconds
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
#endif
}

void AkashicFile::init_header() {
    std::memset(&header_, 0, sizeof(header_));

    // Magic bytes
    std::memcpy(header_.magic, MAGIC_BYTES, 4);

    // Format version
    header_.format_version = FORMAT_VERSION;

    // N_base = 12 (forced by Triple Minimal-Backbone Theorem)
    header_.n_base = ET_N;

    // Koide ratio K = 2/3 (binding stability threshold)
    header_.k_num = 2;
    header_.k_den = 3;

    // Base variance V = 1/12
    header_.v_num = 1;
    header_.v_den = 12;

    // Self-projection: initially the file IS page 0, which is 4096 bytes.
    // 4096 = 2^12, so at N=12: k = 12 (since 2^(12/12) = 2^1 = 2,
    // but we need to project 4096 = 2^12 → k = 12*12 = 144... no.
    // Actually: project r = 4096 at N=12.
    // k = round(12 * log2(4096)) = round(12 * 12) = 144
    // d = 12 / gcd(144, 12) = 12/12 = 1
    // ε = (12*12 - 144) * 1200/12 = 0
    // Self-projection: (N=12, k=144, d=1, ε=0) — exact, d=1 octave.
    // 4096 = 2^12 is a pure power of 2 → d=1, ε=0 exactly.
    header_.self_n = 12;
    header_.self_k = 144;
    header_.self_d = 1;
    header_.self_eps_micros = 0;

    // Generator backbone: initially empty
    header_.total_generators = 0;
    header_.l1_webb_count = 0;
    header_.l2_cascade_count = 0;
    header_.l3_eml_count = 0;
    header_.total_memoized = 0;

    // Coverage: initially 0/0 (no addresses yet)
    header_.covered_addresses = 0;
    header_.total_addresses = 0;

    // K-complexity: initially 0/0
    header_.generator_bytes = 0;
    header_.producible_bytes = 0;

    // Timestamps
    uint64_t now = now_ns();
    header_.created_at_ns = now;
    header_.modified_at_ns = now;

    // D-time at creation: project the creation timestamp.
    // The timestamp in seconds since epoch = now_ns / 10^9.
    // At N=12: k = round(12 * log2(seconds_since_epoch)).
    // For a creation timestamp, we store the N and k of the
    // D-time coordinate. Initially just store N=12, k=0 as placeholder
    // (the actual D-time projection requires Module 2 which would
    // create a circular dependency at file creation time — the full
    // D-time projection happens when the first value is ingested).
    header_.creation_dtime_n = 12;
    header_.creation_dtime_k = 0;

    // Section directory: all sections initially uncreated (offset 0)
    for (auto& offset : header_.section_offsets) {
        offset = 0;
    }

    // Memoization metadata
    header_.memo_capacity = HASH_INITIAL_CAP;
    header_.memo_occupied = 0;
    header_.memo_total_lookups = 0;
    header_.memo_total_hits = 0;

    // File geometry: starts with 1 page (the header itself)
    header_.total_pages = 1;
    header_.free_page_head = 0;  // No free pages

    // Reserved: already zeroed by memset

    // Checksum: computed and stored last
    auto checksum = compute_header_checksum(header_);
    std::memcpy(header_.header_checksum, checksum.data(), SHA256_SIZE);
}

std::array<uint8_t, SHA256_SIZE> AkashicFile::compute_header_checksum(
    const AkashicFileHeader& hdr) {
    // SHA-256 of bytes [0..4063] — everything BEFORE the checksum field.
    // The checksum field is at offset (4096 - 32) = 4064.
    constexpr size_t checksum_offset = FILE_HEADER_SIZE - SHA256_SIZE;
    return SHA256::hash(reinterpret_cast<const char*>(&hdr), checksum_offset);
}

void AkashicFile::create(const std::string& path) {
    if (file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::create",
            "Cannot create — a file is already open",
            "close the current file first");
    }

    // Check if file already exists
    FILE* test = std::fopen(path.c_str(), "rb");
    if (test) {
        std::fclose(test);
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::create",
            "File already exists",
            "path: " + path);
    }

    // Create the file
    file_ = std::fopen(path.c_str(), "w+b");
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::create",
            "Cannot create file",
            "path: " + path + " (check permissions and path validity)");
    }

    path_ = path;

    // Initialize the header with ET constants
    init_header();

    // Write the header as page 0
    write_at(0, &header_, sizeof(header_));

    // Flush to ensure the header is on disk
    std::fflush(file_);

    // Initialize the in-memory memoization store
    memo_store_.initialize(HASH_INITIAL_CAP);

    header_dirty_ = false;
}

void AkashicFile::open(const std::string& path) {
    if (file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::open",
            "Cannot open — a file is already open",
            "close the current file first");
    }

    // Open existing file for read+write
    file_ = std::fopen(path.c_str(), "r+b");
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::open",
            "Cannot open file",
            "path: " + path + " (file may not exist or permissions may be insufficient)");
    }

    path_ = path;

    // Read the header
    read_at(0, &header_, sizeof(header_));

    // Verify magic bytes
    if (std::memcmp(header_.magic, MAGIC_BYTES, 4) != 0) {
        std::fclose(file_);
        file_ = nullptr;
        throw ETError(ETError::Code::BLOB_CORRUPT, "AkashicFile::open",
            "Invalid magic bytes — not a Sempaevum.akashic file",
            "expected: SMVM, got: " +
            std::string(reinterpret_cast<const char*>(header_.magic), 4));
    }

    // Verify format version
    if (header_.format_version != FORMAT_VERSION) {
        std::fclose(file_);
        file_ = nullptr;
        throw ETError(ETError::Code::BLOB_CORRUPT, "AkashicFile::open",
            "Unsupported format version",
            "expected: " + std::to_string(FORMAT_VERSION) +
            ", got: " + std::to_string(header_.format_version));
    }

    // Verify header SHA-256 checksum
    auto expected = compute_header_checksum(header_);
    if (std::memcmp(expected.data(), header_.header_checksum, SHA256_SIZE) != 0) {
        std::fclose(file_);
        file_ = nullptr;
        throw ETError(ETError::Code::HASH_MISMATCH, "AkashicFile::open",
            "Header SHA-256 checksum mismatch — file may be corrupt or tampered",
            "path: " + path);
    }

    // Verify ET constants are correct (they should never change)
    if (header_.n_base != ET_N ||
        header_.k_num != 2 || header_.k_den != 3 ||
        header_.v_num != 1 || header_.v_den != 12) {
        std::fclose(file_);
        file_ = nullptr;
        throw ETError(ETError::Code::BLOB_CORRUPT, "AkashicFile::open",
            "ET constants in header are incorrect — file is structurally invalid",
            "N_base=" + std::to_string(header_.n_base) +
            " K=" + std::to_string(header_.k_num) + "/" + std::to_string(header_.k_den) +
            " V=" + std::to_string(header_.v_num) + "/" + std::to_string(header_.v_den));
    }

    // Initialize the in-memory memoization store from header metadata
    size_t memo_cap = static_cast<size_t>(header_.memo_capacity);
    if (memo_cap == 0) memo_cap = HASH_INITIAL_CAP;
    // Ensure power of 2
    if ((memo_cap & (memo_cap - 1)) != 0) {
        // Round up to next power of 2
        size_t p = 1;
        while (p < memo_cap) p <<= 1;
        memo_cap = p;
    }
    memo_store_.initialize(memo_cap);

    // Load memoization entries from the MEMOIZATION_STORE section
    // into the in-memory hash table. Rebuilds the entire table from disk.
    load_memo_store();

    header_dirty_ = false;
}

void AkashicFile::close() {
    if (!file_) {
        return;  // Already closed — idempotent
    }

    // Persist the memoization store to disk (must precede header flush
    // because it updates header metrics)
    flush_memo_store();

    // Update header timestamps and metrics
    header_.modified_at_ns = now_ns();

    // Flush the header to disk (includes memo metrics + SHA-256)
    flush_header();

    // Sync to physical disk
    std::fflush(file_);

    // Close the file handle
    std::fclose(file_);
    file_ = nullptr;
    path_.clear();
    header_dirty_ = false;
}

void AkashicFile::flush_header() {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::flush_header",
            "No file open");
    }

    // Update modification timestamp
    header_.modified_at_ns = now_ns();

    // Sync memoization metrics
    header_.memo_capacity = static_cast<uint64_t>(memo_store_.capacity());
    header_.memo_occupied = static_cast<uint64_t>(memo_store_.occupied());
    header_.memo_total_lookups = memo_store_.total_lookups();
    header_.memo_total_hits = memo_store_.total_hits();

    // Recompute SHA-256 checksum
    auto checksum = compute_header_checksum(header_);
    std::memcpy(header_.header_checksum, checksum.data(), SHA256_SIZE);

    // Write header to page 0
    write_at(0, &header_, sizeof(header_));

    // Flush OS buffers
    std::fflush(file_);

    header_dirty_ = false;
}

bool AkashicFile::verify_header_checksum() {
    if (!file_) return false;

    // Read the on-disk header
    AkashicFileHeader disk_header{};
    read_at(0, &disk_header, sizeof(disk_header));

    // Compute expected checksum
    auto expected = compute_header_checksum(disk_header);

    // Compare
    return std::memcmp(expected.data(), disk_header.header_checksum, SHA256_SIZE) == 0;
}

uint64_t AkashicFile::allocate_page(PageType type, SectionID section) {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::allocate_page",
            "No file open");
    }

    uint64_t offset;

    // Check the free page list first
    if (header_.free_page_head != 0) {
        // Reuse a free page
        offset = header_.free_page_head;

        // Read the free page to get its next pointer
        Page free_page{};
        read_page(offset, free_page);

        // Update free list head to point to the next free page
        header_.free_page_head = free_page.header.next_page;

        // Clear the page for reuse
        free_page.clear();
        free_page.header.page_type = static_cast<uint8_t>(type);
        free_page.header.section_id = static_cast<uint8_t>(section);
        free_page.compute_body_checksum();
        write_at(offset, &free_page, sizeof(free_page));
    } else {
        // Allocate at end of file
        offset = header_.total_pages * PAGE_SIZE;
        header_.total_pages++;

        // Extend the file
        extend_file(header_.total_pages * PAGE_SIZE);

        // Write a clean page
        Page new_page{};
        new_page.clear();
        new_page.header.page_type = static_cast<uint8_t>(type);
        new_page.header.section_id = static_cast<uint8_t>(section);
        new_page.compute_body_checksum();
        write_at(offset, &new_page, sizeof(new_page));
    }

    header_dirty_ = true;
    return offset;
}

void AkashicFile::read_page(uint64_t offset, Page& page) const {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::read_page",
            "No file open");
    }

    // Validate offset alignment
    if (offset % PAGE_SIZE != 0) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::read_page",
            "Page offset not aligned to page boundary",
            "offset: " + std::to_string(offset) +
            " (must be multiple of " + std::to_string(PAGE_SIZE) + ")");
    }

    // Validate offset is within file bounds
    if (offset >= header_.total_pages * PAGE_SIZE) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::read_page",
            "Page offset beyond file end",
            "offset: " + std::to_string(offset) +
            ", file size: " + std::to_string(header_.total_pages * PAGE_SIZE));
    }

    read_at(offset, &page, sizeof(page));
}

void AkashicFile::write_page(uint64_t offset, Page& page) {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::write_page",
            "No file open");
    }

    // Validate offset alignment
    if (offset % PAGE_SIZE != 0) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::write_page",
            "Page offset not aligned to page boundary",
            "offset: " + std::to_string(offset));
    }

    // Compute body CRC-32 before writing
    page.compute_body_checksum();

    write_at(offset, &page, sizeof(page));
}

bool AkashicFile::verify_page(uint64_t offset) const {
    Page page{};
    read_page(offset, page);
    return page.verify_body_checksum();
}

uint64_t AkashicFile::section_offset(SectionID section) const {
    auto idx = static_cast<size_t>(section);
    if (idx >= SECTION_COUNT) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::section_offset",
            "Invalid section ID",
            "id: " + std::to_string(idx));
    }
    return header_.section_offsets[idx];
}

void AkashicFile::set_section_offset(SectionID section, uint64_t offset) {
    auto idx = static_cast<size_t>(section);
    if (idx >= SECTION_COUNT) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::set_section_offset",
            "Invalid section ID",
            "id: " + std::to_string(idx));
    }
    header_.section_offsets[idx] = offset;
    header_dirty_ = true;
}

uint64_t AkashicFile::initialize_section(SectionID section, PageType initial_page_type) {
    if (section_offset(section) != 0) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::initialize_section",
            "Section already initialized",
            "section: " + std::to_string(static_cast<int>(section)));
    }

    uint64_t offset = allocate_page(initial_page_type, section);
    set_section_offset(section, offset);
    return offset;
}

bool AkashicFile::verify_full_integrity(uint64_t* first_corrupt_page) const {
    if (!file_) return false;

    // 1. Verify header SHA-256
    AkashicFileHeader disk_header{};
    read_at(0, &disk_header, sizeof(disk_header));
    auto expected_checksum = compute_header_checksum(disk_header);
    if (std::memcmp(expected_checksum.data(),
                    disk_header.header_checksum, SHA256_SIZE) != 0) {
        if (first_corrupt_page) *first_corrupt_page = 0;
        return false;
    }

    // 2. Verify every data page's CRC-32
    // Pages start at offset PAGE_SIZE (after the header)
    for (uint64_t page_idx = 1; page_idx < disk_header.total_pages; ++page_idx) {
        uint64_t offset = page_idx * PAGE_SIZE;
        Page page{};
        read_at(offset, &page, sizeof(page));

        // Skip free pages (type 0x00) — their body content is irrelevant
        if (page.header.page_type == static_cast<uint8_t>(PageType::FREE)) {
            continue;
        }

        if (!page.verify_body_checksum()) {
            if (first_corrupt_page) *first_corrupt_page = offset;
            return false;
        }
    }

    // 3. Verify section directory consistency:
    // Every section offset must point to a valid page boundary within the file
    for (const auto& sect_offset : disk_header.section_offsets) {
        if (sect_offset == 0) continue;  // Section not created

        if (sect_offset % PAGE_SIZE != 0) {
            if (first_corrupt_page) *first_corrupt_page = sect_offset;
            return false;
        }

        if (sect_offset >= disk_header.total_pages * PAGE_SIZE) {
            if (first_corrupt_page) *first_corrupt_page = sect_offset;
            return false;
        }
    }

    return true;
}

// ── Persistent Memoization ────────────────────────────────────────────

void AkashicFile::flush_memo_store() {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::flush_memo_store",
            "No file open");
    }

    // Serialize all occupied entries
    std::vector<uint8_t> all_data;
    all_data.reserve(memo_store_.occupied() * 300);  // ~300 bytes typical per entry

    uint64_t entry_count = 0;
    for (const auto& entry : memo_store_.entries()) {
        if (!entry.occupied) continue;
        auto blob = memo_serial::serialize(entry);
        // Prefix each entry with its serialized length (varint) for framing
        uint8_t vbuf[9];
        size_t vlen = varint::encode(blob.size(), vbuf);
        all_data.insert(all_data.end(), vbuf, vbuf + vlen);
        all_data.insert(all_data.end(), blob.begin(), blob.end());
        entry_count++;
    }

    // Ensure the MEMOIZATION_STORE section exists
    if (section_offset(SectionID::MEMOIZATION_STORE) == 0) {
        initialize_section(SectionID::MEMOIZATION_STORE, PageType::MEMO_ENTRY);
    }

    // Write serialized data across pages in the section.
    // Each page body holds up to PAGE_BODY_SIZE bytes.
    // Pages are chained via next_page.
    uint64_t current_page_offset = section_offset(SectionID::MEMOIZATION_STORE);
    size_t data_pos = 0;
    uint64_t page_seq = 0;

    while (data_pos < all_data.size() || page_seq == 0) {
        // If we need a new page beyond the first, allocate it
        if (page_seq > 0) {
            uint64_t new_page_offset = allocate_page(PageType::MEMO_ENTRY,
                                                       SectionID::MEMOIZATION_STORE);
            // Link the previous page to this one
            Page prev_page{};
            read_page(current_page_offset, prev_page);
            prev_page.header.next_page = new_page_offset;
            write_page(current_page_offset, prev_page);
            current_page_offset = new_page_offset;
        }

        // Fill this page's body with as much data as fits
        Page page{};
        page.header.page_type = static_cast<uint8_t>(PageType::MEMO_ENTRY);
        page.header.section_id = static_cast<uint8_t>(SectionID::MEMOIZATION_STORE);
        page.header.page_sequence = page_seq;
        page.header.next_page = 0;  // Will be updated if we need more pages

        size_t chunk = std::min(static_cast<size_t>(PAGE_BODY_SIZE),
                                all_data.size() - data_pos);
        if (chunk > 0) {
            std::memcpy(page.body, all_data.data() + data_pos, chunk);
        }
        page.header.used_bytes = static_cast<uint32_t>(chunk);
        page.header.entry_count = (page_seq == 0)
            ? static_cast<uint32_t>(entry_count)  // Total count on first page
            : 0;

        write_page(current_page_offset, page);

        data_pos += chunk;
        page_seq++;

        // If all data written and this was the first iteration, we're done
        if (data_pos >= all_data.size()) break;
    }

    // Update header memo metrics
    header_.memo_capacity = static_cast<uint64_t>(memo_store_.capacity());
    header_.memo_occupied = static_cast<uint64_t>(memo_store_.occupied());
    header_.memo_total_lookups = memo_store_.total_lookups();
    header_.memo_total_hits = memo_store_.total_hits();
    header_dirty_ = true;
}

void AkashicFile::load_memo_store() {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::load_memo_store",
            "No file open");
    }

    uint64_t sect_offset = section_offset(SectionID::MEMOIZATION_STORE);
    if (sect_offset == 0) {
        // Section doesn't exist yet — no entries to load
        return;
    }

    // Read all page bodies into a contiguous buffer by following the chain
    std::vector<uint8_t> all_data;
    uint64_t total_entries = 0;
    uint64_t page_offset = sect_offset;

    while (page_offset != 0) {
        Page page{};
        read_page(page_offset, page);

        // Verify this is a MEMO_ENTRY page
        if (page.header.page_type != static_cast<uint8_t>(PageType::MEMO_ENTRY)) {
            throw ETError(ETError::Code::BLOB_CORRUPT, "AkashicFile::load_memo_store",
                "Expected MEMO_ENTRY page",
                "got page_type: " + std::to_string(page.header.page_type) +
                " at offset: " + std::to_string(page_offset));
        }

        // Verify CRC-32
        if (!page.verify_body_checksum()) {
            throw ETError(ETError::Code::CRC_MISMATCH, "AkashicFile::load_memo_store",
                "Page CRC-32 mismatch in MEMOIZATION_STORE",
                "page offset: " + std::to_string(page_offset));
        }

        // First page's entry_count holds the total entry count
        if (page_offset == sect_offset) {
            total_entries = page.header.entry_count;
        }

        // Append used body bytes to buffer
        if (page.header.used_bytes > 0) {
            all_data.insert(all_data.end(),
                           page.body,
                           page.body + page.header.used_bytes);
        }

        // Follow the chain
        page_offset = page.header.next_page;
    }

    // Deserialize entries from the contiguous buffer
    // Each entry is prefixed with a varint length for framing
    size_t pos = 0;
    uint64_t loaded = 0;

    while (pos < all_data.size() && loaded < total_entries) {
        // Read the framing length
        size_t vread = 0;
        uint64_t entry_len = varint::decode(all_data.data() + pos, &vread);
        pos += vread;

        if (pos + entry_len > all_data.size()) {
            throw ETError(ETError::Code::BLOB_CORRUPT, "AkashicFile::load_memo_store",
                "Entry extends beyond data buffer",
                "entry " + std::to_string(loaded) + " at offset " + std::to_string(pos) +
                ", length " + std::to_string(entry_len) +
                ", buffer size " + std::to_string(all_data.size()));
        }

        // Deserialize the entry
        size_t consumed = 0;
        MemoEntry entry = memo_serial::deserialize(
            all_data.data() + pos,
            static_cast<size_t>(entry_len),
            &consumed);
        pos += consumed;

        // Store into the in-memory hash table
        memo_store_.store(entry);
        loaded++;
    }
}

// ── Private helpers ────────────────────────────────────────────────────

void AkashicFile::write_at(uint64_t offset, const void* data, size_t size) {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::write_at",
            "No file open");
    }

    // Seek to offset
    // Use platform-appropriate 64-bit seek
#ifdef _WIN32
    if (_fseeki64(file_, static_cast<int64_t>(offset), SEEK_SET) != 0) {
#else
    if (fseeko(file_, static_cast<off_t>(offset), SEEK_SET) != 0) {
#endif
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::write_at",
            "Seek failed",
            "offset: " + std::to_string(offset));
    }

    size_t written = std::fwrite(data, 1, size, file_);
    if (written != size) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::write_at",
            "Write failed — possible disk full",
            "expected: " + std::to_string(size) +
            " bytes, wrote: " + std::to_string(written));
    }
}

void AkashicFile::read_at(uint64_t offset, void* data, size_t size) const {
    if (!file_) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::read_at",
            "No file open");
    }

#ifdef _WIN32
    if (_fseeki64(const_cast<FILE*>(file_), static_cast<int64_t>(offset), SEEK_SET) != 0) {
#else
    if (fseeko(const_cast<FILE*>(file_), static_cast<off_t>(offset), SEEK_SET) != 0) {
#endif
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::read_at",
            "Seek failed",
            "offset: " + std::to_string(offset));
    }

    size_t rd = std::fread(data, 1, size, const_cast<FILE*>(file_));
    if (rd != size) {
        throw ETError(ETError::Code::INVALID_INPUT, "AkashicFile::read_at",
            "Read failed — unexpected end of file or I/O error",
            "expected: " + std::to_string(size) +
            " bytes, read: " + std::to_string(rd));
    }
}

void AkashicFile::extend_file(uint64_t new_size) {
    // Seek to end and write zeros to extend
    uint64_t current = file_size();
    if (new_size <= current) return;

    // Write zeros from current end to new_size
    uint64_t to_write = new_size - current;
    std::vector<uint8_t> zeros(static_cast<size_t>(std::min(to_write,
        static_cast<uint64_t>(PAGE_SIZE))), 0);

    uint64_t pos = current;
    while (pos < new_size) {
        size_t chunk = static_cast<size_t>(
            std::min(static_cast<uint64_t>(zeros.size()), new_size - pos));
        write_at(pos, zeros.data(), chunk);
        pos += chunk;
    }
}

uint64_t AkashicFile::file_size() const {
    if (!file_) return 0;

    // Save current position
#ifdef _WIN32
    int64_t saved = _ftelli64(const_cast<FILE*>(file_));
    _fseeki64(const_cast<FILE*>(file_), 0, SEEK_END);
    int64_t size = _ftelli64(const_cast<FILE*>(file_));
    _fseeki64(const_cast<FILE*>(file_), saved, SEEK_SET);
#else
    off_t saved = ftello(const_cast<FILE*>(file_));
    fseeko(const_cast<FILE*>(file_), 0, SEEK_END);
    off_t size = ftello(const_cast<FILE*>(file_));
    fseeko(const_cast<FILE*>(file_), saved, SEEK_SET);
#endif
    return static_cast<uint64_t>(size);
}

} // namespace et::akashic