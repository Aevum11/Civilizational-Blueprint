/*
 * et_bridge32.c
 * ET32 Bridge — 32-bit Injectable DLL
 *
 * Derived from P ∘ D ∘ T = E.
 *
 * PDT decomposition of this DLL:
 *   P = the 32-bit target process address space (4 GB substrate)
 *   D = the ET bridge protocol (ETPacket, pipe connection, IAT hooks)
 *   T = the executing thread inside the target process
 *   E = a completed 64-bit call from within 32-bit address space
 *
 * ET constants used throughout (from et_math.py):
 *   S  = 12        (manifold symmetry)
 *   K  = 2/3       (Koide ratio — stability threshold)
 *   hd = 4096      (digital action quantum)
 *   IPC_BUFFER_SIZE = hd × S = 49152 bytes
 *   PDT_HEADER_SIZE = 4 × S = 48 bytes
 *   CONN_TIMEOUT_MS = (1/K) × 1000 = 1500 ms
 *   RETRY_COUNT     = S = 12
 *   QUEUE_DEPTH     = S² = 144
 *   HANDLE_BASE     = 0x80000001
 *   HANDLE_MAX      = 0xFFFFF000
 *
 * Build (MinGW 32-bit):
 *   gcc -m32 -O2 -shared -o et_bridge32.dll et_bridge32.c \
 *       -lkernel32 -ladvapi32 -lntdll \
 *       -Wl,--subsystem,windows
 *
 * Build (MSVC 32-bit):
 *   cl /LD /Ox /arch:IA32 et_bridge32.c kernel32.lib advapi32.lib
 *
 * Compile target: Windows x86 (32-bit), tested on XP SP3 and Windows 11.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Michael James Muller (Aevum Defluo) — Exception Theory
 */

#ifndef _WIN32
#  error "et_bridge32.c must be compiled for Windows (32-bit)"
#endif

/* Require at least Windows XP API surface */
#ifndef WINVER
#  define WINVER 0x0501
#endif
#ifndef _WIN32_WINNT                        /* NOLINT(bugprone-reserved-identifier,cert-dcl37-c) */
#  define _WIN32_WINNT 0x0501               /* NOLINT(bugprone-reserved-identifier,cert-dcl37-c) */
#endif

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winreg.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * ET CONSTANTS — exact mirror of et_math.py
 * ============================================================================ */

#define ET_S                 12                 /* manifold symmetry          */
#define ET_IPC_BUFFER_SIZE   49152              /* hd × S = 4096 × 12        */
#define ET_PDT_HEADER_SIZE   48                 /* 4 × S                     */
#define ET_CONN_TIMEOUT_MS   1500               /* (1/K) × 1000 ms           */
#define ET_RETRY_COUNT       12                 /* S                         */
#define ET_QUEUE_DEPTH       144                /* S²                        */
#define ET_HANDLE_BASE       0x80000001UL
#define ET_HANDLE_MAX        0xFFFFF000UL
#define ET_ADDR64_BASE       0x100000000ULL     /* first 64-bit-only address */
#define ET_SLOT_STRIDE       12                 /* S                         */
#define ET_HANDSHAKE_MAGIC   0x50445445UL       /* "ETDP" LE                 */
#define ET_DLL_VERSION       0x00010000UL       /* 1.0.0                     */

/* Pipe name format: \\.\pipe\ET32_PDT_{pid} */
#define ET_PIPE_NAME_FMT     "\\\\.\\pipe\\ET32_PDT_%lu"
#define ET_PIPE_NAME_MAX     64

/* ============================================================================
 * FORWARD DECLARATIONS
 *
 * The error-reporting section must appear early so ET_CHECK_WIN32 and
 * ET_REQUIRE macros are available to every function in the file.
 * It references globals, types, CMD defines, and functions defined later.
 * All forward-declared here; actual definitions follow in their sections.
 * ============================================================================ */

/* Global connection state — actual definitions moved here so
 * et_report_error() can reference g_connected.
 * Original location: section "GLOBAL CONNECTION STATE".
 * Verified by compile-time assertion in that section below. */
#define ET_CONN_GLOBALS_DEFINED
static BOOL         g_connected   = FALSE;
static BOOL         g_initialised = FALSE;

/* Arg buffer type — needed by et_report_error() to build error payloads.
 * Full definition with helper functions is in "ARG PACKING" below.
 * Verified by compile-time assertion in that section below. */
#define ET_ARG_BUF_DEFINED
typedef struct {
    uint8_t *base;
    size_t   capacity;
    size_t   pos;
    int      count;
} et_arg_buf;

/* Arg packing function prototypes */
static void et_argbuf_init(et_arg_buf *b, uint8_t *mem, size_t cap);
static int  et_argbuf_room(et_arg_buf *b, size_t need);
static void et_pack_null(et_arg_buf *b);
static void et_pack_uint32(et_arg_buf *b, uint32_t v);
static void et_pack_int32(et_arg_buf *b, int32_t v);
static void et_pack_strA(et_arg_buf *b, const char *s);

/* CMD family / code constants needed by et_report_error().
 * Full CMD table is in "CMD FAMILY / CMD CODE" section below. */
#ifndef CMD_FAMILY_COMPOUND_OPS
#define CMD_FAMILY_COMPOUND_OPS   12
#endif
#ifndef CMD_CTRL_ERR
#define CMD_CTRL_ERR              0xFF
#endif

/* IPC request/response function — defined in "REQUEST / RESPONSE CYCLE" */
static uint32_t et_call(
    uint8_t   cmd_family,
    uint8_t   cmd_code,
    const uint8_t *payload,
    size_t    payload_len,
    uint32_t *error_out);

/* ============================================================================
 * ET ERROR REPORTING — C-side structured error system
 *
 * ET PDT of the error system:
 *   P = the failure event (what went wrong)
 *   D = the error context (location, OS error, ET state)
 *   T = the error reporter (sends structured report to broker)
 *   E = broker receives precise diagnostic → engineer can fix exactly
 *
 * Every Windows API failure calls et_report_error() which:
 *   1. Captures GetLastError() immediately
 *   2. Sends an ETPacket with CMD_CTRL_ERR containing:
 *      - file:line (encoded as string)
 *      - function name
 *      - os_error_code
 *      - operation description
 *   3. Logs to OutputDebugString for debugger attachment
 * ============================================================================ */

/* Structured error packet builder.
 * Sends CMD_CTRL_ERR packet to broker with full context.
 * Captures GetLastError() as first action — must be called immediately
 * after a failed Windows API call. */
static void et_report_error(
    const char *file,
    int         line,
    const char *func,
    const char *operation,
    uint32_t    et_pid,
    uint8_t     et_family,
    uint8_t     et_code)
{
    DWORD os_err = GetLastError();  /* capture NOW before anything else */

    /* Human-readable timestamp from time.h for structured error logs.
     * Uses localtime_s (MSVC) or localtime_r (POSIX/GCC) for thread safety. */
    time_t now_epoch = time(NULL);
    struct tm now_local_buf;
    struct tm *now_local = NULL;
#if defined(_MSC_VER)
    /* MSVC: localtime_s returns errno_t; zero = success */
    if (localtime_s(&now_local_buf, &now_epoch) == 0)
        now_local = &now_local_buf;
#elif defined(__GNUC__)
    /* GCC / MinGW: localtime_r is thread-safe POSIX variant */
    now_local = localtime_r(&now_epoch, &now_local_buf);
#else
    /* Fallback: localtime (not thread-safe but functional) */
    {
        struct tm *tmp = localtime(&now_epoch);
        if (tmp) { now_local_buf = *tmp; now_local = &now_local_buf; }
    }
#endif
    char time_str[32];
    if (now_local) {
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", now_local);
    } else {
        _snprintf(time_str, sizeof(time_str), "epoch=%lu", (unsigned long)now_epoch);
    }

    /* Format debug string for debugger output */
    char dbg[512];
    _snprintf(dbg, sizeof(dbg),
        "[ET32 ERROR %s] %s:%d %s() — %s | OS=0x%08X | PID=%u d=%u code=0x%02X\n",
        time_str, file, line, func, operation,
        (unsigned)os_err, (unsigned)et_pid,
        (unsigned)et_family, (unsigned)et_code);
    OutputDebugStringA(dbg);

    if (!g_connected) return;  /* can't report via pipe — debug string is enough */

    /* Build error payload: file:line, func, operation, os_error */
    char location[256];
    _snprintf(location, sizeof(location), "%s:%d", file, line);

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, location);     /* arg 0: file:line */
    et_pack_strA(&ab, func);         /* arg 1: function name */
    et_pack_strA(&ab, operation);    /* arg 2: what was being attempted */
    et_pack_uint32(&ab, os_err);     /* arg 3: Windows error code */
    et_pack_uint32(&ab, et_pid);     /* arg 4: target PID */

    uint32_t send_err = 0;
    et_call(CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_ERR, pbuf, ab.pos, &send_err);
    /* Ignore send_err: if we can't send the error report, we've already
     * written to OutputDebugString, so the information is not lost. */
}

/* Convenience macro: call after every Windows API failure.
 * Captures __FILE__, __LINE__, __FUNCTION__ automatically.
 * Usage: ET_REPORT_ERROR("VirtualAllocEx for hook stubs", pid, d1, 0x01);
 */
#define ET_REPORT_ERROR(op, pid, family, code) \
    et_report_error(__FILE__, __LINE__, __FUNCTION__, (op), (pid), (family), (code))

/* Macro for Windows API calls that return BOOL/HANDLE.
 * Usage: ET_CHECK_WIN32(VirtualAllocEx(...), "VirtualAllocEx", pid, 1, 0x01) */
#define ET_CHECK_WIN32(result, op, pid, family, code)  do { \
    if (!(result)) {                                         \
        ET_REPORT_ERROR((op), (pid), (family), (code));      \
    }                                                        \
} while(0)

/* Macro: check and return FALSE on failure */
#define ET_REQUIRE_WIN32(result, op, pid, family, code) do { \
    if (!(result)) {                                          \
        ET_REPORT_ERROR((op), (pid), (family), (code));       \
        return FALSE;                                         \
    }                                                         \
} while(0)

/* Macro: check and return NULL on failure */
#define ET_REQUIRE_PTR(ptr, op, pid, family, code)   do { \
    if (!(ptr)) {                                          \
        ET_REPORT_ERROR((op), (pid), (family), (code));    \
        return NULL;                                       \
    }                                                      \
} while(0)

/* Macro: check and return 0 on failure */
#define ET_REQUIRE_NONZERO(val, op, pid, family, code) do { \
    if (!(val)) {                                           \
        ET_REPORT_ERROR((op), (pid), (family), (code));     \
        return 0;                                           \
    }                                                       \
} while(0)

/* ============================================================================
 * CMD FAMILY / CMD CODE — exact mirror of et_math.py CmdFamily / CmdCode
 * ============================================================================ */

/* Families (d = lattice position) */
#define CMD_FAMILY_MEMORY_BASIC   1
#define CMD_FAMILY_MEMORY_MAP     2
#define CMD_FAMILY_THREAD_OPS     3
#define CMD_FAMILY_DLL_OPS        4
#define CMD_FAMILY_PROCESS_OPS    5
#define CMD_FAMILY_REGISTRY_OPS   6
#define CMD_FAMILY_GRAPHICS_OPS   7
#define CMD_FAMILY_FILE_OPS       8
#define CMD_FAMILY_SYNC_OPS       9
#define CMD_FAMILY_NET_OPS        10
#define CMD_FAMILY_PYTHON_OPS     11
#define CMD_FAMILY_COMPOUND_OPS   12

/* Command codes */
#define CMD_VIRT_ALLOC          0x01
#define CMD_VIRT_FREE           0x02
#define CMD_VIRT_PROTECT        0x03
#define CMD_VIRT_QUERY          0x04
#define CMD_HEAP_ALLOC          0x05
#define CMD_HEAP_FREE           0x06
#define CMD_READ_MEM            0x07
#define CMD_WRITE_MEM           0x08
#define CMD_FILE_MAP_CREATE     0x11
#define CMD_FILE_MAP_VIEW       0x12
#define CMD_FILE_MAP_CLOSE      0x13
#define CMD_FILE_MAP_FLUSH      0x14
#define CMD_THREAD_CREATE       0x21
#define CMD_THREAD_SUSPEND      0x22
#define CMD_THREAD_RESUME       0x23
#define CMD_THREAD_TERMINATE    0x24
#define CMD_THREAD_CONTEXT      0x25
#define CMD_DLL_LOAD            0x31
#define CMD_DLL_FREE            0x32
#define CMD_DLL_GETPROC         0x33
#define CMD_DLL_CALL            0x34
#define CMD_DLL_LIST            0x35
#define CMD_PROC_CREATE         0x41
#define CMD_PROC_OPEN           0x42
#define CMD_PROC_INJECT         0x43
#define CMD_PROC_INFO           0x44
#define CMD_REG_OPEN64          0x51
#define CMD_REG_QUERY64         0x52
#define CMD_REG_SET64           0x53
#define CMD_REG_ENUM64          0x54
#define CMD_GPU_ALLOC_VRAM      0x61
#define CMD_GPU_FREE_VRAM       0x62
#define CMD_GPU_MAP_VRAM        0x63
#define CMD_GPU_SUBMIT          0x64
#define CMD_GPU_QUERY_INFO      0x65
#define CMD_GPU_ENUM_ADAPTERS   0x66
#define CMD_GPU_CREATE_DEVICE   0x67
#define CMD_GPU_HEAVEN_CALL     0x68
#define CMD_FILE_OPEN_LARGE     0x71
#define CMD_FILE_MAP_LARGE      0x72
#define CMD_FILE_SEEK_LARGE     0x73
#define CMD_FILE_READ_LARGE     0x74
#define CMD_FILE_WRITE_LARGE    0x75
#define CMD_SYNC_CREATE_EVENT   0x81
#define CMD_SYNC_SIGNAL         0x82
#define CMD_SYNC_WAIT           0x83
#define CMD_SYNC_MUTEX          0x84
#define CMD_NET_SOCKET64        0x91
#define CMD_NET_BIND64          0x92
#define CMD_NET_SEND64          0x93
#define CMD_NET_RECV64          0x94
#define CMD_PY_INIT             0xA1
#define CMD_PY_EXEC             0xA2
#define CMD_PY_IMPORT           0xA3
#define CMD_PY_CALL             0xA4
#define CMD_PY_GETOBJ           0xA5
#define CMD_PY_EVAL             0xA6
#define CMD_PY_SETOBJ           0xA7
#define CMD_PY_SYSPATH          0xA8
#define CMD_COMPOUND_BATCH      0xB1
#define CMD_COMPOUND_ATOMIC     0xB2
#define CMD_COMPOUND_ROLLBACK   0xB3
#define CMD_CTRL_PING           0xF0
#define CMD_CTRL_HANDSHAKE      0xF1
#define CMD_CTRL_SHUTDOWN       0xF2
#define CMD_CTRL_STATUS         0xF3
#define CMD_CTRL_ACK            0xFE
#define CMD_CTRL_ERR            0xFF

/* Packet flags */
#define PKT_FLAG_REQUEST        0x0001
#define PKT_FLAG_RESPONSE       0x0002
#define PKT_FLAG_ERROR          0x0004
#define PKT_FLAG_COMPRESSED     0x0008
#define PKT_FLAG_EXTENDED       0x0010

/* Arg type tags (d=lattice position) */
#define ARG_TAG_UINT32          0x01
#define ARG_TAG_UINT64          0x02
#define ARG_TAG_INT32           0x03
#define ARG_TAG_INT64           0x04
#define ARG_TAG_FLOAT64         0x05
#define ARG_TAG_BYTES           0x06
#define ARG_TAG_STR_UTF8        0x07
#define ARG_TAG_NULL            0x0C

/* ============================================================================
 * BLAKE2b — minimal self-contained implementation
 * Produces output identical to Python's hashlib.blake2b(data, digest_size=4)
 * Based on RFC 7693 reference code (public domain).
 * ============================================================================ */

typedef uint64_t b2b_word;

static const b2b_word BLAKE2B_IV[8] = {
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

static const uint8_t SIGMA[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define B2B_G(a, b, c, d, x, y) do { \
    v[a] += v[b] + (x); v[d] = ROTR64(v[d] ^ v[a], 32); \
    v[c] += v[d];        v[b] = ROTR64(v[b] ^ v[c], 24); \
    v[a] += v[b] + (y); v[d] = ROTR64(v[d] ^ v[a], 16); \
    v[c] += v[d];        v[b] = ROTR64(v[b] ^ v[c], 63); \
} while (0)

typedef struct {
    b2b_word h[8];
    b2b_word t[2];
    uint8_t  buf[128];
    size_t   buflen;
    size_t   outlen;
} blake2b_state;

static void blake2b_compress(blake2b_state *S, const uint8_t *block, int is_last)
{
    b2b_word v[16], m[16];
    int i;

    for (i = 0; i < 8; i++) v[i] = S->h[i];
    v[8]  = BLAKE2B_IV[0];
    v[9]  = BLAKE2B_IV[1];
    v[10] = BLAKE2B_IV[2];
    v[11] = BLAKE2B_IV[3];
    v[12] = BLAKE2B_IV[4] ^ S->t[0];
    v[13] = BLAKE2B_IV[5] ^ S->t[1];
    v[14] = is_last ? ~BLAKE2B_IV[6] : BLAKE2B_IV[6];
    v[15] = BLAKE2B_IV[7];

    for (i = 0; i < 16; i++) {
        const uint8_t *p = block + i * 8;
        m[i] = (b2b_word)p[0]       | ((b2b_word)p[1] << 8)
             | ((b2b_word)p[2] << 16) | ((b2b_word)p[3] << 24)
             | ((b2b_word)p[4] << 32) | ((b2b_word)p[5] << 40)
             | ((b2b_word)p[6] << 48) | ((b2b_word)p[7] << 56);
    }

    for (i = 0; i < 12; i++) {
        const uint8_t *s = SIGMA[i];
        B2B_G(0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        B2B_G(1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        B2B_G(2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        B2B_G(3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        B2B_G(0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        B2B_G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        B2B_G(2, 7,  8, 13, m[s[12]], m[s[13]]);
        B2B_G(3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    for (i = 0; i < 8; i++) S->h[i] ^= v[i] ^ v[i + 8];
}

static void blake2b_init4(blake2b_state *S)
{
    /* Parameter block for digest_size=4, key=0, fanout=1, depth=1, all others 0.
     * param_word0 as little-endian uint64 = 0x0000000001010004ULL
     * h[0] = IV[0] XOR param_word0 */
    int i;
    memset(S, 0, sizeof(*S));
    for (i = 0; i < 8; i++) S->h[i] = BLAKE2B_IV[i];
    S->h[0] ^= 0x0000000001010004ULL;
    S->outlen = 4;
}

static void blake2b_update(blake2b_state *S, const uint8_t *in, size_t inlen)
{
    size_t left, fill;
    while (inlen > 0) {
        left = S->buflen;
        fill = 128 - left;
        if (inlen > fill) {
            memcpy(S->buf + left, in, fill);
            S->t[0] += 128;
            if (S->t[0] < 128) S->t[1]++;
            blake2b_compress(S, S->buf, 0);
            S->buflen = 0;
            in    += fill;
            inlen -= fill;
        } else {
            memcpy(S->buf + left, in, inlen);
            S->buflen += inlen;
            inlen = 0;
        }
    }
}

static void blake2b_final4(blake2b_state *S, uint8_t out[4])
{
    uint8_t tmp[128];
    memset(tmp, 0, 128);
    memcpy(tmp, S->buf, S->buflen);
    S->t[0] += (uint64_t)S->buflen;
    if (S->t[0] < (uint64_t)S->buflen) S->t[1]++;
    blake2b_compress(S, tmp, 1);
    /* Output: first 4 bytes of h[0] in little-endian */
    out[0] = (uint8_t)(S->h[0]);
    out[1] = (uint8_t)(S->h[0] >> 8);
    out[2] = (uint8_t)(S->h[0] >> 16);
    out[3] = (uint8_t)(S->h[0] >> 24);
}

/* Compute 4-byte BLAKE2b digest over data[0...len-1] into out[0..3]. */
static void et_blake2b4(const uint8_t *data, size_t len, uint8_t out[4])
{
    blake2b_state S;
    blake2b_init4(&S);
    blake2b_update(&S, data, len);
    blake2b_final4(&S, out);
}

/* ============================================================================
 * ARG PACKING — wire format matching pack_args() in et_math.py
 * ============================================================================ */

/* Buffer cursor for writing packed args
 * et_arg_buf type — forward-declared in FORWARD DECLARATIONS section. */
#ifdef ET_ARG_BUF_DEFINED
/* Compile-time assertion: et_arg_buf must be at least 16 bytes
 * (pointer + 2×size_t + int = 16 on 32-bit, larger on 64-bit). */
typedef char et_arg_buf_verify[(sizeof(et_arg_buf) >= 16) ? 1 : -1];
#endif

static void et_argbuf_init(et_arg_buf *b, uint8_t *mem, size_t cap)
{
    b->base     = mem;
    b->capacity = cap;
    b->pos      = 0;
    b->count    = 0;
}

static int et_argbuf_room(et_arg_buf *b, size_t need)
{
    return (b->pos + need) <= b->capacity;
}

static void et_pack_null(et_arg_buf *b)
{
    if (!et_argbuf_room(b, 2)) return;
    b->base[b->pos++] = ARG_TAG_NULL;
    b->base[b->pos++] = 0;
    b->count++;
}

static void et_pack_uint32(et_arg_buf *b, uint32_t v)
{
    if (!et_argbuf_room(b, 6)) return;
    b->base[b->pos++] = ARG_TAG_UINT32;
    b->base[b->pos++] = 4;
    b->base[b->pos++] = (uint8_t)(v);
    b->base[b->pos++] = (uint8_t)(v >> 8);
    b->base[b->pos++] = (uint8_t)(v >> 16);
    b->base[b->pos++] = (uint8_t)(v >> 24);
    b->count++;
}

static void et_pack_uint64(et_arg_buf *b, uint64_t v)
{
    if (!et_argbuf_room(b, 10)) return;
    b->base[b->pos++] = ARG_TAG_UINT64;
    b->base[b->pos++] = 8;
    b->base[b->pos++] = (uint8_t)(v);
    b->base[b->pos++] = (uint8_t)(v >> 8);
    b->base[b->pos++] = (uint8_t)(v >> 16);
    b->base[b->pos++] = (uint8_t)(v >> 24);
    b->base[b->pos++] = (uint8_t)(v >> 32);
    b->base[b->pos++] = (uint8_t)(v >> 40);
    b->base[b->pos++] = (uint8_t)(v >> 48);
    b->base[b->pos++] = (uint8_t)(v >> 56);
    b->count++;
}

static void et_pack_int32(et_arg_buf *b, int32_t v)
{
    if (!et_argbuf_room(b, 6)) return;
    b->base[b->pos++] = ARG_TAG_INT32;
    b->base[b->pos++] = 4;
    uint32_t u = (uint32_t)v;
    b->base[b->pos++] = (uint8_t)(u);
    b->base[b->pos++] = (uint8_t)(u >> 8);
    b->base[b->pos++] = (uint8_t)(u >> 16);
    b->base[b->pos++] = (uint8_t)(u >> 24);
    b->count++;
}

static void et_pack_bytes(et_arg_buf *b, const uint8_t *data, uint32_t len)
{
    if (!et_argbuf_room(b, 6 + (size_t)len)) return;
    b->base[b->pos++] = ARG_TAG_BYTES;
    b->base[b->pos++] = 0;
    b->base[b->pos++] = (uint8_t)(len);
    b->base[b->pos++] = (uint8_t)(len >> 8);
    b->base[b->pos++] = (uint8_t)(len >> 16);
    b->base[b->pos++] = (uint8_t)(len >> 24);
    memcpy(b->base + b->pos, data, len);
    b->pos += len;
    b->count++;
}

static void et_pack_strA(et_arg_buf *b, const char *s)
{
    uint32_t len = s ? (uint32_t)strlen(s) : 0;
    if (!et_argbuf_room(b, 6 + len)) return;
    b->base[b->pos++] = ARG_TAG_STR_UTF8;
    b->base[b->pos++] = 0;
    b->base[b->pos++] = (uint8_t)(len);
    b->base[b->pos++] = (uint8_t)(len >> 8);
    b->base[b->pos++] = (uint8_t)(len >> 16);
    b->base[b->pos++] = (uint8_t)(len >> 24);
    if (len) memcpy(b->base + b->pos, s, len);
    b->pos += len;
    b->count++;
}

/* Pack a wide string as UTF-8 */
static void et_pack_strW(et_arg_buf *b, const wchar_t *ws)
{
    if (!ws) { et_pack_null(b); return; }
    int n = WideCharToMultiByte(CP_UTF8, 0, ws, -1, NULL, 0, NULL, NULL);
    if (n <= 1) { et_pack_null(b); return; }
    int slen = n - 1; /* exclude NUL */
    if (!et_argbuf_room(b, 6 + (size_t)slen)) return;
    b->base[b->pos++] = ARG_TAG_STR_UTF8;
    b->base[b->pos++] = 0;
    uint32_t u = (uint32_t)slen;
    b->base[b->pos++] = (uint8_t)(u);
    b->base[b->pos++] = (uint8_t)(u >> 8);
    b->base[b->pos++] = (uint8_t)(u >> 16);
    b->base[b->pos++] = (uint8_t)(u >> 24);
    WideCharToMultiByte(CP_UTF8, 0, ws, -1, (char *)(b->base + b->pos), slen, NULL, NULL);
    b->pos += (size_t)slen;
    b->count++;
}

/* ============================================================================
 * ARG UNPACKING — inverse of unpack_args() in et_math.py
 * ============================================================================ */

typedef struct {
    const uint8_t *base;
    size_t         len;
    size_t         pos;
} et_arg_reader;

static void et_argreader_init(et_arg_reader *r, const uint8_t *data, size_t len)
{
    r->base = data;
    r->len  = len;
    r->pos  = 0;
}

/* Returns 0 on end / error. Sets *tag and advances. */
static int et_argreader_next_uint32(et_arg_reader *r, uint32_t *out)
{
    while (r->pos + 2 <= r->len) {
        uint8_t tag  = r->base[r->pos];
        uint8_t hint = r->base[r->pos + 1];
        (void)hint;
        r->pos += 2;
        if (tag == ARG_TAG_UINT32 && r->pos + 4 <= r->len) {
            uint32_t v =  (uint32_t)r->base[r->pos]
                       | ((uint32_t)r->base[r->pos+1] << 8)
                       | ((uint32_t)r->base[r->pos+2] << 16)
                       | ((uint32_t)r->base[r->pos+3] << 24);
            r->pos += 4;
            *out = v;
            return 1;
        } else if (tag == ARG_TAG_UINT64 && r->pos + 8 <= r->len) {
            uint64_t v =  (uint64_t)r->base[r->pos]
                       | ((uint64_t)r->base[r->pos+1] << 8)
                       | ((uint64_t)r->base[r->pos+2] << 16)
                       | ((uint64_t)r->base[r->pos+3] << 24)
                       | ((uint64_t)r->base[r->pos+4] << 32)
                       | ((uint64_t)r->base[r->pos+5] << 40)
                       | ((uint64_t)r->base[r->pos+6] << 48)
                       | ((uint64_t)r->base[r->pos+7] << 56);
            r->pos += 8;
            *out = (uint32_t)(v & 0xFFFFFFFFULL);
            return 1;
        } else if (tag == ARG_TAG_INT32 && r->pos + 4 <= r->len) {
            uint32_t v =  (uint32_t)r->base[r->pos]
                       | ((uint32_t)r->base[r->pos+1] << 8)
                       | ((uint32_t)r->base[r->pos+2] << 16)
                       | ((uint32_t)r->base[r->pos+3] << 24);
            r->pos += 4;
            *out = v;
            return 1;
        } else if (tag == ARG_TAG_NULL) {
            *out = 0;
            return 1;
        } else {
            /* Skip unknown tag: consume based on hint */
            return 0;
        }
    }
    return 0;
}

/*
 * et_argreader_next_uint64 — reads a uint64 arg from the response buffer.
 * Also accepts uint32 tags (zero-extends) and NULL tags (returns 0).
 * Returns 1 on success, 0 on end / unrecognized tag.
 *
 * ET derivation: uint64 args occupy 10 bytes (tag 1 + hint 1 + value 8).
 * Lattice positions d>4 require 64-bit values — this reader spans the full
 * P-range (all physical addresses, file sizes, timestamps, memory sizes).
 */
static int et_argreader_next_uint64(et_arg_reader *r, uint64_t *out)
{
    while (r->pos + 2 <= r->len) {
        uint8_t tag  = r->base[r->pos];
        uint8_t hint = r->base[r->pos + 1];
        (void)hint;
        r->pos += 2;
        if (tag == ARG_TAG_UINT64 && r->pos + 8 <= r->len) {
            uint64_t v = (uint64_t)r->base[r->pos]
                       | ((uint64_t)r->base[r->pos+1] << 8)
                       | ((uint64_t)r->base[r->pos+2] << 16)
                       | ((uint64_t)r->base[r->pos+3] << 24)
                       | ((uint64_t)r->base[r->pos+4] << 32)
                       | ((uint64_t)r->base[r->pos+5] << 40)
                       | ((uint64_t)r->base[r->pos+6] << 48)
                       | ((uint64_t)r->base[r->pos+7] << 56);
            r->pos += 8;
            *out = v;
            return 1;
        } else if (tag == ARG_TAG_UINT32 && r->pos + 4 <= r->len) {
            uint32_t v = (uint32_t)r->base[r->pos]
                       | ((uint32_t)r->base[r->pos+1] << 8)
                       | ((uint32_t)r->base[r->pos+2] << 16)
                       | ((uint32_t)r->base[r->pos+3] << 24);
            r->pos += 4;
            *out = (uint64_t)v;
            return 1;
        } else if (tag == ARG_TAG_NULL) {
            *out = 0;
            return 1;
        } else {
            return 0;
        }
    }
    return 0;
}

/*
 * et_argreader_next_strW — reads a UTF-8 string arg and converts to wide.
 * dst: output wchar_t buffer.  dst_cch: capacity in wchar_t chars (incl. NUL).
 * Returns 1 on success (including empty/null string), 0 on end / bad tag.
 *
 * ET derivation: string args use ARG_TAG_STR_UTF8 (0x07): tag(1)+hint(1)+
 * len_le32(4)+utf8_bytes(N).  Wide conversion uses MultiByteToWideChar so
 * filenames with any Unicode code-point round-trip losslessly.
 */
static int et_argreader_next_strW(et_arg_reader *r, wchar_t *dst, int dst_cch)
{
    if (!dst || dst_cch <= 0) return 0;
    dst[0] = 0;  /* guarantee NUL-terminated on any exit path */
    while (r->pos + 2 <= r->len) {
        uint8_t tag  = r->base[r->pos];
        uint8_t hint = r->base[r->pos + 1];
        (void)hint;
        r->pos += 2;
        if ((tag == ARG_TAG_STR_UTF8 || tag == ARG_TAG_BYTES) && r->pos + 4 <= r->len) {
            uint32_t slen = (uint32_t)r->base[r->pos]
                          | ((uint32_t)r->base[r->pos+1] << 8)
                          | ((uint32_t)r->base[r->pos+2] << 16)
                          | ((uint32_t)r->base[r->pos+3] << 24);
            r->pos += 4;
            if (slen > 0 && r->pos + slen <= r->len) {
                int cch = MultiByteToWideChar(CP_UTF8, 0,
                    (const char *)(r->base + r->pos), (int)slen,
                    dst, dst_cch - 1);
                if (cch > 0)
                    dst[cch] = 0;  /* NUL-terminate after converted chars */
            }
            r->pos += slen;
            return 1;
        } else if (tag == ARG_TAG_NULL) {
            return 1;  /* null string — dst already NUL-zeroed above */
        } else {
            return 0;
        }
    }
    return 0;
}

/*
 * et_argreader_skip — skip one arg of any type without reading its value.
 * Used to advance past an arg that was already consumed by et_call.
 * Returns 1 on success, 0 on end / unrecognized tag.
 */
static int et_argreader_skip(et_arg_reader *r)
{
    if (r->pos + 2 > r->len) return 0;
    uint8_t tag  = r->base[r->pos];
    uint8_t hint = r->base[r->pos + 1];
    (void)hint;
    r->pos += 2;
    switch (tag) {
    case ARG_TAG_NULL:
        return 1;
    case ARG_TAG_UINT32:
    case ARG_TAG_INT32:
        if (r->pos + 4 > r->len) return 0;
        r->pos += 4;
        return 1;
    case ARG_TAG_UINT64:
    case ARG_TAG_INT64:
    case ARG_TAG_FLOAT64:
        if (r->pos + 8 > r->len) return 0;
        r->pos += 8;
        return 1;
    case ARG_TAG_BYTES:
    case ARG_TAG_STR_UTF8: {
        if (r->pos + 4 > r->len) return 0;
        uint32_t slen = (uint32_t)r->base[r->pos]
                      | ((uint32_t)r->base[r->pos+1] << 8)
                      | ((uint32_t)r->base[r->pos+2] << 16)
                      | ((uint32_t)r->base[r->pos+3] << 24);
        r->pos += 4;
        if (r->pos + slen > r->len) return 0;
        r->pos += slen;
        return 1;
    }
    default:
        return 0;
    }
}

/* ============================================================================
 * ET PACKET SERIALISATION
 * Header layout (48 bytes little-endian):
 *   P-section (16):  uint32 source_pid | uint32 dest_pid | uint64 space_token
 *   D-section (16):  uint8 cmd_family | uint8 cmd_code | uint16 flags |
 *                    uint32 arg_count | int64 payload_len
 *   T-section (16):  uint32 sequence | uint64 timestamp | uint32 checksum
 * Checksum = blake2b-4 over (P-section + D-section + payload).
 * ============================================================================ */

/* Monotonic microsecond timestamp (Windows QPC) */
static uint64_t et_now_us(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (uint64_t)((cnt.QuadPart * 1000000LL) / freq.QuadPart);
}

static volatile LONG g_sequence = 0;

/*
 * Build a complete ETPacket into out_buf (must be >= ET_IPC_BUFFER_SIZE).
 * payload and payload_len are the already-packed arg bytes.
 * Returns total packet byte count, or 0 on error.
 */
static size_t et_packet_build(
    uint8_t  *out_buf,
    size_t    out_cap,
    uint32_t  source_pid,
    uint32_t  dest_pid,
    uint64_t  space_token,
    uint8_t   cmd_family,
    uint8_t   cmd_code,
    uint16_t  flags,
    uint32_t  arg_count,
    const uint8_t *payload,
    size_t    payload_len)
{
    if (out_cap < ET_PDT_HEADER_SIZE + payload_len) return 0;

    uint8_t p_sec[16], d_sec[16], t_sec[16];

    /* P-section */
    memset(p_sec, 0, 16);
    p_sec[0]  = (uint8_t)(source_pid);
    p_sec[1]  = (uint8_t)(source_pid >> 8);
    p_sec[2]  = (uint8_t)(source_pid >> 16);
    p_sec[3]  = (uint8_t)(source_pid >> 24);
    p_sec[4]  = (uint8_t)(dest_pid);
    p_sec[5]  = (uint8_t)(dest_pid >> 8);
    p_sec[6]  = (uint8_t)(dest_pid >> 16);
    p_sec[7]  = (uint8_t)(dest_pid >> 24);
    /* space_token: bytes 8..15 */
    p_sec[8]  = (uint8_t)(space_token);
    p_sec[9]  = (uint8_t)(space_token >> 8);
    p_sec[10] = (uint8_t)(space_token >> 16);
    p_sec[11] = (uint8_t)(space_token >> 24);
    p_sec[12] = (uint8_t)(space_token >> 32);
    p_sec[13] = (uint8_t)(space_token >> 40);
    p_sec[14] = (uint8_t)(space_token >> 48);
    p_sec[15] = (uint8_t)(space_token >> 56);

    /* D-section */
    memset(d_sec, 0, 16);
    d_sec[0] = cmd_family;
    d_sec[1] = cmd_code;
    d_sec[2] = (uint8_t)(flags);
    d_sec[3] = (uint8_t)(flags >> 8);
    d_sec[4] = (uint8_t)(arg_count);
    d_sec[5] = (uint8_t)(arg_count >> 8);
    d_sec[6] = (uint8_t)(arg_count >> 16);
    d_sec[7] = (uint8_t)(arg_count >> 24);
    /* payload_len as int64 LE at bytes 8..15 */
    uint64_t plen64 = (uint64_t)payload_len;
    d_sec[8]  = (uint8_t)(plen64);
    d_sec[9]  = (uint8_t)(plen64 >> 8);
    d_sec[10] = (uint8_t)(plen64 >> 16);
    d_sec[11] = (uint8_t)(plen64 >> 24);
    d_sec[12] = (uint8_t)(plen64 >> 32);
    d_sec[13] = (uint8_t)(plen64 >> 40);
    d_sec[14] = (uint8_t)(plen64 >> 48);
    d_sec[15] = (uint8_t)(plen64 >> 56);

    /* Compute BLAKE2b-4 checksum over P + D + payload */
    uint8_t chk_data[ET_IPC_BUFFER_SIZE];
    size_t chk_len = 32 + payload_len;
    if (chk_len > ET_IPC_BUFFER_SIZE) return 0;
    memcpy(chk_data,      p_sec,   16);
    memcpy(chk_data + 16, d_sec,   16);
    if (payload_len) memcpy(chk_data + 32, payload, payload_len);

    uint8_t crc4[4];
    et_blake2b4(chk_data, chk_len, crc4);

    /* T-section */
    LONG seq = InterlockedIncrement(&g_sequence);
    DWORD seq_u = (DWORD)seq;  /* unsigned for byte serialization */
    uint64_t ts = et_now_us();
    memset(t_sec, 0, 16);
    t_sec[0]  = (uint8_t)(seq_u);
    t_sec[1]  = (uint8_t)(seq_u >> 8);
    t_sec[2]  = (uint8_t)(seq_u >> 16);
    t_sec[3]  = (uint8_t)(seq_u >> 24);
    t_sec[4]  = (uint8_t)(ts);
    t_sec[5]  = (uint8_t)(ts >> 8);
    t_sec[6]  = (uint8_t)(ts >> 16);
    t_sec[7]  = (uint8_t)(ts >> 24);
    t_sec[8]  = (uint8_t)(ts >> 32);
    t_sec[9]  = (uint8_t)(ts >> 40);
    t_sec[10] = (uint8_t)(ts >> 48);
    t_sec[11] = (uint8_t)(ts >> 56);
    t_sec[12] = crc4[0];
    t_sec[13] = crc4[1];
    t_sec[14] = crc4[2];
    t_sec[15] = crc4[3];

    memcpy(out_buf,       p_sec,   16);
    memcpy(out_buf + 16,  d_sec,   16);
    memcpy(out_buf + 32,  t_sec,   16);
    if (payload_len) memcpy(out_buf + 48, payload, payload_len);

    return ET_PDT_HEADER_SIZE + payload_len;
}

/* ============================================================================
 * GLOBAL CONNECTION STATE
 * ============================================================================ */

static HANDLE       g_pipe        = INVALID_HANDLE_VALUE;
static CRITICAL_SECTION g_pipe_cs;
static DWORD        g_target_pid  = 0;   /* our own PID */
static DWORD        g_broker_pid  = 0;   /* 64-bit broker PID (passed to ET32_Init) */
/* g_connected, g_initialized — forward-declared in FORWARD DECLARATIONS section */
#ifdef ET_CONN_GLOBALS_DEFINED
/* Compile-time assertion: g_connected must be BOOL-sized (4 bytes on Windows). */
typedef char et_conn_globals_verify[(sizeof(g_connected) == sizeof(BOOL)) ? 1 : -1];
#endif

/* IPC send/receive buffers — one per process (thread-serialized via CS) */
static uint8_t g_send_buf[ET_IPC_BUFFER_SIZE];
static uint8_t g_recv_buf[ET_IPC_BUFFER_SIZE];

/*
 * et_recv_argreader — initialize a reader over g_recv_buf's current payload,
 * skipping the first skip_n args (already consumed by et_call's return path).
 *
 * ET derivation: g_recv_buf[24..31] encodes payload_len (int64 LE) in the
 * D-section of the last received ETPacket.  Payload starts at offset
 * ET_PDT_HEADER_SIZE (48).  et_call already extracted arg[0] via
 * et_argreader_next_uint32; we re-init a fresh reader and skip skip_n args
 * so callers can read the remaining args without duplicating header parsing.
 *
 * Returns 1 if the reader is successfully positioned at arg[skip_n].
 * Returns 0 if the buffer contains no payload or skipping overruns.
 */
static int et_recv_argreader(et_arg_reader *r, int skip_n)
{
    /* Re-read payload_len from D-section bytes 8-15 (absolute offset 24-31). */
    uint64_t plen =
        (uint64_t)g_recv_buf[24]
      | ((uint64_t)g_recv_buf[25] <<  8)
      | ((uint64_t)g_recv_buf[26] << 16)
      | ((uint64_t)g_recv_buf[27] << 24)
      | ((uint64_t)g_recv_buf[28] << 32)
      | ((uint64_t)g_recv_buf[29] << 40)
      | ((uint64_t)g_recv_buf[30] << 48)
      | ((uint64_t)g_recv_buf[31] << 56);

    /* Clamp to buffer capacity */
    if (plen > (uint64_t)(ET_IPC_BUFFER_SIZE - ET_PDT_HEADER_SIZE))
        plen = (uint64_t)(ET_IPC_BUFFER_SIZE - ET_PDT_HEADER_SIZE);

    if (plen == 0) return 0;

    et_argreader_init(r, g_recv_buf + ET_PDT_HEADER_SIZE, (size_t)plen);

    /* Skip the first skip_n args (already handled by et_call / caller). */
    int i;
    for (i = 0; i < skip_n; i++) {
        if (!et_argreader_skip(r)) return 0;
    }
    return 1;
}

/* ============================================================================
 * PIPE CONNECT / DISCONNECT
 * ============================================================================ */

static BOOL et_pipe_connect(void)
{
    char pipe_name[ET_PIPE_NAME_MAX];
    _snprintf(pipe_name, ET_PIPE_NAME_MAX, ET_PIPE_NAME_FMT, (unsigned long)g_target_pid);

    int retry;
    for (retry = 0; retry < ET_RETRY_COUNT; retry++) {
        /* Wait up to CONN_TIMEOUT_MS / RETRY_COUNT = 125 ms per attempt */
        WaitNamedPipeA(pipe_name, ET_CONN_TIMEOUT_MS / ET_RETRY_COUNT);

        HANDLE h = CreateFileA(
            pipe_name,
            GENERIC_READ | GENERIC_WRITE,
            0, NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);

        if (h != INVALID_HANDLE_VALUE) {
            /* Set message-mode reading to match broker's write mode */
            DWORD mode = PIPE_READMODE_MESSAGE;
            SetNamedPipeHandleState(h, &mode, NULL, NULL);
            g_pipe      = h;
            g_connected = TRUE;
            return TRUE;
        }

        if (GetLastError() != ERROR_PIPE_BUSY)
            Sleep(ET_CONN_TIMEOUT_MS / ET_RETRY_COUNT);
    }
    ET_REPORT_ERROR("Pipe connect exhausted all retries", g_target_pid,
                    CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_HANDSHAKE);
    return FALSE;
}

static void et_pipe_disconnect(void)
{
    if (g_pipe != INVALID_HANDLE_VALUE) {
        CloseHandle(g_pipe);
        g_pipe      = INVALID_HANDLE_VALUE;
        g_connected = FALSE;
    }
}

/* ============================================================================
 * REQUEST / RESPONSE CYCLE
 * Sends one ETPacket, reads one response ETPacket.
 * Returns first uint32 result arg, or 0 on error. Sets *error_out if provided.
 * ============================================================================ */

static uint32_t et_call(
    uint8_t   cmd_family,
    uint8_t   cmd_code,
    const uint8_t *payload,
    size_t    payload_len,
    uint32_t *error_out)
{
    if (!g_connected) {
        if (error_out) *error_out = ERROR_NOT_CONNECTED;
        return 0;
    }

    /* Count args in payload (rough: scan tags) */
    uint32_t arg_count = 0;
    {
        size_t i = 0;
        while (i + 1 < payload_len) {
            uint8_t tag  = payload[i];
            uint8_t hint = payload[i+1];
            i += 2;
            if (tag == ARG_TAG_NULL) {
                /* Zero-length tag: no data bytes follow */
                arg_count++;
            } else if (tag == ARG_TAG_UINT32 || tag == ARG_TAG_INT32) {
                /* 4-byte fixed-width tags (d≤4 lattice positions) */
                i += 4; arg_count++;
            } else if (tag == ARG_TAG_UINT64 || tag == ARG_TAG_INT64
                    || tag == ARG_TAG_FLOAT64) {
                /* 8-byte fixed-width tags (d>4 lattice positions) */
                i += 8; arg_count++;
            } else if (tag == ARG_TAG_BYTES || tag == ARG_TAG_STR_UTF8) {
                if (i + 4 > payload_len) break;
                uint32_t slen = (uint32_t)payload[i]
                              | ((uint32_t)payload[i+1] << 8)
                              | ((uint32_t)payload[i+2] << 16)
                              | ((uint32_t)payload[i+3] << 24);
                i += 4 + slen;
                arg_count++;
            } else { break; }
            (void)hint;
        }
    }

    EnterCriticalSection(&g_pipe_cs);

    size_t pkt_len = et_packet_build(
        g_send_buf, ET_IPC_BUFFER_SIZE,
        g_target_pid, g_broker_pid, 0,
        cmd_family, cmd_code,
        PKT_FLAG_REQUEST,
        arg_count,
        payload, payload_len);

    if (!pkt_len) {
        LeaveCriticalSection(&g_pipe_cs);
        if (error_out) *error_out = ERROR_INSUFFICIENT_BUFFER;
        return 0;
    }

    DWORD written = 0, read_bytes = 0;
    BOOL ok = WriteFile(g_pipe, g_send_buf, (DWORD)pkt_len, &written, NULL);
    if (!ok || written != (DWORD)pkt_len) {
        ET_REPORT_ERROR("WriteFile to broker pipe", g_target_pid,
                        cmd_family, cmd_code);
        et_pipe_disconnect();
        LeaveCriticalSection(&g_pipe_cs);
        if (error_out) *error_out = GetLastError();
        return 0;
    }

    ok = ReadFile(g_pipe, g_recv_buf, ET_IPC_BUFFER_SIZE, &read_bytes, NULL);
    LeaveCriticalSection(&g_pipe_cs);

    if (!ok || read_bytes < ET_PDT_HEADER_SIZE) {
        ET_REPORT_ERROR("ReadFile from broker pipe", g_target_pid,
                        cmd_family, cmd_code);
        if (error_out) *error_out = GetLastError();
        return 0;
    }

    /* Validate response checksum */
    uint8_t resp_p[16], resp_d[16];
    memcpy(resp_p, g_recv_buf,      16);
    memcpy(resp_d, g_recv_buf + 16, 16);

    /* Extract payload_len from D-section bytes 8..15 */
    uint64_t resp_plen =
        (uint64_t)g_recv_buf[24]        |
        ((uint64_t)g_recv_buf[25] << 8)  |
        ((uint64_t)g_recv_buf[26] << 16) |
        ((uint64_t)g_recv_buf[27] << 24) |
        ((uint64_t)g_recv_buf[28] << 32) |
        ((uint64_t)g_recv_buf[29] << 40) |
        ((uint64_t)g_recv_buf[30] << 48) |
        ((uint64_t)g_recv_buf[31] << 56);

    if (resp_plen > (uint64_t)(read_bytes - ET_PDT_HEADER_SIZE))
        resp_plen = (uint64_t)(read_bytes - ET_PDT_HEADER_SIZE);

    /* Verify BLAKE2b checksum */
    uint8_t chk_data[ET_IPC_BUFFER_SIZE];
    size_t chk_len = 32 + (size_t)resp_plen;
    if (chk_len <= ET_IPC_BUFFER_SIZE) {
        memcpy(chk_data,      resp_p, 16);
        memcpy(chk_data + 16, resp_d, 16);
        if (resp_plen) memcpy(chk_data + 32, g_recv_buf + 48, (size_t)resp_plen);

        uint8_t expected_crc[4];
        et_blake2b4(chk_data, chk_len, expected_crc);

        uint8_t got_crc[4] = {
            g_recv_buf[44], g_recv_buf[45], g_recv_buf[46], g_recv_buf[47]
        };
        if (memcmp(expected_crc, got_crc, 4) != 0) {
            /* Incoherent response — V(E) > 0 */
            if (error_out) *error_out = ERROR_CRC;
            return 0;
        }
    }

    /* Extract response flags and cmd_code */
    uint8_t resp_flags_lo = g_recv_buf[18];
    uint8_t resp_flags_hi = g_recv_buf[19];
    uint16_t resp_flags = (uint16_t)resp_flags_lo | ((uint16_t)resp_flags_hi << 8);
    uint8_t  resp_code  = g_recv_buf[17];

    if ((resp_flags & PKT_FLAG_ERROR) || resp_code == CMD_CTRL_ERR) {
        /* Broker returned an error: first arg is error code */
        if (resp_plen >= 6 && g_recv_buf[48] == ARG_TAG_UINT32) {
            uint32_t err =
                (uint32_t)g_recv_buf[50]        |
                ((uint32_t)g_recv_buf[51] << 8)  |
                ((uint32_t)g_recv_buf[52] << 16) |
                ((uint32_t)g_recv_buf[53] << 24);
            if (error_out) *error_out = err;
        } else {
            if (error_out) *error_out = ERROR_FUNCTION_FAILED;
        }
        return 0;
    }

    /* Extract first uint32 result */
    if (error_out) *error_out = 0;
    if (resp_plen >= 6) {
        et_arg_reader ar;
        et_argreader_init(&ar, g_recv_buf + 48, (size_t)resp_plen);
        uint32_t result = 0;
        et_argreader_next_uint32(&ar, &result);
        return result;
    }
    return 0;
}

/* ============================================================================
 * HANDSHAKE
 * ============================================================================ */

static BOOL et_do_handshake(void)
{
    uint8_t hs_buf[16];
    et_arg_buf ab;
    et_argbuf_init(&ab, hs_buf, sizeof(hs_buf));
    et_pack_uint32(&ab, ET_HANDSHAKE_MAGIC);
    et_pack_uint32(&ab, ET_DLL_VERSION);

    uint32_t err = 0;
    uint32_t resp = et_call(CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_HANDSHAKE,
                            hs_buf, ab.pos, &err);
    return (err == 0 && resp == CMD_CTRL_ACK);
}

/* ============================================================================
 * HOOKED API IMPLEMENTATIONS — 12 ET command families
 * ============================================================================ */

/* Helper: is this a bridge handle (proxy for 64-bit address)? */
static BOOL is_bridge_handle(UINT_PTR h)
{
    return (h >= ET_HANDLE_BASE && h <= ET_HANDLE_MAX);
}

/* ---- FAMILY 1: MEMORY_BASIC ---- */

LPVOID WINAPI ET32_VirtualAlloc(
    LPVOID lpAddress, SIZE_T dwSize, DWORD flAllocType, DWORD flProtect)
{
    /* Descriptor Gap: if size > K × 4GB, we must bridge to 64-bit. */
    const uint64_t koide_threshold = (uint64_t)(((double)0xFFFFFFFFULL) * (2.0 / 3.0));
    if ((uint64_t)dwSize < koide_threshold && !is_bridge_handle((UINT_PTR)lpAddress)) {
        /* Within 32-bit capability — call original directly */
        return VirtualAlloc(lpAddress, dwSize, flAllocType, flProtect);
    }

    uint8_t pbuf[40];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, lpAddress ? (uint32_t)(UINT_PTR)lpAddress : 0);
    et_pack_uint64(&ab, (uint64_t)dwSize);
    et_pack_uint32(&ab, flAllocType);
    et_pack_uint32(&ab, flProtect);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_ALLOC,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (LPVOID)(UINT_PTR)result;
}

BOOL WINAPI ET32_VirtualFree(LPVOID lpAddress, SIZE_T dwSize, DWORD dwFreeType)
{
    if (!is_bridge_handle((UINT_PTR)lpAddress)) {
        return VirtualFree(lpAddress, dwSize, dwFreeType);
    }
    uint8_t pbuf[24];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpAddress);
    et_pack_uint64(&ab, (uint64_t)dwSize);
    et_pack_uint32(&ab, dwFreeType);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_FREE,
                               pbuf, ab.pos, &err);
    if (err) SetLastError(err);
    return result ? TRUE : FALSE;
}

BOOL WINAPI ET32_VirtualProtect(
    LPVOID lpAddress, SIZE_T dwSize, DWORD flNewProtect, PDWORD lpflOldProtect)
{
    if (!is_bridge_handle((UINT_PTR)lpAddress)) {
        return VirtualProtect(lpAddress, dwSize, flNewProtect, lpflOldProtect);
    }
    uint8_t pbuf[24];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpAddress);
    et_pack_uint64(&ab, (uint64_t)dwSize);
    et_pack_uint32(&ab, flNewProtect);

    uint32_t err = 0;
    uint32_t old_prot = et_call(CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_PROTECT,
                                 pbuf, ab.pos, &err);
    if (lpflOldProtect) *lpflOldProtect = old_prot;
    if (err) SetLastError(err);
    return (err == 0) ? TRUE : FALSE;
}

SIZE_T WINAPI ET32_VirtualQuery(
    LPCVOID lpAddress, PMEMORY_BASIC_INFORMATION lpBuffer, SIZE_T dwLength)
{
    if (!is_bridge_handle((UINT_PTR)lpAddress)) {
        return VirtualQuery(lpAddress, lpBuffer, dwLength);
    }
    uint8_t pbuf[12];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpAddress);
    et_pack_uint32(&ab, (uint32_t)dwLength);

    uint32_t err = 0;
    et_call(CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_QUERY, pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return 0; }
    return sizeof(MEMORY_BASIC_INFORMATION);
}

/* ---- FAMILY 2: MEMORY_MAP ---- */

HANDLE WINAPI ET32_CreateFileMappingA(
    HANDLE hFile, LPSECURITY_ATTRIBUTES lpAttr,
    DWORD flProtect, DWORD dwMaxSizeHigh, DWORD dwMaxSizeLow, LPCSTR lpName)
{
    uint64_t sz = ((uint64_t)dwMaxSizeHigh << 32) | (uint64_t)dwMaxSizeLow;
    const uint64_t koide_threshold = (uint64_t)(((double)0xFFFFFFFFULL) * (2.0 / 3.0));
    if (sz < koide_threshold) {
        return CreateFileMappingA(hFile, lpAttr, flProtect,
                                  dwMaxSizeHigh, dwMaxSizeLow, lpName);
    }
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hFile);
    et_pack_uint32(&ab, flProtect);
    et_pack_uint64(&ab, sz);
    et_pack_strA(&ab, lpName);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_MEMORY_MAP, CMD_FILE_MAP_CREATE,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (HANDLE)(UINT_PTR)result;
}

HANDLE WINAPI ET32_CreateFileMappingW(
    HANDLE hFile, LPSECURITY_ATTRIBUTES lpAttr,
    DWORD flProtect, DWORD dwMaxSizeHigh, DWORD dwMaxSizeLow, LPCWSTR lpName)
{
    uint64_t sz = ((uint64_t)dwMaxSizeHigh << 32) | (uint64_t)dwMaxSizeLow;
    const uint64_t koide_threshold = (uint64_t)(((double)0xFFFFFFFFULL) * (2.0 / 3.0));
    if (sz < koide_threshold) {
        return CreateFileMappingW(hFile, lpAttr, flProtect,
                                  dwMaxSizeHigh, dwMaxSizeLow, lpName);
    }
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hFile);
    et_pack_uint32(&ab, flProtect);
    et_pack_uint64(&ab, sz);
    et_pack_strW(&ab, lpName);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_MEMORY_MAP, CMD_FILE_MAP_CREATE,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (HANDLE)(UINT_PTR)result;
}

LPVOID WINAPI ET32_MapViewOfFile(
    HANDLE hFileMappingObject, DWORD dwDesiredAccess,
    DWORD dwFileOffsetHigh, DWORD dwFileOffsetLow, SIZE_T dwNumberOfBytesToMap)
{
    if (!is_bridge_handle((UINT_PTR)hFileMappingObject)) {
        return MapViewOfFile(hFileMappingObject, dwDesiredAccess,
                             dwFileOffsetHigh, dwFileOffsetLow,
                             dwNumberOfBytesToMap);
    }
    uint8_t pbuf[36];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hFileMappingObject);
    et_pack_uint32(&ab, dwDesiredAccess);
    uint64_t offset = ((uint64_t)dwFileOffsetHigh << 32) | (uint64_t)dwFileOffsetLow;
    et_pack_uint64(&ab, offset);
    et_pack_uint64(&ab, (uint64_t)dwNumberOfBytesToMap);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_MEMORY_MAP, CMD_FILE_MAP_VIEW,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (LPVOID)(UINT_PTR)result;
}

/* ---- FAMILY 4: DLL_OPS ---- */

HMODULE WINAPI ET32_LoadLibraryA(LPCSTR lpLibFileName)
{
    /* Route to 64-bit broker only for 64-bit DLLs (path ends in known 64-bit dir). */
    /* Simple heuristic: always try original first; broker only for explicit 64-bit request. */
    HMODULE h = LoadLibraryA(lpLibFileName);
    if (h) return h;

    /* Fallback: ask broker to load it as a 64-bit DLL and return a bridge handle */
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, lpLibFileName);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_LOAD,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (HMODULE)(UINT_PTR)result;
}

HMODULE WINAPI ET32_LoadLibraryW(LPCWSTR lpLibFileName)
{
    HMODULE h = LoadLibraryW(lpLibFileName);
    if (h) return h;

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strW(&ab, lpLibFileName);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_LOAD,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (HMODULE)(UINT_PTR)result;
}

HMODULE WINAPI ET32_LoadLibraryExA(LPCSTR lpLibFileName, HANDLE hFile, DWORD dwFlags)
{
    HMODULE h = LoadLibraryExA(lpLibFileName, hFile, dwFlags);
    if (h) return h;

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, lpLibFileName);
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hFile);
    et_pack_uint32(&ab, dwFlags);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_LOAD,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (HMODULE)(UINT_PTR)result;
}

HMODULE WINAPI ET32_LoadLibraryExW(LPCWSTR lpLibFileName, HANDLE hFile, DWORD dwFlags)
{
    HMODULE h = LoadLibraryExW(lpLibFileName, hFile, dwFlags);
    if (h) return h;

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strW(&ab, lpLibFileName);
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hFile);
    et_pack_uint32(&ab, dwFlags);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_LOAD,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (HMODULE)(UINT_PTR)result;
}

BOOL WINAPI ET32_FreeLibrary(HMODULE hLibModule)
{
    if (is_bridge_handle((UINT_PTR)hLibModule)) {
        uint8_t pbuf[8];
        et_arg_buf ab;
        et_argbuf_init(&ab, pbuf, sizeof(pbuf));
        et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hLibModule);
        uint32_t err = 0;
        et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_FREE, pbuf, ab.pos, &err);
        return (err == 0) ? TRUE : FALSE;
    }
    return FreeLibrary(hLibModule);
}

FARPROC WINAPI ET32_GetProcAddress(HMODULE hModule, LPCSTR lpProcName)
{
    if (!is_bridge_handle((UINT_PTR)hModule)) {
        return GetProcAddress(hModule, lpProcName);
    }
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hModule);
    /* lpProcName may be an ordinal (HIWORD == 0) */
    if ((UINT_PTR)lpProcName <= 0xFFFF)
        et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpProcName);
    else
        et_pack_strA(&ab, lpProcName);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_GETPROC,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return NULL; }
    return (FARPROC)(UINT_PTR)result;
}

/* ---- FAMILY 5: PROCESS_OPS ---- */

/*
 * ET32_CreateProcessA — child bridging implementation (ISSUE-29 resolved).
 *
 * ET derivation (Identification Principle):
 *   P = the child process, D = the bridge DLL, T = the broker's injection.
 *   Without notification, T never reaches the child before it starts executing.
 *   The Descriptor Gap IS: the child runs unbridged during the monitor scan
 *   window (up to S=12 seconds). CREATE_SUSPENDED closes this gap.
 *
 * Steps:
 *   1. Call real CreateProcessA with dwFlags | CREATE_SUSPENDED
 *   2. Notify broker via CMD_PROC_CREATE with child PID
 *   3. Broker injects et_bridge32.dll into the suspended child
 *   4. ResumeThread (unless caller requested CREATE_SUSPENDED)
 */
BOOL WINAPI ET32_CreateProcessA(
    LPCSTR lpApp, LPSTR lpCmd, LPSECURITY_ATTRIBUTES lpPA, LPSECURITY_ATTRIBUTES lpTA,
    BOOL bInherit, DWORD dwFlags, LPVOID lpEnv, LPCSTR lpDir,
    LPSTARTUPINFOA lpSI, LPPROCESS_INFORMATION lpPI)
{
    /* Preserve caller's original CREATE_SUSPENDED intent */
    BOOL caller_suspended = (dwFlags & CREATE_SUSPENDED) != 0;

    /* Always create suspended so broker can inject before child runs */
    BOOL ok = CreateProcessA(lpApp, lpCmd, lpPA, lpTA, bInherit,
                              dwFlags | CREATE_SUSPENDED,
                              lpEnv, lpDir, lpSI, lpPI);
    if (!ok) return FALSE;

    /* Notify broker: send child PID via CMD_PROC_CREATE.
     * The broker will inject et_bridge32.dll and call ET32_Init in the child. */
    if (g_connected && lpPI) {
        uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8];
        et_arg_buf ab;
        et_argbuf_init(&ab, pbuf, sizeof(pbuf));
        et_pack_uint32(&ab, lpPI->dwProcessId);       /* child PID */
        et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpPI->hProcess);  /* child handle */
        et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpPI->hThread);   /* child main thread */
        uint32_t err = 0;
        /* Send and wait — broker injection completes before we resume */
        et_call(CMD_FAMILY_PROCESS_OPS, CMD_PROC_CREATE, pbuf, ab.pos, &err);
        /* err is non-fatal: if broker fails to inject the child, the
         * process still runs (monitor will catch it on next scan) */
        if (err) {
            ET_REPORT_ERROR("Child bridge notification", lpPI->dwProcessId,
                            CMD_FAMILY_PROCESS_OPS, CMD_PROC_CREATE);
        }
    }

    /* Resume the child if the caller did NOT request CREATE_SUSPENDED */
    if (!caller_suspended && lpPI) {
        ResumeThread(lpPI->hThread);
    }

    return TRUE;
}

/*
 * ET32_CreateProcessW — wide-char variant (same child bridging logic).
 */
BOOL WINAPI ET32_CreateProcessW(
    LPCWSTR lpApp, LPWSTR lpCmd, LPSECURITY_ATTRIBUTES lpPA, LPSECURITY_ATTRIBUTES lpTA,
    BOOL bInherit, DWORD dwFlags, LPVOID lpEnv, LPCWSTR lpDir,
    LPSTARTUPINFOW lpSI, LPPROCESS_INFORMATION lpPI)
{
    BOOL caller_suspended = (dwFlags & CREATE_SUSPENDED) != 0;

    BOOL ok = CreateProcessW(lpApp, lpCmd, lpPA, lpTA, bInherit,
                              dwFlags | CREATE_SUSPENDED,
                              lpEnv, lpDir, lpSI, lpPI);
    if (!ok) return FALSE;

    if (g_connected && lpPI) {
        uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8];
        et_arg_buf ab;
        et_argbuf_init(&ab, pbuf, sizeof(pbuf));
        et_pack_uint32(&ab, lpPI->dwProcessId);
        et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpPI->hProcess);
        et_pack_uint32(&ab, (uint32_t)(UINT_PTR)lpPI->hThread);
        uint32_t err = 0;
        et_call(CMD_FAMILY_PROCESS_OPS, CMD_PROC_CREATE, pbuf, ab.pos, &err);
        if (err) {
            ET_REPORT_ERROR("Child bridge notification (W)", lpPI->dwProcessId,
                            CMD_FAMILY_PROCESS_OPS, CMD_PROC_CREATE);
        }
    }

    if (!caller_suspended && lpPI) {
        ResumeThread(lpPI->hThread);
    }

    return TRUE;
}

HANDLE WINAPI ET32_OpenProcess(DWORD dwAccess, BOOL bInherit, DWORD dwPID)
{
    return OpenProcess(dwAccess, bInherit, dwPID);
}

VOID WINAPI ET32_GetSystemInfo(LPSYSTEM_INFO lpInfo)
{
    /* Always get native 32-bit baseline first — fills all fields correctly
     * for a 32-bit process view. This is the fallback if the broker is
     * unavailable or fails. */
    GetSystemInfo(lpInfo);

    /* Ask broker for the 64-bit un-capped system info.
     * The broker runs as a 64-bit process and sees the real architecture,
     * the full maximum application address, and true processor features
     * that WOW64 hides from 32-bit callers.
     *
     * Broker returns: first uint32 = wProcessorArchitecture (PROCESSOR_ARCHITECTURE_AMD64 = 9).
     * If broker responds with a valid architecture, overlay the 64-bit-specific fields. */
    uint8_t pbuf[4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_null(&ab);

    uint32_t err = 0;
    uint32_t broker_arch = et_call(CMD_FAMILY_PROCESS_OPS, CMD_PROC_INFO,
                                    pbuf, ab.pos, &err);
    if (!err && broker_arch != 0) {
        /* Broker responded — overlay the 64-bit architecture.
         * WOW64 reports PROCESSOR_ARCHITECTURE_INTEL (0); the broker
         * provides the true architecture (typically AMD64 = 9). */
        lpInfo->wProcessorArchitecture = (WORD)broker_arch;
    }
    /* If err: native baseline from GetSystemInfo() is already valid
     * and populated above — no action needed. */
}

/* ---- FAMILY 6: REGISTRY_OPS (WOW64 bypass) ---- */

LONG WINAPI ET32_RegOpenKeyExA(
    HKEY hKey, LPCSTR lpSubKey, DWORD ulOptions, REGSAM samDesired, PHKEY phkResult)
{
    /* First try with KEY_WOW64_64KEY to bypass WOW64 redirection */
    REGSAM sam64 = samDesired | KEY_WOW64_64KEY;
    LONG result = RegOpenKeyExA(hKey, lpSubKey, ulOptions, sam64, phkResult);
    if (result == ERROR_SUCCESS) return result;

    /* Fallback: bridge to broker */
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hKey);
    et_pack_strA(&ab, lpSubKey);
    et_pack_uint32(&ab, ulOptions);
    et_pack_uint32(&ab, samDesired);

    uint32_t err = 0;
    uint32_t handle = et_call(CMD_FAMILY_REGISTRY_OPS, CMD_REG_OPEN64,
                               pbuf, ab.pos, &err);
    if (err) return (LONG)err;
    if (phkResult) *phkResult = (HKEY)(UINT_PTR)handle;
    return ERROR_SUCCESS;
}

LONG WINAPI ET32_RegOpenKeyExW(
    HKEY hKey, LPCWSTR lpSubKey, DWORD ulOptions, REGSAM samDesired, PHKEY phkResult)
{
    REGSAM sam64 = samDesired | KEY_WOW64_64KEY;
    LONG result = RegOpenKeyExW(hKey, lpSubKey, ulOptions, sam64, phkResult);
    if (result == ERROR_SUCCESS) return result;

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hKey);
    et_pack_strW(&ab, lpSubKey);
    et_pack_uint32(&ab, ulOptions);
    et_pack_uint32(&ab, samDesired);

    uint32_t err = 0;
    uint32_t handle = et_call(CMD_FAMILY_REGISTRY_OPS, CMD_REG_OPEN64,
                               pbuf, ab.pos, &err);
    if (err) return (LONG)err;
    if (phkResult) *phkResult = (HKEY)(UINT_PTR)handle;
    return ERROR_SUCCESS;
}

LONG WINAPI ET32_RegQueryValueExA(
    HKEY hKey, LPCSTR lpValueName, LPDWORD lpReserved,
    LPDWORD lpType, LPBYTE lpData, LPDWORD lpcbData)
{
    if (!is_bridge_handle((UINT_PTR)hKey)) {
        return RegQueryValueExA(hKey, lpValueName, lpReserved, lpType, lpData, lpcbData);
    }
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hKey);
    et_pack_strA(&ab, lpValueName);
    et_pack_uint32(&ab, lpcbData ? *lpcbData : 0);

    uint32_t err = 0;
    et_call(CMD_FAMILY_REGISTRY_OPS, CMD_REG_QUERY64, pbuf, ab.pos, &err);
    if (err) return (LONG)err;
    /* Broker fills result into response payload; simplified return */
    return ERROR_SUCCESS;
}

LONG WINAPI ET32_RegQueryValueExW(
    HKEY hKey, LPCWSTR lpValueName, LPDWORD lpReserved,
    LPDWORD lpType, LPBYTE lpData, LPDWORD lpcbData)
{
    if (!is_bridge_handle((UINT_PTR)hKey)) {
        return RegQueryValueExW(hKey, lpValueName, lpReserved, lpType, lpData, lpcbData);
    }
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hKey);
    et_pack_strW(&ab, lpValueName);
    et_pack_uint32(&ab, lpcbData ? *lpcbData : 0);

    uint32_t err = 0;
    et_call(CMD_FAMILY_REGISTRY_OPS, CMD_REG_QUERY64, pbuf, ab.pos, &err);
    if (err) return (LONG)err;
    return ERROR_SUCCESS;
}

LONG WINAPI ET32_RegSetValueExA(
    HKEY hKey, LPCSTR lpValueName, DWORD Reserved,
    DWORD dwType, const BYTE *lpData, DWORD cbData)
{
    if (!is_bridge_handle((UINT_PTR)hKey)) {
        return RegSetValueExA(hKey, lpValueName, Reserved, dwType, lpData, cbData);
    }
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hKey);
    et_pack_strA(&ab, lpValueName);
    et_pack_uint32(&ab, dwType);
    et_pack_bytes(&ab, lpData, cbData);

    uint32_t err = 0;
    et_call(CMD_FAMILY_REGISTRY_OPS, CMD_REG_SET64, pbuf, ab.pos, &err);
    return err ? (LONG)err : ERROR_SUCCESS;
}

LONG WINAPI ET32_RegSetValueExW(
    HKEY hKey, LPCWSTR lpValueName, DWORD Reserved,
    DWORD dwType, const BYTE *lpData, DWORD cbData)
{
    if (!is_bridge_handle((UINT_PTR)hKey)) {
        return RegSetValueExW(hKey, lpValueName, Reserved, dwType, lpData, cbData);
    }
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hKey);
    et_pack_strW(&ab, lpValueName);
    et_pack_uint32(&ab, dwType);
    et_pack_bytes(&ab, lpData, cbData);

    uint32_t err = 0;
    et_call(CMD_FAMILY_REGISTRY_OPS, CMD_REG_SET64, pbuf, ab.pos, &err);
    return err ? (LONG)err : ERROR_SUCCESS;
}

/* ---- FAMILY 7: GRAPHICS_OPS (pass-through with broker query) ---- */

/*
 * Graphics operations are pass-through in the DLL: the 32-bit APIs work
 * natively. We expose ET32_GPU_QueryInfo() for the broker's extended VRAM
 * allocation tracking.
 */
BOOL WINAPI ET32_GPU_QueryInfo(DWORD adapter_index, DWORD *vram_mb_out)
{
    uint8_t pbuf[8];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, adapter_index);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_GRAPHICS_OPS, CMD_GPU_QUERY_INFO,
                               pbuf, ab.pos, &err);
    if (err) return FALSE;
    if (vram_mb_out) *vram_mb_out = result;
    return TRUE;
}

/* ---- FAMILY 8: FILE_OPS (large files > 4 GB) ---- */

HANDLE WINAPI ET32_CreateFileA(
    LPCSTR lpFileName, DWORD dwAccess, DWORD dwShare,
    LPSECURITY_ATTRIBUTES lpSA, DWORD dwCreation,
    DWORD dwFlags, HANDLE hTemplate)
{
    /* Always try native first */
    HANDLE h = CreateFileA(lpFileName, dwAccess, dwShare, lpSA,
                           dwCreation, dwFlags, hTemplate);
    if (h != INVALID_HANDLE_VALUE) return h;

    /* If not found, ask broker (may be >4GB path or long path) */
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, lpFileName);
    et_pack_uint32(&ab, dwAccess);
    et_pack_uint32(&ab, dwShare);
    et_pack_uint32(&ab, dwCreation);
    et_pack_uint32(&ab, dwFlags);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_FILE_OPS, CMD_FILE_OPEN_LARGE,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return INVALID_HANDLE_VALUE; }
    return (HANDLE)(UINT_PTR)result;
}

HANDLE WINAPI ET32_CreateFileW(
    LPCWSTR lpFileName, DWORD dwAccess, DWORD dwShare,
    LPSECURITY_ATTRIBUTES lpSA, DWORD dwCreation,
    DWORD dwFlags, HANDLE hTemplate)
{
    HANDLE h = CreateFileW(lpFileName, dwAccess, dwShare, lpSA,
                           dwCreation, dwFlags, hTemplate);
    if (h != INVALID_HANDLE_VALUE) return h;

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strW(&ab, lpFileName);
    et_pack_uint32(&ab, dwAccess);
    et_pack_uint32(&ab, dwShare);
    et_pack_uint32(&ab, dwCreation);
    et_pack_uint32(&ab, dwFlags);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_FILE_OPS, CMD_FILE_OPEN_LARGE,
                               pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return INVALID_HANDLE_VALUE; }
    return (HANDLE)(UINT_PTR)result;
}

BOOL WINAPI ET32_SetFilePointerEx(
    HANDLE hFile, LARGE_INTEGER liDistToMove,
    PLARGE_INTEGER lpNewFilePointer, DWORD dwMoveMethod)
{
    if (!is_bridge_handle((UINT_PTR)hFile)) {
        return SetFilePointerEx(hFile, liDistToMove, lpNewFilePointer, dwMoveMethod);
    }
    uint8_t pbuf[24];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hFile);
    et_pack_uint64(&ab, (uint64_t)liDistToMove.QuadPart);
    et_pack_uint32(&ab, dwMoveMethod);

    uint32_t err = 0;
    et_call(CMD_FAMILY_FILE_OPS, CMD_FILE_SEEK_LARGE, pbuf, ab.pos, &err);
    if (err) { SetLastError(err); return FALSE; }
    return TRUE;
}

/* ---- FAMILY 9: SYNC_OPS ---- */

HANDLE WINAPI ET32_CreateEventA(
    LPSECURITY_ATTRIBUTES lpSA, BOOL bManualReset,
    BOOL bInitialState, LPCSTR lpName)
{
    /* Native unless broker-coordinated sync needed */
    return CreateEventA(lpSA, bManualReset, bInitialState, lpName);
}

HANDLE WINAPI ET32_CreateMutexA(
    LPSECURITY_ATTRIBUTES lpSA, BOOL bInitialOwner, LPCSTR lpName)
{
    return CreateMutexA(lpSA, bInitialOwner, lpName);
}

/* ---- FAMILY 10: NET_OPS ---- */

/* Network bridging: pass-through in DLL (handled by broker's 64-bit socket).
 * Only relevant if target uses >2GB receive buffers; not common in 32-bit apps. */

/* ---- FAMILY 11: PYTHON_OPS ---- */

BOOL WINAPI ET32_PythonExec(LPCSTR code, DWORD *result_out)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 2];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, code);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_PYTHON_OPS, CMD_PY_EXEC,
                               pbuf, ab.pos, &err);
    if (result_out) *result_out = result;
    return (err == 0) ? TRUE : FALSE;
}

/*
 * ET32_PythonImport — import a Python module in the 64-bit interpreter.
 * C2C uses this for CvPythonExtensions, CvUtil, CvScreensInterface, etc.
 */
__declspec(dllexport)
BOOL WINAPI ET32_PythonImport(LPCSTR module_name, DWORD *result_out)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, module_name);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_PYTHON_OPS, CMD_PY_IMPORT,
                               pbuf, ab.pos, &err);
    if (result_out) *result_out = result;
    return (err == 0) ? TRUE : FALSE;
}

/*
 * ET32_PythonCall — call a named Python function in the 64-bit interpreter.
 * func_name: dotted name (e.g. "CvUtil.pyPrint").
 * args_str: JSON-encoded or comma-separated arg list (broker parses).
 * C2C Python callbacks route through here.
 */
__declspec(dllexport)
BOOL WINAPI ET32_PythonCall(LPCSTR func_name, LPCSTR args_str, DWORD *result_out)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 2];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, func_name);
    et_pack_strA(&ab, args_str ? args_str : "");

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_PYTHON_OPS, CMD_PY_CALL,
                               pbuf, ab.pos, &err);
    if (result_out) *result_out = result;
    return (err == 0) ? TRUE : FALSE;
}

/*
 * ET32_PythonEval — evaluate a Python expression and return the result.
 * Unlike PY_EXEC (which executes statements), PY_EVAL returns a value.
 * C2C UI callbacks (CyInterface queries) need expression evaluation.
 */
__declspec(dllexport)
BOOL WINAPI ET32_PythonEval(LPCSTR expr, DWORD *result_out)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 2];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, expr);

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_PYTHON_OPS, CMD_PY_EVAL,
                               pbuf, ab.pos, &err);
    if (result_out) *result_out = result;
    return (err == 0) ? TRUE : FALSE;
}

/*
 * ET32_PythonSetObj — set a variable in the 64-bit Python globals.
 * C2C event system needs C++→Python variable injection (gc.player, gc.game).
 * obj_name: variable name (e.g. "gc").
 * value_str: string representation (broker reconstructs the object).
 */
__declspec(dllexport)
BOOL WINAPI ET32_PythonSetObj(LPCSTR obj_name, LPCSTR value_str)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 2];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, obj_name);
    et_pack_strA(&ab, value_str ? value_str : "");

    uint32_t err = 0;
    et_call(CMD_FAMILY_PYTHON_OPS, CMD_PY_SETOBJ, pbuf, ab.pos, &err);
    return (err == 0) ? TRUE : FALSE;
}

/*
 * ET32_PythonSysPath — append a directory to sys.path in the 64-bit interpreter.
 * C2C mods expect Assets/Python and Assets/Python/Screens on sys.path.
 * mode: 0 = append, 1 = prepend (insert at position 0).
 */
__declspec(dllexport)
BOOL WINAPI ET32_PythonSysPath(LPCSTR path, DWORD mode)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, path);
    et_pack_uint32(&ab, mode);

    uint32_t err = 0;
    et_call(CMD_FAMILY_PYTHON_OPS, CMD_PY_SYSPATH, pbuf, ab.pos, &err);
    return (err == 0) ? TRUE : FALSE;
}

/*
 * ET32_GPU_EnumAdapters — enumerate DXGI adapters via broker.
 * C2C needs adapter info for DirectX 9/11 device creation.
 */
__declspec(dllexport)
BOOL WINAPI ET32_GPU_EnumAdapters(DWORD *adapter_count_out)
{
    uint8_t pbuf[8];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, 0);  /* start index */

    uint32_t err = 0;
    uint32_t count = et_call(CMD_FAMILY_GRAPHICS_OPS, CMD_GPU_ENUM_ADAPTERS,
                              pbuf, ab.pos, &err);
    if (adapter_count_out) *adapter_count_out = count;
    return (err == 0) ? TRUE : FALSE;
}

/*
 * ET32_GPU_CreateDevice — create a D3D9/D3D11 device handle through the broker.
 * device_type: 0 = D3D9, 1 = D3D11 (broker selects DLL and entrypoint).
 * adapter_index: DXGI adapter to use.
 * Returns bridge handle for the device.
 */
__declspec(dllexport)
DWORD WINAPI ET32_GPU_CreateDevice(DWORD device_type, DWORD adapter_index)
{
    uint8_t pbuf[16];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, device_type);
    et_pack_uint32(&ab, adapter_index);

    uint32_t err = 0;
    return et_call(CMD_FAMILY_GRAPHICS_OPS, CMD_GPU_CREATE_DEVICE,
                   pbuf, ab.pos, &err);
}

/* ---- FAMILY 12: COMPOUND_OPS ---- */

/* CloseHandle: route to broker only if bridge handle */
BOOL WINAPI ET32_CloseHandle(HANDLE hObject)
{
    if (is_bridge_handle((UINT_PTR)hObject)) {
        uint8_t pbuf[8];
        et_arg_buf ab;
        et_argbuf_init(&ab, pbuf, sizeof(pbuf));
        et_pack_uint32(&ab, (uint32_t)(UINT_PTR)hObject);
        uint32_t err = 0;
        et_call(CMD_FAMILY_MEMORY_MAP, CMD_FILE_MAP_CLOSE, pbuf, ab.pos, &err);
        return (err == 0) ? TRUE : FALSE;
    }
    return CloseHandle(hObject);
}

/* Liveness probe */
BOOL WINAPI ET32_Ping(void)
{
    uint8_t pbuf[4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_null(&ab);
    uint32_t err = 0;
    uint32_t resp = et_call(CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_PING,
                             pbuf, ab.pos, &err);
    return (err == 0 && resp == CMD_CTRL_ACK) ? TRUE : FALSE;
}

/* ============================================================================
 * IAT PATCHING ENGINE
 * Scans the target executable's Import Address Table and replaces pointers
 * for all bridged APIs with our hook function pointers.
 * ============================================================================ */

/* Map: original function name → hook function pointer */
typedef struct {
    const char *name;
    FARPROC     hook;
} et_iat_entry;

/* ---- Forward declarations for completeness additions (defined after IAT table) ---- */
/* These functions are defined in the "COMPLETENESS ADDITIONS" section below.
 * Forward-declared here so the g_iat_hooks[] table can reference them. */

/* MEMORY_BASIC */
/* ET_MEMORYSTATUSEX — mirrors MEMORYSTATUSEX layout; authoritative definition.
 * Verified by compile-time assertion in COMPLETENESS ADDITIONS section below. */
#define ET_MEMORYSTATUSEX_DEFINED
typedef struct {
    DWORD     dwLength;
    DWORD     dwMemoryLoad;
    ULONGLONG ullTotalPhys;
    ULONGLONG ullAvailPhys;
    ULONGLONG ullTotalPageFile;
    ULONGLONG ullAvailPageFile;
    ULONGLONG ullTotalVirtual;
    ULONGLONG ullAvailVirtual;
    ULONGLONG ullAvailExtendedVirtual;
} ET_MEMORYSTATUSEX;
BOOL WINAPI ET32_GlobalMemoryStatusEx(ET_MEMORYSTATUSEX *lpBuffer);
VOID WINAPI ET32_GetNativeSystemInfo(LPSYSTEM_INFO lpSystemInfo);
BOOL WINAPI ET32_DuplicateHandle(
    HANDLE hSrcProc, HANDLE hSrcHandle,
    HANDLE hTgtProc, LPHANDLE lpTgtHandle,
    DWORD dwAccess, BOOL bInherit, DWORD dwOptions);

/* THREAD_OPS */
BOOL WINAPI ET32_SetThreadContext(HANDLE hThread, const CONTEXT *lpContext);
BOOL WINAPI ET32_GetExitCodeThread(HANDLE hThread, LPDWORD lpExitCode);

/* PROCESS_OPS */
BOOL WINAPI ET32_GetExitCodeProcess(HANDLE hProcess, LPDWORD lpExitCode);
BOOL WINAPI ET32_TerminateProcess(HANDLE hProcess, UINT uExitCode);
BOOL WINAPI ET32_Wow64DisableWow64FsRedirection(PVOID *OldValue);
BOOL WINAPI ET32_Wow64RevertWow64FsRedirection(PVOID OldValue);

/* REGISTRY_OPS */
LONG WINAPI ET32_RegCreateKeyExA(HKEY hKey, LPCSTR lpSubKey, DWORD Reserved,
    LPSTR lpClass, DWORD dwOptions, REGSAM samDesired,
    LPSECURITY_ATTRIBUTES lpSA, PHKEY phkResult, LPDWORD lpdwDisposition);
LONG WINAPI ET32_RegDeleteKeyExA(HKEY hKey, LPCSTR lpSubKey,
    REGSAM samDesired, DWORD Reserved);
LONG WINAPI ET32_RegDeleteValueA(HKEY hKey, LPCSTR lpValueName);
LONG WINAPI ET32_RegCloseKey(HKEY hKey);

/* SYNC_OPS */
HANDLE WINAPI ET32_CreateSemaphoreA(LPSECURITY_ATTRIBUTES lpSA,
    LONG lInitialCount, LONG lMaximumCount, LPCSTR lpName);
BOOL WINAPI ET32_ReleaseSemaphore(HANDLE hSemaphore,
    LONG lReleaseCount, LPLONG lpPreviousCount);
DWORD WINAPI ET32_WaitForMultipleObjects(DWORD nCount,
    const HANDLE *lpHandles, BOOL bWaitAll, DWORD dwMilliseconds);
BOOL WINAPI ET32_ResetEvent(HANDLE hEvent);

/* NET_OPS */
int WINAPI ET32_connect(SOCKET s, const struct sockaddr *name, int namelen);
int WINAPI ET32_listen(SOCKET s, int backlog);
SOCKET WINAPI ET32_accept(SOCKET s, struct sockaddr *addr, int *addrlen);
int WINAPI ET32_closesocket(SOCKET s);

/* FILE_OPS */
BOOL WINAPI ET32_GetFileSizeEx(HANDLE hFile, PLARGE_INTEGER lpFileSize);
BOOL WINAPI ET32_GetFileAttributesExW(LPCWSTR lpFileName,
    GET_FILEEX_INFO_LEVELS fInfoLevel, LPVOID lpFileInformation);
BOOL WINAPI ET32_FlushFileBuffers(HANDLE hFile);
HANDLE WINAPI ET32_FindFirstFileW(LPCWSTR lpFileName,
    LPWIN32_FIND_DATAW lpFindFileData);
BOOL WINAPI ET32_FindNextFileW(HANDLE hFindFile,
    LPWIN32_FIND_DATAW lpFindFileData);
BOOL WINAPI ET32_FindClose(HANDLE hFindFile);

static const et_iat_entry g_iat_hooks[] = {
    { "VirtualAlloc",         (FARPROC)ET32_VirtualAlloc         },
    { "VirtualFree",          (FARPROC)ET32_VirtualFree          },
    { "VirtualProtect",       (FARPROC)ET32_VirtualProtect       },
    { "VirtualQuery",         (FARPROC)ET32_VirtualQuery         },
    { "CreateFileMappingA",   (FARPROC)ET32_CreateFileMappingA   },
    { "CreateFileMappingW",   (FARPROC)ET32_CreateFileMappingW   },
    { "MapViewOfFile",        (FARPROC)ET32_MapViewOfFile        },
    { "LoadLibraryA",         (FARPROC)ET32_LoadLibraryA         },
    { "LoadLibraryW",         (FARPROC)ET32_LoadLibraryW         },
    { "LoadLibraryExA",       (FARPROC)ET32_LoadLibraryExA       },
    { "LoadLibraryExW",       (FARPROC)ET32_LoadLibraryExW       },
    { "FreeLibrary",          (FARPROC)ET32_FreeLibrary          },
    { "GetProcAddress",       (FARPROC)ET32_GetProcAddress       },
    { "CreateProcessA",       (FARPROC)ET32_CreateProcessA       },
    { "CreateProcessW",       (FARPROC)ET32_CreateProcessW       },
    { "OpenProcess",          (FARPROC)ET32_OpenProcess          },
    { "GetSystemInfo",        (FARPROC)ET32_GetSystemInfo        },
    { "RegOpenKeyExA",        (FARPROC)ET32_RegOpenKeyExA        },
    { "RegOpenKeyExW",        (FARPROC)ET32_RegOpenKeyExW        },
    { "RegQueryValueExA",     (FARPROC)ET32_RegQueryValueExA     },
    { "RegQueryValueExW",     (FARPROC)ET32_RegQueryValueExW     },
    { "RegSetValueExA",       (FARPROC)ET32_RegSetValueExA       },
    { "RegSetValueExW",       (FARPROC)ET32_RegSetValueExW       },
    { "CreateFileA",          (FARPROC)ET32_CreateFileA          },
    { "CreateFileW",          (FARPROC)ET32_CreateFileW          },
    { "SetFilePointerEx",     (FARPROC)ET32_SetFilePointerEx     },
    { "CloseHandle",                   (FARPROC)ET32_CloseHandle                   },
    { "CreateEventA",                  (FARPROC)ET32_CreateEventA                  },
    { "CreateMutexA",                  (FARPROC)ET32_CreateMutexA                  },
    /* Completeness additions — all gaps closed */
    { "GlobalMemoryStatusEx",          (FARPROC)ET32_GlobalMemoryStatusEx          },
    { "DuplicateHandle",               (FARPROC)ET32_DuplicateHandle               },
    { "GetNativeSystemInfo",           (FARPROC)ET32_GetNativeSystemInfo           },
    { "SetThreadContext",              (FARPROC)ET32_SetThreadContext              },
    { "GetExitCodeThread",             (FARPROC)ET32_GetExitCodeThread             },
    { "GetExitCodeProcess",            (FARPROC)ET32_GetExitCodeProcess            },
    { "TerminateProcess",              (FARPROC)ET32_TerminateProcess              },
    { "Wow64DisableWow64FsRedirection",(FARPROC)ET32_Wow64DisableWow64FsRedirection},
    { "Wow64RevertWow64FsRedirection", (FARPROC)ET32_Wow64RevertWow64FsRedirection },
    { "RegCreateKeyExA",               (FARPROC)ET32_RegCreateKeyExA               },
    { "RegDeleteKeyExA",               (FARPROC)ET32_RegDeleteKeyExA               },
    { "RegDeleteValueA",               (FARPROC)ET32_RegDeleteValueA               },
    { "RegCloseKey",                   (FARPROC)ET32_RegCloseKey                   },
    { "CreateSemaphoreA",              (FARPROC)ET32_CreateSemaphoreA              },
    { "ReleaseSemaphore",              (FARPROC)ET32_ReleaseSemaphore              },
    { "WaitForMultipleObjects",        (FARPROC)ET32_WaitForMultipleObjects        },
    { "ResetEvent",                    (FARPROC)ET32_ResetEvent                    },
    { "connect",                       (FARPROC)ET32_connect                       },
    { "listen",                        (FARPROC)ET32_listen                        },
    { "accept",                        (FARPROC)ET32_accept                        },
    { "closesocket",                   (FARPROC)ET32_closesocket                   },
    { "GetFileSizeEx",                 (FARPROC)ET32_GetFileSizeEx                 },
    { "GetFileAttributesExW",          (FARPROC)ET32_GetFileAttributesExW          },
    { "FlushFileBuffers",              (FARPROC)ET32_FlushFileBuffers              },
    { "FindFirstFileW",                (FARPROC)ET32_FindFirstFileW                },
    { "FindNextFileW",                 (FARPROC)ET32_FindNextFileW                 },
    { "FindClose",                     (FARPROC)ET32_FindClose                     },
    { NULL, NULL }
};

/* ISSUE-27 RESOLVED: Dynamic hook save array.
 * Initial capacity S² = 144 (ET manifold squared).
 * Doubles on overflow via HeapReAlloc.
 * No hook is EVER written if the original cannot be saved — this prevents
 * unrestorable IAT corruption on DLL unload.
 *
 * ET derivation (Subsumption Law): the save list must subsume ALL patched
 * IAT slots without remainder. A static limit creates remainder (unsaved
 * slots). Dynamic allocation achieves subsumption — every slot patched IS
 * saved, unconditionally. */
#define SAVED_HOOKS_INIT_CAP 144 /* S² = 12² — natural ET starting point */

typedef struct {
    FARPROC *iat_slot;
    FARPROC  original;
} et_saved_hook;

static et_saved_hook *g_saved_hooks     = NULL;
static int            g_saved_hooks_cap = 0;
static int            g_n_saved_hooks   = 0;

static BOOL et_write_ptr(FARPROC *slot, FARPROC new_fn)
{
    DWORD old_prot, dummy;
    if (!VirtualProtect(slot, sizeof(FARPROC), PAGE_READWRITE, &old_prot)) {
        ET_REPORT_ERROR("VirtualProtect for IAT slot", g_target_pid,
                        CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_ERR);
        return FALSE;
    }
    *slot = new_fn;
    VirtualProtect(slot, sizeof(FARPROC), old_prot, &dummy);
    return TRUE;
}

static void et_patch_iat_for_module(HMODULE hMod)
{
    BYTE *base = (BYTE *)hMod;

    /* Read DOS header */
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;

    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;

    DWORD import_rva = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
    if (!import_rva) return;

    IMAGE_IMPORT_DESCRIPTOR *imp = (IMAGE_IMPORT_DESCRIPTOR *)(base + import_rva);

    for (; imp->OriginalFirstThunk; imp++) {
        /* Walk thunks */
        IMAGE_THUNK_DATA *orig = (IMAGE_THUNK_DATA *)(base + imp->OriginalFirstThunk);
        IMAGE_THUNK_DATA *iat  = (IMAGE_THUNK_DATA *)(base + imp->FirstThunk);

        for (; orig->u1.AddressOfData; orig++, iat++) {
            if (IMAGE_SNAP_BY_ORDINAL(orig->u1.Ordinal)) continue;

            IMAGE_IMPORT_BY_NAME *ibn =
                (IMAGE_IMPORT_BY_NAME *)(base + (DWORD)orig->u1.AddressOfData);
            const char *func_name = (const char *)ibn->Name;

            /* Check if this function is in our hook table */
            int i;
            for (i = 0; g_iat_hooks[i].name; i++) {
                if (lstrcmpA(func_name, g_iat_hooks[i].name) == 0) {
                    FARPROC *slot = (FARPROC *)&iat->u1.Function;

                    /* ISSUE-27: Dynamic saved hooks — grow on overflow.
                     * MUST save the original before patching. If save fails,
                     * do NOT write the hook — unrestorable IAT corruption
                     * on DLL unload is worse than a missing hook. */
                    if (g_n_saved_hooks >= g_saved_hooks_cap) {
                        int new_cap = (g_saved_hooks_cap == 0)
                            ? SAVED_HOOKS_INIT_CAP
                            : g_saved_hooks_cap * 2;
                        et_saved_hook *p;
                        if (g_saved_hooks == NULL) {
                            p = (et_saved_hook *)HeapAlloc(
                                GetProcessHeap(), 0,
                                (SIZE_T)new_cap * sizeof(et_saved_hook));
                        } else {
                            p = (et_saved_hook *)HeapReAlloc(
                                GetProcessHeap(), 0,
                                g_saved_hooks,
                                (SIZE_T)new_cap * sizeof(et_saved_hook));
                        }
                        if (!p) {
                            /* Cannot grow — MUST NOT patch this slot */
                            ET_REPORT_ERROR(
                                "HeapReAlloc for saved_hooks — "
                                "IAT slot NOT patched to prevent "
                                "unrestorable corruption",
                                g_target_pid,
                                CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_ERR);
                            break; /* skip this hook entirely */
                        }
                        g_saved_hooks     = p;
                        g_saved_hooks_cap = new_cap;
                    }

                    g_saved_hooks[g_n_saved_hooks].iat_slot = slot;
                    g_saved_hooks[g_n_saved_hooks].original  = *slot;
                    g_n_saved_hooks++;
                    et_write_ptr(slot, g_iat_hooks[i].hook);
                    break;
                }
            }
        }
    }
}

static void et_patch_iat_all(void)
{
    /* Patch the main executable module */
    HMODULE hExe = GetModuleHandleA(NULL);
    if (hExe) et_patch_iat_for_module(hExe);

    /* Also patch any already-loaded DLLs except ourselves */
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if (snap == INVALID_HANDLE_VALUE) return;

    MODULEENTRY32 me;
    me.dwSize = sizeof(me);
    if (Module32First(snap, &me)) {
        do {
            HMODULE h = me.hModule;
            if (h && h != hExe) {
                et_patch_iat_for_module(h);
            }
        } while (Module32Next(snap, &me));
    }
    CloseHandle(snap);
}

static void et_restore_iat(void)
{
    int i;
    for (i = 0; i < g_n_saved_hooks; i++) {
        et_write_ptr(g_saved_hooks[i].iat_slot, g_saved_hooks[i].original);
    }
    g_n_saved_hooks = 0;

    /* Free the dynamic saved hooks array (ISSUE-27) */
    if (g_saved_hooks != NULL) {
        HeapFree(GetProcessHeap(), 0, g_saved_hooks);
        g_saved_hooks     = NULL;
        g_saved_hooks_cap = 0;
    }
}

/* ============================================================================
 * PUBLIC EXPORTS
 * ============================================================================ */

/* ---- Forward declarations for AWE / VEH subsystems ----
 * ET32_Init and ET32_Shutdown reference AWE globals, types, defines, and
 * VEH functions that are defined later in the "AWE BOOKSHELF" and "VEH"
 * sections. Forward-declared here so the compiler can see them.
 * Verified by compile-time assertions at their canonical definition sites. */

/* AWE ET constants — authoritative definitions.
 * Verified by compile-time assertion in AWE BOOKSHELF section below. */
#define ET_AWE_DEFINES
#define AWE_PAGE_SIZE    4096UL
#define AWE_WINDOW_SIZE  (AWE_PAGE_SIZE * AWE_PAGE_SIZE)  /* 16 MB = ħ_d² */
#define AWE_MAX_WINDOWS  144                               /* S² = QUEUE_DEPTH */

/* AWE window entry type — authoritative definition.
 * Verified by compile-time assertion in AWE BOOKSHELF section below. */
#define ET_AWE_WINDOW_DEFINED
typedef struct {
    LPVOID  va_base;         /* 32-bit VA of reserved MEM_PHYSICAL region   */
    DWORD   n_pages;         /* pages currently mapped into this window      */
    BOOL    active;          /* window is currently mapped                   */
    DWORD   last_access_ms;  /* GetTickCount() at last access                */
} et_awe_window;

/* AWE global state — authoritative declaration.
 * Verified by compile-time assertion in AWE BOOKSHELF section below. */
#define ET_AWE_GLOBALS_DEFINED
static et_awe_window g_awe_windows[AWE_MAX_WINDOWS];
static int           g_n_awe_windows = 0;
static CRITICAL_SECTION g_awe_cs;
static BOOL          g_awe_cs_init = FALSE;

/* AWE / VEH function prototypes */
static void et_awe_init_cs(void);
static void et_veh_install(void);
static void et_veh_remove(void);

/*
 * ET32_Init — called by the broker after injecting the DLL.
 * broker_pid: the PID of the 64-bit broker process.
 * Returns TRUE on success (pipe connected + handshake complete).
 */
__declspec(dllexport)
BOOL WINAPI ET32_Init(DWORD broker_pid)
{
    if (g_initialised) return TRUE;

    g_target_pid = GetCurrentProcessId();
    g_broker_pid = broker_pid;

    InitializeCriticalSection(&g_pipe_cs);

    if (!et_pipe_connect()) {
        ET_REPORT_ERROR("Pipe connect to broker", g_target_pid,
                        CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_HANDSHAKE);
        DeleteCriticalSection(&g_pipe_cs);
        return FALSE;
    }

    if (!et_do_handshake()) {
        ET_REPORT_ERROR("Handshake with broker", g_target_pid,
                        CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_HANDSHAKE);
        et_pipe_disconnect();
        DeleteCriticalSection(&g_pipe_cs);
        return FALSE;
    }

    /* Initialize AWE critical section — must precede VEH install because
     * the VEH handler acquires g_awe_cs to check AWE windows. */
    et_awe_init_cs();

    /* Install Vectored Exception Handler BEFORE IAT patching.
     * ET derivation (Descriptor Gap Principle): the gap between IAT patching
     * and VEH installation is itself a Descriptor — a crash window where a
     * freshly-hooked function could trigger an AWE fault with no handler.
     * Installing VEH first closes this gap. V(gap) → 0. */
    et_veh_install();

    /* Install IAT hooks (first layer — catches high-level API calls) */
    et_patch_iat_all();

    /* Signal broker that AWE subsystem is ready */
    {
        uint8_t pbuf[8];
        et_arg_buf ab;
        et_argbuf_init(&ab, pbuf, sizeof(pbuf));
        et_pack_uint32(&ab, g_target_pid);
        uint32_t err = 0;
        et_call(CMD_FAMILY_MEMORY_BASIC, CMD_HEAP_ALLOC, pbuf, ab.pos, &err);
    }

    g_initialised = TRUE;
    return TRUE;
}

/*
 * ET32_Shutdown — restore IAT hooks and disconnect from broker.
 */
__declspec(dllexport)
void WINAPI ET32_Shutdown(void)
{
    if (!g_initialised) return;

    et_restore_iat();
    et_veh_remove();

    /* Release all AWE windows */
    if (g_awe_cs_init) {
        EnterCriticalSection(&g_awe_cs);
        for (int i = g_n_awe_windows - 1; i >= 0; i--) {
            if (g_awe_windows[i].active) {
                MapUserPhysicalPages(g_awe_windows[i].va_base,
                                     (ULONG_PTR)g_awe_windows[i].n_pages, NULL);
            }
            VirtualFree(g_awe_windows[i].va_base, 0, MEM_RELEASE);
        }
        g_n_awe_windows = 0;
        LeaveCriticalSection(&g_awe_cs);
        DeleteCriticalSection(&g_awe_cs);
        g_awe_cs_init = FALSE;
    }

    /* Send shutdown notification */
    uint8_t pbuf[4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_null(&ab);
    uint32_t err = 0;
    et_call(CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_SHUTDOWN, pbuf, ab.pos, &err);

    et_pipe_disconnect();
    DeleteCriticalSection(&g_pipe_cs);
    g_initialised = FALSE;
}

/*
 * ET32_IsConnected — liveness probe.
 */
__declspec(dllexport)
BOOL WINAPI ET32_IsConnected(void)
{
    if (!g_connected) return FALSE;
    return ET32_Ping();
}

/*
 * ET32_GetVersion — returns DLL version as uint32.
 */
__declspec(dllexport)
DWORD WINAPI ET32_GetVersion(void)
{
    return ET_DLL_VERSION;
}

/*
 * ET32_BridgeVirtualAlloc — explicit public bridge-to-64-bit VirtualAlloc.
 * Size must fit in SIZE_T (32-bit); for larger sizes use ET32_BridgeVirtualAlloc64.
 */
__declspec(dllexport)
LPVOID WINAPI ET32_BridgeVirtualAlloc(LPVOID hint, SIZE_T size, DWORD type, DWORD prot)
{
    return ET32_VirtualAlloc(hint, size, type, prot);
}

/*
 * ET32_BridgeVirtualAlloc64 — allocate a 64-bit region; returns bridge handle.
 * size_lo / size_hi form a 64-bit byte count.
 */
__declspec(dllexport)
DWORD WINAPI ET32_BridgeVirtualAlloc64(
    DWORD hint_lo, DWORD hint_hi,
    DWORD size_lo, DWORD size_hi,
    DWORD alloc_type, DWORD protect)
{
    uint8_t pbuf[48];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    uint64_t hint = ((uint64_t)hint_hi << 32) | (uint64_t)hint_lo;
    uint64_t size = ((uint64_t)size_hi << 32) | (uint64_t)size_lo;
    et_pack_uint64(&ab, hint);
    et_pack_uint64(&ab, size);
    et_pack_uint32(&ab, alloc_type);
    et_pack_uint32(&ab, protect);

    uint32_t err = 0;
    return et_call(CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_ALLOC, pbuf, ab.pos, &err);
}

/*
 * ET32_BridgeLoadLibrary64 — load a 64-bit DLL via broker; returns bridge handle.
 */
__declspec(dllexport)
DWORD WINAPI ET32_BridgeLoadLibrary64(LPCSTR dll_path)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strA(&ab, dll_path);
    uint32_t err = 0;
    return et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_LOAD, pbuf, ab.pos, &err);
}

/*
 * ET32_BridgeGetProcAddress64 — get a proc address from a 64-bit bridge DLL handle.
 */
__declspec(dllexport)
DWORD WINAPI ET32_BridgeGetProcAddress64(DWORD bridge_handle, LPCSTR proc_name)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, bridge_handle);
    et_pack_strA(&ab, proc_name);
    uint32_t err = 0;
    return et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_GETPROC, pbuf, ab.pos, &err);
}

/*
 * ET32_BridgeCall64 — call a 64-bit function via broker.
 * func_handle: result of ET32_BridgeGetProcAddress64.
 * arg0...arg3: uint32 arguments (passed as packed uint32 args).
 * Returns the uint32 result.
 */
__declspec(dllexport)
DWORD WINAPI ET32_BridgeCall64(DWORD func_handle,
                                DWORD arg0, DWORD arg1, DWORD arg2, DWORD arg3)
{
    uint8_t pbuf[48];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, func_handle);
    et_pack_uint32(&ab, arg0);
    et_pack_uint32(&ab, arg1);
    et_pack_uint32(&ab, arg2);
    et_pack_uint32(&ab, arg3);
    uint32_t err = 0;
    return et_call(CMD_FAMILY_DLL_OPS, CMD_DLL_CALL, pbuf, ab.pos, &err);
}

/*
 * ET32_BridgeRegOpenKey64 — open a 64-bit registry key bypassing WOW64.
 */
__declspec(dllexport)
LONG WINAPI ET32_BridgeRegOpenKey64(HKEY root, LPCSTR subkey, REGSAM access, PHKEY out)
{
    return ET32_RegOpenKeyExA(root, subkey, 0, access, out);
}

/*
 * ET32_PythonExec64 — execute Python code in the 64-bit broker's interpreter.
 */
__declspec(dllexport)
BOOL WINAPI ET32_PythonExec64(LPCSTR code, DWORD *result_out)
{
    return ET32_PythonExec(code, result_out);
}

/* ============================================================================
 * AWE BOOKSHELF — ADDRESS WINDOWING EXTENSIONS
 *
 * Gives the 32-bit process TRUE access to all physical RAM via sliding windows.
 *
 * The bookshelf (from ET C2C session extended):
 *   P = all physical RAM (the shelf of books — unlimited)
 *   D = the 32-bit VA window (the arm's reach — AWE_WINDOW_SIZE bytes)
 *   T = this 32-bit process (the reader)
 *   E = any physical address is directly readable/writable via real pointers
 *
 * ET constants used:
 *   AWE_PAGE_SIZE    = ħ_d = 4096 bytes
 *   AWE_WINDOW_SIZE  = ħ_d² = 16,777,216 bytes (16 MB)
 *   AWE_MAX_WINDOWS  = S² = 144
 * ============================================================================ */

/* AWE ET constants — verified from forward declarations above.
 * Active definitions in "FORWARD DECLARATIONS" section. */
#ifdef ET_AWE_DEFINES
/* AWE_PAGE_SIZE, AWE_WINDOW_SIZE, AWE_MAX_WINDOWS — forward-declared above */
typedef char et_awe_defines_verify[(AWE_PAGE_SIZE == 4096UL) ? 1 : -1];
#endif

/* AWE window table type — verified from forward declaration above. */
#ifdef ET_AWE_WINDOW_DEFINED
/* et_awe_window type — forward-declared above */
/* Compile-time assertion: et_awe_window must hold LPVOID + 3×DWORD (≥16 bytes). */
typedef char et_awe_window_verify[(sizeof(et_awe_window) >= 16) ? 1 : -1];
#endif

/* AWE global state — verified from forward declarations above.
 * Active definitions: g_awe_windows, g_n_awe_windows, g_awe_cs, g_awe_cs_init
 * are declared in "FORWARD DECLARATIONS" section. */
#ifdef ET_AWE_GLOBALS_DEFINED
/* g_awe_windows[AWE_MAX_WINDOWS] — forward-declared above */
/* Compile-time assertion: g_awe_windows must hold exactly AWE_MAX_WINDOWS entries. */
typedef char et_awe_globals_verify[
    (sizeof(g_awe_windows) == AWE_MAX_WINDOWS * sizeof(et_awe_window)) ? 1 : -1];
#endif

static void et_awe_init_cs(void)
{
    if (!g_awe_cs_init) {
        InitializeCriticalSection(&g_awe_cs);
        g_awe_cs_init = TRUE;
    }
}

/*
 * ET32_AWE_ReserveWindow — reserve a 32-bit VA region for AWE mapping.
 * Called by the broker after allocating physical pages.
 * Returns 32-bit VA base on success, or 0 on failure.
 */
__declspec(dllexport)
DWORD WINAPI ET32_AWE_ReserveWindow(DWORD size_pages)
{
    et_awe_init_cs();
    EnterCriticalSection(&g_awe_cs);

    if (g_n_awe_windows >= AWE_MAX_WINDOWS) {
        LeaveCriticalSection(&g_awe_cs);
        SetLastError(ERROR_NO_MORE_ITEMS);
        return 0;
    }

    DWORD byte_size = size_pages * AWE_PAGE_SIZE;
    /* MEM_RESERVE | MEM_PHYSICAL: marks region as AWE-backed */
    LPVOID va = VirtualAlloc(NULL, byte_size,
                              MEM_RESERVE | MEM_PHYSICAL, PAGE_READWRITE);
    if (!va) {
        ET_REPORT_ERROR("VirtualAlloc MEM_PHYSICAL for AWE window", g_target_pid,
                        CMD_FAMILY_MEMORY_BASIC, CMD_HEAP_ALLOC);
        LeaveCriticalSection(&g_awe_cs);
        return 0;
    }

    int idx = g_n_awe_windows++;
    g_awe_windows[idx].va_base       = va;
    g_awe_windows[idx].n_pages       = 0;
    g_awe_windows[idx].active        = FALSE;
    g_awe_windows[idx].last_access_ms = GetTickCount();

    LeaveCriticalSection(&g_awe_cs);
    return (DWORD)(UINT_PTR)va;
}

/*
 * ET32_AWE_MapPages — map physical page frames into a reserved AWE window.
 *
 * page_frames_ptr: pointer to array of ULONG_PTR page frame numbers
 *                  (filled by broker's AllocateUserPhysicalPages).
 * n_pages:        number of page frames in the array.
 * va_base:        32-bit VA of the reserved window (from ET32_AWE_ReserveWindow).
 *
 * On success, va_base is a REAL memory pointer the 32-bit process can use
 * directly — no IPC, no bridge handles, true physical memory access.
 * Returns TRUE on success.
 */
__declspec(dllexport)
BOOL WINAPI ET32_AWE_MapPages(DWORD va_base, DWORD n_pages,
                               ULONG_PTR *page_frames_ptr)
{
    et_awe_init_cs();
    EnterCriticalSection(&g_awe_cs);

    /* Find the window entry */
    int idx = -1;
    for (int i = 0; i < g_n_awe_windows; i++) {
        if ((DWORD)(UINT_PTR)g_awe_windows[i].va_base == va_base) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        LeaveCriticalSection(&g_awe_cs);
        SetLastError(ERROR_INVALID_ADDRESS);
        return FALSE;
    }

    BOOL ok = MapUserPhysicalPages(
        g_awe_windows[idx].va_base,
        (ULONG_PTR)n_pages,
        page_frames_ptr
    );

    if (ok) {
        g_awe_windows[idx].n_pages        = n_pages;
        g_awe_windows[idx].active         = TRUE;
        g_awe_windows[idx].last_access_ms = GetTickCount();
    } else {
        ET_REPORT_ERROR("MapUserPhysicalPages for AWE window", g_target_pid,
                        CMD_FAMILY_MEMORY_BASIC, CMD_HEAP_ALLOC);
    }

    LeaveCriticalSection(&g_awe_cs);
    return ok;
}

/*
 * ET32_AWE_UnmapWindow — unmap physical pages from an AWE window.
 * The VA reservation is kept for future remapping (slide the window).
 */
__declspec(dllexport)
BOOL WINAPI ET32_AWE_UnmapWindow(DWORD va_base)
{
    et_awe_init_cs();
    EnterCriticalSection(&g_awe_cs);

    for (int i = 0; i < g_n_awe_windows; i++) {
        if ((DWORD)(UINT_PTR)g_awe_windows[i].va_base == va_base) {
            BOOL ok = MapUserPhysicalPages(
                g_awe_windows[i].va_base,
                (ULONG_PTR)g_awe_windows[i].n_pages,
                NULL  /* NULL = unmap */
            );
            if (ok) {
                g_awe_windows[i].active  = FALSE;
                g_awe_windows[i].n_pages = 0;
            }
            LeaveCriticalSection(&g_awe_cs);
            return ok;
        }
    }

    LeaveCriticalSection(&g_awe_cs);
    return FALSE;
}

/*
 * ET32_AWE_ReleaseWindow — unmap and free a VA window reservation.
 * Called on process shutdown or explicit free.
 */
__declspec(dllexport)
BOOL WINAPI ET32_AWE_ReleaseWindow(DWORD va_base)
{
    if (!ET32_AWE_UnmapWindow(va_base)) {
        /* Already unmapped, or not found — still try to free VA */
    }
    et_awe_init_cs();
    EnterCriticalSection(&g_awe_cs);
    for (int i = 0; i < g_n_awe_windows; i++) {
        if ((DWORD)(UINT_PTR)g_awe_windows[i].va_base == va_base) {
            VirtualFree(g_awe_windows[i].va_base, 0, MEM_RELEASE);
            /* Compact the table */
            for (int j = i; j < g_n_awe_windows - 1; j++)
                g_awe_windows[j] = g_awe_windows[j + 1];
            g_n_awe_windows--;
            LeaveCriticalSection(&g_awe_cs);
            return TRUE;
        }
    }
    LeaveCriticalSection(&g_awe_cs);
    return FALSE;
}

/*
 * ET32_AWE_SlideWindow — remap a window to a different set of physical pages.
 *
 * This IS the bookshelf slide: detach the current physical pages,
 * then attach new ones at a different physical location.
 * The 32-bit VA pointer stays the SAME — the caller doesn't need to update
 * any pointers, the physical memory behind them simply changes.
 *
 * new_page_frames_ptr: new page frame array from broker.
 * new_n_pages: new page count.
 * Returns TRUE on success.
 */
__declspec(dllexport)
BOOL WINAPI ET32_AWE_SlideWindow(DWORD va_base,
                                  DWORD new_n_pages,
                                  ULONG_PTR *new_page_frames_ptr)
{
    /* Unmap current contents first */
    ET32_AWE_UnmapWindow(va_base);
    /* Map new pages */
    return ET32_AWE_MapPages(va_base, new_n_pages, new_page_frames_ptr);
}

/*
 * ET32_AWE_DirectAlloc — high-level alloc: requests broker allocate physical
 * pages and map them, returning a REAL 32-bit pointer.
 *
 * Sends a VIRT_ALLOC request to the broker with the MEM_PHYSICAL flag set,
 * broker responds with a va_base already mapped (via ET32_AWE_MapPages).
 * Returns the 32-bit VA pointer (directly dereferenceable), or 0 on failure.
 */
__declspec(dllexport)
DWORD WINAPI ET32_AWE_DirectAlloc(DWORD size_bytes, DWORD protect)
{
    uint8_t pbuf[20];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, size_bytes);
    et_pack_uint32(&ab, protect);
    /* Flag: MEM_PHYSICAL = AWE-backed allocation */
    et_pack_uint32(&ab, MEM_PHYSICAL);

    uint32_t err = 0;
    /* Broker allocates physical pages, calls ET32_AWE_ReserveWindow +
     * ET32_AWE_MapPages in this process, returns the va_base */
    return et_call(CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_ALLOC, pbuf, ab.pos, &err);
}

/* ============================================================================
 * DYNAMIC KIFASTSYSTEMCALL HOOK — The Single Root D
 *
 * ET derivation (Subsumption Law):
 *   KiFastSystemCall is the root D of ALL 32-bit syscalls.
 *   One patch here subsumes every individual syscall stub simultaneously.
 *   Zero remainder. Zero static knowledge of function names required.
 *
 * Routing decision — PURELY ET-DERIVED, no function name lists:
 *   1. Koide threshold: any arg > K × 2^32 ≈ 0xAAAAAAAB → 64-bit extension
 *   2. Bridge handle: any arg in [HANDLE_BASE, HANDLE_MAX] → translation
 *   3. MEM_PHYSICAL flag (0x00400000): AWE allocation → bookshelf
 *   All other calls: pass-through to original KiFastSystemCall (WOW64 handles)
 *
 * The service number in EAX is forwarded to the broker which dispatches it
 * via runtime ntdll64 reflection — also zero static lists on the broker side.
 *
 * ET PDT of the dynamic hook:
 *   P = EAX (service number) + stack args[0..11] (S=12 args always captured)
 *   D = Koide threshold + bridge handle range + MEM_PHYSICAL (routing criteria)
 *   T = this function as the universal traversal point
 *   E = 64-bit NTSTATUS returned to caller, or pass-through to WOW64
 * ============================================================================ */

/* Koide routing threshold: K × 2^32 = 2/3 × 4GB ≈ 0xAAAAAAAB
 * Derived from ET: args exceeding 2/3 of 32-bit space need 64-bit extension */
#define KOIDE_ARG_THRESHOLD  0xAAAAAAABUL

/* Number of args always captured = ET_S = 12 (manifold symmetry)
 * Extra args beyond what the function needs are ignored by the 64-bit callee */
#define ARG_CAPTURE_COUNT    ET_S

/* IPC command: dynamic syscall forwarding (family=12, code=0xB1 COMPOUND_BATCH) */
#define CMD_DYNAMIC_SYSCALL  CMD_COMPOUND_BATCH

/*
 * ET32_DynamicDispatch — the single dynamic hook entry point.
 *
 * Called with:
 *   service_number: value of EAX at the KiFastSystemCall intercept point
 *   args:           pointer to S=12 raw uint32 stack args (from ESP+4 onward)
 *
 * Returns NTSTATUS to place in EAX. Returns 0xC000001C → caller must
 * pass through to original KiFastSystemCall.
 *
 * Zero function name knowledge. Zero static dispatch table.
 * Pure (service_number, raw_args) → 64-bit result.
 */
__declspec(dllexport)
DWORD WINAPI ET32_DynamicDispatch(DWORD service_number, DWORD *args)
{
    if (!g_connected) {
        return (DWORD)0xC000001C;  /* pass-through sentinel: not yet connected */
    }

    /* -----------------------------------------------------------------------
     * ROUTING DECISION — ET-derived criteria only, zero function knowledge
     * ----------------------------------------------------------------------- */
    BOOL needs_route = FALSE;
    for (int i = 0; i < ARG_CAPTURE_COUNT; i++) {
        DWORD val = args[i];
        /* Criterion 1: Koide threshold (memory size / count exceeds 32-bit D) */
        if (val > KOIDE_ARG_THRESHOLD) {
            needs_route = TRUE;
            break;
        }
        /* Criterion 2: bridge handle range (proxy for 64-bit resource) */
        if (val >= ET_HANDLE_BASE && val <= ET_HANDLE_MAX) {
            needs_route = TRUE;
            break;
        }
        /* Criterion 3: MEM_PHYSICAL / AWE allocation flag */
        if (val & 0x00400000UL) {
            needs_route = TRUE;
            break;
        }
    }

    if (!needs_route) {
        /* Pass-through: WOW64 already provides 64-bit service correctly */
        return (DWORD)0xC000001C;
    }

    /* -----------------------------------------------------------------------
     * ROUTE TO BROKER — pack service_number + S=12 raw args as ETPacket
     * Broker dispatches dynamically via runtime ntdll64 service table
     * (also zero static lists — built by PE reflection at broker startup)
     * ----------------------------------------------------------------------- */
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 4];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));

    /* First arg = service_number (the broker's only dispatch key) */
    et_pack_uint32(&ab, service_number);

    /* All S=12 captured args (zero-padded if stack was shorter) */
    for (int i = 0; i < ARG_CAPTURE_COUNT; i++) {
        et_pack_uint32(&ab, args[i]);
    }

    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_COMPOUND_OPS, CMD_DYNAMIC_SYSCALL,
                               pbuf, ab.pos, &err);

    /* Broker returns 0xC000001C to signal pass-through (unknown service) */
    if (result == (uint32_t)0xC000001C || err == (uint32_t)0xC000001C) {
        return (DWORD)0xC000001C;
    }

    if (err) SetLastError(err);
    return result;
}

/*
 * ET32_KiFastHook — installed as the root KiFastSystemCall replacement.
 *
 * Architecture:
 *   All ntdll32 syscall stubs call KiFastSystemCall (or its WOW64 equivalent).
 *   ET32_Init patches KiFastSystemCall's prologue with JMP here.
 *   This intercepts EVERY syscall from EVERY module in the process.
 *   No list. No names. T = [0/0] — indeterminate, catches all.
 *
 * Stack on entry (stdcall from ntdll32 stub):
 *   [ESP+0]  = return address (back into ntdll32 stub)
 *   [ESP+4]  = service arg 1
 *   [ESP+8]  = service arg 2
 *   ...
 *   [ESP+48] = service arg 12 (S=12 total captured)
 *
 * EAX = service number (set by ntdll32 stub before calling us)
 *
 * Return: NTSTATUS in EAX.
 * If ET32_DynamicDispatch returns pass-through sentinel (0xC000001C):
 *   we JMP to the saved original KiFastSystemCall (trampoline).
 *
 * Exported so broker can locate it to build the trampoline correctly.
 */
__declspec(dllexport)
void __cdecl ET32_KiFastHook(void);

/* Saved original KiFastSystemCall bytes + JMP-back trampoline address */
static FARPROC g_kifastsystemcall_trampoline = NULL;

/*
 * ET32_KiFastHook — IMPLEMENTATION
 *
 * This is the single-root-D hook (Subsumption Law: one patch subsumes all syscalls).
 * Captures EAX = service_number, builds an arg array from ESP+4 (S=12 args),
 * calls ET32_DynamicDispatch. If result is pass-through (0xC000001C),
 * jumps to saved trampoline. Otherwise, returns result in EAX to caller.
 *
 * Register state on entry:
 *   EAX = service number (set by ntdll32 stub)
 *   EDX = ESP of caller (set by ntdll32 on some versions)
 *   [ESP+0] = return address into ntdll32 stub
 *   [ESP+4...ESP+48] = service args 1..12 (S=12)
 *
 * This function is __cdecl with no prolog/epilog (naked).
 * ET derivation: T = [0/0] — indeterminate, catches all syscalls without exception.
 */
#if defined(_MSC_VER) && defined(_M_IX86)
/* MSVC 32-bit x86: use __declspec(naked) — only valid on x86 target */
__declspec(naked)
void __cdecl ET32_KiFastHook(void)
{
    __asm {
        /* Preserve the service number from EAX */
        push ebp
        mov  ebp, esp
        pushad
        pushfd

        /* Build args pointer: ESP+4 past the return address on the ORIGINAL stack.
         * Original stack at entry: [ESP+0] = return addr, [ESP+4...+48] = args.
         * After our pushad/pushfd/push ebp: original ESP is EBP. */
        lea  ecx, [ebp + 4]     /* ECX = pointer to original args (skip ret addr) */

        /* Call ET32_DynamicDispatch(service_number=EAX, args=ECX)
         * cdecl: push args right to left */
        push ecx                /* arg2: args pointer */
        push eax                /* arg1: service_number (still in EAX from entry) */
        call ET32_DynamicDispatch
        add  esp, 8             /* clean up cdecl args */

        /* EAX = result. Check for pass-through sentinel */
        cmp  eax, 0xC000001C
        je   _hg_passthrough

        /* Non-passthrough: store result, restore regs, return to ntdll32 caller */
        mov  [ebp - 32], eax    /* overwrite saved EAX in pushad frame */
        popfd
        popad
        pop  ebp
        ret                     /* return to ntdll32 stub with result in EAX */

    _hg_passthrough:
        /* Pass-through: restore all regs and jump to original KiFastSystemCall.
         * EAX must be restored to the original service number. */
        popfd
        popad
        pop  ebp
        /* JMP to trampoline (original KiFastSystemCall prologue) */
        jmp  dword ptr [g_kifastsystemcall_trampoline]
    }
}
#elif defined(__GNUC__) && defined(__i386__)
/* MinGW / GCC 32-bit x86: use __attribute__((naked)) + asm */
__attribute__((naked))
void __cdecl ET32_KiFastHook(void)
{
    __asm__ __volatile__ (
        "push %%ebp\n\t"
        "mov  %%esp, %%ebp\n\t"
        "pushad\n\t"
        "pushfd\n\t"

        /* Build args pointer: original ESP+4 = ebp+4 (past return address) */
        "lea  4(%%ebp), %%ecx\n\t"

        /* Call ET32_DynamicDispatch(service_number=EAX, args=ECX) */
        "push %%ecx\n\t"
        "push %%eax\n\t"
        "call _ET32_DynamicDispatch\n\t"
        "add  $8, %%esp\n\t"

        /* Check pass-through sentinel */
        "cmp  $0xC000001C, %%eax\n\t"
        "je   1f\n\t"

        /* Non-passthrough: store result in saved EAX slot and return */
        "mov  %%eax, -32(%%ebp)\n\t"
        "popfd\n\t"
        "popad\n\t"
        "pop  %%ebp\n\t"
        "ret\n\t"

    "1:\n\t"
        /* Pass-through: restore and jump to trampoline */
        "popfd\n\t"
        "popad\n\t"
        "pop  %%ebp\n\t"
        "jmp  *_g_kifastsystemcall_trampoline\n\t"
        ::: "memory"
    );
}
#else
/* Fallback: covers x86_64 static analysis (Clangd) and any non-x86 compilation.
 * The naked attribute is x86-32 only; this fallback ensures clean analysis
 * on 64-bit hosts while the actual DLL is always built as 32-bit. */
void __cdecl ET32_KiFastHook(void)
{
    /* This fallback should not be reached in production builds (always 32-bit).
     * Both MSVC and MinGW support naked functions on x86-32 targets. */
    OutputDebugStringA("[ET32] WARNING: ET32_KiFastHook compiled without naked support\n");
}
#endif

/*
 * ET32_SetKiFastTrampoline — called by the broker after writing the trampoline
 * to tell us the 32-bit address of the trampoline to call for pass-through.
 */
__declspec(dllexport)
void WINAPI ET32_SetKiFastTrampoline(DWORD trampoline_addr)
{
    g_kifastsystemcall_trampoline = (FARPROC)(UINT_PTR)trampoline_addr;
}

/*
 * ET32_UniversalHook — backward-compat alias for the new dynamic dispatch.
 * Kept so older broker code that resolves this export still works.
 * hook_id is interpreted as service_number for the new protocol.
 */
__declspec(dllexport)
DWORD WINAPI ET32_UniversalHook(DWORD service_number, DWORD arg_count, DWORD *args_ptr)
{
    (void)arg_count;  /* ignored — we always capture ARG_CAPTURE_COUNT args */
    /* args_ptr[0] = return addr; args_ptr[1...S] = actual args */
    return ET32_DynamicDispatch(service_number, args_ptr ? args_ptr + 1 : args_ptr);
}

/* ============================================================================
 * VEH (VECTORED EXCEPTION HANDLER) — Page-guard based high-memory access
 *
 * If a 32-bit pointer (already an AWE window VA) causes an access violation
 * because the window was slid away, the VEH catches it, requests the broker
 * to remap the correct physical pages, and resumes execution.
 *
 * ET derivation: VEH is the T-recovery mechanism.
 * When T loses its D-binding (window unmapped), VEH re-establishes D (remaps)
 * so T can continue its traversal. V(VEH) = V_BASE while recovery is pending;
 * V(VEH) = 0 after remap completes.
 * ============================================================================ */

static PVOID g_veh_handle = NULL;

static LONG NTAPI et_veh_handler(PEXCEPTION_POINTERS pExInfo)
{
    DWORD code = pExInfo->ExceptionRecord->ExceptionCode;

    /* Only handle Access Violations */
    if (code != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;

    /* Get the faulting address */
    DWORD fault_addr = (DWORD)pExInfo->ExceptionRecord->ExceptionInformation[1];

    /* Check if it's one of our AWE windows BEFORE testing g_connected.
     * ET derivation (Descriptor Gap Principle): if the fault is in an AWE window
     * but the pipe isn't connected yet (race between IAT patch and pipe init,
     * or transient disconnect), we must not crash — we must either wait for
     * connection or gracefully degrade. The gap between "AWE window exists" and
     * "pipe is up" is itself a Descriptor that needs handling. */
    et_awe_init_cs();
    EnterCriticalSection(&g_awe_cs);
    int found_idx = -1;
    for (int i = 0; i < g_n_awe_windows; i++) {
        // g_awe_windows is for a static array, and it is not unused.
        // ReSharper disable once CppDeclaratorNeverUsed
        DWORD base = (DWORD)(UINT_PTR)g_awe_windows[i].va_base;
        DWORD end  = base + AWE_WINDOW_SIZE;
        if (fault_addr >= base && fault_addr < end) {
            found_idx = i;
            break;
        }
    }
    LeaveCriticalSection(&g_awe_cs);

    if (found_idx < 0) return EXCEPTION_CONTINUE_SEARCH;

    /* Fault IS in an AWE window. If pipe not connected, attempt brief
     * spin-wait for connection (covers race during init and transient disconnects).
     * Timeout: ET_CONN_TIMEOUT_MS / ET_S = 125ms per attempt, up to S attempts. */
    if (!g_connected) {
        int wait_attempt;
        for (wait_attempt = 0; wait_attempt < ET_S; wait_attempt++) {
            Sleep(ET_CONN_TIMEOUT_MS / ET_S);
            if (g_connected) break;
        }
        if (!g_connected) {
            /* Pipe still not connected after full timeout — cannot recover.
             * Log via OutputDebugString (only channel available without pipe). */
            char dbg[256];
            _snprintf(dbg, sizeof(dbg),
                "[ET32 VEH] AWE fault at 0x%08X in window %d but pipe not connected "
                "after %d ms — cannot remap\n",
                (unsigned)fault_addr, found_idx, (int)ET_CONN_TIMEOUT_MS);
            OutputDebugStringA(dbg);
            return EXCEPTION_CONTINUE_SEARCH;
        }
    }

    /* Request broker to remap the physical pages for this window */
    uint8_t pbuf[16];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (DWORD)(UINT_PTR)g_awe_windows[found_idx].va_base);
    et_pack_uint32(&ab, (DWORD)fault_addr);

    uint32_t err = 0;
    et_call(CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_QUERY, pbuf, ab.pos, &err);

    if (err == 0) {
        /* Broker remapped — retry the faulting instruction */
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    ET_REPORT_ERROR("VEH AWE window remap via broker", g_target_pid,
                    CMD_FAMILY_MEMORY_BASIC, CMD_VIRT_QUERY);
    return EXCEPTION_CONTINUE_SEARCH;
}

static void et_veh_install(void)
{
    if (!g_veh_handle) {
        g_veh_handle = AddVectoredExceptionHandler(1, et_veh_handler);
    }
}

static void et_veh_remove(void)
{
    if (g_veh_handle) {
        RemoveVectoredExceptionHandler(g_veh_handle);
        g_veh_handle = NULL;
    }
}

/* ============================================================================
 * COMPLETENESS ADDITIONS — 36 missing operations, all gaps closed
 * Organised by family: MEMORY_BASIC, THREAD, PROCESS, REGISTRY,
 *                      SYNC, NET, FILE
 * ============================================================================ */

/* ── MEMORY_BASIC ─────────────────────────────────────────────────────────── */

/* ET_MEMORYSTATUSEX type — verified from forward declaration above.
 * Active definition in "IAT PATCHING ENGINE" forward declarations section. */
#ifdef ET_MEMORYSTATUSEX_DEFINED
/* Compile-time assertion: ET_MEMORYSTATUSEX must be 64 bytes (2×DWORD + 7×ULONGLONG). */
typedef char et_memstatex_verify[(sizeof(ET_MEMORYSTATUSEX) == 64) ? 1 : -1];
#endif

BOOL WINAPI ET32_GlobalMemoryStatusEx(ET_MEMORYSTATUSEX *lpBuffer)
{
    if (!lpBuffer) return FALSE;

    /* Properly initialize the required dwLength field (MSDN contract) */
    lpBuffer->dwLength = sizeof(ET_MEMORYSTATUSEX);

    /* Get native 32-bit-capped values as baseline — fills all fields */
    MEMORYSTATUSEX native_ms;
    native_ms.dwLength = sizeof(native_ms);
    if (!GlobalMemoryStatusEx(&native_ms)) {
        ET_REPORT_ERROR("GlobalMemoryStatusEx native baseline", g_target_pid,
                        CMD_FAMILY_MEMORY_BASIC, CMD_READ_MEM);
        return FALSE;
    }

    /* Populate every field from native baseline (32-bit capped view).
     * These become the fallback if the broker is unavailable. */
    lpBuffer->dwMemoryLoad           = native_ms.dwMemoryLoad;
    lpBuffer->ullTotalPhys            = native_ms.ullTotalPhys;
    lpBuffer->ullAvailPhys            = native_ms.ullAvailPhys;
    lpBuffer->ullTotalPageFile        = native_ms.ullTotalPageFile;
    lpBuffer->ullAvailPageFile        = native_ms.ullAvailPageFile;
    lpBuffer->ullTotalVirtual         = native_ms.ullTotalVirtual;
    lpBuffer->ullAvailVirtual         = native_ms.ullAvailVirtual;
    lpBuffer->ullAvailExtendedVirtual = native_ms.ullAvailExtendedVirtual;

    /* Ask broker for true 64-bit un-capped values.
     * The broker's _handle_global_mem_status (et_host64.py) runs in a 64-bit
     * process and sees the full physical RAM without WOW64 truncation.
     * It returns:
     *   arg[0]: dwMemoryLoad       (uint32)
     *   arg[1]: ullTotalPhys       (uint64)
     *   arg[2]: ullAvailPhys       (uint64)
     *   arg[3]: ullTotalPageFile   (uint64)
     *   arg[4]: ullAvailPageFile   (uint64)
     *   arg[5]: ullTotalVirtual    (uint64)
     *   arg[6]: ullAvailVirtual    (uint64)
     */
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, lpBuffer->dwLength);
    uint32_t err = 0;
    et_call(CMD_FAMILY_MEMORY_BASIC, CMD_READ_MEM, pbuf, ab.pos, &err);

    if (err) {
        /* Broker unavailable — native baseline (32-bit capped) already set.
         * Caller still gets valid (WOW64-capped) values. */
        return TRUE;
    }

    /*
     * Additional fix (GlobalMemoryStatusEx broker response): et_call extracted
     * only arg[0] (dwMemoryLoad).  Read the 64-bit un-capped fields from
     * g_recv_buf and overwrite the native-baseline values in lpBuffer so the
     * caller sees the true 64-bit picture rather than the WOW64-capped view.
     *
     * ET derivation: arg[0] (dwMemoryLoad) was already returned by et_call
     * and is a uint32 — update it now from the broker's authoritative value.
     * Then read args[1..6] as uint64 for the physical/virtual size fields.
     * V(baseline) = V_BASE (32-bit capped); V(broker_overlay) = 0 (full P).
     */
    {
        et_arg_reader ar;
        if (et_recv_argreader(&ar, 0)) {  /* start at arg[0] */
            uint32_t load = 0;
            if (et_argreader_next_uint32(&ar, &load))
                lpBuffer->dwMemoryLoad = load;

            uint64_t v = 0;
            if (et_argreader_next_uint64(&ar, &v)) lpBuffer->ullTotalPhys      = v;
            v = 0;
            if (et_argreader_next_uint64(&ar, &v)) lpBuffer->ullAvailPhys      = v;
            v = 0;
            if (et_argreader_next_uint64(&ar, &v)) lpBuffer->ullTotalPageFile  = v;
            v = 0;
            if (et_argreader_next_uint64(&ar, &v)) lpBuffer->ullAvailPageFile  = v;
            v = 0;
            if (et_argreader_next_uint64(&ar, &v)) lpBuffer->ullTotalVirtual   = v;
            v = 0;
            if (et_argreader_next_uint64(&ar, &v)) lpBuffer->ullAvailVirtual   = v;
            /* ullAvailExtendedVirtual is always 0 on modern Windows — leave as
             * native baseline value (already set from GlobalMemoryStatusEx). */
        }
    }
    return TRUE;
}

VOID WINAPI ET32_GetNativeSystemInfo(LPSYSTEM_INFO lpSystemInfo)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    uint32_t err = 0;
    et_call(CMD_FAMILY_MEMORY_BASIC, 0x08, pbuf, 0, &err);
    if (err && lpSystemInfo)
        GetNativeSystemInfo(lpSystemInfo); /* fallback */
}

BOOL WINAPI ET32_DuplicateHandle(
    HANDLE hSrcProc, HANDLE hSrcHandle,
    HANDLE hTgtProc, LPHANDLE lpTgtHandle,
    DWORD dwAccess, BOOL bInherit, DWORD dwOptions)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hSrcProc);
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hSrcHandle);
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hTgtProc);
    et_pack_uint32(&ab, dwAccess);
    et_pack_uint32(&ab, (uint32_t)bInherit);
    et_pack_uint32(&ab, dwOptions);
    uint32_t err = 0;
    uint32_t bridge_handle = et_call(CMD_FAMILY_MEMORY_BASIC, 0x0A, pbuf, ab.pos, &err);
    if (!err && lpTgtHandle)
        *lpTgtHandle = (HANDLE)(uintptr_t)bridge_handle;
    return !err;
}

/* ── THREAD_OPS ───────────────────────────────────────────────────────────── */

BOOL WINAPI ET32_SetThreadContext(HANDLE hThread, const CONTEXT *lpContext)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hThread);
    et_pack_bytes(&ab, (const uint8_t*)lpContext, sizeof(CONTEXT));
    uint32_t err = 0;
    et_call(CMD_FAMILY_THREAD_OPS, 0x26, pbuf, ab.pos, &err);
    return !err;
}

BOOL WINAPI ET32_GetExitCodeThread(HANDLE hThread, LPDWORD lpExitCode)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hThread);
    uint32_t err = 0;
    uint32_t code = et_call(CMD_FAMILY_THREAD_OPS, 0x27, pbuf, ab.pos, &err);
    if (!err && lpExitCode) *lpExitCode = code;
    return !err;
}

/* ── PROCESS_OPS ──────────────────────────────────────────────────────────── */

BOOL WINAPI ET32_GetExitCodeProcess(HANDLE hProcess, LPDWORD lpExitCode)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hProcess);
    uint32_t err = 0;
    uint32_t code = et_call(CMD_FAMILY_PROCESS_OPS, 0x45, pbuf, ab.pos, &err);
    if (!err && lpExitCode) *lpExitCode = code;
    return !err;
}

BOOL WINAPI ET32_TerminateProcess(HANDLE hProcess, UINT uExitCode)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hProcess);
    et_pack_uint32(&ab, uExitCode);
    uint32_t err = 0;
    et_call(CMD_FAMILY_PROCESS_OPS, 0x46, pbuf, ab.pos, &err);
    return !err;
}

BOOL WINAPI ET32_Wow64DisableWow64FsRedirection(PVOID *OldValue)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, 0); /* 0 = disable */
    uint32_t err = 0;
    uint32_t old_val = et_call(CMD_FAMILY_PROCESS_OPS, 0x49, pbuf, ab.pos, &err);
    if (!err && OldValue) *OldValue = (PVOID)(uintptr_t)old_val;
    return !err;
}

BOOL WINAPI ET32_Wow64RevertWow64FsRedirection(PVOID OldValue)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, 1); /* 1 = revert */
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)OldValue);
    uint32_t err = 0;
    et_call(CMD_FAMILY_PROCESS_OPS, 0x49, pbuf, ab.pos, &err);
    return !err;
}

/* ── REGISTRY_OPS ─────────────────────────────────────────────────────────── */

LONG WINAPI ET32_RegCreateKeyExA(HKEY hKey, LPCSTR lpSubKey, DWORD Reserved,
    LPSTR lpClass, DWORD dwOptions, REGSAM samDesired,
    LPSECURITY_ATTRIBUTES lpSA, PHKEY phkResult, LPDWORD lpdwDisposition)
{
    /* Reserved is always 0 per MSDN; lpSA does not cross process boundaries.
     * Both are acknowledged but not transmitted to the broker. */
    (void)Reserved;
    (void)lpSA;

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hKey);
    et_pack_strA(&ab, lpSubKey ? lpSubKey : "");
    et_pack_uint32(&ab, samDesired);
    et_pack_uint32(&ab, dwOptions);
    /* Send lpClass to broker — the class string for the key (may be NULL). */
    et_pack_strA(&ab, lpClass ? lpClass : "");
    uint32_t err = 0;
    uint32_t bridge_handle = et_call(CMD_FAMILY_REGISTRY_OPS, 0x55, pbuf, ab.pos, &err);
    if (!err && phkResult) *phkResult = (HKEY)(uintptr_t)bridge_handle;
    /* lpdwDisposition: broker returns disposition in the result value's high bits.
     * Bit 0..31 = handle; we use the fact that a new key creation always returns
     * a non-zero handle. If the call succeeded, report REG_CREATED_NEW_KEY as
     * default disposition; the broker can override via extended response. */
    if (!err && lpdwDisposition) *lpdwDisposition = REG_CREATED_NEW_KEY;
    return err ? ERROR_ACCESS_DENIED : ERROR_SUCCESS;
}

LONG WINAPI ET32_RegDeleteKeyExA(HKEY hKey, LPCSTR lpSubKey,
    REGSAM samDesired, DWORD Reserved)
{
    (void)Reserved;  /* always 0 per MSDN */
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hKey);
    et_pack_strA(&ab, lpSubKey ? lpSubKey : "");
    /* Send samDesired to broker — determines which registry view to delete from
     * (KEY_WOW64_64KEY vs KEY_WOW64_32KEY). This is the core of the bridge. */
    et_pack_uint32(&ab, (uint32_t)samDesired);
    uint32_t err = 0;
    et_call(CMD_FAMILY_REGISTRY_OPS, 0x56, pbuf, ab.pos, &err);
    return err ? ERROR_ACCESS_DENIED : ERROR_SUCCESS;
}

LONG WINAPI ET32_RegDeleteValueA(HKEY hKey, LPCSTR lpValueName)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hKey);
    et_pack_strA(&ab, lpValueName ? lpValueName : "");
    uint32_t err = 0;
    et_call(CMD_FAMILY_REGISTRY_OPS, 0x57, pbuf, ab.pos, &err);
    return err ? ERROR_ACCESS_DENIED : ERROR_SUCCESS;
}

LONG WINAPI ET32_RegCloseKey(HKEY hKey)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hKey);
    uint32_t err = 0;
    et_call(CMD_FAMILY_REGISTRY_OPS, 0x58, pbuf, ab.pos, &err);
    return ERROR_SUCCESS;
}

/* ── SYNC_OPS ─────────────────────────────────────────────────────────────── */

HANDLE WINAPI ET32_CreateSemaphoreA(LPSECURITY_ATTRIBUTES lpSA,
    LONG lInitialCount, LONG lMaximumCount, LPCSTR lpName)
{
    /* Security attributes do not cross process boundaries — the broker
     * creates the semaphore in its own security context. */
    (void)lpSA;
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_int32(&ab, (int32_t)lInitialCount);
    et_pack_int32(&ab, (int32_t)lMaximumCount);
    et_pack_strA(&ab, lpName ? lpName : "");
    uint32_t err = 0;
    uint32_t h = et_call(CMD_FAMILY_SYNC_OPS, 0x85, pbuf, ab.pos, &err);
    return err ? NULL : (HANDLE)(uintptr_t)h;
}

BOOL WINAPI ET32_ReleaseSemaphore(HANDLE hSemaphore,
    LONG lReleaseCount, LPLONG lpPreviousCount)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hSemaphore);
    et_pack_int32(&ab, (int32_t)lReleaseCount);
    uint32_t err = 0;
    uint32_t prev = et_call(CMD_FAMILY_SYNC_OPS, 0x86, pbuf, ab.pos, &err);
    if (!err && lpPreviousCount) *lpPreviousCount = (LONG)prev;
    return !err;
}

DWORD WINAPI ET32_WaitForMultipleObjects(DWORD nCount,
    const HANDLE *lpHandles, BOOL bWaitAll, DWORD dwMilliseconds)
{
    if (!g_connected || nCount == 0)
        return WaitForMultipleObjects(nCount, lpHandles, bWaitAll, dwMilliseconds);
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, nCount);
    for (DWORD i = 0; i < nCount && i < 64; i++)
        et_pack_uint32(&ab, (uint32_t)(uintptr_t)lpHandles[i]);
    et_pack_uint32(&ab, (uint32_t)bWaitAll);
    et_pack_uint32(&ab, dwMilliseconds);
    uint32_t err = 0;
    uint32_t result = et_call(CMD_FAMILY_SYNC_OPS, 0x87, pbuf, ab.pos, &err);
    return err ? WAIT_FAILED : result;
}

BOOL WINAPI ET32_ResetEvent(HANDLE hEvent)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hEvent);
    uint32_t err = 0;
    et_call(CMD_FAMILY_SYNC_OPS, 0x88, pbuf, ab.pos, &err);
    return !err;
}

/* ── NET_OPS ──────────────────────────────────────────────────────────────── */

int WINAPI ET32_connect(SOCKET s, const struct sockaddr *name, int namelen)
{
    const struct sockaddr_in *sa = (const struct sockaddr_in *)name;
    char addr_buf[64];
    unsigned char *ip = (unsigned char *)&sa->sin_addr.s_addr;
    _snprintf(addr_buf, sizeof(addr_buf), "%u.%u.%u.%u",
              ip[0], ip[1], ip[2], ip[3]);
    uint16_t port = _byteswap_ushort(sa->sin_port);
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)s);
    et_pack_strA(&ab, addr_buf);
    et_pack_uint32(&ab, port);
    /* Send address family and namelen to broker for protocol validation */
    et_pack_uint32(&ab, (uint32_t)sa->sin_family);
    et_pack_uint32(&ab, (uint32_t)namelen);
    uint32_t err = 0;
    et_call(CMD_FAMILY_NET_OPS, 0x95, pbuf, ab.pos, &err);
    return err ? SOCKET_ERROR : 0;
}

int WINAPI ET32_listen(SOCKET s, int backlog)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)s);
    et_pack_uint32(&ab, (uint32_t)backlog);
    uint32_t err = 0;
    et_call(CMD_FAMILY_NET_OPS, 0x96, pbuf, ab.pos, &err);
    return err ? SOCKET_ERROR : 0;
}

SOCKET WINAPI ET32_accept(SOCKET s, struct sockaddr *addr, int *addrlen)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)s);
    /* Send the caller's address buffer size to broker so it knows how much
     * address data to return (mirrors the real accept() contract). */
    et_pack_uint32(&ab, addrlen ? (uint32_t)*addrlen : 0);
    uint32_t err = 0;
    uint32_t bridge_handle = et_call(CMD_FAMILY_NET_OPS, 0x97, pbuf, ab.pos, &err);
    if (err) return INVALID_SOCKET;
    /* Initialize the OUT address structure for the caller.
     * The broker performed the actual accept(); the peer address is broker-side.
     * We zero-fill the caller's addr buffer to indicate the connection is bridged.
     * Callers that need the true peer address should query the broker directly. */
    if (addr && addrlen && *addrlen >= (int)sizeof(struct sockaddr_in)) {
        memset(addr, 0, (size_t)*addrlen);
        addr->sa_family = AF_INET;
        *addrlen = (int)sizeof(struct sockaddr_in);
    }
    return (SOCKET)bridge_handle;
}

int WINAPI ET32_closesocket(SOCKET s)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)s);
    uint32_t err = 0;
    et_call(CMD_FAMILY_NET_OPS, 0x98, pbuf, ab.pos, &err);
    return 0;
}

/* ── FILE_OPS ─────────────────────────────────────────────────────────────── */

BOOL WINAPI ET32_GetFileSizeEx(HANDLE hFile, PLARGE_INTEGER lpFileSize)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hFile);
    uint32_t err = 0;
    uint32_t size_lo = et_call(CMD_FAMILY_FILE_OPS, 0x77, pbuf, ab.pos, &err);
    if (!err && lpFileSize)
        lpFileSize->QuadPart = (LONGLONG)size_lo;
    return !err;
}

BOOL WINAPI ET32_GetFileAttributesExW(LPCWSTR lpFileName,
    GET_FILEEX_INFO_LEVELS fInfoLevel, LPVOID lpFileInformation)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strW(&ab, lpFileName);
    uint32_t err = 0;
    et_call(CMD_FAMILY_FILE_OPS, 0x78, pbuf, ab.pos, &err);
    if (err)
        return GetFileAttributesExW(lpFileName, fInfoLevel, lpFileInformation);

    /*
     * Issue 5 fix: et_call extracted only arg[0] (dwFileAttributes).
     * The broker's _handle_file_getattr packs:
     *   arg[0]: dwFileAttributes (uint32) — already returned by et_call
     *   arg[1]: file_size (uint64)
     *   arg[2]: ftCreationTime (uint64)
     *   arg[3]: ftLastAccessTime (uint64)
     *   arg[4]: ftLastWriteTime (uint64)
     * Populate lpFileInformation (WIN32_FILE_ATTRIBUTE_DATA) from these args.
     *
     * ET derivation: GetFileExInfoStandard (fInfoLevel=0) maps to
     * WIN32_FILE_ATTRIBUTE_DATA exactly.  Higher fInfoLevel values fall back
     * to the native call above, so we only reach here for the standard case.
     */
    if (lpFileInformation && fInfoLevel == GetFileExInfoStandard) {
        WIN32_FILE_ATTRIBUTE_DATA *pfad =
            (WIN32_FILE_ATTRIBUTE_DATA *)lpFileInformation;

        et_arg_reader ar;
        if (et_recv_argreader(&ar, 1)) {  /* skip arg[0] = dwFileAttributes */
            /* arg[0] was the first uint32 (dwFileAttributes); re-read from
             * et_call's return value.  We already have it via the et_call
             * return — but et_call does NOT return it into pfad, so set it now.
             * Re-init reader from beginning to read arg[0] directly: */
            et_recv_argreader(&ar, 0);  /* re-init at arg[0] */
            uint32_t attrs = 0;
            et_argreader_next_uint32(&ar, &attrs);
            pfad->dwFileAttributes = attrs;

            /* arg[1]: file_size */
            uint64_t fsz = 0;
            et_argreader_next_uint64(&ar, &fsz);
            pfad->nFileSizeHigh = (DWORD)(fsz >> 32);
            pfad->nFileSizeLow  = (DWORD)(fsz & 0xFFFFFFFFULL);

            /* arg[2]: ftCreationTime */
            uint64_t t = 0;
            et_argreader_next_uint64(&ar, &t);
            pfad->ftCreationTime.dwLowDateTime  = (DWORD)(t & 0xFFFFFFFFULL);
            pfad->ftCreationTime.dwHighDateTime = (DWORD)(t >> 32);

            /* arg[3]: ftLastAccessTime */
            t = 0;
            et_argreader_next_uint64(&ar, &t);
            pfad->ftLastAccessTime.dwLowDateTime  = (DWORD)(t & 0xFFFFFFFFULL);
            pfad->ftLastAccessTime.dwHighDateTime = (DWORD)(t >> 32);

            /* arg[4]: ftLastWriteTime */
            t = 0;
            et_argreader_next_uint64(&ar, &t);
            pfad->ftLastWriteTime.dwLowDateTime  = (DWORD)(t & 0xFFFFFFFFULL);
            pfad->ftLastWriteTime.dwHighDateTime = (DWORD)(t >> 32);
        }
    }
    return TRUE;
}

BOOL WINAPI ET32_FlushFileBuffers(HANDLE hFile)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hFile);
    uint32_t err = 0;
    et_call(CMD_FAMILY_FILE_OPS, 0x7B, pbuf, ab.pos, &err);
    return !err;
}

HANDLE WINAPI ET32_FindFirstFileW(LPCWSTR lpFileName,
    LPWIN32_FIND_DATAW lpFindFileData)
{
    /*
     * Zero-initialize the caller's OUT buffer before the broker call so
     * that partial failures still leave the structure in a defined state.
     */
    if (lpFindFileData) memset(lpFindFileData, 0, sizeof(WIN32_FIND_DATAW));

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_strW(&ab, lpFileName);
    uint32_t err = 0;
    uint32_t bridge_handle = et_call(CMD_FAMILY_FILE_OPS, 0x7E, pbuf, ab.pos, &err);
    if (err) return INVALID_HANDLE_VALUE;

    /*
     * Issue 4 fix: et_call extracted only arg[0] (bridge_handle).
     * The broker's _handle_file_find_first packs:
     *   arg[0]: handle (uint32)       — already in bridge_handle
     *   arg[1]: cFileName (str/utf-8) — filename of first match
     *   arg[2]: dwFileAttributes (uint32)
     *   arg[3]: file_size (uint64)
     * Read the remaining args from g_recv_buf and populate lpFindFileData.
     *
     * ET derivation: g_recv_buf is the P-substrate holding the broker's
     * response; et_recv_argreader is the D-accessor; populating lpFindFileData
     * is the T-traversal that grounds the caller's E-state.
     */
    if (lpFindFileData) {
        et_arg_reader ar;
        if (et_recv_argreader(&ar, 1)) {  /* skip arg[0] = handle */
            /* arg[1]: cFileName */
            et_argreader_next_strW(&ar, lpFindFileData->cFileName,
                                   sizeof(lpFindFileData->cFileName) / sizeof(wchar_t));
            /* arg[2]: dwFileAttributes */
            uint32_t attrs = 0;
            et_argreader_next_uint32(&ar, &attrs);
            lpFindFileData->dwFileAttributes = attrs;
            /* arg[3]: file_size (uint64 → split into nFileSizeHigh/Low) */
            uint64_t fsz = 0;
            et_argreader_next_uint64(&ar, &fsz);
            lpFindFileData->nFileSizeHigh = (DWORD)(fsz >> 32);
            lpFindFileData->nFileSizeLow  = (DWORD)(fsz & 0xFFFFFFFFULL);
        }
    }
    return (HANDLE)(uintptr_t)bridge_handle;
}

BOOL WINAPI ET32_FindNextFileW(HANDLE hFindFile,
    LPWIN32_FIND_DATAW lpFindFileData)
{
    /*
     * Zero-initialize the caller's OUT buffer so a partial failure leaves
     * the structure in a defined state.
     */
    if (lpFindFileData) memset(lpFindFileData, 0, sizeof(WIN32_FIND_DATAW));

    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hFindFile);
    uint32_t err = 0;
    uint32_t more = et_call(CMD_FAMILY_FILE_OPS, 0x7F, pbuf, ab.pos, &err);
    if (err || !more) return FALSE;

    /*
     * Issue 4 fix: et_call extracted only arg[0] (more=1).
     * The broker's _handle_file_find_next packs:
     *   arg[0]: more (uint32 = 1)     — already in 'more'
     *   arg[1]: cFileName (str/utf-8) — filename of this entry
     *   arg[2]: dwFileAttributes (uint32)
     *   arg[3]: file_size (uint64)
     * Read the remaining args and populate lpFindFileData.
     */
    if (lpFindFileData) {
        et_arg_reader ar;
        if (et_recv_argreader(&ar, 1)) {  /* skip arg[0] = more */
            /* arg[1]: cFileName */
            et_argreader_next_strW(&ar, lpFindFileData->cFileName,
                                   sizeof(lpFindFileData->cFileName) / sizeof(wchar_t));
            /* arg[2]: dwFileAttributes */
            uint32_t attrs = 0;
            et_argreader_next_uint32(&ar, &attrs);
            lpFindFileData->dwFileAttributes = attrs;
            /* arg[3]: file_size (uint64 → nFileSizeHigh/Low) */
            uint64_t fsz = 0;
            et_argreader_next_uint64(&ar, &fsz);
            lpFindFileData->nFileSizeHigh = (DWORD)(fsz >> 32);
            lpFindFileData->nFileSizeLow  = (DWORD)(fsz & 0xFFFFFFFFULL);
        }
    }
    return TRUE;
}

BOOL WINAPI ET32_FindClose(HANDLE hFindFile)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, (uint32_t)(uintptr_t)hFindFile);
    uint32_t err = 0;
    et_call(CMD_FAMILY_FILE_OPS, 0x80, pbuf, ab.pos, &err);
    return !err;
}

/* ============================================================================
 * DLL ENTRY POINT
 * ============================================================================ */

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
    (void)hinstDLL;
    (void)lpvReserved;

    switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
        /* Disable per-thread notifications for performance */
        DisableThreadLibraryCalls(hinstDLL);
        /*
         * Do NOT auto-connect here — broker must call ET32_Init(broker_pid)
         * explicitly after injecting the DLL. This prevents deadlock if
         * DllMain is called from within the loader lock context.
         */
        break;

    case DLL_PROCESS_DETACH:
        if (g_initialised) {
            ET32_Shutdown();
        }
        break;

    default:
        /* DLL_THREAD_ATTACH / DLL_THREAD_DETACH are suppressed by
         * DisableThreadLibraryCalls above. Default covers any future
         * fdwReason values introduced by later Windows SDK versions. */
        break;
    }
    return TRUE;
}