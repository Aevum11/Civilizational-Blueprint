import React, { useState } from "react";

// ═══════════════════════════════════════════════════════════════
// ET SHADOW HOSTING — HASSE DIAGRAM OF div(12)
// All 12 harmonic families with force + phase names
// Michael James Muller — Aevum Defluo — Exception Theory LLC
// ═══════════════════════════════════════════════════════════════

export default function N12Hasse() {
  const [dark, setDark] = useState(false);
  const t = dark ? {
    bg: "#0a0e14", title: "#f0f4fc", sub: "#e8c44a", desc: "#a0aec0",
    sFill: "#e8ecf4", sStroke: "#4da6ff", sText: "#0a0e14", sPhi: "#7a8a9e",
    sForce: "#f0f4fc", sPhase: "#c0ccdd", sXi: "#e8c44a",
    cFill: "#141a24", cStroke: "#a0aec0", cText: "#e0e8f0",
    cForce: "#e0e8f0", cPhase: "#a0aec0", cHost: "#ff9100",
    hasse: "#3a4558", mirror: "#d084ff", hosting: "#ff9100",
    ucLine: "#ffb300", green: "#00e676", red: "#ff1744",
    cascBg: "#0a0e14", stepN: "#a0aec0", panelM: "#a0aec0",
    ann: "#c0ccdd", legend: "#a0aec0", summary: "#a0aec0",
    footer: "#5a6a7e", footBorder: "#1a2230", selfMirror: "#d084ff",
    structTitle: "#f0f4fc",
  } : {
    bg: "#faf8f4", title: "#0a0a2e", sub: "#8a6a10", desc: "#4a4a6a",
    sFill: "#0a0a2e", sStroke: "#2a2a6e", sText: "#ffffff", sPhi: "#9090cc",
    sForce: "#0a0a2e", sPhase: "#3a3a5a", sXi: "#8a6a10",
    cFill: "#ede8e0", cStroke: "#5a5a7a", cText: "#2a2a3a",
    cForce: "#2a2a3a", cPhase: "#5a5a6a", cHost: "#d84300",
    hasse: "#1a1a3a", mirror: "#8e24aa", hosting: "#d84300",
    ucLine: "#e65100", green: "#00831e", red: "#d50000",
    cascBg: "#faf8f4", stepN: "#666", panelM: "#555",
    ann: "#3a3a5a", legend: "#3a3a5a", summary: "#4a4a6a",
    footer: "#6a6a7a", footBorder: "#c0bca4", selfMirror: "#8e24aa",
    structTitle: "#0a0a2e",
  };
  const W = 760, H = 1420;

  // ═══ ET PRIMITIVES — ontological constants, hardcoded per their status ═══
  // (origins trivial: |Π|=3 · S=C(3,2)+C(3,3)=4 · N=|Π|·S · A0=(N−1)²+S²)
  const PI_CARD = 3, S_ST = 4;
  const N = PI_CARD * S_ST;                    // 12
  const A0 = (N - 1) ** 2 + S_ST ** 2;         // 137
  // κ-weights (IC-102/103, exact): κ=0 → 3/4 · κ=±1 combined → 1/4

  // ─── derived machinery — everything below computed from N (Rule 33) ───
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const lcm = (a, b) => (a / gcd(a, b)) * b;
  const range1 = n => Array.from({ length: n }, (_, i) => i + 1);
  const divisorsN = range1(N).filter(d => N % d === 0);            // τ=6 families
  const shadowsN  = range1(N).filter(m => N % m !== 0);            // {5,7,8,9,10,11}
  const phi = m => range1(m).filter(k => gcd(k, m) === 1).length;
  const dOf = k => N / gcd(k, N);                                   // gcd(0,N)=N → d(0)=1
  const resOf = m => Array.from({ length: N }, (_, k) => k).filter(k => dOf(k) === m);
  const hostOf = m => N / gcd(m, N);
  // ── LOSSLESS ARITHMETIC — BigInt exact rationals (Mike's standard:
  // string→exact→string, float FORBIDDEN in every value chain; IEEE numbers
  // appear ONLY in SVG pixel geometry, never in a theory value, comparison,
  // or displayed digit) ──
  const bgcd = (a, b) => (b ? bgcd(b, a % b) : (a < 0n ? -a : a));
  const fred = ([n, d]) => { const g = bgcd(n < 0n ? -n : n, d); return [n / g, d / g]; };
  const fadd = ([a, b], [c, d]) => fred([a * d + c * b, b * d]);
  const fmul = ([a, b], [c, d]) => fred([a * c, b * d]);
  const trunc3 = (n, d) => {                 // exact truncation, never rounding
    const q = (n * 1000n) / d;
    return (q / 1000n).toString() + "." + (q % 1000n).toString().padStart(3, "0");
  };
  const xiDen = m => BigInt((m - 1) ** 2 + S_ST ** 2);
  const xiStr = m => trunc3(BigInt(A0), xiDen(m));

  // layout (print geometry) + physical identifications (RC-13) — the only inputs
  const LAYOUT = { 12: [340, 130], 4: [180, 340], 6: [500, 340], 2: [210, 545],
    3: [470, 545], 1: [340, 740], 5: [240, 70], 7: [440, 70], 11: [340, 26],
    9: [60, 330], 10: [660, 330], 8: [630, 530] };
  const LABELS = { 1: ["Gravity", "Scalar / SSB"], 2: ["Tritone", "Spin-2"],
    3: ["Strong", "Instanton"], 4: ["Weak", "SU(2)_W"], 6: ["Hexadic", "Spin-½"],
    12: ["EM", "Photon / U(1)"], 5: ["Quintic", "E₈ Icosahedral"],
    7: ["Septic", "Octonionic / G₂"], 8: ["Gluon Octet", "SU(3) Adjoint"],
    9: ["Nonic", "CKM Mixing"], 10: ["Decic", "10D Majorana"],
    11: ["Undecimal", "11D Majorana"] };

  const simple = divisorsN.map(m => ({ m, x: LAYOUT[m][0], y: LAYOUT[m][1],
    phi: phi(m), force: LABELS[m][0], phase: LABELS[m][1], xi: xiStr(m), res: resOf(m) }));
  const complex = shadowsN.map(m => ({ m, x: LAYOUT[m][0], y: LAYOUT[m][1],
    host: hostOf(m), nc: lcm(N, m), force: LABELS[m][0], phase: LABELS[m][1], xi: xiStr(m) }));

  const nodeMap = {};
  simple.forEach(s => { nodeMap[s.m] = s; });
  complex.forEach(c => { nodeMap[c.m] = c; });

  // Hasse covering relations — derived from divisibility
  const hasse = [];
  divisorsN.forEach(a => divisorsN.forEach(b => {
    if (a < b && b % a === 0 &&
        !divisorsN.some(c => a < c && c < b && c % a === 0 && b % c === 0)) hasse.push([a, b]);
  }));

  // Mirror pairs m + m' = N (self-mirror N/2 drawn as halo)
  const mirrors = range1(N - 1).filter(m => m < N - m).map(m => ({ m1: m, m2: N - m }));

  const R_SIMPLE = 22;
  const R_SHADOW = 15;

  // Cascade position sequences — derived: k_n = (g·n) mod N, generators
  // (N/2+1, N/2−1) non-trivial pair (MAG-19) and (1, N−1) trivial pair
  const seqOf = g => range1(N).map(n => (g * n) % N);
  const cascades = [
    { g: N / 2 + 1, label: `g=${N / 2 + 1} fifths`,  pair: "nt", col: "#2979ff", pos: seqOf(N / 2 + 1) },
    { g: N / 2 - 1, label: `g=${N / 2 - 1} fourths`, pair: "nt", col: "#00b0ff", pos: seqOf(N / 2 - 1) },
    { g: 1,         label: "g=1 fwd",                pair: "tr", col: "#00c853", pos: seqOf(1) },
    { g: N - 1,     label: `g=${N - 1} bwd`,         pair: "tr", col: "#76ff03", pos: seqOf(N - 1) },
  ];

  // Position IS the harmonic family: k → m=k (k=0 → m=N octave closure)
  const k2m = (k) => k === 0 ? N : k;
  const famCol = { 1: "#c62828", 2: "#ef6c00", 3: "#7b1fa2", 4: "#1565c0", 5: "#4a148c",
    6: "#00897b", 7: "#1a237e", 8: "#4e342e", 9: "#bf360c", 10: "#006064", 11: "#33691e", 12: "#1a1a2e" };

  // FULL EM→m TABLE — LOSSLESS. Efficiency measure derived from IC-102: base ε
  // is uniform on ±R/2N native cells, so the composed offset is TRIANGULAR on
  // ±W = R/N; the weight of native carry κ is an EXACT rational, reproducing
  // the base weights (3/4, 1/8, 1/8) identically at W = 1. NO channel is
  // closed; the shadows are ε-residues — reached inside the half-cell (D),
  // exactly ON ∂I (the boundary-sworn Octet), or by the act (κ beyond ∂I).
  const resN = resOf(N);
  const fullTable = range1(N).map(m => {
    const R = lcm(N, m), W = R / N;                 // exact small integers
    const lift = resN.map(u => W * u);
    const dR = s => R / gcd(s === 0 ? R : s, R);
    const Wb = BigInt(W);
    const wOf = j => {                              // triangle cell weight, exact
      j = Math.abs(j);
      if (j === 0) return [4n * Wb - 1n, 4n * Wb * Wb];
      if (j < W)  return [Wb - BigInt(j), Wb * Wb];
      return [1n, 8n * Wb * Wb];                    // boundary sliver at j = W
    };
    let T = [0n, 1n]; let kmin = -1; let c0 = 0;
    lift.forEach(r1 => lift.forEach(r2 => {
      for (let kap = -W; kap <= W; kap++) {
        if (dR(((r1 + r2 + kap) % R + R) % R) !== m) continue;
        T = fadd(T, fmul([1n, 16n], wOf(kap)));
        if (kmin < 0 || Math.abs(kap) < kmin) kmin = Math.abs(kap);
        if (kap === 0) c0++;
      }
    }));
    const [En, Ed] = fred([T[0] * BigInt(A0), T[1] * xiDen(m)]);   // ξ(N) = 1 exact
    const band = (kmin === 0 || 2 * N * kmin < R) ? "D" : (2 * N * kmin === R ? "∂I" : "T");
    return { m, R, c0, kmin, band,
      kap: c0 > 0 ? "κ=0" : (band === "D" ? "ε-band" : (band === "∂I" ? "on ∂I" : "κ-act")),
      eff: trunc3(En, Ed), open: true };
  });

  // DRAWN CHANNELS — EVERY family gets its arrow: green = deterministic (κ=0 or
  // ε-band inside the half-cell), purple = exactly ON ∂I (the boundary-sworn
  // Octet), red = T-act required. The self-channel m=N is the one non-line: its
  // direction vector is 0/0, and [0/0] IS the Traverser (Sempaevum §2.4) —
  // rendered as the red T-act ring, no curve.
  const transfers = fullTable
    .filter(f => f.m !== N)
    .map(f => ({ tgt: f.m, kap: f.kap, eff: f.eff, band: f.band }));
  const selfCh = fullTable.find(f => f.m === N);


  // Unit circle phase traversal (IC-112): 1→4→2→6→12
  const ucSeq = [1, 4, 2, 6, 12];

  return (
    <div style={{
      background: t.bg, minHeight: "100vh", padding: "12px 8px",
      fontFamily: "'SF Mono', 'Fira Code', monospace",
    }}>
      <div style={{ textAlign: "center", marginBottom: 8, position: "relative" }}>
        <button onClick={() => setDark(!dark)} style={{
          position: "absolute", right: 8, top: 0, background: "none", border: `1px solid ${t.desc}`,
          color: t.desc, fontSize: 9, padding: "2px 8px", borderRadius: 4, cursor: "pointer",
        }}>{dark ? "☀ light" : "● dark"}</button>
        <div style={{ color: t.sub, fontSize: 9, letterSpacing: 2 }}>THE LCM TOWER · ℓ=0</div>
        <div style={{ color: t.title, fontSize: 15, fontWeight: "bold" }}>
          {`N=${N} · τ=${divisorsN.length} · ${divisorsN.length}/${N} active · ξ(m) = ${A0}/((m−1)²+${S_ST ** 2})`}
        </div>
        <div style={{ color: t.desc, fontSize: 7, marginTop: 2 }}>
          Host: d = 12/gcd(m, 12) · Mirrors: m + m' = 12 · Axis-agnostic (IC-110)
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, margin: "0 auto", display: "block" }}>

        {/* ── HASSE EDGES ── */}
        {hasse.map(([d1, d2], i) => {
          const a = nodeMap[d1], b = nodeMap[d2];
          return <line key={`h${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={t.hasse} strokeWidth="2" opacity="0.25" />;
        })}

        {/* ── MIRROR CROSSINGS ── */}
        {mirrors.map((mp, i) => {
          const a = nodeMap[mp.m1], b = nodeMap[mp.m2];
          return <line key={`m${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={t.mirror} strokeWidth="1.2" strokeDasharray="5,4" opacity="0.35" />;
        })}

        {/* ── SHADOW HOSTING LINES ── */}
        {complex.map(c => {
          const h = nodeMap[c.host];
          return <line key={`s${c.m}`} x1={c.x} y1={c.y} x2={h.x} y2={h.y}
            stroke={t.hosting} strokeWidth="1.2" strokeDasharray="3,2" opacity="0.55" />;
        })}

        {/* ── m=N/2 SELF-MIRROR ── */}
        <circle cx={nodeMap[N / 2].x} cy={nodeMap[N / 2].y} r={30} fill="none"
          stroke={t.selfMirror} strokeWidth="1.2" strokeDasharray="3,2" opacity="0.5" />

        {/* ── TRANSFER TENSOR FROM m=12 (IC-104/106/107) ── */}
        <defs>
          <marker id="aG" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill={t.green} /></marker>
          <marker id="aR" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill={t.red} /></marker>
          <marker id="aV" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill={t.mirror} /></marker>
          <marker id="aO" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill={t.ucLine} /></marker>
          <marker id="aB" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#2979ff" /></marker>
          <marker id="aLB" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#00b0ff" /></marker>
          <marker id="aDG" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#00c853" /></marker>
          <marker id="aLG" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#76ff03" /></marker>
        </defs>
        {transfers.map((tf, i) => {
          const src = nodeMap[N], tgt = nodeMap[tf.tgt];
          const dx = tgt.x - src.x, dy = tgt.y - src.y;
          const len = Math.sqrt(dx*dx + dy*dy);
          const nx = dx/len, ny = dy/len;
          const px = -ny, py = nx;
          const latMap = { 1: -10, 2: -16, 3: 10, 4: 0, 5: -8, 6: -12, 7: 8,
            8: 26, 9: -20, 10: 20, 11: 0 };
          const lateralOff = latMap[tf.tgt] || 0;
          const insT = tgt.phi !== undefined ? 26 : 19;   // shadow spheres are smaller
          const sx = src.x + nx * 26 + px * lateralOff, sy = src.y + ny * 26 + py * lateralOff;
          const ex = tgt.x - nx * insT + px * lateralOff, ey = tgt.y - ny * insT + py * lateralOff;
          const col = tf.band === "D" ? t.green : (tf.band === "∂I" ? t.mirror : t.red);
          const mid = tf.band === "D" ? "aG" : (tf.band === "∂I" ? "aV" : "aR");
          const mx = (sx + ex) / 2 + px * 12;
          const my = (sy + ey) / 2 + py * 12;
          return (
            <g key={`tr${i}`}>
              <line x1={sx} y1={sy} x2={ex} y2={ey}
                stroke={col} strokeWidth="2" opacity="0.75" markerEnd={`url(#${mid})`} />
              <text x={mx} y={my} fill={col} fontSize="5.5" textAnchor="middle" fontWeight="bold">
                {tf.kap} E={tf.eff}
              </text>
            </g>
          );
        })}

        {/* ── SELF-CHANNEL: [0/0] = T — the T-act ring (no line; the singularity IS the signature) ── */}
        <circle cx={nodeMap[N].x} cy={nodeMap[N].y} r={27} fill="none"
          stroke={t.red} strokeWidth="1.2" strokeDasharray="2,3" opacity="0.8" />
        <text x={nodeMap[N].x + 110} y={nodeMap[N].y + 27} fill={t.red} fontSize="5.5" textAnchor="middle">
          {`self ${selfCh.kap} E=${selfCh.eff} · [0/0]→T-act`}
        </text>

        {/* ── UNIT CIRCLE TRAVERSAL (IC-112): 1→4→2→6→12 ── */}
        {ucSeq.slice(0, -1).map((m, i) => {
          const a = nodeMap[m], b = nodeMap[ucSeq[i + 1]];
          const dx = b.x - a.x, dy = b.y - a.y;
          const len = Math.sqrt(dx*dx + dy*dy);
          const nx = dx/len, ny = dy/len;
          const px = -ny, py = nx;
          const off = 10;
          return (
            <g key={`uc${i}`}>
              <line
                x1={a.x + nx*26 + px*off} y1={a.y + ny*26 + py*off}
                x2={b.x - nx*26 + px*off} y2={b.y - ny*26 + py*off}
                stroke={t.ucLine} strokeWidth="2" opacity="0.7" strokeDasharray="6,3"
                markerEnd="url(#aO)" />
              <text x={(a.x+b.x)/2 + px*(off+10)} y={(a.y+b.y)/2 + py*(off+10)}
                fill={t.ucLine} fontSize="6.5" fontWeight="bold" textAnchor="middle">{i+1}</text>
            </g>
          );
        })}
        <text x={505} y={500} fill={t.ucLine} fontSize="6" opacity="0.7" textAnchor="middle">
          m=3 excluded
        </text>
        <text x={505} y={509} fill={t.ucLine} fontSize="5.5" opacity="0.6" textAnchor="middle">
          from U(1) traversal
        </text>

        {/* ── COMPLEX SHADOW NODES ── */}
        {complex.map(c => (
          <g key={`cn${c.m}`}>
            <circle cx={c.x} cy={c.y} r={R_SHADOW}
              fill={t.cFill} stroke={t.cStroke} strokeWidth="1.8" strokeDasharray="4,2" />
            <text x={c.x} y={c.y + 1} fill={t.cText} fontSize="11" fontWeight="bold"
              textAnchor="middle" dominantBaseline="middle">{c.m}</text>
            {/* Labels below shadow sphere */}
            <text x={c.x} y={c.y + R_SHADOW + 10} fill={t.cForce} fontSize="7" fontWeight="bold"
              textAnchor="middle">{c.force}</text>
            <text x={c.x} y={c.y + R_SHADOW + 19} fill={t.cPhase} fontSize="6"
              textAnchor="middle">{c.phase}</text>
            <text x={c.x} y={c.y + R_SHADOW + 28} fill={t.cHost} fontSize="5.5"
              textAnchor="middle">@d={c.host} · N={c.nc} · ξ={c.xi}</text>
          </g>
        ))}

        {/* ── SIMPLE FAMILY NODES ── */}
        {simple.map(s => (
          <g key={`sn${s.m}`}>
            <circle cx={s.x} cy={s.y} r={R_SIMPLE}
              fill={t.sFill} stroke={t.sStroke} strokeWidth="2.5" />
            <text x={s.x} y={s.y - 3} fill={t.sText} fontSize="14" fontWeight="bold"
              textAnchor="middle">{s.m}</text>
            <text x={s.x} y={s.y + 10} fill={t.sPhi} fontSize="7"
              textAnchor="middle">φ={s.phi}</text>
            {/* Labels below simple sphere */}
            <text x={s.x} y={s.y + R_SIMPLE + 12} fill={t.sForce} fontSize="8.5" fontWeight="bold"
              textAnchor="middle">{s.force}</text>
            <text x={s.x} y={s.y + R_SIMPLE + 22} fill={t.sPhase} fontSize="7"
              textAnchor="middle">{s.phase}</text>
            <text x={s.x} y={s.y + R_SIMPLE + 32} fill={t.sXi} fontSize="6"
              textAnchor="middle">ξ={s.xi} · Res={"{"+s.res.join(",")+"}"}</text>
          </g>
        ))}

        {/* ═══ CASCADE ARROWS ON HASSE — 4 GENERATORS, 4 DIFFERENT PATHS ═══ */}
        {/* Each position k IS harmonic family m=k. ALL 12 families visited. */}
        {/* Step markers sit AROUND the sphere edges, not inside */}
        {(() => {
          const arrowIds = { 7: "aB", 5: "aLB", 1: "aDG", 11: "aLG" };
          // Angle offsets: each generator gets a different perimeter position
          const angles = { 7: -2.4, 5: -0.7, 1: 0.7, 11: 2.4 }; // radians
          return cascades.map(casc => {
            const ang = angles[casc.g];
            const pts = casc.pos.map((k, si) => {
              const fam = k2m(k);
              const nd = nodeMap[fam];
              const r = nd.phi !== undefined ? R_SIMPLE + 10 : R_SHADOW + 9;
              return { x: nd.x + Math.cos(ang) * r, y: nd.y + Math.sin(ang) * r, k, fam, n: si + 1 };
            });
            return (
              <g key={`cp${casc.g}`}>
                {pts.slice(0, -1).map((p, i) => (
                  <line key={`cl${casc.g}s${i}`}
                    x1={p.x} y1={p.y} x2={pts[i+1].x} y2={pts[i+1].y}
                    stroke={casc.col} strokeWidth="1.4" opacity="0.55"
                    markerEnd={`url(#${arrowIds[casc.g]})`} />
                ))}
                {pts.map((p, i) => (
                  <g key={`cm${casc.g}n${i}`}>
                    <circle cx={p.x} cy={p.y} r={6}
                      fill={t.cascBg} stroke={casc.col} strokeWidth="1.6" />
                    <text x={p.x} y={p.y + 3} fill={casc.col} fontSize="5.5" fontWeight="bold"
                      textAnchor="middle">{p.n}</text>
                  </g>
                ))}
              </g>
            );
          });
        })()}

        {/* ═══ CASCADE POSITION SEQUENCES — EACH IS A DIFFERENT PATH ═══ */}
        <text x={380} y={840} fill={t.structTitle} fontSize="10" fontWeight="bold" textAnchor="middle">
          CASCADE POSITION SEQUENCES — k_n = (g·n) mod 12
        </text>
        <text x={380} y={853} fill={t.desc} fontSize="6.5" textAnchor="middle">
          Each generator traverses a DIFFERENT path through ALL 12 harmonic families. Position k = family m.
        </text>

        {/* Non-trivial pair header */}
        <text x={70} y={875} fill="#1565c0" fontSize="8" fontWeight="bold">NON-TRIVIAL PAIR (5, 7)</text>

        {cascades.filter(c => c.pair === "nt").map((casc, ci) => {
          const baseY = 890 + ci * 48;
          return (
            <g key={`cas${casc.g}`}>
              <text x={70} y={baseY} fill={casc.col} fontSize="7" fontWeight="bold">{casc.label}</text>
              {casc.pos.map((k, si) => {
                const bx = 70 + si * 54;
                const fam = k2m(k);
                const fc = famCol[fam] || "#666";
                return (
                  <g key={`p${casc.g}s${si}`}>
                    {si > 0 && <line x1={bx - 20} y1={baseY + 13} x2={bx - 3} y2={baseY + 13}
                      stroke={casc.col} strokeWidth="0.7" opacity="0.4" />}
                    <text x={bx + 9} y={baseY + 7} fill={t.stepN} fontSize="4.5" textAnchor="middle">n={si+1}</text>
                    <rect x={bx} y={baseY + 8} width={18} height={11} rx={2}
                      fill={fc} opacity={dark ? "0.25" : "0.1"} stroke={fc} strokeWidth="0.5" />
                    <text x={bx + 9} y={baseY + 17} fill={fc} fontSize="8" fontWeight="bold" textAnchor="middle">{k}</text>
                    <text x={bx + 9} y={baseY + 25} fill={t.panelM} fontSize="4" textAnchor="middle">m={fam}</text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Trivial pair header */}
        <text x={70} y={995} fill="#2e7d32" fontSize="8" fontWeight="bold">TRIVIAL PAIR (1, 11)</text>

        {cascades.filter(c => c.pair === "tr").map((casc, ci) => {
          const baseY = 1010 + ci * 48;
          return (
            <g key={`cas${casc.g}`}>
              <text x={70} y={baseY} fill={casc.col} fontSize="7" fontWeight="bold">{casc.label}</text>
              {casc.pos.map((k, si) => {
                const bx = 70 + si * 54;
                const fam = k2m(k);
                const fc = famCol[fam] || "#666";
                return (
                  <g key={`p${casc.g}s${si}`}>
                    {si > 0 && <line x1={bx - 20} y1={baseY + 13} x2={bx - 3} y2={baseY + 13}
                      stroke={casc.col} strokeWidth="0.7" opacity="0.4" />}
                    <text x={bx + 9} y={baseY + 7} fill={t.stepN} fontSize="4.5" textAnchor="middle">n={si+1}</text>
                    <rect x={bx} y={baseY + 8} width={18} height={11} rx={2}
                      fill={fc} opacity={dark ? "0.25" : "0.1"} stroke={fc} strokeWidth="0.5" />
                    <text x={bx + 9} y={baseY + 17} fill={fc} fontSize="8" fontWeight="bold" textAnchor="middle">{k}</text>
                    <text x={bx + 9} y={baseY + 25} fill={t.panelM} fontSize="4" textAnchor="middle">m={fam}</text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* ═══ ANNOTATIONS ═══ */}
        <text x={70} y={1120} fill={t.mirror} fontSize="7.5" fontWeight="bold">MIRRORS m+m'=12</text>
        <text x={30} y={1133} fill={t.ann} fontSize="6.5">1↔11 · 2↔10 · 3↔9 · 4↔8 · 5↔7 · 6↔6(self)</text>

        <text x={430} y={1120} fill={t.hosting} fontSize="7.5" fontWeight="bold">HOST d=12/gcd(m,12)</text>
        <text x={430} y={1133} fill={t.ann} fontSize="6.5">m=5,7,11→d=12 · m=9→d=4 · m=10→d=6 · m=8→d=3</text>

        <text x={380} y={1155} fill={t.ann} fontSize="7" textAnchor="middle">
          {`Σφ(d) = ${divisorsN.map(phi).join("+")} = ${divisorsN.reduce((a, d) => a + phi(d), 0)} = N · Distribution [3,1,1,1,0,0] unique to ET · Sublattice Visitation Thm`}
        </text>

        {/* ═══ STRUCTURAL SUMMARY ═══ */}
        <text x={380} y={1178} fill={t.structTitle} fontSize="7" textAnchor="middle" fontWeight="bold">
          {transfers.map(tf => `EM→${LABELS[tf.tgt][0].toLowerCase()}: ${tf.kap} E=${tf.eff}`).join(" · ")}
        </text>
        <text x={380} y={1190} fill={t.summary} fontSize="6" textAnchor="middle">
          Gravity+strong channels: D-arithmetic (deterministic). Weak channel: T-agency exclusively (IC-107).
        </text>
        <text x={380} y={1202} fill={t.summary} fontSize="6" textAnchor="middle">
          U(1) traversal: m=1→4→2→6→12 · m=3 (strong/instanton) excluded — topological, non-perturbative
        </text>
        <text x={380} y={1214} fill={t.summary} fontSize="6" textAnchor="middle">
          ξ monotonically decreasing (IC-109): gravity 8.562 → EM 1.000 · Axis-agnostic (IC-110)
        </text>

        {/* ═══ FULL EM→m TABLE — shadows at native R = lcm(N,m) ═══ */}
        <text x={380} y={1226} fill={t.structTitle} fontSize="6.5" textAnchor="middle" fontWeight="bold">
          FULL EM→m TABLE · shadows evaluated at native R = lcm(N,m) · shadows are ε-residues at base, k-classes at home
        </text>
        {fullTable.map((f, i) => {
          const col = f.band === "D" ? t.green : (f.band === "∂I" ? t.mirror : t.red);
          const bx = 130 + (i % 6) * 100, by = 1236 + Math.floor(i / 6) * 10;
          return (
            <text key={`ft${f.m}`} x={bx} y={by} fill={col} fontSize="5.5" textAnchor="middle">
              {`m${f.m}@${f.R}: ${f.kap}${f.open ? " E=" + f.eff : ""}`}
            </text>
          );
        })}

        {/* ═══ LEGEND ═══ */}
        <text x={380} y={1266} fill={t.structTitle} fontSize="9" fontWeight="bold" textAnchor="middle">LEGEND</text>

        <circle cx={35} cy={1286} r={6} fill={t.sFill} stroke={t.sStroke} strokeWidth="1.5" />
        <text x={48} y={1289} fill={t.legend} fontSize="6">Simple (m|12)</text>
        <circle cx={160} cy={1286} r={6} fill={t.cFill} stroke={t.cStroke} strokeWidth="1" strokeDasharray="4,2" />
        <text x={173} y={1289} fill={t.legend} fontSize="6">Complex (m∤12)</text>
        <line x1={285} y1={1286} x2={297} y2={1286} stroke={t.hasse} strokeWidth="1.5" />
        <text x={303} y={1289} fill={t.legend} fontSize="6">Hasse</text>
        <line x1={360} y1={1286} x2={372} y2={1286} stroke={t.hosting} strokeWidth="0.8" strokeDasharray="2,2" />
        <text x={378} y={1289} fill={t.legend} fontSize="6">Shadow hosting</text>
        <line x1={475} y1={1286} x2={487} y2={1286} stroke={t.mirror} strokeWidth="0.8" strokeDasharray="5,4" />
        <text x={493} y={1289} fill={t.legend} fontSize="6">Mirror</text>

        <circle cx={35} cy={1304} r={8} fill="none" stroke={t.selfMirror} strokeWidth="0.7" strokeDasharray="3,2" />
        <text x={48} y={1307} fill={t.legend} fontSize="6">Self-mirror (m=6)</text>
        <line x1={160} y1={1304} x2={172} y2={1304} stroke={t.ucLine} strokeWidth="1.2" strokeDasharray="6,3" />
        <text x={178} y={1307} fill={t.legend} fontSize="6">U(1) traversal</text>
        <line x1={285} y1={1304} x2={297} y2={1304} stroke={t.green} strokeWidth="1" />
        <text x={303} y={1307} fill={t.legend} fontSize="6">κ=0 (D-arithmetic)</text>
        <line x1={400} y1={1304} x2={412} y2={1304} stroke={t.red} strokeWidth="1" />
        <text x={418} y={1307} fill={t.legend} fontSize="6">κ-act (T only)</text>
        <line x1={520} y1={1304} x2={532} y2={1304} stroke={t.mirror} strokeWidth="1" />
        <text x={538} y={1307} fill={t.legend} fontSize="6">on ∂I (boundary-sworn)</text>

        <rect x={28} y={1318} width={12} height={8} rx={2} fill="#2979ff" opacity="0.25" stroke="#2979ff" strokeWidth="0.8" />
        <text x={46} y={1325} fill={t.legend} fontSize="6">(5,7) non-trivial pair</text>
        <rect x={160} y={1318} width={12} height={8} rx={2} fill="#00c853" opacity="0.25" stroke="#00c853" strokeWidth="0.8" />
        <text x={178} y={1325} fill={t.legend} fontSize="6">(1,11) trivial pair</text>

        <text x={380} y={1340} fill={t.footer} fontSize="6" textAnchor="middle">
          Force (bold) = real D-axis · Phase = imaginary T-axis · Position colors match family nodes
        </text>
      </svg>

      <div style={{
        textAlign: "center", color: t.footer, fontSize: 7, marginTop: 6, paddingTop: 6,
        borderTop: `1px solid ${t.footBorder}`
      }}>
        P ∘ D ∘ T = E — Exception Theory — Michael James Muller — Aevum Defluo
      </div>
    </div>
  );
}
