import React from "react";

// ═══════════════════════════════════════════════════════════════
// ET SHADOW HOSTING — HASSE DIAGRAM OF div(12)
// All 12 harmonic families with force + phase names
// Michael James Muller — Aevum Defluo — Exception Theory LLC
// ═══════════════════════════════════════════════════════════════

export default function N12Hasse() {
  const W = 680, H = 1260;

  // Simple families — Hasse lattice positions with wide vertical gaps for labels
  const simple = [
    { m: 12, x: 280, y: 90,  phi: 4, force: "EM",      phase: "Photon / U(1)",  xi: "1.000",  res: [1,5,7,11] },
    { m: 4,  x: 140, y: 280, phi: 2, force: "Weak",     phase: "SU(2)_W",        xi: "5.480",  res: [3,9] },
    { m: 6,  x: 420, y: 280, phi: 2, force: "Hexadic",  phase: "Spin-½",         xi: "3.341",  res: [2,10] },
    { m: 2,  x: 170, y: 460, phi: 1, force: "Tritone",  phase: "Spin-2",         xi: "8.059",  res: [6] },
    { m: 3,  x: 390, y: 460, phi: 2, force: "Strong",   phase: "Instanton",      xi: "6.850",  res: [4,8] },
    { m: 1,  x: 280, y: 630, phi: 1, force: "Gravity",  phase: "Scalar / SSB",   xi: "8.563",  res: [0] },
  ];

  // Complex shadow families — positioned beside their hosts
  const complex = [
    { m: 5,  x: 200, y: 55,  host: 12, nc: 60,  force: "Quintic",      phase: "E₈ Icosahedral",    xi: "4.281" },
    { m: 7,  x: 360, y: 55,  host: 12, nc: 84,  force: "Septic",       phase: "Octonionic / G₂",   xi: "2.635" },
    { m: 11, x: 280, y: 22,  host: 12, nc: 132, force: "Undecimal",    phase: "11D Majorana",       xi: "1.181" },
    { m: 9,  x: 40,  y: 270, host: 4,  nc: 36,  force: "Nonic",        phase: "CKM Mixing",         xi: "1.713" },
    { m: 10, x: 560, y: 270, host: 6,  nc: 60,  force: "Decic",        phase: "10D Majorana",       xi: "1.412" },
    { m: 8,  x: 530, y: 450, host: 3,  nc: 24,  force: "Gluon Octet",  phase: "SU(3) Adjoint",      xi: "2.108" },
  ];

  const nodeMap = {};
  simple.forEach(s => { nodeMap[s.m] = s; });
  complex.forEach(c => { nodeMap[c.m] = c; });

  // Hasse covering relations
  const hasse = [[1,2],[1,3],[2,4],[2,6],[3,6],[4,12],[6,12]];

  // Mirror pairs
  const mirrors = [
    { m1: 1, m2: 11 }, { m1: 2, m2: 10 },
    { m1: 3, m2: 9 },  { m1: 4, m2: 8 }, { m1: 5, m2: 7 },
  ];

  const R_SIMPLE = 22;
  const R_SHADOW = 15;

  // Cascade position sequences — EACH IS A DIFFERENT TRAVERSAL
  const cascades = [
    { g: 7,  label: "g=7 fifths",   pair: "nt", col: "#1565c0", pos: [7,2,9,4,11,6,1,8,3,10,5,0] },
    { g: 5,  label: "g=5 fourths",  pair: "nt", col: "#64b5f6", pos: [5,10,3,8,1,6,11,4,9,2,7,0] },
    { g: 1,  label: "g=1 fwd",      pair: "tr", col: "#2e7d32", pos: [1,2,3,4,5,6,7,8,9,10,11,0] },
    { g: 11, label: "g=11 bwd",     pair: "tr", col: "#81c784", pos: [11,10,9,8,7,6,5,4,3,2,1,0] },
  ];

  // Position IS the harmonic family: k → m=k (k=0 → m=12 octave closure)
  const k2m = (k) => k === 0 ? 12 : k;
  const famCol = { 1: "#c62828", 2: "#ef6c00", 3: "#7b1fa2", 4: "#1565c0", 5: "#4a148c",
    6: "#00897b", 7: "#1a237e", 8: "#4e342e", 9: "#bf360c", 10: "#006064", 11: "#33691e", 12: "#1a1a2e" };

  // Transfer tensor (IC-104, IC-106, IC-107)
  const transfers = [
    { tgt: 1, kap: "κ=0", eff: "1.606", gt: true },
    { tgt: 3, kap: "κ=0", eff: "1.284", gt: true },
    { tgt: 4, kap: "κ≠0", eff: "0.343", gt: false },
  ];

  // Unit circle phase traversal (IC-112): 1→4→2→6→12
  const ucSeq = [1, 4, 2, 6, 12];

  return (
    <div style={{
      background: "#faf8f4", minHeight: "100vh", padding: "12px 8px",
      fontFamily: "'SF Mono', 'Fira Code', monospace",
    }}>
      <div style={{ textAlign: "center", marginBottom: 8 }}>
        <div style={{ color: "#8a7a5a", fontSize: 9, letterSpacing: 2 }}>THE LCM TOWER · ℓ=0</div>
        <div style={{ color: "#1a1a2e", fontSize: 15, fontWeight: "bold" }}>N=12 · τ=6 · 6/12 active · ξ(m) = 137/((m−1)²+16)</div>
        <div style={{ color: "#6a6a7a", fontSize: 7, marginTop: 2 }}>
          Host: d = 12/gcd(m, 12) · Mirrors: m + m' = 12 · Axis-agnostic (IC-110)
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, margin: "0 auto", display: "block" }}>

        {/* ── HASSE EDGES ── */}
        {hasse.map(([d1, d2], i) => {
          const a = nodeMap[d1], b = nodeMap[d2];
          return <line key={`h${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="#2a2a3a" strokeWidth="1.8" opacity="0.1" />;
        })}

        {/* ── MIRROR CROSSINGS ── */}
        {mirrors.map((mp, i) => {
          const a = nodeMap[mp.m1], b = nodeMap[mp.m2];
          return <line key={`m${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="#7b1fa2" strokeWidth="1" strokeDasharray="5,4" opacity="0.15" />;
        })}

        {/* ── SHADOW HOSTING LINES ── */}
        {complex.map(c => {
          const h = nodeMap[c.host];
          return <line key={`s${c.m}`} x1={c.x} y1={c.y} x2={h.x} y2={h.y}
            stroke="#e65100" strokeWidth="0.9" strokeDasharray="2,2" opacity="0.3" />;
        })}

        {/* ── m=6 SELF-MIRROR ── */}
        <circle cx={420} cy={280} r={30} fill="none"
          stroke="#7b1fa2" strokeWidth="0.8" strokeDasharray="3,2" opacity="0.3" />

        {/* ── TRANSFER TENSOR FROM m=12 (IC-104/106/107) ── */}
        <defs>
          <marker id="aG" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill="#1b5e20" /></marker>
          <marker id="aR" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill="#c62828" /></marker>
          <marker id="aO" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill="#ff6f00" /></marker>
          <marker id="aB" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#1565c0" /></marker>
          <marker id="aLB" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#64b5f6" /></marker>
          <marker id="aDG" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#2e7d32" /></marker>
          <marker id="aLG" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#81c784" /></marker>
        </defs>
        {transfers.map((t, i) => {
          const src = nodeMap[12], tgt = nodeMap[t.tgt];
          const dx = tgt.x - src.x, dy = tgt.y - src.y;
          const len = Math.sqrt(dx*dx + dy*dy);
          const nx = dx/len, ny = dy/len;
          const px = -ny, py = nx;
          const lateralOff = t.tgt === 1 ? -10 : t.tgt === 3 ? 10 : 0;
          const sx = src.x + nx * 26 + px * lateralOff, sy = src.y + ny * 26 + py * lateralOff;
          const ex = tgt.x - nx * 26 + px * lateralOff, ey = tgt.y - ny * 26 + py * lateralOff;
          const col = t.gt ? "#1b5e20" : "#c62828";
          const mid = t.gt ? "aG" : "aR";
          const mx = (sx + ex) / 2 + px * 12;
          const my = (sy + ey) / 2 + py * 12;
          return (
            <g key={`tr${i}`}>
              <line x1={sx} y1={sy} x2={ex} y2={ey}
                stroke={col} strokeWidth="1.2" opacity="0.5" markerEnd={`url(#${mid})`} />
              <text x={mx} y={my} fill={col} fontSize="5.5" textAnchor="middle" fontWeight="bold">
                {t.kap} E={t.eff}
              </text>
            </g>
          );
        })}

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
                stroke="#ff6f00" strokeWidth="1.4" opacity="0.5" strokeDasharray="6,3"
                markerEnd="url(#aO)" />
              <text x={(a.x+b.x)/2 + px*(off+10)} y={(a.y+b.y)/2 + py*(off+10)}
                fill="#ff6f00" fontSize="6.5" fontWeight="bold" textAnchor="middle">{i+1}</text>
            </g>
          );
        })}
        <text x={480} y={500} fill="#ff6f00" fontSize="6" opacity="0.7">
          m=3 excluded
        </text>
        <text x={480} y={509} fill="#ff6f00" fontSize="5.5" opacity="0.6">
          from U(1) traversal
        </text>

        {/* ── COMPLEX SHADOW NODES ── */}
        {complex.map(c => (
          <g key={`cn${c.m}`}>
            <circle cx={c.x} cy={c.y} r={R_SHADOW}
              fill="#f0ede8" stroke="#6a6a7a" strokeWidth="1.3" strokeDasharray="4,2" />
            <text x={c.x} y={c.y + 1} fill="#3a3a4a" fontSize="11" fontWeight="bold"
              textAnchor="middle" dominantBaseline="middle">{c.m}</text>
            {/* Labels below shadow sphere */}
            <text x={c.x} y={c.y + R_SHADOW + 10} fill="#4a4a5a" fontSize="7" fontWeight="bold"
              textAnchor="middle">{c.force}</text>
            <text x={c.x} y={c.y + R_SHADOW + 19} fill="#7a7a8a" fontSize="6"
              textAnchor="middle">{c.phase}</text>
            <text x={c.x} y={c.y + R_SHADOW + 28} fill="#e65100" fontSize="5.5"
              textAnchor="middle">@d={c.host} · N={c.nc} · ξ={c.xi}</text>
          </g>
        ))}

        {/* ── SIMPLE FAMILY NODES ── */}
        {simple.map(s => (
          <g key={`sn${s.m}`}>
            <circle cx={s.x} cy={s.y} r={R_SIMPLE}
              fill="#1a1a2e" stroke="#3a3a5e" strokeWidth="2.5" />
            <text x={s.x} y={s.y - 3} fill="#ffffff" fontSize="14" fontWeight="bold"
              textAnchor="middle">{s.m}</text>
            <text x={s.x} y={s.y + 10} fill="#aaaacc" fontSize="7"
              textAnchor="middle">φ={s.phi}</text>
            {/* Labels below simple sphere */}
            <text x={s.x} y={s.y + R_SIMPLE + 12} fill="#1a1a2e" fontSize="8.5" fontWeight="bold"
              textAnchor="middle">{s.force}</text>
            <text x={s.x} y={s.y + R_SIMPLE + 22} fill="#5a5a6a" fontSize="7"
              textAnchor="middle">{s.phase}</text>
            <text x={s.x} y={s.y + R_SIMPLE + 32} fill="#8a7a5a" fontSize="6"
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
              const fam = k === 0 ? 12 : k;
              const nd = nodeMap[fam];
              const r = nd.phi !== undefined ? R_SIMPLE + 10 : R_SHADOW + 9;
              return { x: nd.x + Math.cos(ang) * r, y: nd.y + Math.sin(ang) * r, k, fam, n: si + 1 };
            });
            return (
              <g key={`cp${casc.g}`}>
                {pts.slice(0, -1).map((p, i) => (
                  <line key={`cl${casc.g}s${i}`}
                    x1={p.x} y1={p.y} x2={pts[i+1].x} y2={pts[i+1].y}
                    stroke={casc.col} strokeWidth="1" opacity="0.4"
                    markerEnd={`url(#${arrowIds[casc.g]})`} />
                ))}
                {pts.map((p, i) => (
                  <g key={`cm${casc.g}n${i}`}>
                    <circle cx={p.x} cy={p.y} r={6}
                      fill="#faf8f4" stroke={casc.col} strokeWidth="1.2" />
                    <text x={p.x} y={p.y + 3} fill={casc.col} fontSize="5.5" fontWeight="bold"
                      textAnchor="middle">{p.n}</text>
                  </g>
                ))}
              </g>
            );
          });
        })()}

        {/* ═══ CASCADE POSITION SEQUENCES — EACH IS A DIFFERENT PATH ═══ */}
        <text x={340} y={720} fill="#1a1a2e" fontSize="10" fontWeight="bold" textAnchor="middle">
          CASCADE POSITION SEQUENCES — k_n = (g·n) mod 12
        </text>
        <text x={340} y={733} fill="#6a6a7a" fontSize="6.5" textAnchor="middle">
          Each generator traverses a DIFFERENT path through ALL 12 harmonic families. Position k = family m.
        </text>

        {/* Non-trivial pair header */}
        <text x={30} y={755} fill="#1565c0" fontSize="8" fontWeight="bold">NON-TRIVIAL PAIR (5, 7)</text>

        {cascades.filter(c => c.pair === "nt").map((casc, ci) => {
          const baseY = 770 + ci * 48;
          return (
            <g key={`cas${casc.g}`}>
              <text x={30} y={baseY} fill={casc.col} fontSize="7" fontWeight="bold">{casc.label}</text>
              {casc.pos.map((k, si) => {
                const bx = 30 + si * 54;
                const fam = k2m(k);
                const fc = famCol[fam] || "#666";
                return (
                  <g key={`p${casc.g}s${si}`}>
                    {si > 0 && <line x1={bx - 20} y1={baseY + 13} x2={bx - 3} y2={baseY + 13}
                      stroke={casc.col} strokeWidth="0.7" opacity="0.4" />}
                    <text x={bx + 9} y={baseY + 7} fill="#aaa" fontSize="4.5" textAnchor="middle">n={si+1}</text>
                    <rect x={bx} y={baseY + 8} width={18} height={11} rx={2}
                      fill={fc} opacity="0.1" stroke={fc} strokeWidth="0.5" />
                    <text x={bx + 9} y={baseY + 17} fill={fc} fontSize="8" fontWeight="bold" textAnchor="middle">{k}</text>
                    <text x={bx + 9} y={baseY + 25} fill="#999" fontSize="4" textAnchor="middle">m={fam}</text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Trivial pair header */}
        <text x={30} y={875} fill="#2e7d32" fontSize="8" fontWeight="bold">TRIVIAL PAIR (1, 11)</text>

        {cascades.filter(c => c.pair === "tr").map((casc, ci) => {
          const baseY = 890 + ci * 48;
          return (
            <g key={`cas${casc.g}`}>
              <text x={30} y={baseY} fill={casc.col} fontSize="7" fontWeight="bold">{casc.label}</text>
              {casc.pos.map((k, si) => {
                const bx = 30 + si * 54;
                const fam = k2m(k);
                const fc = famCol[fam] || "#666";
                return (
                  <g key={`p${casc.g}s${si}`}>
                    {si > 0 && <line x1={bx - 20} y1={baseY + 13} x2={bx - 3} y2={baseY + 13}
                      stroke={casc.col} strokeWidth="0.7" opacity="0.4" />}
                    <text x={bx + 9} y={baseY + 7} fill="#aaa" fontSize="4.5" textAnchor="middle">n={si+1}</text>
                    <rect x={bx} y={baseY + 8} width={18} height={11} rx={2}
                      fill={fc} opacity="0.1" stroke={fc} strokeWidth="0.5" />
                    <text x={bx + 9} y={baseY + 17} fill={fc} fontSize="8" fontWeight="bold" textAnchor="middle">{k}</text>
                    <text x={bx + 9} y={baseY + 25} fill="#999" fontSize="4" textAnchor="middle">m={fam}</text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* ═══ ANNOTATIONS ═══ */}
        <text x={30} y={1000} fill="#7b1fa2" fontSize="7.5" fontWeight="bold">MIRRORS m+m'=12</text>
        <text x={30} y={1013} fill="#5a5a6a" fontSize="6.5">1↔11 · 2↔10 · 3↔9 · 4↔8 · 5↔7 · 6↔6(self)</text>

        <text x={360} y={1000} fill="#e65100" fontSize="7.5" fontWeight="bold">HOST d=12/gcd(m,12)</text>
        <text x={360} y={1013} fill="#5a5a6a" fontSize="6.5">m=5,7,11→d=12 · m=9→d=4 · m=10→d=6 · m=8→d=3</text>

        <text x={340} y={1035} fill="#5a5a6a" fontSize="7" textAnchor="middle">
          Σφ(d) = 1+1+2+2+2+4 = 12 = N · Distribution [3,1,1,1,0,0] unique to ET
        </text>

        {/* ═══ STRUCTURAL SUMMARY ═══ */}
        <text x={340} y={1058} fill="#1a1a2e" fontSize="7" textAnchor="middle" fontWeight="bold">
          EM→gravity: κ=0 E=1.606 · EM→strong: κ=0 E=1.284 · EM→weak: κ≠0 E=0.343
        </text>
        <text x={340} y={1070} fill="#6a6a7a" fontSize="6" textAnchor="middle">
          Gravity+strong channels: D-arithmetic (deterministic). Weak channel: T-agency exclusively (IC-107).
        </text>
        <text x={340} y={1082} fill="#6a6a7a" fontSize="6" textAnchor="middle">
          U(1) traversal: m=1→4→2→6→12 · m=3 (strong/instanton) excluded — topological, non-perturbative
        </text>
        <text x={340} y={1094} fill="#6a6a7a" fontSize="6" textAnchor="middle">
          ξ monotonically decreasing (IC-109): gravity 8.563 → EM 1.000 · Axis-agnostic (IC-110)
        </text>

        {/* ═══ LEGEND ═══ */}
        <text x={340} y={1120} fill="#1a1a2e" fontSize="9" fontWeight="bold" textAnchor="middle">LEGEND</text>

        <circle cx={35} cy={1140} r={6} fill="#1a1a2e" stroke="#3a3a5e" strokeWidth="1.5" />
        <text x={48} y={1143} fill="#4a4a5a" fontSize="6">Simple (m|12)</text>
        <circle cx={160} cy={1140} r={6} fill="#f0ede8" stroke="#6a6a7a" strokeWidth="1" strokeDasharray="4,2" />
        <text x={173} y={1143} fill="#4a4a5a" fontSize="6">Complex (m∤12)</text>
        <line x1={285} y1={1140} x2={297} y2={1140} stroke="#2a2a3a" strokeWidth="1.5" />
        <text x={303} y={1143} fill="#4a4a5a" fontSize="6">Hasse</text>
        <line x1={360} y1={1140} x2={372} y2={1140} stroke="#e65100" strokeWidth="0.8" strokeDasharray="2,2" />
        <text x={378} y={1143} fill="#4a4a5a" fontSize="6">Shadow hosting</text>
        <line x1={475} y1={1140} x2={487} y2={1140} stroke="#7b1fa2" strokeWidth="0.8" strokeDasharray="5,4" />
        <text x={493} y={1143} fill="#4a4a5a" fontSize="6">Mirror</text>

        <circle cx={35} cy={1158} r={8} fill="none" stroke="#7b1fa2" strokeWidth="0.7" strokeDasharray="3,2" />
        <text x={48} y={1161} fill="#4a4a5a" fontSize="6">Self-mirror (m=6)</text>
        <line x1={160} y1={1158} x2={172} y2={1158} stroke="#ff6f00" strokeWidth="1.2" strokeDasharray="6,3" />
        <text x={178} y={1161} fill="#4a4a5a" fontSize="6">U(1) traversal</text>
        <line x1={285} y1={1158} x2={297} y2={1158} stroke="#1b5e20" strokeWidth="1" />
        <text x={303} y={1161} fill="#4a4a5a" fontSize="6">κ=0 (D-arith, E{">"}1)</text>
        <line x1={400} y1={1158} x2={412} y2={1158} stroke="#c62828" strokeWidth="1" />
        <text x={418} y={1161} fill="#4a4a5a" fontSize="6">κ≠0 (T-act, E{"<"}1)</text>

        <rect x={28} y={1172} width={12} height={8} rx={2} fill="#1565c0" opacity="0.15" stroke="#1565c0" strokeWidth="0.5" />
        <text x={46} y={1179} fill="#4a4a5a" fontSize="6">(5,7) non-trivial pair</text>
        <rect x={160} y={1172} width={12} height={8} rx={2} fill="#2e7d32" opacity="0.15" stroke="#2e7d32" strokeWidth="0.5" />
        <text x={178} y={1179} fill="#4a4a5a" fontSize="6">(1,11) trivial pair</text>

        <text x={340} y={1200} fill="#8a8a9a" fontSize="6" textAnchor="middle">
          Force (bold) = real D-axis · Phase = imaginary T-axis · Position colors match family nodes
        </text>
      </svg>

      <div style={{
        textAlign: "center", color: "#8a8a9a", fontSize: 7, marginTop: 6, paddingTop: 6,
        borderTop: "1px solid #d0ccc4"
      }}>
        P ∘ D ∘ T = E — Exception Theory — Michael James Muller — Aevum Defluo
      </div>
    </div>
  );
}
