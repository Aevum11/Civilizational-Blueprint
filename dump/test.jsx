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
    { m: 4,  x: 140, y: 280, phi: 2, force: "Weak",     phase: "SU(2)_W",       xi: "5.480",  res: [3,9] },
    { m: 6,  x: 420, y: 280, phi: 2, force: "Hexadic",  phase: "Spin-½",         xi: "3.341",  res: [2,10] },
    { m: 2,  x: 170, y: 460, phi: 1, force: "Tritone",  phase: "Spin-2",         xi: "8.059",  res: [6] },
    { m: 3,  x: 390, y: 460, phi: 2, force: "Strong",   phase: "Instanton",      xi: "6.850",  res: [4,8] },
    { m: 1,  x: 280, y: 630, phi: 1, force: "Gravity",  phase: "Scalar / SSB",   xi: "8.563",  res: [0] },
  ];

  // Complex shadow families — positioned beside their hosts
  const complex = [
    { m: 5,  x: 200, y: 55,  host: 12, nc: 60,  force: "Quintic",      phase: "E₈ Icosahedral",    xi: "4.281" },
    { m: 7,  x: 360, y: 55,  host: 12, nc: 84,  force: "Septic",       phase: "Octonionic / G₂",   xi: "2.635" },
    { m: 11, x: 280, y: 22,  host: 12, nc: 132, force: "Undecimal",    phase: "11D Majorana",        xi: "1.181" },
    { m: 9,  x: 40,  y: 270, host: 4,  nc: 36,  force: "Nonic",        phase: "CKM Mixing",          xi: "1.713" },
    { m: 10, x: 560, y: 270, host: 6,  nc: 60,  force: "Decic",        phase: "10D Majorana",        xi: "1.412" },
    { m: 8,  x: 530, y: 450, host: 3,  nc: 24,  force: "Gluon Octet",  phase: "SU(3) Adjoint",       xi: "2.108" },
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
    { g: 7,  label: "g=7 fifths",   pair: "nt", col: "#2563eb", pos: [7,2,9,4,11,6,1,8,3,10,5,0] }, // Deep Blue
    { g: 5,  label: "g=5 fourths",  pair: "nt", col: "#3b82f6", pos: [5,10,3,8,1,6,11,4,9,2,7,0] }, // Lighter Blue
    { g: 1,  label: "g=1 fwd",      pair: "tr", col: "#16a34a", pos: [1,2,3,4,5,6,7,8,9,10,11,0] }, // Solid Green
    { g: 11, label: "g=11 bwd",     pair: "tr", col: "#22c55e", pos: [11,10,9,8,7,6,5,4,3,2,1,0] }, // Lighter Green
  ];

  // Position IS the harmonic family: k → m=k (k=0 → m=12 octave closure)
  // Refined palette for vibrancy and contrast
  const k2m = (k) => k === 0 ? 12 : k;
  const famCol = { 
    1: "#dc2626", 2: "#ea580c", 3: "#9333ea", 4: "#2563eb", 5: "#5b21b6",
    6: "#0d9488", 7: "#3730a3", 8: "#78350f", 9: "#991b1b", 10: "#155e75", 
    11: "#15803d", 12: "#0f172a" 
  };

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
      background: "linear-gradient(to bottom, #fdfcfb, #f1f5f9)", 
      minHeight: "100vh", padding: "24px 8px",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
    }}>
      <div style={{ textAlign: "center", marginBottom: 12 }}>
        <div style={{ color: "#78716c", fontSize: 10, letterSpacing: 3, fontWeight: "bold" }}>THE LCM TOWER · ℓ=0</div>
        <div style={{ color: "#0f172a", fontSize: 16, fontWeight: "900", marginTop: 4 }}>N=12 · τ=6 · 6/12 active · ξ(m) = 137/((m−1)²+16)</div>
        <div style={{ color: "#64748b", fontSize: 8, marginTop: 4 }}>
          Host: d = 12/gcd(m, 12) · Mirrors: m + m' = 12 · Axis-agnostic (IC-110)
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, margin: "0 auto", display: "block" }}>

        <defs>
          {/* Drop shadows for improved node visibility */}
          <filter id="shadow-simple" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="4" stdDeviation="4" floodOpacity="0.2" floodColor="#0f172a" />
          </filter>
          <filter id="shadow-complex" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity="0.1" floodColor="#475569" />
          </filter>

          {/* Markers */}
          <marker id="aG" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill="#16a34a" />
          </marker>
          <marker id="aR" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill="#dc2626" />
          </marker>
          <marker id="aO" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <path d="M0,0 L6,2 L0,4" fill="#ea580c" />
          </marker>
          <marker id="aB" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#2563eb" />
          </marker>
          <marker id="aLB" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#3b82f6" />
          </marker>
          <marker id="aDG" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#16a34a" />
          </marker>
          <marker id="aLG" markerWidth="5" markerHeight="3" refX="4" refY="1.5" orient="auto">
            <path d="M0,0 L5,1.5 L0,3" fill="#22c55e" />
          </marker>
        </defs>

        {/* ── HASSE EDGES ── */}
        {hasse.map(([d1, d2], i) => {
          const a = nodeMap[d1], b = nodeMap[d2];
          return <line key={`h${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="#334155" strokeWidth="2" opacity="0.15" />;
        })}

        {/* ── MIRROR CROSSINGS ── */}
        {mirrors.map((mp, i) => {
          const a = nodeMap[mp.m1], b = nodeMap[mp.m2];
          return <line key={`m${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="#9333ea" strokeWidth="1.2" strokeDasharray="5,4" opacity="0.25" />;
        })}

        {/* ── SHADOW HOSTING LINES ── */}
        {complex.map(c => {
          const h = nodeMap[c.host];
          return <line key={`s${c.m}`} x1={c.x} y1={c.y} x2={h.x} y2={h.y}
            stroke="#ea580c" strokeWidth="1" strokeDasharray="3,3" opacity="0.4" />;
        })}

        {/* ── m=6 SELF-MIRROR ── */}
        <circle cx={420} cy={280} r={32} fill="none"
          stroke="#9333ea" strokeWidth="1" strokeDasharray="4,3" opacity="0.4" />

        {/* ── TRANSFER TENSOR FROM m=12 (IC-104/106/107) ── */}
        {transfers.map((t, i) => {
          const src = nodeMap[12], tgt = nodeMap[t.tgt];
          const dx = tgt.x - src.x, dy = tgt.y - src.y;
          const len = Math.sqrt(dx*dx + dy*dy);
          const nx = dx/len, ny = dy/len;
          const px = -ny, py = nx;
          const lateralOff = t.tgt === 1 ? -12 : t.tgt === 3 ? 12 : 0;
          const sx = src.x + nx * 28 + px * lateralOff, sy = src.y + ny * 28 + py * lateralOff;
          const ex = tgt.x - nx * 28 + px * lateralOff, ey = tgt.y - ny * 28 + py * lateralOff;
          const col = t.gt ? "#16a34a" : "#dc2626";
          const mid = t.gt ? "aG" : "aR";
          const mx = (sx + ex) / 2 + px * 14;
          const my = (sy + ey) / 2 + py * 14;
          return (
            <g key={`tr${i}`}>
              <line x1={sx} y1={sy} x2={ex} y2={ey}
                stroke={col} strokeWidth="1.5" opacity="0.6" markerEnd={`url(#${mid})`} />
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
          const off = 12;
          return (
            <g key={`uc${i}`}>
              <line
                x1={a.x + nx*28 + px*off} y1={a.y + ny*28 + py*off}
                x2={b.x - nx*28 + px*off} y2={b.y - ny*28 + py*off}
                stroke="#ea580c" strokeWidth="1.5" opacity="0.6" strokeDasharray="6,3"
                markerEnd="url(#aO)" />
              <text x={(a.x+b.x)/2 + px*(off+12)} y={(a.y+b.y)/2 + py*(off+12)}
                fill="#ea580c" fontSize="7" fontWeight="bold" textAnchor="middle">{i+1}</text>
            </g>
          );
        })}
        <text x={480} y={500} fill="#ea580c" fontSize="6.5" opacity="0.8" fontWeight="bold">
          m=3 excluded
        </text>
        <text x={480} y={510} fill="#ea580c" fontSize="6" opacity="0.7">
          from U(1) traversal
        </text>

        {/* ── COMPLEX SHADOW NODES ── */}
        {complex.map(c => (
          <g key={`cn${c.m}`}>
            <circle cx={c.x} cy={c.y} r={R_SHADOW}
              fill="#f8fafc" stroke="#64748b" strokeWidth="1.5" strokeDasharray="4,2" filter="url(#shadow-complex)" />
            <text x={c.x} y={c.y + 1} fill="#334155" fontSize="12" fontWeight="bold"
              textAnchor="middle" dominantBaseline="middle">{c.m}</text>
            {/* Labels below shadow sphere */}
            <text x={c.x} y={c.y + R_SHADOW + 12} fill="#1e293b" fontSize="7.5" fontWeight="bold"
              textAnchor="middle">{c.force}</text>
            <text x={c.x} y={c.y + R_SHADOW + 21} fill="#475569" fontSize="6.5"
              textAnchor="middle">{c.phase}</text>
            <text x={c.x} y={c.y + R_SHADOW + 30} fill="#ea580c" fontSize="6" fontWeight="bold"
              textAnchor="middle">@d={c.host} · N={c.nc} · ξ={c.xi}</text>
          </g>
        ))}

        {/* ── SIMPLE FAMILY NODES ── */}
        {simple.map(s => (
          <g key={`sn${s.m}`}>
            <circle cx={s.x} cy={s.y} r={R_SIMPLE}
              fill="#0f172a" stroke="#475569" strokeWidth="2.5" filter="url(#shadow-simple)" />
            <text x={s.x} y={s.y - 3} fill="#ffffff" fontSize="15" fontWeight="bold"
              textAnchor="middle">{s.m}</text>
            <text x={s.x} y={s.y + 10} fill="#cbd5e1" fontSize="7.5" fontWeight="bold"
              textAnchor="middle">φ={s.phi}</text>
            {/* Labels below simple sphere */}
            <text x={s.x} y={s.y + R_SIMPLE + 14} fill="#0f172a" fontSize="9" fontWeight="900"
              textAnchor="middle">{s.force}</text>
            <text x={s.x} y={s.y + R_SIMPLE + 24} fill="#475569" fontSize="7.5" fontWeight="bold"
              textAnchor="middle">{s.phase}</text>
            <text x={s.x} y={s.y + R_SIMPLE + 34} fill="#78716c" fontSize="6.5"
              textAnchor="middle">ξ={s.xi} · Res={"{"+s.res.join(",")+"}"}</text>
          </g>
        ))}

        {/* ═══ CASCADE ARROWS ON HASSE — 4 GENERATORS, 4 DIFFERENT PATHS ═══ */}
        {(() => {
          const arrowIds = { 7: "aB", 5: "aLB", 1: "aDG", 11: "aLG" };
          const angles = { 7: -2.4, 5: -0.7, 1: 0.7, 11: 2.4 }; 
          return cascades.map(casc => {
            const ang = angles[casc.g];
            const pts = casc.pos.map((k, si) => {
              const fam = k === 0 ? 12 : k;
              const nd = nodeMap[fam];
              const r = nd.phi !== undefined ? R_SIMPLE + 12 : R_SHADOW + 10;
              return { x: nd.x + Math.cos(ang) * r, y: nd.y + Math.sin(ang) * r, k, fam, n: si + 1 };
            });
            return (
              <g key={`cp${casc.g}`}>
                {pts.slice(0, -1).map((p, i) => (
                  <line key={`cl${casc.g}s${i}`}
                    x1={p.x} y1={p.y} x2={pts[i+1].x} y2={pts[i+1].y}
                    stroke={casc.col} strokeWidth="1.2" opacity="0.4"
                    markerEnd={`url(#${arrowIds[casc.g]})`} />
                ))}
                {pts.map((p, i) => (
                  <g key={`cm${casc.g}n${i}`}>
                    <circle cx={p.x} cy={p.y} r={6.5}
                      fill="#f8fafc" stroke={casc.col} strokeWidth="1.5" />
                    <text x={p.x} y={p.y + 3} fill={casc.col} fontSize="6" fontWeight="bold"
                      textAnchor="middle">{p.n}</text>
                  </g>
                ))}
              </g>
            );
          });
        })()}

        {/* ═══ CASCADE POSITION SEQUENCES — EACH IS A DIFFERENT PATH ═══ */}
        <text x={340} y={720} fill="#0f172a" fontSize="11" fontWeight="900" textAnchor="middle">
          CASCADE POSITION SEQUENCES — k_n = (g·n) mod 12
        </text>
        <text x={340} y={734} fill="#475569" fontSize="7" textAnchor="middle">
          Each generator traverses a DIFFERENT path through ALL 12 harmonic families. Position k = family m.
        </text>

        {/* Non-trivial pair header */}
        <text x={30} y={755} fill="#2563eb" fontSize="8.5" fontWeight="900">NON-TRIVIAL PAIR (5, 7)</text>

        {cascades.filter(c => c.pair === "nt").map((casc, ci) => {
          const baseY = 770 + ci * 48;
          return (
            <g key={`cas${casc.g}`}>
              <text x={30} y={baseY} fill={casc.col} fontSize="7.5" fontWeight="bold">{casc.label}</text>
              {casc.pos.map((k, si) => {
                const bx = 30 + si * 54;
                const fam = k2m(k);
                const fc = famCol[fam] || "#64748b";
                return (
                  <g key={`p${casc.g}s${si}`}>
                    {si > 0 && <line x1={bx - 20} y1={baseY + 13} x2={bx - 3} y2={baseY + 13}
                      stroke={casc.col} strokeWidth="1" opacity="0.3" />}
                    <text x={bx + 9} y={baseY + 7} fill="#94a3b8" fontSize="5" fontWeight="bold" textAnchor="middle">n={si+1}</text>
                    <rect x={bx} y={baseY + 8} width={18} height={12} rx={3}
                      fill={fc} opacity="0.15" stroke={fc} strokeWidth="0.8" />
                    <text x={bx + 9} y={baseY + 17.5} fill={fc} fontSize="8" fontWeight="bold" textAnchor="middle">{k}</text>
                    <text x={bx + 9} y={baseY + 26} fill="#64748b" fontSize="4.5" fontWeight="bold" textAnchor="middle">m={fam}</text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Trivial pair header */}
        <text x={30} y={875} fill="#16a34a" fontSize="8.5" fontWeight="900">TRIVIAL PAIR (1, 11)</text>

        {cascades.filter(c => c.pair === "tr").map((casc, ci) => {
          const baseY = 890 + ci * 48;
          return (
            <g key={`cas${casc.g}`}>
              <text x={30} y={baseY} fill={casc.col} fontSize="7.5" fontWeight="bold">{casc.label}</text>
              {casc.pos.map((k, si) => {
                const bx = 30 + si * 54;
                const fam = k2m(k);
                const fc = famCol[fam] || "#64748b";
                return (
                  <g key={`p${casc.g}s${si}`}>
                    {si > 0 && <line x1={bx - 20} y1={baseY + 13} x2={bx - 3} y2={baseY + 13}
                      stroke={casc.col} strokeWidth="1" opacity="0.3" />}
                    <text x={bx + 9} y={baseY + 7} fill="#94a3b8" fontSize="5" fontWeight="bold" textAnchor="middle">n={si+1}</text>
                    <rect x={bx} y={baseY + 8} width={18} height={12} rx={3}
                      fill={fc} opacity="0.15" stroke={fc} strokeWidth="0.8" />
                    <text x={bx + 9} y={baseY + 17.5} fill={fc} fontSize="8" fontWeight="bold" textAnchor="middle">{k}</text>
                    <text x={bx + 9} y={baseY + 26} fill="#64748b" fontSize="4.5" fontWeight="bold" textAnchor="middle">m={fam}</text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* ═══ ANNOTATIONS ═══ */}
        <text x={30} y={1000} fill="#9333ea" fontSize="8" fontWeight="bold">MIRRORS m+m'=12</text>
        <text x={30} y={1014} fill="#475569" fontSize="7">1↔11 · 2↔10 · 3↔9 · 4↔8 · 5↔7 · 6↔6(self)</text>

        <text x={360} y={1000} fill="#ea580c" fontSize="8" fontWeight="bold">HOST d=12/gcd(m,12)</text>
        <text x={360} y={1014} fill="#475569" fontSize="7">m=5,7,11→d=12 · m=9→d=4 · m=10→d=6 · m=8→d=3</text>

        <text x={340} y={1035} fill="#475569" fontSize="7.5" textAnchor="middle">
          Σφ(d) = 1+1+2+2+2+4 = 12 = N · Distribution [3,1,1,1,0,0] unique to ET
        </text>

        {/* ═══ STRUCTURAL SUMMARY ═══ */}
        <text x={340} y={1058} fill="#0f172a" fontSize="7.5" textAnchor="middle" fontWeight="bold">
          EM→gravity: κ=0 E=1.606 · EM→strong: κ=0 E=1.284 · EM→weak: κ≠0 E=0.343
        </text>
        <text x={340} y={1071} fill="#475569" fontSize="6.5" textAnchor="middle">
          Gravity+strong channels: D-arithmetic (deterministic). Weak channel: T-agency exclusively (IC-107).
        </text>
        <text x={340} y={1084} fill="#475569" fontSize="6.5" textAnchor="middle">
          U(1) traversal: m=1→4→2→6→12 · m=3 (strong/instanton) excluded — topological, non-perturbative
        </text>
        <text x={340} y={1097} fill="#475569" fontSize="6.5" textAnchor="middle">
          ξ monotonically decreasing (IC-109): gravity 8.563 → EM 1.000 · Axis-agnostic (IC-110)
        </text>

        {/* ═══ LEGEND ═══ */}
        <text x={340} y={1125} fill="#0f172a" fontSize="9.5" fontWeight="900" textAnchor="middle">LEGEND</text>

        <circle cx={35} cy={1145} r={6.5} fill="#0f172a" stroke="#475569" strokeWidth="1.5" />
        <text x={48} y={1148} fill="#334155" fontSize="6.5" fontWeight="bold">Simple (m|12)</text>
        <circle cx={160} cy={1145} r={6.5} fill="#f8fafc" stroke="#64748b" strokeWidth="1.5" strokeDasharray="4,2" />
        <text x={173} y={1148} fill="#334155" fontSize="6.5" fontWeight="bold">Complex (m∤12)</text>
        <line x1={285} y1={1145} x2={297} y2={1145} stroke="#334155" strokeWidth="2" opacity="0.3" />
        <text x={303} y={1148} fill="#334155" fontSize="6.5" fontWeight="bold">Hasse</text>
        <line x1={360} y1={1145} x2={372} y2={1145} stroke="#ea580c" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
        <text x={378} y={1148} fill="#334155" fontSize="6.5" fontWeight="bold">Shadow hosting</text>
        <line x1={475} y1={1145} x2={487} y2={1145} stroke="#9333ea" strokeWidth="1.2" strokeDasharray="5,4" opacity="0.4" />
        <text x={493} y={1148} fill="#334155" fontSize="6.5" fontWeight="bold">Mirror</text>

        <circle cx={35} cy={1163} r={8.5} fill="none" stroke="#9333ea" strokeWidth="1" strokeDasharray="4,3" opacity="0.6" />
        <text x={48} y={1166} fill="#334155" fontSize="6.5" fontWeight="bold">Self-mirror (m=6)</text>
        <line x1={160} y1={1163} x2={172} y2={1163} stroke="#ea580c" strokeWidth="1.5" strokeDasharray="6,3" opacity="0.8" />
        <text x={178} y={1166} fill="#334155" fontSize="6.5" fontWeight="bold">U(1) traversal</text>
        <line x1={285} y1={1163} x2={297} y2={1163} stroke="#16a34a" strokeWidth="1.5" opacity="0.8" />
        <text x={303} y={1166} fill="#334155" fontSize="6.5" fontWeight="bold">κ=0 (D-arith, E{">"}1)</text>
        <line x1={400} y1={1163} x2={412} y2={1163} stroke="#dc2626" strokeWidth="1.5" opacity="0.8" />
        <text x={418} y={1166} fill="#334155" fontSize="6.5" fontWeight="bold">κ≠0 (T-act, E{"<"}1)</text>

        <rect x={28} y={1178} width={13} height={9} rx={2.5} fill="#2563eb" opacity="0.15" stroke="#2563eb" strokeWidth="0.8" />
        <text x={48} y={1185} fill="#334155" fontSize="6.5" fontWeight="bold">(5,7) non-trivial pair</text>
        <rect x={160} y={1178} width={13} height={9} rx={2.5} fill="#16a34a" opacity="0.15" stroke="#16a34a" strokeWidth="0.8" />
        <text x={180} y={1185} fill="#334155" fontSize="6.5" fontWeight="bold">(1,11) trivial pair</text>

        <text x={340} y={1208} fill="#64748b" fontSize="6.5" textAnchor="middle">
          Force (bold) = real D-axis · Phase = imaginary T-axis · Position colors match family nodes
        </text>
      </svg>

      <div style={{
        textAlign: "center", color: "#94a3b8", fontSize: 7.5, marginTop: 10, paddingTop: 10,
        borderTop: "1px solid #e2e8f0", fontWeight: "bold"
      }}>
        P ∘ D ∘ T = E — Exception Theory — Michael James Muller — Aevum Defluo
      </div>
    </div>
  );
}