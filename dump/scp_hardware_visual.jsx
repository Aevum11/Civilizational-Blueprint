import { useState } from "react";

const SCP_Hardware = () => {
  const [showInternals, setShowInternals] = useState(true);
  const [displayActive, setDisplayActive] = useState(true);
  const [hoveredPart, setHoveredPart] = useState(null);

  const partInfo = {
    display: { name: "Free-air holographic display", desc: "VCSEL arrays + NaYF₄:Yb,Er nanoparticle aerosol. Light floating in transparent air. 239M updates/sec, 3.12μm, no pixels, no frames." },
    chip: { name: "ET-native chip array", desc: "4-8 custom dies fabricated by Mike. 36K-110K LAU cores. 12-state Webb gates. Permanent memoization. Zero accumulated error." },
    quantum: { name: "Diamond NV quantum module", desc: "CVD diamond with NV centers. 532nm laser init. 2.87GHz microwave. Room-temperature qubits. Materials ① diamond + ② nitrogen." },
    analog: { name: "Log-domain analog board", desc: "12-level resistor ladder + precision refs ③④⑤⑥⑦. Projection Π_N and pullback Π_N⁻¹. Universal codec IC-127." },
    storage: { name: "Seed store (raw NAND)", desc: "2-32TB raw NAND flash. Lattice-addressed seeds. Persistent memoization table. Deduplication built in." },
    power: { name: "Power supply", desc: "Wall → transformer → V_VEV=0.912V, V₀=0.456V, 3.3V, 5V. ~15W total. Built entirely from discrete components." },
    keyboard: { name: "Keyboard", desc: "104 mechanical switches in 13×8 matrix. FPGA-scanned directly. Sub-μs latency. No USB. Each keypress = a T-act." },
    enclosure: { name: "Enclosure", desc: "Aluminum + mu-metal + acrylic. Descriptor filter: excludes D_EM, D_magnetic, D_photon, D_vibration. Zero cooling needed." },
    vcsel: { name: "VCSEL arrays", desc: "128×128 to 512×512 per color. 980nm IR. Invisible beams cross in nanoparticle volume → visible upconversion." },
  };

  const InfoPanel = () => {
    const info = hoveredPart ? partInfo[hoveredPart] : null;
    return (
      <div style={{
        position: "absolute", bottom: 12, left: 12, right: 12,
        background: "var(--surface-1)", border: "1px solid var(--border)",
        borderRadius: 10, padding: "10px 14px", minHeight: 52,
        transition: "opacity 0.2s", opacity: info ? 1 : 0.4,
      }}>
        <div style={{ fontWeight: 500, fontSize: 13, color: "var(--text-primary)", marginBottom: 2 }}>
          {info ? info.name : "Hover over any component"}
        </div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
          {info ? info.desc : "to see its ET-native specification"}
        </div>
      </div>
    );
  };

  const glow = (active) => active ? "0 0 20px rgba(239,159,39,0.4), 0 0 40px rgba(239,159,39,0.15)" : "none";

  return (
    <div style={{ position: "relative", width: "100%", paddingBottom: "110%", fontFamily: "var(--font-sans)" }}>
      <div style={{
        position: "absolute", inset: 0,
        perspective: 900, perspectiveOrigin: "50% 35%",
      }}>
        {/* Scene container with 3D transform */}
        <div style={{
          width: "100%", height: "100%",
          transformStyle: "preserve-3d",
          transform: "rotateX(12deg) rotateY(-15deg)",
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", paddingTop: 20,
        }}>

          {/* ═══ HOLOGRAPHIC DISPLAY VOLUME ═══ */}
          <div
            onMouseEnter={() => setHoveredPart("display")}
            onMouseLeave={() => setHoveredPart(null)}
            onClick={() => setDisplayActive(!displayActive)}
            style={{
              width: 220, height: 180,
              background: "rgba(180,220,255,0.04)",
              border: "1px solid rgba(150,200,255,0.15)",
              borderRadius: 12,
              position: "relative", cursor: "pointer",
              boxShadow: glow(displayActive),
              transition: "box-shadow 0.5s",
              marginBottom: -2, zIndex: 10,
            }}
          >
            {/* Floating light points inside */}
            {displayActive && [
              { x: 35, y: 30, s: 6, c: "#5dca5d", d: 0 },
              { x: 70, y: 55, s: 4, c: "#5dcaa5", d: 0.3 },
              { x: 120, y: 40, s: 8, c: "#ef9f27", d: 0.6 },
              { x: 90, y: 80, s: 5, c: "#85b7eb", d: 0.1 },
              { x: 150, y: 65, s: 7, c: "#ed93b1", d: 0.8 },
              { x: 55, y: 110, s: 5, c: "#afa9ec", d: 0.4 },
              { x: 130, y: 100, s: 6, c: "#5dca5d", d: 0.2 },
              { x: 100, y: 130, s: 4, c: "#ef9f27", d: 0.7 },
              { x: 160, y: 120, s: 5, c: "#85b7eb", d: 0.5 },
              { x: 45, y: 145, s: 3, c: "#ed93b1", d: 0.9 },
              { x: 180, y: 45, s: 4, c: "#afa9ec", d: 0.15 },
              { x: 75, y: 155, s: 6, c: "#5dcaa5", d: 0.55 },
            ].map((p, i) => (
              <div key={i} style={{
                position: "absolute", left: p.x, top: p.y,
                width: p.s, height: p.s, borderRadius: "50%",
                background: p.c,
                boxShadow: `0 0 ${p.s*2}px ${p.c}, 0 0 ${p.s*4}px ${p.c}40`,
                animation: `float ${2+p.d*2}s ease-in-out ${p.d}s infinite alternate`,
              }} />
            ))}
            <div style={{
              position: "absolute", top: 6, left: 0, right: 0,
              textAlign: "center", fontSize: 10, color: "rgba(150,200,255,0.5)",
              letterSpacing: 1,
            }}>HOLOGRAPHIC VOLUME</div>

            {/* VCSEL array indicators on sides */}
            <div
              onMouseEnter={(e) => { e.stopPropagation(); setHoveredPart("vcsel"); }}
              onMouseLeave={() => setHoveredPart("display")}
              style={{
                position: "absolute", left: -18, top: "50%", transform: "translateY(-50%)",
                width: 14, height: 60, background: "rgba(239,159,39,0.3)",
                border: "1px solid rgba(239,159,39,0.5)", borderRadius: 3,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              <div style={{ fontSize: 8, color: "#ef9f27", writingMode: "vertical-lr", transform: "rotate(180deg)" }}>VCSEL</div>
            </div>
            <div style={{
              position: "absolute", right: -18, top: "50%", transform: "translateY(-50%)",
              width: 14, height: 60, background: "rgba(239,159,39,0.3)",
              border: "1px solid rgba(239,159,39,0.5)", borderRadius: 3,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <div style={{ fontSize: 8, color: "#ef9f27", writingMode: "vertical-lr", transform: "rotate(180deg)" }}>VCSEL</div>
            </div>
          </div>

          {/* ═══ MAIN ENCLOSURE ═══ */}
          <div
            onMouseEnter={() => setHoveredPart("enclosure")}
            onMouseLeave={() => setHoveredPart(null)}
            style={{
              width: 300, height: 200,
              background: "linear-gradient(135deg, var(--surface-0) 0%, var(--surface-1) 100%)",
              border: "2px solid var(--border-strong)",
              borderRadius: 10,
              position: "relative",
              boxShadow: "0 8px 32px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.1)",
            }}
          >
            {/* ET LLC label */}
            <div style={{
              position: "absolute", top: 6, left: 0, right: 0,
              textAlign: "center", fontSize: 9, color: "var(--text-muted)",
              letterSpacing: 2, fontWeight: 500,
            }}>EXCEPTION THEORY LLC</div>

            {showInternals && (
              <div style={{ padding: "22px 10px 8px", display: "flex", gap: 6, height: "calc(100% - 30px)" }}>
                {/* Chip carrier board */}
                <div
                  onMouseEnter={(e) => { e.stopPropagation(); setHoveredPart("chip"); }}
                  onMouseLeave={() => setHoveredPart("enclosure")}
                  style={{
                    flex: 2, background: "rgba(29,158,117,0.12)",
                    border: "1px solid rgba(29,158,117,0.35)", borderRadius: 6,
                    padding: 4, display: "flex", flexDirection: "column", gap: 3,
                  }}
                >
                  <div style={{ fontSize: 8, color: "#1d9e75", textAlign: "center", fontWeight: 500 }}>ET CHIP ARRAY</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 3, flex: 1 }}>
                    {[1,2,3,4].map(i => (
                      <div key={i} style={{
                        background: "rgba(29,158,117,0.25)", borderRadius: 3,
                        border: "1px solid rgba(29,158,117,0.4)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 7, color: "#1d9e75",
                      }}>Die {i}</div>
                    ))}
                  </div>
                  <div style={{ fontSize: 7, color: "#1d9e75", textAlign: "center" }}>36K+ LAU cores</div>
                </div>

                {/* Quantum module */}
                <div
                  onMouseEnter={(e) => { e.stopPropagation(); setHoveredPart("quantum"); }}
                  onMouseLeave={() => setHoveredPart("enclosure")}
                  style={{
                    flex: 1.2, background: "rgba(127,119,221,0.12)",
                    border: "1px solid rgba(127,119,221,0.35)", borderRadius: 6,
                    padding: 4, display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
                  }}
                >
                  <div style={{ fontSize: 8, color: "#7f77dd", fontWeight: 500 }}>QUANTUM</div>
                  {/* Diamond crystal */}
                  <div style={{
                    width: 24, height: 24, transform: "rotate(45deg)",
                    background: "rgba(175,169,236,0.3)", border: "1px solid #afa9ec",
                    borderRadius: 3, marginTop: 4,
                  }} />
                  <div style={{ fontSize: 7, color: "#afa9ec" }}>Diamond NV</div>
                  <div style={{
                    width: 6, height: 18, background: "rgba(93,202,165,0.5)",
                    borderRadius: 2, marginTop: 2,
                  }} />
                  <div style={{ fontSize: 7, color: "#5dcaa5" }}>532nm</div>
                </div>

                {/* Analog board */}
                <div
                  onMouseEnter={(e) => { e.stopPropagation(); setHoveredPart("analog"); }}
                  onMouseLeave={() => setHoveredPart("enclosure")}
                  style={{
                    flex: 1.5, background: "rgba(216,90,48,0.1)",
                    border: "1px solid rgba(216,90,48,0.3)", borderRadius: 6,
                    padding: 4, display: "flex", flexDirection: "column", gap: 2,
                  }}
                >
                  <div style={{ fontSize: 8, color: "#d85a30", textAlign: "center", fontWeight: 500 }}>ANALOG</div>
                  {/* Resistor symbols */}
                  <div style={{ display: "flex", gap: 2, justifyContent: "center", flexWrap: "wrap", flex: 1 }}>
                    {["φ","2^⅐","⁷⁄₆","2^⅛","2^¹⁄₁₁"].map((r, i) => (
                      <div key={i} style={{
                        padding: "2px 4px", background: "rgba(216,90,48,0.2)",
                        borderRadius: 2, fontSize: 7, color: "#d85a30",
                        border: "1px solid rgba(216,90,48,0.3)",
                      }}>{r}</div>
                    ))}
                  </div>
                  <div style={{ fontSize: 7, color: "#d85a30", textAlign: "center" }}>Refs ③-⑦</div>
                  <div style={{ fontSize: 7, color: "#d85a30", textAlign: "center" }}>Π_N + IC-127</div>
                </div>
              </div>
            )}

            {/* Bottom section: storage + power */}
            <div style={{
              position: "absolute", bottom: 0, left: 0, right: 0,
              display: "flex", gap: 4, padding: "0 10px 6px",
            }}>
              <div
                onMouseEnter={(e) => { e.stopPropagation(); setHoveredPart("storage"); }}
                onMouseLeave={() => setHoveredPart("enclosure")}
                style={{
                  flex: 1, background: "rgba(99,153,34,0.15)",
                  border: "1px solid rgba(99,153,34,0.3)", borderRadius: 4,
                  padding: "2px 4px", fontSize: 7, color: "#639922", textAlign: "center",
                }}>NAND 2-32TB</div>
              <div
                onMouseEnter={(e) => { e.stopPropagation(); setHoveredPart("power"); }}
                onMouseLeave={() => setHoveredPart("enclosure")}
                style={{
                  flex: 1, background: "rgba(136,135,128,0.15)",
                  border: "1px solid rgba(136,135,128,0.3)", borderRadius: 4,
                  padding: "2px 4px", fontSize: 7, color: "var(--text-secondary)", textAlign: "center",
                }}>PSU ~15W</div>
            </div>

            {/* Seed Protocol port on side */}
            <div style={{
              position: "absolute", right: -12, top: "40%",
              width: 10, height: 20, background: "rgba(29,158,117,0.3)",
              border: "1px solid rgba(29,158,117,0.5)", borderRadius: "0 4px 4px 0",
            }} />
            <div style={{
              position: "absolute", right: -60, top: "40%",
              fontSize: 8, color: "#1d9e75", transform: "translateY(-50%)",
            }}>Seed Protocol<br/>→ other SCPs</div>
          </div>

          {/* ═══ KEYBOARD ═══ */}
          <div
            onMouseEnter={() => setHoveredPart("keyboard")}
            onMouseLeave={() => setHoveredPart(null)}
            style={{
              width: 280, height: 55, marginTop: 16,
              background: "var(--surface-1)",
              border: "1px solid var(--border-strong)",
              borderRadius: 6, padding: "4px 8px",
              display: "flex", flexDirection: "column", gap: 2,
              cursor: "pointer",
            }}
          >
            {[0,1,2,3].map(row => (
              <div key={row} style={{
                display: "flex", gap: 2,
                marginLeft: row * 4,
              }}>
                {Array.from({ length: row === 3 ? 8 : (row === 0 ? 14 : 13) }).map((_, i) => (
                  <div key={i} style={{
                    flex: (row === 3 && (i === 0 || i === 7)) ? 2 : 1,
                    height: 9, background: "var(--surface-0)",
                    border: "1px solid var(--border)",
                    borderRadius: 1.5,
                  }} />
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div style={{
          position: "absolute", top: 8, right: 8,
          display: "flex", gap: 8, zIndex: 20,
        }}>
          <button onClick={() => setShowInternals(!showInternals)} style={{
            background: "var(--surface-1)", border: "1px solid var(--border)",
            borderRadius: 6, padding: "4px 10px", fontSize: 11,
            color: "var(--text-secondary)", cursor: "pointer",
          }}>{showInternals ? "Close case" : "Open case"}</button>
          <button onClick={() => setDisplayActive(!displayActive)} style={{
            background: "var(--surface-1)", border: "1px solid var(--border)",
            borderRadius: 6, padding: "4px 10px", fontSize: 11,
            color: "var(--text-secondary)", cursor: "pointer",
          }}>{displayActive ? "Display off" : "Display on"}</button>
        </div>

        {/* Title */}
        <div style={{
          position: "absolute", top: 8, left: 12,
          fontSize: 13, fontWeight: 500, color: "var(--text-primary)",
        }}>SCP — physical hardware</div>

        <InfoPanel />
      </div>

      <style>{`
        @keyframes float {
          0% { transform: translateY(0px) scale(1); opacity: 0.7; }
          100% { transform: translateY(-8px) scale(1.3); opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default SCP_Hardware;
