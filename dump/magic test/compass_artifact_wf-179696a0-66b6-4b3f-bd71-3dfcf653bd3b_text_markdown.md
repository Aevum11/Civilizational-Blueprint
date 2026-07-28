# The Metallic Arts of Scadrial and Sanderson's Hard Magic Framework: A Comprehensive Technical Reference

Brandon Sanderson's *Mistborn* novels, set on the planet Scadrial within the larger Cosmere, feature what is widely cited as the canonical example of a "hard" magic system. Three interlocking Metallic Arts — Allomancy, Feruchemy, and Hemalurgy — share a single elemental focus (sixteen base metals organized as eight metal/alloy pairs, plus the God Metals) but draw from different metaphysical sources, follow different conservation laws, and produce radically different effects. This document lays out their full operating parameters, the meta-framework Sanderson uses to design them, and the Cosmere-level cosmology that ties them together.

---

## 1. ALLOMANCY

### 1.1 Fundamental Mechanics

Allomancy is the art of "burning" metals to draw on the Investiture (raw magical energy) of the Shard Preservation. The Allomancer is not the source of the power; the metal is a focus, and Preservation's Investiture is the fuel. Sanderson has analogized the metals to nozzles on a Play-Doh extruder: the press (the Allomancer's soul) supplies pressure, and each metal "shapes" the released power into a different ability.

To activate the magic, the Allomancer must:

1. **Ingest the metal** in a usable form. Metals are typically swallowed as small flakes suspended in alcohol or oil (a "metal vial"). Solid lumps will pass through the digestive tract, but flakes are absorbed and held in the stomach where they can be reached by the Allomancer's reserve.
2. **The metal must be Allomantically pure.** Every burnable metal is either an elemental metal or a precise alloy. Burning an impure alloy makes the Allomancer sick (lessens efficiency, causes nausea); burning a metal that is not Allomantically viable at all (silver, lead, platinum, etc.) cannot be done — the body simply does not recognize it as a fuel — and ingesting non-viable metals like lead is just toxic.
3. **The Allomancer chooses to "burn"** the reserve, which begins metabolizing it and producing the effect. Burn rate is roughly steady; copper is the slowest-burning of the eight basic metals, atium burns extremely fast.
4. **Flaring** is the act of consciously consuming the metal faster for a stronger but shorter-lived effect.

### 1.2 The 16-Metal Table: Four Quadrants

The sixteen base Allomantic metals are organized as a 4×4 grid of complementary pairs:

| Quadrant | Pulling (base metal) | Pushing (alloy) | Internal Pulling | Internal Pushing |
|---|---|---|---|---|
| **Physical** | Iron (external pull) | Steel (external push) | Tin (internal pull) | Pewter (internal push) |
| **Mental** | Zinc (external pull) | Brass (external push) | Copper (internal pull) | Bronze (internal push) |
| **Temporal (Hybrid)** | Cadmium (external pull) | Bendalloy (external push) | Gold (internal pull) | Electrum (internal push) |
| **Enhancement** | Chromium (external pull) | Nicrosil (external push) | Aluminum (internal pull) | Duralumin (internal push) |

Two organizing principles run through the table:

- **Pulling vs. Pushing.** Every base elemental metal pulls (concentrates, draws toward, enhances input); every alloy pushes (projects, expels, dampens, releases).
- **Internal vs. External.** Internal metals act on the Allomancer's own body or soul; external metals act on the world or on other people.

The Lord Ruler's regime suppressed knowledge of more than half of these, so during Era 1 (the original trilogy), commoners knew only the eight basic metals (iron, steel, tin, pewter, zinc, brass, copper, bronze) plus gold and atium; aluminum, duralumin, chromium, nicrosil, cadmium, bendalloy, and electrum were either secret, undiscovered, or known only to obligators and the Steel Ministry. Atium and malatium were also incorrectly classified as Temporal metals before the Catacendre; they are now properly understood as God Metals.

### 1.3 What Each Metal Does When Burned

**Physical Quadrant**
- **Iron (External Pulling — Lurcher).** Pulls nearby pieces of metal toward the Allomancer's "center of self" (a Spiritual rather than purely physical center of gravity). Pulls are strictly axial — straight along a line between Allomancer and target. Iron also lets the user *see* metal: faint blue lines radiate from the Allomancer's chest to every nearby piece of metal, with thickness proportional to mass. Cannot affect aluminum or some of its alloys.
- **Steel (External Pushing — Coinshot).** Pushes metals directly away. The same blue-line vision applies. Steelpushing is the iconic Mistborn flying ability: an Allomancer pushes against a coin or anchor on the ground, effectively "pushing the planet," and is hurled upward.
- **Tin (Internal Pulling — Tineye).** Enhances all five senses simultaneously. Tineyes can see through Scadrial's mists (a property derived from Allomancy's Preservation source), see in low light (but not absolute darkness), and pick up faint sounds and smells. Senses cannot be enhanced selectively — tin amplifies environmental cold, heat, hunger, and pain along with vision and hearing. Flaring tin gives a sensory burst that can clear a head fogged by fatigue or mental influence.
- **Pewter (Internal Pushing — Thug/Pewterarm).** Boosts physical strength, speed, balance, endurance, and resistance to pain and damage. Pewter "drags" — the body operates beyond normal limits while burning, then crashes ("pewter drag") afterward.

**Mental Quadrant**
- **Zinc (External Pulling — Rioter).** Pulls/inflames a target's emotions, "Rioting" specific feelings to greater intensity. Cannot directly read minds; the Rioter chooses a general emotion (fear, anger, trust) and amplifies it.
- **Brass (External Pushing — Soother).** Damps, "Soothes," specific emotions. With sufficient skill (and especially with duralumin), a Soother can bring a person's will so low that the target effectively obeys; this is also how kandra and koloss are controlled.
- **Copper (Internal Pulling — Smoker/Coppercloud).** Creates an invisible bubble (a "coppercloud") that hides the Allomantic pulses given off by anyone burning metal inside it. It also makes the Smoker themself immune to emotional Allomancy. Copperclouds also block other forms of Investiture detection — Awakener life sense, secretspren, even the rhythms a singer can attune to. Multiple Smokers stack. Copper is the slowest-burning of the basic metals.
- **Bronze (Internal Pushing — Seeker).** Lets the Allomancer hear/feel "Allomantic pulses" — the rhythmic emanations given off by other burning Allomancers. Different metals have distinct pulse signatures. Most Seekers cannot pierce a coppercloud, but a Seeker with a Hemalurgic bronze spike, a bronze savant, or one fortified by duralumin or nicrosil can.

**Temporal/Hybrid Quadrant**
- **Gold (Internal Pulling — Augur).** Shows the Allomancer a vision of who they could have been if they had made different choices in the past — an alternate self. Disorienting and considered useless or even psychologically dangerous by most.
- **Electrum (Internal Pushing — Oracle).** Shows possible futures of *oneself* — the Oracle's own atium-style shadow, a few seconds out. Electrum's most important property is that it is the **counter to atium**: an Oracle burning electrum prevents an enemy Seer's atium from working clearly, because the futures fork into a tangled web. Often called "poor man's atium." Ideal allomantic electrum has a 9:11 gold-to-silver ratio.
- **Cadmium (External Pulling — Pulser).** Creates a stationary bubble (room-sized) inside which time passes more slowly relative to the outside. Anyone partly inside is included. Objects entering or leaving are deflected unpredictably because parts of them traverse different time rates. Bubbles cannot be moved unless the Pulser is on a sufficiently massive moving body (e.g., a planet or train).
- **Bendalloy (External Pushing — Slider).** The opposite: a smaller bubble inside which time passes faster. Same edge effects (deflection, popping if the Slider walks out). Bendalloy is an alloy of bismuth, lead, tin, and cadmium (50/26.7/13.3/10). Originally Brandon planned the bubbles to redshift/blueshift light passing through, but this was retconned to a Spiritual Realm energy transfer to avoid the implication that observers would be microwaved. **Cadmium and bendalloy bubbles overlapping cancel out**; multiple same-type bubbles multiply their effect.

**Enhancement Quadrant**
- **Aluminum (Internal Pulling — Aluminum Gnat).** Burns away all of the Allomancer's other metal reserves instantly, with no other effect. Effectively useless on its own, but aluminum the *substance* is enormously important across the Cosmere: it is magically inert, blocks Investiture, blocks Steelpushes/Ironpulls, blocks emotional Allomancy (aluminum-lined hats), creates dead zones inside cadmium/bendalloy bubbles, generates no atium shadow, cannot be Forged or Soulcast, blocks Adhesion and Gravitation Surges, and can possibly prevent Hemalurgic decay if a spike is encased in it. Known elsewhere as "ralkalest" (Sel) and historically as "starmetal" (Azir).
- **Duralumin (Internal Pushing — Duralumin Gnat).** When burned alone, does nothing useful — but burned simultaneously with another metal, it instantly consumes that metal's entire reserve and produces a single massive burst of its effect (a duralumin-fueled Soothe can dominate a person; a duralumin-fueled Steelpush can launch an Allomancer enormous distances; a duralumin-fueled atium burn extends future-sight further). Duralumin is the alloy of aluminum (with copper, etc.).
- **Chromium (External Pulling — Leecher).** When the Leecher touches another Allomancer, that Allomancer's metal reserves are wiped — chromium is essentially "aluminum applied to someone else."
- **Nicrosil (External Pushing — Nicroburst).** When the Nicroburst touches an Allomancer who is burning metals, those metals are instantly all-consumed in a duralumin-style burst affecting the *target*. With duralumin/nicrosil, a Mistborn can briefly pierce a coppercloud.

### 1.4 The God Metals and Other Specials

- **Atium (Ruin's God Metal).** A Misting who burns atium is a **Seer**. The standard "atium" most people knew was actually an atium-electrum alloy mined from the Pits of Hathsin; pure atium has different properties. Burning the alloy produces "atium shadows" — visual projections of every nearby being's actions a few seconds in the future — and simultaneously expands the user's mind to process the flood of information, allowing them to dodge attacks even from behind. The only practical counter is electrum or another atium burner (whose competing prediction creates a recursive splitting of the atium shadow). Atium burns very fast. Pure atium, when burned, instead grants a vision into the Spiritual Realm. Atium is condensed Investiture of Ruin and does not draw on Preservation.
- **Malatium (the "Eleventh Metal").** Atium-gold alloy. Burned, it shows a vision of who *another person* was in their past or could have been. Powered by Ruin's Investiture, not Preservation's. Discovered/leaked by Kelsier; rumors of it (started possibly by Ruin) helped destabilize the Lord Ruler. When flared at a critical moment it can grant a glimpse of another person's Spiritual Realm Connection.
- **Lerasium (Preservation's God Metal).** A normal person who burns a bead of lerasium becomes a full-power Mistborn permanently; the change is hardcoded into their Spiritual DNA and inheritable (though it dilutes over generations). The Lord Ruler took ten beads from the Well of Ascension; nine he gave to kings, one he consumed himself. Lerasium alloyed with one of the sixteen base metals turns the burner into a Misting of that metal. A Mistborn or Misting burning lerasium has their existing powers drastically increased. Burning enough lerasium to become a savant would make one Ascend to become Preservation. Cannot be alloyed with atium to make harmonium by mundane means; the two only combine via Investiture-driven processes (notably trellium reactions on harmonium).
- **Cerrobend.** This is *not* a separate magical metal. "Cerrobend" was the original working name for what became bendalloy — Brandon was forced to change it because Cerrobend is a real-world trademarked alloy name. Sometimes confused with lerasium because both begin with similar phonemes.

### 1.5 The Mechanics of Steelpushing and Ironpulling

Pushing and pulling on metal is not magnetism (it works on bronze, gold, etc., which aren't ferromagnetic) and is not pure gravitation, though it shares Newton's third law:

- **Pushes/pulls are strictly axial** along the line from the Allomancer's "center of self" (a Spiritual concept, near but not identical to the physical center of gravity) to the target metal.
- **Force is roughly proportional to the Allomancer's mass.** A heavier Allomancer pushes harder. This is why iron Feruchemy (storing weight) combos so devastatingly with steel/iron Allomancy.
- **Reciprocity:** if the target is more massive (or anchored to something more massive — e.g., a coin nailed to the ground, anchored to the planet), the Allomancer is thrown; if the target is lighter, it flies; if comparable, both move.
- **Anchoring matters.** A coin lying on the floor is "anchored" to the planet through friction/contact and behaves as essentially infinitely massive. The same coin in mid-air will simply fly away.
- **Force is inversely proportional to distance**, with a "zenith" — a maximum altitude beyond which a steelpush cannot lift the user further. Duralumin or larger anchors raise the zenith.
- **Metal partly inside a living person's body resists being pushed/pulled.** This is why Inquisitors can be pushed but not easily, and why Feruchemists sometimes implant their metalminds. Resistance scales with the target's level of Investiture (Susebron, who holds many Breaths, is harder to affect than a Drab). A sufficiently strong push (e.g., one mist-fueled or duralumin-amplified) can overcome this.
- **Skilled Allomancers** can push on individual sections of an object, deflect bullets along curved paths through follow-up pushes, push through copper wires to flicker electric lights (Era 2), and "ride" a fired bullet by pushing on it as it leaves the barrel to drive it through cover.
- **Aluminum and aluminum alloys cannot be pushed or pulled.**
- **Visualization:** burning iron or steel produces faint blue lines radiating from the Allomancer's chest to every nearby metal, thickness proportional to size.

### 1.6 Mistborn vs. Mistings

- **Mistings** are Allomancers who can burn only one metal. Each metal has a distinct name (Coinshot for steel, Lurcher for iron, Tineye for tin, Thug/Pewterarm for pewter, Soother for brass, Rioter for zinc, Smoker for copper, Seeker for bronze, Augur for gold, Oracle for electrum, Pulser for cadmium, Slider for bendalloy, Aluminum Gnat for aluminum, Duralumin Gnat for duralumin, Leecher for chromium, Nicroburst for nicrosil, Seer for atium).
- **Mistborn** can burn all sixteen metals. They are extraordinarily rare. Mistborn are the descendants (genetically) of the ten people the Lord Ruler dosed with lerasium.
- **Pre-Catacendre Era 1:** every Allomancer is descended from those original ten lerasium recipients (with rare exceptions of late-trilogy mist-Snapped commoners). Allomancy has been diluting for a thousand years; Mistborn are vanishingly rare by Vin's time.
- **Post-Catacendre Era 2:** Mistborn are essentially extinct (legendary). Most metalborn are Mistings. Allomantic strength continued to fade.
- A **Twinborn** has one Misting power and one Ferring power; a **Compounder** is a Twinborn whose two powers share a metal.

### 1.7 Snapping

Allomancy is hardcoded into the spiritweb but generally dormant. To unlock it, the Allomancer must "Snap" — a traumatic event (severe physical pain, near-death, intense emotional distress) creates a "crack" in the soul through which the power can flow. Realmatically, Snapping is the moment Investiture connects to a previously latent capacity.

- In the Final Empire, noble children of confirmed Mistborn lineages were beaten as a planned test for Snapping.
- Late in the original trilogy, Preservation's mists themselves begin Snapping people who otherwise would never have triggered, as a machine-like mechanism Preservation set up before relinquishing his consciousness. Ruin corrupts this so it kills some — the so-called mist-sickness or "Deepness." Exactly 16% of mist-exposed Scadrians Snap (the prevalence of 16 throughout the system being a deliberate clue Preservation left). One-sixteenth of those (the "mistfallen") are sick the longest; the longest-sick prove to be the rare atium Mistings (Seers).
- Snapping by extreme joy or other strong emotions is also possible.
- Burning lerasium hardcodes Allomancy without requiring Snapping; the bead's size is proportional to power.
- After the Final Ascension, **Harmony altered how Snapping works**; the modern method is unspecified but no longer requires (or kills people through) the mists. Once Snapped, Allomancers immediately have access to full power; they do not need to "level up" the way Knights Radiant do.

### 1.8 Savantism

An Allomancer who flares a metal continuously over long periods becomes physiologically warped by the constant inflow of Investiture; their spirit becomes infused, with effects bleeding into the physical. This is **savantism**.

- Savants gain raw power and often unique secondary effects.
- They develop heavy dependence — extinguishing the metal causes withdrawal-like effects.
- For most metals, savantism is irreversibly damaging.
- **Spook** is the canonical tin savant; his senses became so acute that ordinary daylight blinded him and ambient sound deafened him (he wore blindfolds and earplugs); during his savant phase he could anticipate attacks almost atium-style by reading air currents.
- **Copper savants** are an exception: copperclouds from a Smoker savant are far harder to pierce, and copper savantism is comparatively non-debilitating.
- A bronze savant can pierce a coppercloud naturally.
- **Lerasium savantism** is essentially impossible to achieve by burning more — the amount needed would Ascend the user to Preservation.
- **Feruchemical savants** are also nearly impossible (Feruchemy doesn't draw external power) unless one Compounds. Surgebinders can become savants, and every Surge can produce one.

---

## 2. FERUCHEMY

### 2.1 Fundamental Mechanics

Feruchemy is the **end-neutral** art: nothing is gained from outside, nothing is permanently lost. The Feruchemist converts an attribute of themselves into Investiture, *stores* it inside a metal in physical contact ("filling," "storing"), and later *taps* the metalmind to retrieve it. Metalminds are keyed to the Feruchemist's **Identity** and ordinarily inaccessible to anyone else.

- **Storing** an attribute drains it: a Feruchemist storing strength feels weak, storing speed moves slowly, storing memory forgets, storing senses goes blind/deaf, storing health gets sick, etc.
- **Tapping** retrieves it, often at amplified rates (storing 50% of strength for an hour can yield 150% strength for less than an hour).
- **Compression cost.** If you tap faster than you stored, some attribute is lost to enable the temporal compression. Storing 50% for 1 hour and tapping at 200% will not get you a full 30 minutes; the harder you compress, the more you bleed.
- **No theoretical upper limit** on tap rate, but storage rate is naturally limited by the Feruchemist's physical capacity.
- **Metalmind size** affects capacity, but non-linearly (with thresholds rather than smooth scaling). Breaking a metalmind splits the charge across pieces because the stored Investiture behaves like a gas in a jar.
- **Feruchemists can sense how much is stored** in their own metalminds simply by touching them.
- **Identity-keying:** another Feruchemist can sense (but not use) someone else's metalminds, but only if they share the relevant power. An aluminummind storing all of one's Identity can produce subsequent metalminds usable by *anyone* with that power.
- **Molten metal** can be stored and tapped (with obvious risks). Alloying a Feruchemically charged metal locks the original Investiture inside (it occupies "space" but can't be retrieved).
- **Alloy purity** matters as in Allomancy: impure alloys hold less.

### 2.2 The Sixteen Feruchemical Powers

| Quadrant | Metal | Stores | Ferring Name |
|---|---|---|---|
| Physical | Iron | Physical weight | Skimmer |
| Physical | Steel | Physical speed | Steelrunner |
| Physical | Tin | Sensitivity of one sense (one tinmind per sense) | Windwhisperer |
| Physical | Pewter | Physical strength | Brute |
| Mental | Zinc | Mental speed (thinking) | Sparker |
| Mental | Brass | Warmth (body heat) | Firesoul |
| Mental | Copper | Memories | Archivist |
| Mental | Bronze | Wakefulness | Sentry |
| Hybrid | Gold | Health (healing) | Bloodmaker |
| Hybrid | Electrum | Determination | Pinnacle |
| Hybrid | Cadmium | Breath (oxygen) | Gasper |
| Hybrid | Bendalloy | Nutrition / calories | Subsumer |
| Spiritual | Chromium | Fortune (luck) | Spinner |
| Spiritual | Nicrosil | Investiture | Soulbearer |
| Spiritual | Aluminum | Identity | Trueself |
| Spiritual | Duralumin | Spiritual Connection | Connector |

Notes on specific edge cases:

- **Tin** stores one sense per metalmind, not all five together — and unlike Allomantic tin, it does not let you sense things otherwise unsensable; it merely amplifies what's already available.
- **Brass** Ferrings storing warmth become cold but produce an internal furnace later; they may potentially be semi-fireproof when tapping.
- **Copper** memories don't decay over time *in the metalmind* (though they degrade in the Feruchemist's own brain after being stored, since they're filed individually rather than as a single reserve; Keepers train their natural memory and create indexed copperminds).
- **Gold** Ferrings (Bloodmakers) heal extraordinarily; Compounders of gold (like Miles Hundredlives) are nearly unkillable.
- **Nicrosil Feruchemy** is unique and barely understood even by Terris: it allows the storage of Investiture itself. This implies one can effectively store *other powers*, and is the foundation of unsealed metalminds and the post-Catacendre Southern Scadrian technology.
- **Aluminum** stores Identity; an Aluminum Trueself who fully stores their Identity can then create metalminds usable by other Feruchemists with the relevant power.
- **Duralumin** stores Connection — used for instant trust-building or to make oneself "forgotten."

### 2.3 Feruchemists vs. Ferrings

- A **full Feruchemist** can use all sixteen Feruchemical metals. Originally, Feruchemy was confined to the Terris people and was bred for; full Feruchemists were the norm there.
- After the Catacendre, the slaughter of the Synod and centuries of Terris-Scadrian intermingling diluted the genetic line. The Allomancy genes interfere with Feruchemy genes, breaking it into single-power **Ferrings**. By Era 2, full Feruchemists are essentially extinct.

### 2.4 Compounding

A Feruchemist who can Allomantically burn the *same metal* they Feruchemically charge can perform **Compounding**: store a Feruchemical attribute in a metalmind, then Allomantically burn that metalmind. The end-positive Allomantic burn draws additional Preservation Investiture to amplify the Feruchemical attribute, breaking end-neutrality. The result: a tiny stored amount can yield enormous output.

- **The Lord Ruler** Compounded all sixteen metals, making him simultaneously immortal (gold), perpetually young (atium-Feruchemy: stored youth), inhumanly strong (pewter), and so on. He maintained a thousand-year reign on this basis.
- **Miles Hundredlives** is a gold-only Compounder; he heals nearly any wound instantly.
- The Bands of Mourning are a legendary Compounding artifact.
- Compounding traditionally describes Allomancy enhancing Feruchemy, but Sanderson has confirmed the inverse direction is also possible.
- A full Mistborn-and-Feruchemist combination is "almost completely impossible" because Mistborn and full Feruchemy spiritual DNAs occupy overlapping space; the Lord Ruler's case was a unique consequence of his specific Ascension.

### 2.5 Resonances

A Twinborn (Misting + Ferring) gains a third, secondary effect from the interaction of the two arts called a **Resonance**. The most famous example: Wax Ladrian, a Crasher (steel Misting + iron Ferring), can form a "steel bubble" deflecting metal around himself — originally written as a Resonance, now framed as a savant ability. Each of the 16×16 = 256 Twinborn combinations has a unique Resonance name and effect, though only a handful (notably Crasher) are explicitly named in canon.

---

## 3. HEMALURGY

### 3.1 Fundamental Mechanics

Hemalurgy is the **end-negative** art of Ruin: powers are stolen from one person and grafted onto another, with permanent net loss in transfer. It is the magic of theft.

The procedure:

1. A metal **spike** is driven into a specific **bind point** on a donor's body, contacting moving blood. The spike must come into contact with moving blood — that's why the art is called "Hemalurgy" (blood-magic). Most commonly the spike pierces the heart, killing the donor.
2. The spike rips off a piece of the donor's **spiritweb** (Spiritual aspect/sDNA), keyed to the type of metal used, the bind point used to remove, and **Intent**. Intent matters: the Hemalurgist must mean to perform the theft.
3. The charged spike is then driven into a specific bind point on the recipient. The spike "staples" the foreign sDNA fragment into the recipient's spiritweb, granting them the stolen attribute (less whatever is lost to decay and transfer).
4. Death is *not* strictly required: a charged spike can be made from a living donor, leaving them in a state worse than a Drab. But for maximum charge, killing the donor is standard, and direct heart-to-heart transfer is most efficient.
5. Spikes can be charged via thrown spike or fired spike-gun if the projectile lands on the right bind point with the right Intent.

### 3.2 Bind Points

There are roughly **200–300 bind points** in the human body, each useful for distinct effects. The location of removal and the location of insertion both matter; the same metal at different bind points produces different stolen attributes. Right and left eye bind points are slightly different. There are no bind points in the mouth/digestive tract (per current notes).

Burning atium (and thus seeing into the Spiritual Realm) helps a Hemalurgist place spikes correctly. The bind-point map is extremely difficult to determine empirically without Investiture-aided sight.

### 3.3 The Hemalurgic Table (Sixteen Spike Effects)

| Metal | Steals |
|---|---|
| Iron | Strength (raw human attribute) |
| Steel | A Physical Allomantic power (iron, steel, tin, or pewter) |
| Tin | Senses (raw human attribute) |
| Pewter | A Physical Feruchemical power |
| Zinc | Emotional fortitude / mental strength |
| Brass | A Cognitive (Mental) Feruchemical power |
| Copper | Mental fortitude, memory, or intelligence |
| Bronze | A Mental Allomantic power |
| Chromium | Destiny / Fortune (Spiritual human attribute) |
| Nicrosil | A Spiritual Allomantic power *or* raw Investiture (Breaths, etc.) — extraordinarily versatile |
| Aluminum | Identity / wipes powers |
| Duralumin | Connection / a Spiritual Feruchemical power |
| Gold | A Hybrid Feruchemical power (gold, electrum, cadmium, bendalloy — including healing) |
| Electrum | An Enhancement Allomantic power (aluminum, duralumin, chromium, nicrosil) |
| Cadmium | A Temporal (Hybrid) Allomantic power |
| Bendalloy | A Spiritual Feruchemical power |
| **Atium** | Any power (must be "refined") |
| **Lerasium** | All abilities |

The pairing pattern: base elemental metals tend to steal raw human attributes or specific kinds of Feruchemy; alloys tend to steal Allomantic powers. The four quadrants (Physical/Mental/Hybrid/Enhancement) determine which subset is accessible. Note that early editions of *The Hero of Ages* erroneously had pewter spikes granting healing; later editions corrected this to gold spikes. Brandon has indicated pewter spikes can also reach into the Hybrid quadrant for some effects.

Hemalurgy can also steal:
- Surgebinding (difficult — requires spiking both the Surgebinder and their spren; the spren retains autonomy and can break the bond)
- Breaths (Awakening) via nicrosil
- Divine Breaths from Returned
- Forms from singers (only as a copy, not as theft per se)
- Innate strength/abilities from non-Allomancers (this is how koloss were made)

### 3.4 Hemalurgic Decay

Once removed from a body, a charged spike begins losing potency by the **Law of Hemalurgic Decay**. This proceeds like a half-life: there's an initial sharp drop when first removed, then progressively slower decay. Over centuries a spike degrades to a shadow of its original power — Wax's earring (originally Vin's) gives only a trickle of zinc by Era 2. Methods to slow decay:

- Direct heart-to-heart transfer (the spike is essentially never out of a body).
- **Coating in fresh blood** — Spook discovered this; near-perfect preservation, though the blood must be replenished.
- **Aluminum encasing** is hypothesized to prevent decay entirely.
- **Inside a body** the charge is stable.

Spikes physically broken in pieces split the charge among the pieces, with additional loss. Once a spike is used to grant a power, you cannot reuse it for a different power simply by moving it — the act of placement at first insertion locks the function.

### 3.5 Costs and Side Effects

Every Hemalurgic spike tears holes in the recipient's spiritweb. These holes:

- Allow Ruin (or another sufficiently powerful Shard) to whisper to or directly **control** the bearer. One spike: hearing/manifestations. Two spikes: control becomes possible but resistible (kandra removed their spikes when they felt Ruin's voice). Four-plus spikes: near-total control by a willing Shard.
- Allow Allomantic emotional control (Soother/Rioter) to dominate the bearer with greatly reduced effort.
- Twist the body into something inhuman, especially when the spike steals raw attributes from non-Allomancers/Feruchemists. Inquisitor hearts are displaced; brains warp around eye-spikes. Koloss faces grow, skin stretches, eyes recede.
- A wearer with fewer spikes and a strong will can resist control for periods.
- Spikes can also savant (per WoB: "yes, [savantism] is possible with Hemalurgy").

### 3.6 Hemalurgic Constructs

Four canonical types on Scadrial:

1. **Steel Inquisitors.** 9–11 spikes of various metals, each granting a specific Allomantic or Feruchemical power. Their iconic eye-spikes are steel/iron, granting them Allomantic iron and steel sight (they "see" the blue Allomantic lines of all surrounding metal in lieu of normal vision). A **linchpin spike** between the shoulder blades holds their fraying spiritweb together; removal kills them. Mistborn or Seekers were preferred candidates because stacking with an additional bronze spike makes them coppercloud-piercing. Tattoos around their eye sockets denoted rank. Made by hammering a charged spike from a freshly killed Misting/Feruchemist directly into the new Inquisitor (minimizing decay).

2. **Koloss.** Four iron spikes, each charged with stolen *human strength* from regular people (not Allomancers — Sanderson realized killing Allomancers to make foot-soldiers was a poor cost-benefit ratio). Koloss lose intelligence, gain immense strength, and continue growing for life until their skin can no longer contain them. Susceptible to Soothing/Rioting control, especially via duralumin- or nicrosil-amplified emotional Allomancy. **Post-Catacendre,** Harmony reshaped them into a true breeding race; their offspring are "koloss-blooded" and only become full koloss if they choose the spike-rite at maturity.

3. **Kandra.** Two spikes ("the Blessings"), distinguishing them from non-sapient mistwraiths from which they are made. Spikes grant attributes (Presence, Potency, Stability, Awareness — varies by Blessing). Two spikes is the minimum to reach human-like sapience without being trivially dominated by Ruin; they were able to remove their own spikes when they felt Ruin's grip tightening. The First Generation of kandra were Rashek's original Terris friends, transformed during his Ascension. Removing both spikes reverts a kandra to a mistwraith. They eat bodies to copy them, since they can't generate hair, bone, or carapace; they sometimes wear "True Bodies" of crafted bone, including translucent quartz sets. After the Catacendre, Inquisitor spikes were melted down into the kandra-given earrings worn by Pathians (which connect them faintly to Harmony).

4. **Hemalurgic monsters / Era 2 constructs (e.g., koloss-blooded, spiked humans).** Era 2 introduces Allomancy/Feruchemy granted to ordinary people via single charged spikes, generally with less spiritweb damage than the gross Inquisitor/koloss procedures.

### 3.7 Why Hemalurgy is End-Negative

Ruin built Hemalurgy with a flaw (the holes in the soul) that lets him exert influence; this is intrinsic to its destructive Intent. Unlike Allomancy (where Preservation supplies the energy without depleting the user) or Feruchemy (conserved transfer through time), every Hemalurgic operation loses some Investiture forever — and the constructs it creates are perpetually weaker than the original donor was.

---

## 4. SANDERSON'S LAWS OF MAGIC

These are not laws of physics within the Cosmere but Sanderson's meta-framework for *designing* magic systems as a writer. He stresses they are guidelines, not rules.

### 4.1 First Law (2007)

> *An author's ability to solve conflict with magic is directly proportional to how well the reader understands said magic.*

This is the formal articulation of the **soft vs. hard magic** distinction. If a magic system's rules are clearly defined to the reader (hard magic — Mistborn), the author can have characters resolve conflicts by clever application of those rules without producing deus ex machina. If the magic is mysterious (soft magic — Gandalf, the Force in the original Star Wars), conflicts should be resolved by character, theme, or non-magical action; using soft magic to solve plot problems feels like cheating. Magic "explained to the reader before it is used to resolve a conflict" is the ideal.

### 4.2 Second Law (2012)

> *Limitations are more important than powers.* (Limitations > Powers)

What makes a magic system interesting is not what it can do but what it cannot. Sanderson explicitly subdivides the negative-space rules into three categories:

- **Limitations** — things the magic intrinsically *can't* do. (Steelpushes only work on metal; pushes are strictly axial; you can't push metal that's inside someone's body easily; copperclouds block detection.)
- **Weaknesses** — things the magic is vulnerable to. (Aluminum negates Allomancy; copperclouds hide Allomancers; emotional Allomancy can be blocked; metals can run out.)
- **Costs** — the price of using the magic. (Allomancers must continually consume metal; Feruchemists must drain themselves first; Hemalurgists destroy donors and corrupt recipients; Wheel of Time channelers go insane.)

Iconic Sanderson example: Allomancers can't push or pull on metals inside a body, can only push/pull along strict lines, and can't push without an anchor. These constraints generate tension and force creative problem-solving.

### 4.3 Third Law (2013)

> *Expand what you already have before you add something new.*

Worldbuilders should deepen existing systems rather than pile on additional ones. The three sub-principles:

- **Extrapolation** — when introducing a new ability, ask what its second- and third-order consequences are. If iron can be Pushed, what about coins, doorknobs, water-pipes, hand-drills?
- **Interconnection** — make systems connect. Allomancy, Feruchemy, and Hemalurgy share the same sixteen metals and the same realmatic substrate; characters with one tend to interact meaningfully with characters who have another. Compounding only works because the systems share metals.
- **Streamlining** — cut excess and redundant rules; let simpler core principles produce richer combinatorial results. Mistborn's 16-metal grid is a powerful streamlined design that yields hundreds of named Misting/Ferring/Twinborn types.

### 4.4 The "Zeroth Law"

In conversation Sanderson has identified an informal Zeroth Law: *err on the side of what is awesome*. The other laws should bend to a sufficiently compelling idea rather than the reverse. This is a nod to Asimov's Zeroth Law of Robotics.

---

## 5. COSMERE-LEVEL CONTEXT

### 5.1 Investiture

**Investiture** is the underlying magical energy of the Cosmere — Sanderson's "mana" or unified-field substance. Properties:

- **It cannot be created or destroyed**, only transformed; it has its own laws of thermodynamics.
- It exists across three Realms: Physical, Cognitive, and Spiritual. Magic is largely the manipulation of Investiture across realms.
- Every object in the Cosmere has a **spiritweb** — the Spiritual aspect, made of Investiture and Connections, an idealized Platonic version of the object.
- **Innate Investiture** is the extra Spiritual energy in a person beyond the baseline spiritweb, providing a conduit to the Spiritual Realm. All Scadrians have some innate Investiture from Preservation; Allomancers have more. Drabs (Nalthians who gave away their Breath) are below baseline.
- Investiture can be **kinetic** (actively doing something, detectable by Allomantic bronze, Taldain sand) or **static** (a charged metalmind, dormant Lights).
- It can manifest as solids (atium, lerasium), liquids, or **gases** (Stormlight, Breath, Scadrian mists). Gaseous Investiture is the primary fuel for Awakening and Surgebinding, and can fuel any system as a stand-in.
- Investiture **resists** other Investiture: a heavily Invested object (the King of Hallandren with thousands of Breaths, an aluminum bar, a charged metalmind) is harder to magically affect.
- Each Shard has a unique **Tone** and **Rhythm** that pulses through their Investiture; Allomantic pulses (felt by Seekers) carry Preservation's tone.

### 5.2 Shards of Adonalsium

Long ago, an entity called **Adonalsium** was Shattered on Yolen by sixteen conspirators; its power split into sixteen **Shards**, each a vast fragment of the divine, with a specific **Intent** that increasingly dominates its Vessel's psyche. Known Shards include Preservation, Ruin, Honor, Cultivation, Odium, Endowment, Devotion, Dominion, Autonomy, Whimsy, Valor, Virtuosity, Mercy, Invention, etc., and the hybrid Shards Harmony (Preservation + Ruin) and Retribution (Honor + Odium).

Each Shard's Investiture is keyed to its Intent, Identity, and Tone, and it has at least partial influence over all Investiture in the Cosmere matching its Intent.

### 5.3 The Three Metallic Arts and Their Shards

Scadrial was created jointly by **Leras** (Preservation) and **Ati** (Ruin), and the three Metallic Arts map onto the dialectic of these two Shards plus their later combination, **Harmony** (Sazed):

- **Allomancy** is of **Preservation**. It is **end-positive**: Investiture flows into the Allomancer from the Shard via the metal as catalyst. Nothing is consumed except the small physical metal, which is a key, not the fuel. This is why mass duralumin/lerasium can deplete or super-charge an Allomancer — their access pipe to Preservation is being widened.
- **Hemalurgy** is of **Ruin**. It is **end-negative**: every transfer destroys some Investiture; the holes in the soul are the price; Ruin built it precisely so that anyone using its power becomes more controllable by him.
- **Feruchemy** is the art of **balance**, drawing on both Shards equally. It is **end-neutral**: power is shifted across time, not created or destroyed. Of the three, only Feruchemy was known to humans on Scadrial before the climactic conflict between Preservation and Ruin.

The number sixteen is woven through all three systems (sixteen metals, sixteen Shards, sixteen percent of mist-exposed people Snapping, sixteen years between major events, etc.) — Preservation deliberately seeded this as a clue to humans about the Cosmere's structure.

### 5.4 Harmony and the Future of Scadrian Magic

After Vin killed Ati and herself perished in the process of holding Preservation, **Sazed** picked up both Shards simultaneously, becoming **Harmony** — the first dual-Shard Vessel in the Cosmere. He used the combined power to perform the Catacendre, restoring Scadrial to a habitable state and eventually transforming the koloss and kandra into true breeding races. The opposing Intents make Harmony slow to act, and over centuries Ruin's influence has gradually grown within him; the original Terris prophecy of the Hero of Ages stated "his name shall be **Discord**," foreshadowing this drift.

Harmony altered Snapping after the Final Ascension, has restricted access to atium and lerasium (both faded into legend), and oversees Era 2's modernized Scadrial. Era 2 also discovers **harmonium (ettmetal)**, a violently reactive metal that is the alloyed God Metal of Harmony itself; alloying atium and lerasium does *not* yield harmonium, but **trellium** (Autonomy's bavadinium, the God Metal of the Shard Autonomy) splits ettmetal into atium and lerasium with explosive energy release — the basis of the Bands of Mourning's restoration of Mistborn. The interaction of these god metals is the foundation of Era 3.

Other Shardic systems on other worlds use entirely different focuses (Stormlight + spren bonds for Surgebinding on Roshar, Breath + color for Awakening on Nalthis, AonDor + location on Sel, Forging via stamps on Sel), but all draw on the same underlying Investiture. Scadrian metals can interact with foreign Investitures: aluminum, for instance, blocks magic universally; nicrosil spikes can steal Breaths; Hemalurgy can in principle steal Surgebinding. This cross-Shard interaction is becoming an increasing focus of late-Cosmere stories.

---

## 6. SUMMARY OF DESIGN ELEGANCE

What makes Sanderson's Scadrian system the canonical example of hard magic is the way each of his three Laws is satisfied in cascading layers:

- **Sixteen base metals** organized into a **4×4 quadrant grid** (Physical/Mental/Temporal/Enhancement × Pulling/Pushing × Internal/External) — a streamlined skeleton.
- **Three orthogonal applications** of those same metals — Allomancy (Preservation, end-positive, ingested), Feruchemy (balance, end-neutral, contact-stored), Hemalurgy (Ruin, end-negative, spike-implanted) — interconnection at the deepest level.
- **Hard limitations** — pure alloys only, axial Pushes, mass-proportional force, anchoring requirements, finite reserves, bind-point precision, identity-keying — that constantly force characters to think creatively.
- **God Metals** (atium, lerasium, malatium, harmonium) that bend the rules in defined ways, providing escalation without breaking the framework.
- **Compounding, Twinborn Resonances, Hemalurgic constructs, savantism, and Snapping** as second-order phenomena that emerge from the interactions of the basic rules — exactly the Third-Law expansion Sanderson advocates.

The result is a magic system in which the reader can, by the end of *The Hero of Ages*, reason about scenarios the characters haven't yet encountered and predict outcomes — and in which the climactic resolutions come from clever applications of established rules rather than arbitrary new powers. It is the practical demonstration of all three of Sanderson's Laws operating at once.