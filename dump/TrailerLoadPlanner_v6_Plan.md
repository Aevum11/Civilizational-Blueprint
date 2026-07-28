# TrailerLoadPlanner v5 → v6 Upgrade Plan

**Base file:** 639-line v5 Mike uploaded  
**Method:** str_replace edits ONLY — never recreate the file  

---

## 1. Dual-Slot Persistence (Manual + Auto Save)

Two independent save slots. Auto save never touches manual save.

### Storage Keys

| Slot | Keys | Written by |
|------|------|------------|
| Auto | `tlp-auto-trailer`, `tlp-auto-items`, `tlp-auto-setup` | useEffect on every state change |
| Manual | `tlp-manual-trailer`, `tlp-manual-items`, `tlp-manual-setup` | User taps Save button |

### Behavior

**Auto save:** Three useEffect hooks fire on state change. Each writes to the `tlp-auto-*` key. Debounce not needed — window.storage is async and fast enough. Guards: skip writes while `loading` is true (prevents overwriting on mount).

**Manual save:** A "💾 Save" button in the items step and plan step. Writes all three keys to `tlp-manual-*` in one action. Shows brief "Saved ✓" confirmation (state-based, clears after 2 seconds via setTimeout).

**Load on startup:**
1. Check if `tlp-manual-items` exists
2. Check if `tlp-auto-items` exists
3. If both exist: load manual (it's the intentional save)
4. If only auto exists: load auto
5. If only manual exists: load manual
6. If neither: fresh start

**Reset All:** Clears both slots. Confirms with user first.

### Edits Required

- Add `useEffect` to import (line 1)
- Add `loading` state (default true), `savedMsg` state (default "")
- Add load-on-mount useEffect with the priority logic above
- Add three auto-save useEffects (trailer, items, setup form state)
- Add `manualSave` handler that writes all three manual keys + sets savedMsg
- Add Save button in items step (near the Calculate button)
- Add Save button in plan step (near the bottom)
- Update Reset All to delete all 6 storage keys
- Update `_id` to `Date.now()%1e6` or max of loaded item IDs + 1

---

## 2. Exact Weight Option

Optional lbs input alongside weight feel categories.

### Data Model Change

Item gains `exactLbs` field: `null` if not entered, number if entered.

```
{id, name, l, w, h, weight: "heavy", exactLbs: 75, category: "furniture", surface: "flat", cutouts: []}
```

### Helper

```javascript
const itemWeight = (it) => it.exactLbs || WT[it.weight].lbs;
```

### Edits Required

- Add `itemWeight` helper after `nid` (line 23 area)
- Add `cExact` state variable
- Modify `addItem`: include `exactLbs: parseFloat(cExact)||null` in item object
- Modify `startEdit`: set `cExact` from item's `exactLbs`
- Modify `cancelEdit`: reset `cExact` to ""
- Replace ALL `WT[item.weight].lbs` in computePlan (line 157) with `itemWeight(item)`
- Replace ALL `WT[p.item.weight].lbs` in computePlan weight accumulators (line 164) with `itemWeight(p.item)`
- Replace `WT[item.weight].lbs` in findPos (line 215) with `itemWeight(item)`
- Replace `WT[p.item.weight].lbs` in buildHeatmap (line 259) with `itemWeight(p.item)`
- Replace `WT[p.item.weight].lbs` in cogHeight calculation (line 171) with `itemWeight(p.item)`
- Update `runW` useMemo to use `itemWeight` instead of `WT[it.weight].lbs`
- Add input field after weight buttons in items UI: `"Or enter exact weight (lbs)"`
- Update item list display: show `${it.exactLbs} lbs` when set, else `wc.label`
- Update selected item detail in plan view similarly

---

## 3. Undo Delete

### Edits Required

- Add `undoStack` state (empty array)
- Modify `deleteItem`: push the deleted item onto undoStack before filtering
- Add undo button between item form and item list: `"↩ Undo delete: [name]"` (only shown when undoStack has items)
- `undoDelete` handler: pop last item from stack, add back to items

---

## 4. Error Handling

### Edits Required

- Add `err` state (empty string)
- Add `errBox` computed element: red-bordered card showing `err` when non-empty
- Modify `setTrailerDims`:
  - Set err if any dimension ≤ 0: `"All dimensions must be positive."`
  - Set err if dimensions implausibly large (L>600, W>120, H>120): `"Dimensions seem too large."`
  - Clear err on success
- Modify `addItem`:
  - Set err if name empty: `"Enter a name."`
  - Set err if dimensions ≤ 0: `"All dimensions must be positive."`
  - Set err if no category: `"Select a category."`
  - Set err if no weight: `"Select a weight feel."`
  - Set warning (non-blocking) if item dimensions > trailer dimensions
  - Clear err on success
- Render `errBox` at top of trailer step and items step

---

## 5. Simple/Advanced Onboarding

### Edits Required

- Add `showAdv` state (default false)
- Add "▶ Refine Shape" toggle button between the Bounding Dimensions card and the Cross-Section card
- Wrap the four advanced cards (Cross-Section, Width Along Length, Obstacles, Door) in `{showAdv && <>...</>}`
- The Bounding Dimensions card (L×W×H + weight limit) stays always visible
- Button text toggles: "▶ Refine Shape" / "▼ Hide Shape Details"

---

## 6. 3D Isometric Floor Fix

The iso view currently draws a rectangular floor polygon. It should use the actual measured shape.

### Edits Required

In the iso view (line 528 area), replace the rectangular floor polygon:

```jsx
<polygon points={pp([ip(0,0,0),ip(W,0,0),ip(W,L,0),ip(0,L,0)])} .../>
```

With a polygon generated from the length profile:

```javascript
const cs = (trailer.cs&&trailer.cs.length>=2) ? [...trailer.cs].sort((a,b)=>a.h-b.h) : [{h:0,w:W},{h:H,w:W}];
const lp = (trailer.lp&&trailer.lp.length>=1) ? [...trailer.lp].sort((a,b)=>a.d-b.d) : [{d:0,w:W},{d:L,w:W}];
const fW = cs[0].w;
const floorPts = [];
const floorSteps = 30;
for(let i=0; i<=floorSteps; i++){
  const y = (i/floorSteps)*L;
  const wAtY = interp(lp, y, "d", "w");
  const sw = fW>0 ? Math.min(wAtY/fW,1)*fW : 0;
  const cx = W/2;
  floorPts.push({l: ip(cx-sw/2, y, 0), r: ip(cx+sw/2, y, 0)});
}
// Build path: left edge forward, right edge backward
let floorPath = `M ${floorPts[0].l[0]},${floorPts[0].l[1]}`;
for(const fp of floorPts) floorPath += ` L ${fp.l[0]},${fp.l[1]}`;
floorPath += ` L ${floorPts[floorPts.length-1].r[0]},${floorPts[floorPts.length-1].r[1]}`;
for(let i=floorPts.length-1; i>=0; i--) floorPath += ` L ${floorPts[i].r[0]},${floorPts[i].r[1]}`;
floorPath += " Z";
```

Then render as `<path d={floorPath} .../>` instead of the rectangular polygon.

Also move the existing `shade` helper (currently inline in the iso view at line 537) to the utility section — already handled in Edit 2 of this plan.

---

## 7. Loading Order List — Visibility Dimming

### Edits Required

In the loading order list (line 628 area), add visibility check and opacity:

```jsx
const isVis = plan._visIds && plan._visIds.has(p.item.id);
// Add to the style: opacity: isVis ? 1 : 0.35
```

Items not in the current step+weight filter are dimmed but still visible, so the user always sees the complete loading sequence.

---

## Edit Order

Apply in this sequence to minimize conflicts:

1. Import change (line 1) — useEffect
2. Utility helpers (line 23 area) — itemWeight, shade
3. State variables (line 292 area) — cExact, undoStack, err, loading, savedMsg, showAdv
4. Persistence hooks (after state, before handlers) — load + auto-save useEffects
5. Handler modifications — addItem, startEdit, cancelEdit, deleteItem, setTrailerDims, calc, manualSave, undoDelete
6. computePlan/findPos/buildHeatmap — itemWeight replacements
7. Trailer setup UI — onboarding toggle + advanced wrapper + error display
8. Items UI — exact weight input, undo button, error display, save button
9. Plan view — 3D floor fix, loading list dimming, save button
10. Reset All — clear all 6 storage keys

Each edit is a single str_replace operation on the exact text in the 639-line file. Verify line count after each group.
