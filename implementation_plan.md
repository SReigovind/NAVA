# NAVA NAVA — Implementation Plan
## Session: 12 May 2026

> This plan covers the UI uplift, immediate bug fixes, and near-term feature work.
> Items are ordered by priority and grouped by effort.

---

## Phase 0 — Immediate Bug Fixes (this session, ~30 min)

These are small, targeted fixes with no design ambiguity.

| # | Fix | File(s) | Change |
|---|---|---|---|
| 1 | Remove text labels from overview plant health dots | `OverviewPanel.jsx` | Delete `.ov-dot-type` and `.ov-dot-label` spans; keep dot + `title` tooltip only |
| 2 | Center col 1 content in Diagnose + VNIR 1:1:1 cards | `styles.css` | Add `justify-content: center; align-items: center` to `.diag-body` and `.vnir-body` |
| 3 | Fix VNIR col 1 scrollable frame | `styles.css` | Remove `max-height: 340px; overflow-y: auto` from `.vnir-body`; let content breathe naturally |
| 4 | Chat default name = short date + time | `ChatPanel.jsx` | Change `makeLabel()` to return `"12 May · 14:32"` format |
| 5 | Landing page: single "Get Started" CTA (remove Sign In button) | `Landing.jsx` | Remove the Sign In `<button>` |
| 6 | Landing page: add footer | `Landing.jsx` | Add `<footer>` with project name, institution, year, brief description |

---

## Phase 1 — Chat Word-by-Word Animation (~1 hour)

Implement a streaming-style reveal effect for assistant messages in `ChatPanel.jsx`.

### How it works
- When a new assistant message arrives, store it in a `pendingMessage` ref.
- Render it word-by-word using `setInterval` at ~40 ms/word (configurable).
- While animating, show a blinking cursor `▍` at the end of the revealed text.
- `renderMarkdown` is applied to the **fully revealed** text after animation completes.
- During animation, render only plain text (avoid partial markdown tags breaking mid-parse).
- User messages are never animated.

### State additions
```js
const [animating, setAnimating] = useState(false);
const [revealedText, setRevealedText] = useState("");
const [pendingWords, setPendingWords] = useState([]);
```

### Edge cases
- If the user sends another message while animating, skip to end immediately.
- Scroll-to-bottom should track during animation.

---

## Phase 2 — Landing Page Redesign (~2 hours)

### Goals
- First impression: feels like a professional agri-tech product, not a student project.
- Animated hero section — subtle floating elements, not distracting.
- Clear information hierarchy.

### Sections

#### Hero
- Full-viewport height.
- Left: NAVA logo + tag + headline + subheadline + single "Get Started →" CTA.
- Right: animated visual — a stylized crop health dashboard mockup (CSS-only, no images needed) OR a subtle animated background grid.
- Entrance animation: headline fades up, CTA slides in after 300ms delay.

#### Feature Strip
- 3 cards in a row (already exists, redesign the card style).
- Each card: icon (SVG or styled div, not emoji), title, one-liner.
- Subtle hover: card lifts with box shadow.

#### How It Works
- 3-step numbered flow: "Create a field → Add crops → Run scans & ask NAVA".
- Horizontal on desktop, connected by a dashed line.

#### Footer (new)
- Project name: **NAVA — Neural Agricultural Virtual Assistant**
- Institution: Kerala Agricultural University / MSc AI & ML
- Year: 2026
- Brief: "An AI-powered crop health and advisory platform."
- Links: none needed for now (no live site).

### Animations
- CSS `@keyframes fadeInUp` for hero content on load.
- Feature cards: `transition: transform 0.2s, box-shadow 0.2s` on hover.
- Background: subtle animated gradient shift on hero (slow, 8s cycle).

---

## Phase 3 — Post-Login User Dashboard (~2 hours)

Currently the `/fields` route is a plain list. Replace with a dashboard layout.

### Layout
```
┌─────────────────────────────────────────────────────┐
│  HEADER                                              │
├─────────────────────────────────────────────────────┤
│  Welcome back, [Name]           [+ New Field]        │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Fields  │ │  Crops   │ │ Scans    │ │Concerns│ │
│  │    3     │ │    8     │ │   24     │ │   2    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
│                                                      │
│  YOUR FIELDS                                         │
│  ┌──────────────────┐ ┌──────────────────┐          │
│  │ North Plot       │ │ South Plot       │          │
│  │ 3 crops · Loamy  │ │ 2 crops · Red    │          │
│  │ ● Concern: 1     │ │ ● All clear      │          │
│  └──────────────────┘ └──────────────────┘          │
│                                                      │
│  RECENT ACTIVITY                          (all →)    │
│  🔬 Late Blight · Row-1 · Tomato · 2 hrs ago        │
│  📡 OK · Plant-A · Pepper · Yesterday               │
└─────────────────────────────────────────────────────┘
```

### Data needed
- `GET /api/fields` — already exists.
- For each field: `GET /api/crops?field_id=X` — already exists.
- Cross-field events: `GET /api/events?limit=10` (without field filter) — already works.
- Aggregate stats computed client-side.

### Aggregate stat card logic
- Total fields: `fields.length`
- Total crops: sum of crops across all fields (parallel fetch)
- Total scans: count of events with `event_type = "diagnose" | "vnir"`
- Active concerns: disease events where label does NOT contain "healthy"

### Field card redesign
- Remove emoji prefix badges.
- Show crop count, soil type, last activity date.
- Show a health indicator: green dot = all clear, red dot = N concerns.

---

## Phase 4 — Field-Level Dashboard (~1.5 hours)

`FieldDetail.jsx` currently shows just metadata + crop grid. Upgrade to a dashboard layout.

### Layout
```
┌─────────────────────────────────────────────────────┐
│  ← Fields   North Plot            [Edit] [+ Crop]   │
├──────────────────────────┬──────────────────────────┤
│  FIELD INFO              │  FIELD HEALTH SUMMARY    │
│  Location: Wayanad       │  Crops: 3                │
│  Area: 2 acres           │  Plants: 7               │
│  Soil: Loamy             │  🔴 Concerns: 2          │
│  Notes: [editable]       │  🟢 Healthy: 5           │
├──────────────────────────┴──────────────────────────┤
│  CROPS                                              │
│  ┌────────────────┐ ┌────────────────┐             │
│  │ Tomato         │ │ Pepper         │             │
│  │ Fruiting       │ │ Vegetative     │             │
│  │ 2 plants · ⚠️1 │ │ 3 plants · ✓  │             │
│  └────────────────┘ └────────────────┘             │
├─────────────────────────────────────────────────────┤
│  RECENT FIELD ACTIVITY                              │
│  Last 5 events across all crops in this field       │
└─────────────────────────────────────────────────────┘
```

### Data needed
- Already available via existing endpoints.
- Aggregate events per field: `GET /api/events?field_id=X&limit=20`.

---

## Phase 5 — Micro-Animations Polish (~1 hour)

Subtle animations that feel alive without being distracting.

| Element | Animation | CSS |
|---|---|---|
| Stat cards on dashboard | Fade-in-up with stagger (0ms, 60ms, 120ms, 180ms) | `@keyframes fadeInUp` + `animation-delay` |
| Field cards on hover | Lift `translateY(-3px)` + green glow shadow | `transition: transform 0.2s, box-shadow 0.2s` |
| Sidebar nav item active | Slide-in indicator bar from left | `::before` pseudo-element `scaleX` |
| Dot indicators | Scale-up pulse once on first appearance | `@keyframes dotPop` |
| Modal open | Scale from 0.95 + fade-in | `@keyframes modalEnter` |
| Chat bubble appear | Slide-in from bottom (user) or left (assistant) | `@keyframes bubbleIn` |

---

## Out of Scope (waiting on data)

- **RAG / Knowledge Base** — requires curated agricultural PDFs in `knowledge_base/sources/`. Once data is available: ingest → ChromaDB → `/api/chat` retrieval step before LLM call.
- **Multilingual** — Bhashini API integration (Malayalam/Tamil/Hindi).
- **Push notifications** — requires service worker.
- **Mobile layout** — explicitly deferred per user decision.

---

## Execution Order

```
[TODAY]  Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
```

Start with Phase 0 (bugs) first — they are already implemented during planning.
