# NAVA NAVA — UI Design System

> Documents the visual design language, CSS architecture, theming system, and
> component patterns used in the React frontend.

---

## Design Philosophy

The interface is designed to feel like a **digital agronomist's dashboard** — professional,
data-dense, and legible — rather than a consumer app. Key principles:

- **No emojis in data cards.** Status is communicated through colour, typography, and shape.
- **Colour is semantic.** Green = healthy/ok, Red = disease/stress/critical, Blue = calibrating/informational, Grey = absent data.
- **Text is the primary medium.** Labels, measurements, and delta values are formatted precisely; images supplement text, not replace it.
- **Micro-animations are restrained.** Confidence gauge fills animate on mount; hover effects are subtle (scale, border highlight).

---

## Theming

The app supports **Light** and **Dark** mode via a `data-theme` attribute on `<html>`.
The preferred mode is persisted in `localStorage` under the key `nava_theme`.

```css
:root               { /* Dark mode defaults */ }
[data-theme="light"] { /* Overrides */ }
```

### Core Tokens

| Token | Dark | Light | Usage |
|---|---|---|---|
| `--bg-primary` | `#0c1117` | `#f0f4f0` | Page background |
| `--bg-secondary` | `#131a20` | `#e8efe8` | Sidebar, header |
| `--bg-card` | `#1a2332` | `#ffffff` | Cards, panels |
| `--bg-glass` | `rgba(255,255,255,0.04)` | `rgba(255,255,255,0.8)` | Glassmorphism overlays |
| `--bg-input` | `#0f1923` | `#f0f4f0` | Form inputs |
| `--text-primary` | `#e8f0e8` | `#1a2e1a` | Headings, labels |
| `--text-secondary` | `#9eb89e` | `#2d5a2d` | Body text |
| `--text-muted` | `#4a6a4a` | `#6b8f6b` | Hints, meta |
| `--text-accent` | `#4ade80` | `#166534` | Highlighted values |
| `--green-400` | `#4ade80` | `#16a34a` | Primary brand colour |
| `--green-500` | `#22c55e` | `#15803d` | Buttons, accents |
| `--border-default` | `rgba(255,255,255,0.06)` | `rgba(0,0,0,0.08)` | Card/panel borders |
| `--border-hover` | `rgba(255,255,255,0.15)` | `rgba(0,0,0,0.2)` | Interactive hover |
| `--shadow-glow` | `0 0 20px rgba(74,222,128,0.1)` | similar | Card hover glow |

### App Header

- **Dark mode:** translucent `rgba(12,17,23,0.9)` with `backdrop-filter: blur(16px)`.
- **Light mode:** solid `#064e3b` (deep forest green) — forced so that white logo text and
  ghost buttons remain readable against a bright page background.

---

## Layout Architecture

### App Shell

```
<html data-theme="dark|light">
  <body>
    <div class="app-layout">
      <header class="app-header" />       ← sticky, 57px tall
      <main class="app-main" />           ← padded, max-width 1200px (standard pages)
      OR
      <div style="flex:1; overflow:hidden"> ← noPadding (crop workspace)
```

### Crop Workspace

The crop detail page bypasses `app-main` and renders a full-height sidebar layout:

```
.crop-workspace  (display: flex; height: calc(100vh - 57px))
├── .crop-sidebar.open/collapsed   (220px / 56px; flex-direction: column)
│   ├── .sidebar-top               (collapse toggle + back button)
│   ├── .sidebar-crop-info         (crop name + stage pill)
│   ├── .sidebar-nav               (nav items with active state)
│   └── .sidebar-footer            (Edit Crop button)
└── .crop-main                     (flex: 1; overflow: hidden)
    ├── .crop-main-header          (tool title + breadcrumb)
    └── .crop-tool-body            (overflow-y: auto; padding: 24px 28px)
```

---

## Component Patterns

### Buttons

| Class | Use |
|---|---|
| `.btn.btn-primary` | Main CTA (green gradient) |
| `.btn.btn-secondary` | Secondary action (outlined) |
| `.btn.btn-ghost` | Minimal, transparent |
| `.btn.btn-sm` | Compact size |
| `.danger-hover` | Ghost that turns red on hover |

### Badges / Pills

| Class | Colour |
|---|---|
| `.badge-green` | Green background — healthy state |
| `.badge-red` | Red background — disease / alert |
| `.badge-neutral` | Grey background — informational |

### Status Chips (Disease Detection)

| Class | Colour | Use |
|---|---|---|
| `.chip-reliable` | Green | Model output is reliable |
| `.chip-unreliable` | Amber | Low confidence result |
| `.chip-action` | Red | Disease detected, action needed |

### Dot Indicators (Overview)

| Class | Colour | Disease meaning | VNIR meaning |
|---|---|---|---|
| `.dot-green` | `#10b981` + glow | Healthy | OK / Healthy |
| `.dot-red` | `#ef4444` + glow | Disease detected | Warning / Stress / Critical |
| `.dot-blue` | `#3b82f6` + glow | — | Calibrating |
| `.dot-gray` | `#6b7280` | No scan | No scan |

Dots scale to `1.4×` on hover. Each carries a native `title` tooltip.

---

## Result Card System

### 1:1:1 Row Layout

```css
.result-row   { display: flex; gap: 12px; align-items: stretch; }
.result-col   { flex: 1; min-width: 0; display: flex; flex-direction: column; }
```

Stacks vertically below 700 px viewport width.

### Disease Detection Col 1 (`.diag-status-col`)

```
┌─ 4px severity bar (green/red gradient) ─────────────┐
│                                                       │
│  ● HEALTHY / ● DISEASE DETECTED   ← .diag-status-tag│
│                                                       │
│  Late Blight                      ← .diag-label      │
│  Tomato                           ← .diag-crop-tag   │
│                                                       │
│  Model Confidence          99%                        │
│  ████████████████░░░       ← .diag-conf-track        │
│                                                       │
│  ● Reliable   Action needed  ← .diag-chip-row        │
└───────────────────────────────────────────────────────┘
```

Label formatting: strip leading `cropname_` prefix, replace `_` with space, title-case.

### VNIR Monitor Col 1 (`.vnir-status-col`)

Tier system drives border colour and tag styling:

| Tier | CSS class | Colour |
|---|---|---|
| ok | `.vnir-ok` | Green |
| warning | `.vnir-warning` | Orange |
| critical | `.vnir-critical` | Red |
| calibrating | `.vnir-calibrate` | Blue |

```
┌─ 4px gradient severity bar ─────────────────────────┐
│                                                       │
│  ● OK / STRESS DETECTED / …  ← .vnir-status-tag     │
│  Full status string           ← .vnir-status-text    │
│                                                       │
│  MEASUREMENTS                 ← section title        │
│  ┌─────────┬─────────┬──────┐                        │
│  │  0.7423 │  142.3  │ 198.7│ ← 3-col mini grid     │
│  │VNIR Ratio│Avg Green│Avg VNIR│                     │
│  └─────────┴─────────┴──────┘                        │
│                                                       │
│  Leaf state: GREEN                                    │
│                                                       │
│  VS. REFERENCE                ← section title        │
│  Baseline       -1.2%                                 │
│  Global avg     +0.8%                                 │
│  Rolling avg    -0.5%         ← red if |Δ| > 5%     │
│  Checkpoint     -1.0%                                 │
└───────────────────────────────────────────────────────┘
```

---

## Chat Markdown Renderer

`renderMarkdown(text)` in `ChatPanel.jsx` is a zero-dependency JSX renderer:

| Input | Output |
|---|---|
| `**bold**` | `<strong>` |
| `*italic*` | `<em>` |
| `` `code` `` | `<code class="inline-code">` |
| `- bullet` or `• bullet` | `<div class="md-bullet">` |
| `### heading` | `<div class="md-h3">` |
| Empty line | `<div style="height: 0.5em">` |

Applied only to `assistant` role messages; user messages render as plain text.

---

## History Section Pattern

Both `DiagnosePanel` and `MonitorPanel` include a `HistorySection` component:

- Renders a collapsible toggle button (`📋 History ▼/▲`).
- On open, fetches events filtered by `plant_id` and `event_type`.
- Renders each event as a row: coloured dot + label + metadata + `×` delete button.
- `×` calls `DELETE /api/events/{id}` and removes the entry from local state.
- A `historyKey` counter in the parent is incremented after each scan to trigger a reload.
