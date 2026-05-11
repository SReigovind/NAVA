# NAVA NAVA — Next Steps & Roadmap

> Tracks open tasks, known issues, planned features, and technical debt.
> Update this file at the start and end of each session.

---

## Status: As of 12 May 2026

---

## 🔴 Known Issues

| # | Issue | Location | Notes |
|---|---|---|---|
| 1 | First chat after a new session sometimes sends without crop context | `ChatPanel.jsx` → `ChatService` | `session_context` may not be set before the first message if `crop_id` is null |
| 2 | VNIR calibrating phase shows "vs. Reference" section as empty | `MonitorPanel.jsx` | Correct behaviour but could show an explicit "baseline building" progress bar |
| 3 | OverviewPanel does not auto-refresh after scans (requires manual page navigation) | `OverviewPanel.jsx` | Could subscribe to a crop-level event or use a refresh interval |
| 4 | `DELETE /api/events/{id}` does not update `shared_context` | `fields.py` router | After deletion, the auto-generated context becomes stale until next crop change |

---

## 🟡 Technical Debt

| # | Item | Location | Priority |
|---|---|---|---|
| 1 | `shared_context` regeneration is synchronous and runs on every CRUD op | `fields.py` / `_refresh_field_context` | Low — acceptable for current scale |
| 2 | `renderMarkdown` in `ChatPanel` does not handle nested formatting e.g. `**_bold italic_**` | `ChatPanel.jsx` | Low |
| 3 | `PlantSelector` fetches plants on every mount (no caching) | `PlantSelector.jsx` | Low |
| 4 | CSS is a single 60 KB file appended incrementally — should be split by component | `styles.css` | Medium |
| 5 | `getVnirTier` in `MonitorPanel` uses string matching on status — fragile if model changes status wording | `MonitorPanel.jsx` | Medium — should use a canonical tier field from the pipeline |

---

## 🟢 Next Features (Prioritised)

### High Priority

- **OverviewPanel auto-refresh after scans**
  After running detection or VNIR, the parent `CropDetail` should signal `OverviewPanel`
  to reload events. Could use a shared state or a lightweight event bus.

- **VNIR baseline progress indicator**
  When calibrating, show "3 / 5 scans complete" instead of empty metrics.
  `vs_baseline` is `null` during calibration — count from `vnir_history` length.

- **Field-level event context in auto-context**
  `auto_generate_field_context` currently assembles crop metadata but not event history.
  Add a "Recent health events across all crops" block to help NAVA answer field-level
  questions without a specific crop context.

### Medium Priority

- **VNIR ratio trend chart**
  Per-plant line chart of ratio over time, sourced from `vnir_history`.
  Useful for showing improvement or degradation trends.
  Could use a lightweight library like `recharts`.

- **Export / field report PDF**
  Generate a one-page PDF snapshot of a field: crops list, plant health statuses, recent
  detection results, VNIR trends.

- **Bulk plant creation**
  Instead of creating plants one by one, allow entering "Row-1 through Row-10" and
  auto-creating named plants.

### Low Priority

- **Mobile layout**
  The sidebar workspace stacks at 700 px but is not fully optimised for phones.
  The chat session rail needs horizontal scroll or bottom-bar treatment on mobile.

- **Offline/PWA mode**
  Cache the SPA shell and last-loaded field data for offline viewing.

- **Multilingual chat**
  Integrate Bhashini API for Malayalam, Tamil, Hindi input/output.
  Preserve disease name entities (do not translate "late blight").

- **Push notifications**
  When a disease detection event is saved, trigger a browser notification.

---

## Architecture Decisions Pending

| Decision | Options | Notes |
|---|---|---|
| Chart library for VNIR trends | `recharts`, `Chart.js`, `d3.js` | `recharts` is React-native and well-maintained |
| Image storage | Current: base64 in response only (not persisted) | For history thumbnails, may need to store processed images in `logs/` |
| Real-time updates | Polling vs SSE vs WebSocket | SSE simplest for scan result streaming |
| Multi-user chat | Currently shared `mozhi_sessions.db` | Should shard per-user like `user_data.db` |

---

## Running Checklist Before Each Session

- [ ] Activate virtual environment (`.nava` in project root)
- [ ] `cd nava` — verify `.env` has valid `HF_API_KEY`
- [ ] Build frontend if JSX changed: `cd nava_core/gathi/frontend && npm run build`
- [ ] Start server: `python run.py`
- [ ] Open `http://localhost:8000` and verify logo loads (checks static serving)
- [ ] Log in and run a quick smoke test: create field → crop → plant → diagnose
