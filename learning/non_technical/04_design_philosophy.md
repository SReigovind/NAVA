# Design Philosophy

> **Subfolder:** `non_technical/`
> **Cross-references:** [01_problem_and_vision.md](01_problem_and_vision.md) | [05_market_and_differentiation.md](05_market_and_differentiation.md) | [technical/02_disease_detection_pipeline.md](../technical/02_disease_detection_pipeline.md) | [technical/06_llm_and_prompt_engineering.md](../technical/06_llm_and_prompt_engineering.md)

---

## The Central Question: Who Gets Hurt If the System Is Wrong?

Every design decision in NAVA can be traced back to one foundational question: who gets hurt if this system makes a mistake, and how do we limit that harm?

The farmer who uses NAVA is not a software developer. They cannot inspect the model's weights or read the source code. They have to take NAVA's outputs at face value. If NAVA is confidently wrong — if it diagnoses Late Blight when the plant has a nutrient deficiency, or says the plant is healthy when it is critically stressed — the farmer acts on that wrong diagnosis. They apply the wrong treatment, spend money they cannot afford to waste, and may lose a significant portion of their harvest.

This asymmetry of consequences — the user bears the cost of errors they have no way to detect — shapes every part of NAVA's design.

---

## Principle 1: Surface Uncertainty Explicitly

Most AI systems are designed to hide their uncertainty. Showing a percentage confidence feels more transparent, but farmers don't know what 73% confidence means in practice. Does it mean the model is right 73% of the time? Or that it is almost right but with some ambiguity?

NAVA takes a different approach: **turn the confidence into a qualitative judgment and show it prominently.**

When a prediction's softmax probability falls below the configured threshold (80% by default), the result is flagged as `UNRELIABLE` and displayed to the user in explicit, plain language: "The AI has low confidence — treat this result with caution." No Grad-CAM heatmap is generated, because generating an explanation for an unreliable result implies a confidence in the explanation that isn't warranted.

Above the threshold, the confidence maps to a natural language phrase:
- 90%+ → "The AI is very confident about this result"
- 75–90% → "The AI is fairly confident"
- 60–75% → "The AI has moderate confidence"
- Below threshold → "UNRELIABLE"

The farmer is always shown how much to trust the output. And the system always recommends consulting a human expert for any high-confidence result that requires treatment.

**Why not just show the percentage?** A percentage is a number without context. "87% confidence" means nothing to someone who has never trained a classifier. "The AI is fairly confident" combined with the visual heatmap gives the farmer an actionable intuition: the model thinks it knows what this is, here is where it's looking, use your judgement.

---

## Principle 2: Explain, Don't Just Predict

Explainability is not optional for NAVA — it is a safety feature.

The Grad-CAM heatmap overlay is not cosmetic. It is the mechanism by which the farmer can verify that the model's reasoning corresponds to reality. If the heatmap highlights a region of the leaf where there is a visible lesion, the farmer's confidence in the diagnosis should increase. If the heatmap highlights the stem, the background, or an unrelated part of the leaf, the farmer should be skeptical.

This gives users a lightweight sanity check that requires no technical knowledge: does the highlighted region *look* like the problem to my eyes? If yes, the diagnosis is plausible. If no, something is wrong.

No other plant disease AI tool in the NAVA competitive set (as of Phase 1, 2024) provided this level of explanation to end users.

---

## Principle 3: Verified Sources, Not Internet Knowledge

NAVA's chat capability was designed from the beginning to avoid the failure mode of a general-purpose LLM: fluent, confident, and potentially wrong.

The solution is structural, not just instructional. NAVA does not tell the LLM "please be careful about making up facts." It retrieves the factual content from verified documents and injects it into the context. The farmer can see which document the information came from. The LLM is instructed to treat that material as authoritative.

This means NAVA's knowledge base is curated, not open-ended:
- Kerala Agricultural University's Package of Practices — the official extension guidance for Kerala's crops
- Disease management guides from peer-reviewed sources
- Pest management references appropriate for the crops and regions in question

The consequence: NAVA can only give advice that it can ground in a verified document. It cannot give advice about a crop that isn't in its RAG knowledge base. This is a deliberate limitation. It is better to say "I don't have specific guidance on this" than to confabulate guidance that sounds correct but isn't.

---

## Principle 4: Memory Belongs to the Farm, Not the Session

Most chat interfaces lose their memory when the tab is closed. This is fine for a general-purpose assistant, but for a farm management tool, it is a significant limitation.

NAVA's session memory is persistent and cumulative. When a farmer asks "what happened to Plant-3 last month?" NAVA has the answer — because it compressed and retained every previous conversation about that plant. When NAVA extracts a concrete action from a conversation ("farmer applied Carbendazim at 0.5g/L on 15 May"), it writes that action to the crop's notes field, where it becomes part of the permanent farm record.

This design means that NAVA gets more useful over time, not less. The longer a farmer uses it, the more context it has about their specific farm, and the more precisely calibrated its advice becomes.

---

## Principle 5: The System Should Know When It Doesn't Know

NAVA implements two forms of epistemic humility:

### Disease Detection
The confidence gate prevents the system from acting confident when it isn't. An `UNRELIABLE` result is not a failure — it is an honest acknowledgment that the image is ambiguous and the farmer should seek a second opinion. A confidently wrong diagnosis is far more dangerous than an uncertain correct one.

### VNIR Stress Monitoring
The VNIR pipeline requires 5 calibration scans before it starts generating stress alerts. During calibration, the user sees "Calibrating: N scans remaining" — they know the system is still learning the baseline for this specific plant. This prevents a false alarm in the first week of use, before the system has enough data to make meaningful comparisons.

Similarly, the zero-ratio guard prevents failed scans (no leaf detected) from corrupting the statistical baseline. A single failed scan should not trigger a spurious "stress detected" alert.

### Chat
When a chat question requires agricultural knowledge that is not in the RAG knowledge base, NAVA acknowledges the gap rather than improvising. The LLM is instructed in the system prompt not to give advice that cannot be grounded in the retrieved material for factual questions, and to be explicit about the limits of its knowledge.

---

## Principle 6: Deployability Over Scalability Theatre

A common failure mode in applied AI research is building systems that require infrastructure most target users cannot provide: cloud GPU clusters, Kubernetes, managed databases, commercial API subscriptions.

NAVA was explicitly designed to be deployable on a single modest server — a machine equivalent to or cheaper than a Raspberry Pi 5 with 8GB RAM. Specific decisions that flow from this:

- **CPU-only ML inference** — both the EfficientNet classifier and the Thanal ONNX model run on CPU. No GPU required.
- **SQLite over PostgreSQL** — SQLite requires zero infrastructure. No separate database server, no connection pooling, no database admin. A single file.
- **ChromaDB over Pinecone or Weaviate** — ChromaDB can run as a pure local persistent client. No cloud account, no API key, no per-query billing.
- **Open-Meteo and Nominatim for weather/geocoding** — both are fully free, require no API key, and can be called from any machine with internet access.
- **ONNX Runtime for VNIR inference** — enables inference without a full PyTorch installation, reducing deployment weight.

The vision is a system that an NGO or agricultural extension service could deploy in a rural district, on local hardware, with no ongoing cloud costs beyond the Hugging Face LLM API calls.

---

## Principle 7: The UI Should Protect Users from the AI

The frontend was designed with one overarching rule: never let the AI feel more authoritative than it should.

Specific UX decisions that embody this:
- **Confidence phrasing over raw percentages** — as described above
- **Grad-CAM as a sanity check** — the farmer can visually verify the model's reasoning
- **RAG source carousel** — the farmer can read the source document passage that grounded the AI's advice
- **VNIR calibration messaging** — the farmer knows the system is still learning before it starts making assertions
- **Timestamped weather** — the farmer knows how fresh the weather data is; it is not presented as live without context
- **"Consult an expert" recommendation** — always shown for high-stakes disease detections, regardless of confidence level

The UI is not trying to impress users. It is trying to equip them to make good decisions.
