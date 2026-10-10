# Research-driven improvements

What we added to the pipeline from the research review (`NCERT podcast generator research papers.md`), why, where it lives, and how it is tested. Everything below runs with no new dependencies. Test suite: `pytest` (42 tests, no network).

## 1. Grounding: every line must be traceable to the chapter

| Change | Research basis | Where |
|---|---|---|
| **Verbatim quotes on facts.** Extraction returns, for each fact, the exact sentence it comes from. The quote is kept only if it really occurs in the section (compared after lower-casing and stripping punctuation), so any quote shown to a student is the chapter's own words. | Slobodkin et al., *Attribute First, then Generate* (ACL 2024): choosing evidence before writing gives precise, checkable citations. | `script_builder._extract`, `prompts.extract` |
| **"Report what the author says" framing.** Grounding rule 1 now calls the source "the chapter author's account" and gives a counterfactual example (if the source said 400 m/s, say 400). | Zhou et al. (2023): opinion-style framing improves faithfulness. Xie et al. (2023): models slip back to memory when the source *almost* matches what they know, as with a revised NCERT edition. | `prompts.GROUNDING_RULES` |
| **Exact-match number check.** A free check, run next to the LLM judge: every number in a line's caption must appear in the chapter with digit boundaries, so 343 doesn't pass because 3430 exists. | Same papers: near-matching numbers are the most common memory leak, and a deterministic check costs nothing. | `script_builder._rule_issues` |
| **Untrusted-input defences.** Cleaning removes invisible characters (keeping the joiner characters that Indic scripts need) and any of our own prompt tags (`<source>` and others) from the paste. The rule check also rejects lines containing links, emails or phone numbers. | Hines et al., *Spotlighting*; Greshake et al.: pasted text can carry hidden instructions. A faithful pipeline would faithfully repeat a planted link, because it is "supported by the source". | `clean.INVISIBLE`, `clean.PROMPT_TAGS`, `script_builder.CONTACT` |
| **Pipeline order.** Whole-episode editing happens before verification, so every line that reaches the audio has been checked, including edited ones. | Huang et al. (ICLR 2024): no stage may approve its own output. | `script_builder.build_script` |

## 2. Verification and fixing

| Change | Research basis | Where |
|---|---|---|
| **Line kinds with per-kind rules.** `question`, `fact`, `misconception`, `refutation`, `analogy`, `recap`, `banter`. Facts, refutations and recaps must be supported. Questions and banter must not slip in claims. Analogies must add no new claim. | VeriScore: not every sentence is a verifiable claim. FaithDial: most dialogue hallucination comes from "personality" (anecdotes, opinions as facts), which the writer prompt now forbids. | `prompts.KIND_RULES`, `prompts.verify` |
| **Judge instruction.** The judge marks as unsupported any claim not in the source "even if it is true". | Chain-of-Verification; HHEM's design: true-but-unsourced must fail, which is exactly the no-memory rule. | `prompts.verify` |
| **Minimal-edit fixes.** The fixer changes only the unsupported part and keeps speaker, tone and length. | Gao et al., *RARR*: revise while preserving the original. | `prompts.fix` |
| **Fallback to the chapter's words.** A line still failing after two fix rounds is replaced by the verbatim quotes of its facts, which are grounded by construction. A failing line that teaches no fact is dropped. Only lines that can't be rescued go to `unresolved_issues`. Rescued ones are listed in `auto_resolved`. | *Dialog Inpainting*: the host gives a near-verbatim paraphrase of the source sentence. | `script_builder._fallback` |
| **Merged turns keep the strictest kind.** When one host's two lines merge, the merged line keeps the kind that is checked hardest, so a fact can't hide inside an unchecked misconception. | Follows from per-kind verification. | `script_builder.KINDS` order |
| **Reasoning only for judging.** OpenRouter reasoning is off for extraction, writing, fixing and normalising. It is on, at low effort, for editing and verifying. | Chen et al., *Do NOT Think That Much*; TALE. In our own test, DeepSeek at default effort spent 16k tokens on one verify call and returned nothing. Low effort caught both errors we planted, in about 3 s. | `providers.OpenAICompatibleLLM.complete` |

## 3. Teaching moves in the dialogue

| Change | Research basis | Where |
|---|---|---|
| **Deep questions first.** The planner gives each segment a why/how question that a fact in the segment answers. Meera asks it before Arjun explains. | Driscoll et al. (2003), Gholson et al. (2009): overheard deep questions drove learning (replicated in grades 8–11); shallow questions did nothing. | `prompts.plan`, `prompts.DIALOGUE_RULES` |
| **Refuted misconceptions.** Optionally one per segment, only a belief students genuinely hold before learning the topic, tied to the fact that contradicts it. Meera voices it before the correcting idea, and Arjun refutes it in the next line. A misconception without a refutation from Arjun within two lines is removed, so a student never hears a wrong idea left standing. | Muller et al. (2008): dialogue and refutation beat plain exposition by d ≈ 0.8, most for low-prior-knowledge students. | `prompts.plan`, `script_builder._normalise_plan`, `script_builder._pair_misconceptions` |
| **No question-and-answer ping-pong.** At most half of Meera's lines are questions. She also guesses, connects ideas and sums up, and Arjun reacts before adding. The run log prints the count. | Book2Dial (2024): textbook dialogues read like "a series of separate QA pairs". Chi et al.: observers learn from the tutee's attempts. | `prompts.DIALOGUE_RULES`, log line in `build_script` |
| **Safer analogies.** At most one per segment, one or two features compared, and Arjun says where the comparison stops working. Recaps use the textbook's own terms. | Shao et al. (2025): only 16 of 30 plainly generated analogies were correct. | `prompts.DIALOGUE_RULES` |
| **Episode map and signalling.** The opening gives a one-sentence map of the episode, and lists are signposted ("there are two ways. First..."). | PaperWave (outline improves comprehension); Alpizar et al. (signalling, d = .38). | `prompts._POSITION`, `prompts.DIALOGUE_RULES` |
| **Fillers only in speech.** A light filler ("Right,", "Haan,") may start `spoken_text`, mostly in Meera's lines. Captions stay clean. | MoonCast: spontaneity markers raise naturalness ratings. PodEval: they cost density, so they're kept light. | `prompts.DIALOGUE_RULES` |

## 4. Text-to-speech safety

| Change | Research basis | Where |
|---|---|---|
| **Digit-free spoken text.** Dashes and ellipses become pauses. Any line whose `spoken_text` still has digits, sub- or superscripts, maths or currency symbols, Greek letters, brackets, or a word repeated twice in a row goes through one normalisation call. That call has few-shot examples: ratios, formulae, genotypes ("double capital T"), units, rupees, Roman numerals, activity and figure numbers. | PolyNorm (few-shot LLM normalisation beat a production rule system). Neekhara et al.: LLM-based TTS repeats or skips words, especially on repeated tokens. | `script_builder.SPOKEN_PROBLEM`, `script_builder._normalise_spoken`, `prompts.normalise` |
| **Voice allow-list.** Voice names are checked against the 45 voices in Bodhan's API reference before any call. | Indic-Speak model card: an unknown voice name can silently produce a degraded, averaged voice. | `providers.BODHAN_VOICES` |

## 5. Audio stitching, QC and captions

| Change | Research basis | Where |
|---|---|---|
| **Trim and level every clip.** Each clip's own leading and trailing silence is cut, leaving 40 ms. The clip is then gain-matched to a common level before stitching, so the hosts don't jump in loudness at every turn. The master is still normalised to −16 LUFS, and the measured value goes into `manifest.loudness_lufs`. | Apple podcast spec (−16 LKFS, true peak ≤ −1 dB). The review notes that per-clip gain matching is the step most stitched podcasts miss. | `render.trim_and_level`, `render._loudness` |
| **Natural turn gaps.** About 0.22 s between speakers, 0.12 s after a quick reaction, 0.35 s after a question, ±50 ms of fixed per-line jitter, and 0.8 s between segments. | Levinson & Torreira (2015): turn gaps cluster at 0–200 ms (mean 236 ms in English). | `render.gap_after` |
| **Duration QC with retakes.** A clip whose length is outside 0.5×–1.8× of what that voice's own median speaking rate predicts is re-synthesised up to twice, and the take closest to the expected length is kept. A clip still off sets `needs_review` and is listed in `manifest.qc`. | Indic-Speak coverage: some outputs repeat or drop content. Zuo et al.: duration abnormality is a cheap black-box signal. | `render.duration_outliers`, `Renderer._retake` |
| **Better captions.** Cues aim for at most two 42-character lines and never span a speaker change. A cue shorter than 0.83 s joins a neighbour, which can push it slightly over. Time is shared by how long each sentence takes to *say*, not to read ("3:1" is 3 characters on screen but about 16 spoken). | Netflix children's timed-text guide; the review's own recommendation. | `render.build_cues` |
| **Manifest for the player.** `facts` (with chapter quotes), so tapping a line can show "from the chapter: …"; `qc`; `loudness_lufs`. Lines already carry `kind` and `fact_ids`, which quizzes and "Ask the hosts" can build on later. | Attribute First (evidence on tap); Adesope & Nesbit (on-screen key terms help). | `render.Renderer.render` |

## 6. Effect on our test chapter

Class 9 Science, Chapter 10 *Sound*, sections 10.1–10.2.1 (1,430 words), `--minutes 6`:

| Run | Words | Meera's lines that are questions | Misconceptions refuted | Notes |
|---|---|---|---|---|
| Before research changes (`out/sound5`) | 914 | not measured | 0 | Rendered with real voices, 7:00 |
| First research run (`out/sound6`) | 900 | 23 of 27 | 1 | Question-and-answer ping-pong |
| `out/sound7` | 1,017 | 14 of 32 | 5 | Some misconceptions were strawmen |
| `out/sound8` | 794 | 16 of 29 | 3 | "Threads of Curiosity" box skipped; clipped sentences |
| **`out/sound9`** | **989** | **12 of 28** | **2** | 37 of 40 facts quote-verified; 0 left for review; 1 line rescued with the chapter's words |

Each problem row led to a prompt fix:
- **sound6:** Meera's role was broadened.
- **sound7:** misconceptions are now only genuine prior beliefs, voiced before the correcting idea.
- **sound8:** boxed stories are always extracted, and the editor must keep sentences complete.

Results still vary between runs of the same input.

## 7. Not done yet, and why

| Idea from the review | Why not yet |
|---|---|
| Local sub-1B checker (HHEM-2.1 / MiniCheck) screening every line | Needs PyTorch and transformers (~2 GB) and is English-only. The number check plus the LLM judge cover the same ground for now. |
| ASR listen-back of every clip (Bodhan Indic-Transcribe) | Needs that model's own key. About ₹0.10 per audio minute. The duration check catches the worst failures meanwhile. |
| Word-level captions via forced alignment (MMS) | Needs PyTorch. Spoken-length timing is the cheap interim fix the review suggests. |
| Prompt-prefix caching and `session_id` sticky routing | LLM cost is already a few cents per episode. Would need every prompt restructured around a shared prefix. |
| Quizzes with feedback; "Ask the hosts" | Out of scope for this phase. The review ranks quizzes as the strongest learning lever, so they're the first candidate after the frontend. |
| Hand-labelled gold set of 50–100 NCERT lines | Needed before any checker threshold can be tuned honestly. A good task to split across the team. |

## Known weaknesses

- A refutation can still slip in a small outside claim ("light doesn't carry sound"), and the judge passed it once.
- Meera sometimes repeats Arjun's point back too often.
- Lines removed during fixing can leave one host speaking twice in a row. It sounds fine in audio and shows as two bubbles in the transcript.
- Facts whose source sentence is broken up by PDF figure labels can't be quote-verified. They still go through the fact-check, but they can't use the quote fallback.
