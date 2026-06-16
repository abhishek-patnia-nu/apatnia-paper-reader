# AGENTS.md

This is a **paper-reading repo**. The owner reads AI/ML papers as section-by-section
markdown notes (often on an iPad, viewing the GitHub-rendered pages). You help write and
**update** those notes.

This file exists so a cold session — especially a **Codex cloud agent triggered from
the iPad** — can make a correct edit from a one-line prompt without losing the conventions.

## Source of truth

**[GUIDELINES.md](GUIDELINES.md) is authoritative.** Read it before doing substantive work.
It defines the full process, the section format, and the **Knowledge Profile** (concepts the
reader has already mastered — skip re-explaining these). The same rules live in
[.cursor/rules/paper-reader.mdc](.cursor/rules/paper-reader.mdc) for the Cursor editor.

## Two modes of work

1. **Reading-time updates (the common case on iPad).** The reader spots something while
   reading and prompts a change — e.g. *"expand the all-gather diagram in fsdp section 3"* or
   *"the 515GB example in deepseek section 0.1 should also show the per-GPU split"*. Find the
   right file, make the edit, honor the conventions below, and **commit straight to `main`**
   with a clear message. No PR, no queue — the change should just appear in the repo.
2. **Writing a new paper / new section.** This is the full, paced process in `GUIDELINES.md`:
   questionnaire first, one section per turn, stop and wait. **Do not** batch-write sections.
   Reading-time updates are exempt from the pacing rule — just apply them.

## Repo map

```
GUIDELINES.md                  # process + Knowledge Profile (source of truth)
.cursor/rules/paper-reader.mdc # same rules, for Cursor
<paper>/
  paper_info.md                # metadata, abstract summary, section status table
  section_*.md                 # the notes, one file per section
  artifacts/                   # clipped figure images (figure_N.png), referenced not reproduced
guides/                        # implementation-idea notes, not bound to one paper
```

Current papers: `nvfp4-pretraining`, `pytorch-fsdp`, `deepseek-v4-attention`,
`adam-optimizer`, `gated-delta-networks`. (When in doubt about which file a terse prompt
refers to, match on the paper folder name + section number.)

## Conventions (must honor on every edit)

These notes are authored for the **GitHub markdown renderer** (viewed in-browser / on iPad):

- **Calibration over completeness.** Match the existing depth of the file. Skip concepts in
  the Knowledge Profile or marked "solid" in that paper's `paper_info.md` calibration table;
  give deep background (worked examples, code, diagrams) only to "new" concepts. Don't pad a
  section the reader already understands.
- **Explain with code, not prose or heavy math.** The reader understands code best: carry
  mechanisms in short PyTorch-style snippets with a **shape comment on every line**
  (`# (t, d_h)`), one framing sentence per block. Keep only equations that anchor to the
  paper's notation (cited eq numbers). See §0.1 of `deepseek-v4-attention` for the
  reference example of this style.
- **Links:** standard relative markdown — `[text](section_2.md)`, **not** Obsidian
  `[[wiki-links]]`. Section anchors follow GitHub slug rules (lowercase, punctuation dropped,
  spaces → hyphens). Keep the prev/next nav links at the bottom of each section correct.
- **Math:** `$$...$$` for display, `$...$` for inline (GitHub renders via MathJax). **No
  spaces just inside inline delimiters** — `$F$`, not `$ F $`.
- **Diagrams:** fenced ` ```mermaid ` blocks for *new* explanatory visuals (concept maps,
  flowcharts, architecture). **Quote any node/edge label with special characters.**
- **Paper figures:** never reproduce a paper's charts as ASCII or Mermaid. Reference the
  figure and, if it isn't clipped yet, ask the reader to add it to `artifacts/`. Embed as
  `![Figure N](artifacts/figure_N.png)`.
- **Tracking:** if you finish or add a section, update that paper's `paper_info.md` status
  table. After completing a whole paper, update the Knowledge Profile in `GUIDELINES.md`.

## Committing

For reading-time updates, commit directly to `main` with a concise message describing the
note change (e.g. `deepseek §0.1: add per-GPU split to the KV-cache memory example`). The
goal is that the reader sees the update in the repo without any review gate.
