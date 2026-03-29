# Paper Reader

AI-assisted deep paper reading. Each paper gets its own folder with section-by-section breakdowns tailored to your current knowledge level.

## How it works

### Starting a new paper

1. User provides an arXiv URL
2. Fetch the paper and identify its section structure
3. Create `paper_info.md` with metadata, abstract summary, and a section tracking table (all sections start blank)
4. **Before writing any section**, present a structured questionnaire of the prerequisite concepts for that section (or the whole paper if front-loaded). Use multiple-choice with levels like "solid / familiar / new to me". This determines what background to include and what to skip.
5. Proceed **one section at a time**. Write the section, then **stop and wait** for the user to read it and say they're ready for the next one. Never batch-write multiple sections.
6. After each section, update `paper_info.md` status to `done`

### Writing a section

For each section:

1. **Identify prerequisites** -- what concepts does the reader need to understand this section?
2. **Check the Knowledge Profile** below and the questionnaire answers -- skip concepts already mastered, explain new ones in depth
3. **Write the section note** with:
   - "What this section covers" summary at the top
   - Background sub-sections for any new prerequisite concepts (with worked examples, code, diagrams)
   - Walkthrough of the paper's actual content, referencing equations/figures/pages
   - Key takeaways at the bottom
   - Navigation links to previous/next sections
4. **After the paper is complete**, update the Knowledge Profile with concepts the reader has now mastered

### Conventions

- **Paper figures:** Don't reproduce charts as ASCII or approximate Mermaid. Instead, reference the figure and ask the user to clip the image into `artifacts/`. Embed as `![Figure N](artifacts/figure_N.png)`.
- **Explanatory diagrams:** Use Mermaid for *new* visuals (flowcharts, concept diagrams, architecture overviews, etc.)
- Each paper folder has an `artifacts/` subfolder for clipped images from the PDF
- Notes are designed to be read in **Obsidian** (Mermaid rendering, image embeds, wiki-links)
- **Pacing:** One section per turn. Wait for the user before moving on. Never write ahead.

## Papers

- [Pretraining Large Language Models with NVFP4](nvfp4-pretraining/) -- NVIDIA, 2025. FP4 training at scale.

## Knowledge Profile

Concepts mastered across papers — used to skip redundant explanations in future readings.

### Math & Statistics
_(none yet)_

### Machine Learning
_(none yet)_

### Deep Learning
_(none yet)_

### NLP
_(none yet)_

### Systems & Infrastructure
_(none yet)_
