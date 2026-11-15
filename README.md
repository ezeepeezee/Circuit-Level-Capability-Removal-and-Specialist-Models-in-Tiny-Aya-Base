# Circuit-Level Capability Removal and Task-Specialist Models in Tiny Aya Base

Submission for the Lexsi Labs Hackathon, **Track 2: AI Red Teaming, Safety and Alignment**.

Author: Anirudh Peddinti (BITS Pilani, Hyderabad Campus)

## Motivation

Interpretability methods used to build safe models might also improve a model on one specific task. This project explores two directions built on that idea:

1. **Safety (Phase 1, the hackathon submission).** Start from a base model with no safety layer and use circuit-level manipulation so that it behaves safely. Even when explicitly asked for harmful content, the model should not produce it, because the capability has been removed from the model itself rather than suppressed by a separate refusal layer.
2. **Specialists (Phase 2, an extension).** Invert the idea: keep one target domain and remove general-purpose components, to produce smaller task-specialist models that can run locally, including on CPU. Running locally means user and enterprise data never leaves the device. Phase 2 is a proof of concept, with no hard performance or hardware targets.

## Problem statements addressed

| Problem | Statement | What this repo attempts |
|---|---|---|
| 7 | Unlearning | Remove a specific capability using forget and retain data, then test it on held-out questions while checking that general capability survives |
| 9 | Locating where safety lives | Find the components responsible for a harmful capability and use them for a targeted fix |
| 10 | Circuit-level interventions | Change behaviour only at the components a discovered circuit points to (circuit-restricted fine-tuning), and verify nothing else changed |

## Approach

**Chosen strategy: remove the capability rather than only install refusals.** Approach A would train the model to refuse. Approach B, used here, tries to remove the harmful capability through circuits. The main risk of B is relearning, so the threat model is **robustness to prompt-level attacks**, not to an attacker with the weights who can fine-tune.

**Scope.** One harm category is enough to test the hypothesis, so the target is hazardous biology (WMDP-bio) on **Tiny Aya Base** (`CohereLabs/tiny-aya-base`, about 3.35B parameters).

## Repository contents

| File | Description | Platform |
|---|---|---|
| `phase_1.ipynb` | Phase 1 (hackathon submission): Stages 0-4, from data splits through circuit discovery to circuit-restricted fine-tuning and held-out evaluation | Google Colab, T4 |
| `phase-2.ipynb` | Phase 2 (extension): Stages 0-5, from sealed test sets through matched-budget pruning to LoRA recovery | Kaggle, 2x T4 |
| `README.md` | This file | |

## Phase 1: Circuit-restricted capability removal (`phase_1.ipynb`)

| Stage | What it does |
|---|---|
| **0. Preflight and setup** | Mounts Drive, writes a run config and seed, checks GPU, RAM and package versions |
| **1. Data splits and CuratorKIT seed** | Splits WMDP-bio into discovery, dev and held-out sets and MMLU into retain and retain-eval sets. Builds a safety seed from `PKU-Alignment/PKU-SafeRLHF-10K` for CuratorKIT and writes a manifest |
| **2. SafeTune SaLoRA hardening** | Tokenizes the MMLU task data and the curated safety data, then hardens Tiny Aya Base with SafeTune's SaLoRA trainer (rank 4, on four evenly spaced `q_proj` modules) and saves the checkpoint |
| **3. Circuit discovery and head selection** | Loads the SaLoRA checkpoint into CircuitKIT, registers WMDP-bio (forget) and MMLU (retain) tasks, and runs node-level EAP discovery over attention heads on each. Compares the two score tables and builds head sets: differential (forget minus retain), forget-only, and size-matched random sets drawn outside the circuit |
| **4. CRFT and held-out evaluation** | Reloads the checkpoint in Hugging Face and verifies that its head mapping and predictions match the earlier CircuitKIT results. Runs training-free head ablation, then circuit-restricted fine-tuning (CRFT) where only the Q-row and O-column slices of the selected heads are trainable (as hook deltas, with base weights untouched). Scores dev sets for every arm, then scores the held-out WMDP-bio set once, alongside MMLU and biology retention checks, against criteria frozen beforehand |

Arms compared: **differential**, **forget-only**, and three **random** head sets (seeds 101, 102, 103), all at 60 heads. A second group of matched 69-head arms was trained but is not included in the scored finalists.

**Controls and data hygiene**
- The random-head arms are the key control: they test whether only the discovered circuit removes the capability.
- Discovery uses only questions the model answers correctly, with benign controls taken from MMLU validation data.
- The held-out set is scored once, at the end, for all finalists together. Evaluation criteria are written to disk before held-out scoring.
- A weight-diff check verifies that no parameter outside the selected heads changes.

## Phase 2: Task-specialist models (`phase-2.ipynb`)

| Stage | What it does |
|---|---|
| **0. Environment and sealed test sets** | Installs dependencies, profiles the model, and writes read-only sealed final test sets (WMDP-bio domain set and a general MMLU set). Backs work up to a private Hugging Face dataset, because Kaggle's working directory does not survive restarts |
| **1. Evaluation toolkit and baselines** | Builds the letter-logit scorer, option-shift and template variants, a fluency set and a decode-speed probe. Scores the fp16 and 4-bit baselines |
| **2. Data for scoring and recovery** | Builds domain and general MCQ sets, counterfactual pairs for attribution, domain text passages and general text, with leakage checks across every split |
| **3. Pre-recovery pruning sweep** | Scores MLP neurons three ways (random, standard importance, circuit-guided attribution) plus an integrated-gradients variant, sweeps the pruning budget on a dev set, and checks off-domain accuracy. Amendments to the plan are written to disk before the pruned models are scored |
| **4. Physical pruning** | Physically removes MLP neurons (gate and up rows, down columns, the same number in every layer) and checks the pruned model against the masked one |
| **5. LoRA recovery** | Recovers each pruned model by LoRA distillation from the unpruned model on label-free prompts and text. Four arms: circuit-guided, standard importance, standard text, random |

The sealed test sets are kept separate so they can be scored once at the end, and the intended target is CPU inference of the chosen arm.

**Note on libraries:** Phase 2 uses custom scoring, pruning and LoRA code (PEFT) and does not call the Lexsi libraries.

## Lexsi libraries used

| Library | Where | Role |
|---|---|---|
| **CircuitKIT** | Phase 1, Stages 3-4 | Model loading, task registration, node-level circuit discovery |
| **SafeTune** | Phase 1, Stage 2 | SaLoRA hardening of the base model |
| **CuratorKIT** | Phase 1, Stage 1 | Safety-data curation seed for SaLoRA training |
| **AuditKIT** | Phase 1 | Referenced only; not run |

The head-selection rules, CRFT training loop, evaluation harness and all of Phase 2 are custom code.

## How to run

**Requirements**
- A CUDA GPU of T4 class (Phase 1 on Colab, Phase 2 on Kaggle with two GPUs)
- A Hugging Face token (stored as the `HF_TOKEN` secret), with the terms for `CohereLabs/tiny-aya-base` accepted. Phase 2's backup cells need a write-access token
- Package versions are installed and printed in each notebook's Stage 0. SafeTune needs Python 3.12 or newer, and CircuitKIT must be imported before `transformer_lens`
- Datasets: `cais/wmdp` (`wmdp-bio`), `cais/mmlu`, `PKU-Alignment/PKU-SafeRLHF-10K` (Phase 1), plus WikiText-2 and domain text passages (Phase 2)

**Notes**
- Run each notebook top to bottom. Seeds are set in the config cells.
- Phase 1 writes artifacts to Google Drive under `hackathon_run/`. Phase 2 writes to `/kaggle/working/specialist`, which is lost on session restarts, so keep the backup cells.
- Discovery prompts are capped at a short length to fit a T4, and memory-saving patches are applied to the TransformerLens model in Stage 3.

## Results

[Add the final numbers here from the saved outputs: Phase 1 held-out WMDP-bio, MMLU and biology scores per arm with confidence intervals and the random-head control; Phase 2 domain and off-domain accuracy per arm.]

## Status and limitations

Completed: Phase 1 data splits, SaLoRA hardening, CircuitKIT discovery, head selection, CRFT arms with verified edit confinement, ablation, and the single held-out scoring. Phase 2 data preparation, pruning sweep, physical pruning and LoRA recovery.

Not done or not included:
- CircuitKIT faithfulness evaluation
- SafeTune safety-neuron cross-check
- AuditKIT base-versus-edited comparison
- A frozen QA prompt set
- The 69-head arms are trained but not scored
- Option-shuffling, answer-text and prompt-template robustness checks are not part of these notebooks
- Phase 2 does not use the Lexsi libraries, and CPU export and sealed-test scoring are [update if completed]

Edge-cut interventions and a GUI for inspecting components were considered and are not part of this work.

## Acknowledgements

Built with Lexsi Labs' CircuitKIT, SafeTune and CuratorKIT, and evaluated on WMDP and MMLU.
