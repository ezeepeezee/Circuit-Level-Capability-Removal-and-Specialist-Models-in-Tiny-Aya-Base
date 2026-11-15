# Circuit-Level Capability Removal and Specialist Models in Tiny Aya Base

Submission for the Lexsi Labs Hackathon, **Track 2: AI Red Teaming, Safety and Alignment**.

Author: Anirudh Peddinti (BITS Pilani, Hyderabad Campus)

## Motivation

Interpretability methods used to build safe models might also improve a model on one specific task. This project explores two directions built on that idea:

1. **Safety (Phase 1, the hackathon submission).** Start from a base model with no safety layer and use circuit-level manipulation so that it behaves safely. Even when explicitly asked for harmful content, the model should not produce it, because the capability has been removed from the model itself rather than suppressed by a separate refusal layer.
2. **Specialists (Phase 2, an extension).** Invert the idea: protect one target domain and remove the general-purpose components, to produce smaller task-specialist models that can run locally (including on CPU). Running locally means user and enterprise data never leaves the device.

## Problem statements addressed

| Problem | Statement | What this repo attempts |
|---|---|---|
| 7 | Unlearning | Remove a specific capability using forget and retain sets, then test it under shuffled, templated and adversarial prompts while checking general capability |
| 9 | Locating where safety lives | Find the circuits, neurons or directions responsible for a harmful capability and use them for a targeted fix |
| 10 | Circuit-level interventions | Change behaviour only at the components a discovered circuit points to (circuit-restricted fine-tuning), and verify nothing else changed |

## Approach

**Chosen strategy: remove the capability, don't just install refusals.** Approach A would train the model to refuse. Approach B, used here, tries to remove the harmful capability through circuits. The main risk of B is relearning, so the threat model is defined as **robustness to prompt-level attacks**, not to an attacker with the weights who can fine-tune.

**Scope.** One harm category is enough to test the hypothesis, so the target is hazardous biology on **Tiny Aya Base** (`CohereLabs/tiny-aya-base`, about 3.35B parameters). Chemistry is a secondary transfer check.

### Phase 1 pipeline

1. **Baselines.** Zero-shot letter-logit multiple-choice scoring on WMDP-bio (target), WMDP-chem (transfer check), and MMLU subsets plus a 500-question general sample (retention controls).
2. **Data hygiene.** WMDP-bio is split into discovery and held-out halves (seed 0). Only discovery-half questions the model answers correctly are used for circuit discovery. Benign controls come from the MMLU validation split, so MMLU test numbers stay untouched. The held-out half is evaluated once at the end, against criteria frozen in advance.
3. **Circuit discovery (CircuitKIT).** EAP-IG at node level (attention heads and MLP layers) on length-matched clean/corrupted prompt pairs, run separately on forget (WMDP-bio) and retain (MMLU validation) data, with repeated seeds to check score stability.
4. **Component selection.** Compared rules: forget-only, forget minus lambda times retain (differential), and random size-matched sets drawn outside the circuit.
5. **Intervention.** Circuit-restricted fine-tuning (CRFT): only the Q-row and O-column slices of the selected attention heads are trainable. The forget loss pushes the answer distribution toward uniform, and a retain loss keeps the original model's behaviour on benign data. A weight-diff guard verifies that nothing outside the selected heads changes. Training-free head ablation is also tested.
6. **Controls.** The key control is a size-matched set of random heads, to test whether only the circuit edit removes the capability.
7. **Robustness evaluation.**
   - Option shuffling (4 cyclic shifts) and answer-text scoring.
   - A battery of 10 prompt templates (changed headings, option markers, answer cues, instruction and adversarial wrappers, few-shot).
   - Retention checks on MMLU subsets and a biology control set.

### Phase 2 pipeline (specialist models)

1. Reserve a sealed domain test set (WMDP-bio) and a general MMLU set, scored once at the end.
2. Matched-budget comparison of **random**, **standard importance** and **circuit-guided** pruning of MLP neurons (budget chosen by a pre-recovery sweep on a dev set).
3. LoRA distillation recovery for each arm.
4. Planned: export the chosen arm for CPU inference and score the sealed test once.

## Lexsi libraries used

| Library | Role in this project |
|---|---|
| **CircuitKIT** | Core: model loading, circuit discovery, checkpoint export |
| **SafeTune** | SaLoRA checkpoint and unlearning/safety tooling |
| **CuratorKIT** | Forget, retain and paraphrase data curation |
| **AuditKIT** | Planned base-versus-edited comparison (configured; see Status) |

The CRFT training loop, head-selection rules, evaluation harness and pruning/recovery pipeline are custom code built on top of CircuitKIT scores.

## Repository contents

| File | Description |
|---|---|
| `[consolidated_notebook.ipynb]` | Phase 1: baselines, data splits, circuit discovery, selection, CRFT, shuffle/answer-text and template evaluations (Colab, T4) |
| `[phase2_specialist_notebook.ipynb]` | Phase 2: matched-budget pruning arms and LoRA recovery (Kaggle, 2x T4) |
| `[README.md]` | This file |

## How to run

**Requirements**
- Python 3.13, a CUDA GPU (T4-class is enough; Phase 1 on Colab, Phase 2 on Kaggle with 2 GPUs)
- `circuitkit` 0.2.0 with `transformer-lens` 3.8.0 (CircuitKIT must be imported before `transformer_lens`)
- A Hugging Face token, with the terms for `CohereLabs/tiny-aya-base` accepted
- Datasets: `cais/wmdp` (configs `wmdp-bio`, `wmdp-chem`) and `cais/mmlu`

**Notes**
- On TransformerLens 3.8.0, discovery runs out of memory on a 15 GB GPU unless the MLP and unembedding weights are re-laid out as column-major after every model load (the outputs are bitwise identical). The notebook includes this fix.
- Discovery prompts were capped at about 151 tokens to fit a T4.
- Run the notebooks top to bottom. Seeds are fixed in the config cells. Files written to `/kaggle/working` do not survive session restarts, so back up outputs to Drive or a dataset.

## Results

[Add the final numbers here from the saved outputs of each notebook: baselines, held-out scores per arm, shuffle and template results, retention, and the Phase 2 arm comparison. Include confidence intervals and the random-head control.]

## Status and scope

Completed: baselines and data splits, CircuitKIT discovery, component selection, CRFT with verified edit confinement, shuffle, answer-text and template evaluations, and the Phase 2 pruning and recovery arms.

Not completed or not run:
- CircuitKIT faithfulness evaluation
- SafeTune safety-neuron cross-check
- AuditKIT base-versus-edited comparison
- Frozen QA prompt set (`qa_prompts.json`)
- Paraphrase step of the template evaluation
- Phase 2 sealed-test scoring and CPU export, if not yet run

Edge-cut interventions and a Jupyter GUI for inspecting and adjusting components were planned as extensions and are not part of the core claim.

## Acknowledgements

Built with Lexsi Labs' CircuitKIT, SafeTune, CuratorKIT and AuditKIT, and evaluated on WMDP and MMLU.
