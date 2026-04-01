# MemFaith Implementation Scope Lock

## Canonical Semester Core

The shipped core project is the **chunk-ablation CCS pipeline**, not ROME editing and not the chunk scorer.

Required core outputs:

- a deterministic long-context constructor
- a deterministic chunker for `K in {0, 2, 4, 8}`
- a full-context baseline answer `A_full`
- leave-one-chunk-out ablations `A_-i`
- answer-comparison logic for classification and QA
- CCS computation and aggregate summaries
- chunk-label export for the extension track

## Project Hierarchy

- memory agents: motivation and framing
- chunk ablation + CCS: main semester method
- learned pruning: optional extension after the core is stable

## What This Repo Now Covers

- runnable CCS scaffolding under `src/memfaith`
- FEVER-style smoke data for quick end-to-end validation
- HotpotQA-style smoke data for multi-hop validation
- export utilities for Mohamad's chunk-label pipeline
- protocol docs and comparison policy for Dev and Jade

## What Is Still Intentionally External

- large-scale FEVER evidence retrieval from Wikipedia dumps
- large-scale HotpotQA raw data download
- GPU-backed batched inference at full project scale
- final paper figures generated from real model runs

Those are operational next steps, not architectural gaps.
