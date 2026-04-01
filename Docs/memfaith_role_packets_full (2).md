# MemFaith — Role Packets and Detailed Checklists
**Team:** Dev Sanghvi · Ansh Dabral · Mohamad Kreidieh · Jade Yan  
**Window:** Week of March 30 through week of April 20  

This document expands the work-division plan into role-specific packets. It is intentionally detailed enough that each teammate can use it like a weekly operating document.

---

## Dev Sanghvi — role packet

### Mission
Keep the project technically coherent and own the causal-faithfulness interpretation from start to finish.

### Your success condition
By April 20, you should be able to explain:
- what CCS is,
- why the project is causal and not just retrieval,
- why FEVER and HotpotQA were chosen,
- what the main result means,
- and whether the pruning extension actually worked or not.

### Week-by-week checklist

#### Week of Mar 30
- [ ] Finalize canonical project statement
- [ ] Finalize K definition and notation
- [ ] Approve FEVER and HotpotQA protocol drafts
- [ ] Define exactly how answer change will be judged on each dataset
- [ ] Write `docs/project_scope.md`
- [ ] Write `docs/experiment_contract.md`

#### Week of Apr 6
- [ ] Review FEVER results for logical correctness
- [ ] Verify that CCS is computed correctly
- [ ] Approve first plots
- [ ] Tag 5–10 case studies for manual inspection
- [ ] Write first draft of `analysis/fever_findings.md`

#### Week of Apr 13
- [ ] Compare FEVER vs HotpotQA results
- [ ] Decide whether extension gate opens
- [ ] Draft the conclusions section
- [ ] Draft limitation bullets grounded in the actual runs

#### Week of Apr 20
- [ ] Finalize interpretation document
- [ ] Finalize final-presentation narrative
- [ ] Own answer bank for likely questions
- [ ] Verify all files align with the same project framing

### Anti-patterns to avoid
- do not let the project become “just train a discriminator”
- do not let the answer comparison function stay vague
- do not let terminology drift across documents

---

## Ansh Dabral — role packet

### Mission
Build the experiment runner that makes the project real.

### Your success condition
By April 20, the team should be able to rerun FEVER and HotpotQA CCS experiments from script, without manual repair.

### Week-by-week checklist

#### Week of Mar 30
- [ ] Build long-context assembly utilities
- [ ] Build chunking module
- [ ] Create stable experiment schema
- [ ] Implement a smoke-test runner on FEVER with tiny sample
- [ ] Add caching for prompts / outputs

#### Week of Apr 6
- [ ] Implement leave-one-chunk-out runner
- [ ] Execute FEVER across all K settings
- [ ] Store results in final schema
- [ ] Verify resumability and deterministic IDs
- [ ] Export first logs for Jade and Dev

#### Week of Apr 13
- [ ] Extend same pipeline to HotpotQA
- [ ] Support stratified exports
- [ ] Add per-example latency and token accounting
- [ ] Add optional chunk-label export for Mohamad

#### Week of Apr 20
- [ ] Clean all scripts
- [ ] Build one-command rerun scripts
- [ ] Verify caches and manifests
- [ ] Package reproducibility folder

### Anti-patterns to avoid
- do not hardcode dataset-specific assumptions into the core runner
- do not let schemas mutate mid-project
- do not delay logging until after the experiment logic is done

---

## Mohamad Kreidieh — role packet

### Mission
Turn chunk-level causal signals into a context-pruning extension, but only if the core project is stable.

### Your success condition
By April 20, either:
- there is a working pruning prototype with basic results, or
- there is a fully specified extension design with label-generation utilities and partial experiments.

### Week-by-week checklist

#### Week of Mar 30
- [ ] Draft chunk-label formats:
  - binary causal
  - scalar causal weight
  - rank-based chunk ordering
- [ ] Define a lightweight scorer design space
- [ ] Write `docs/extension_design.md`

#### Week of Apr 6
- [ ] Build chunk-label extraction utility from FEVER CCS logs
- [ ] Validate label balance and sparsity
- [ ] Prepare scorer training pipeline, but do not start large training yet

#### Week of Apr 13
- [ ] If extension gate is approved, train first chunk scorer
- [ ] Run pruning on FEVER at 2–3 pruning ratios
- [ ] Export comparison tables: performance retained vs context reduced

#### Week of Apr 20
- [ ] If extension is alive, polish results
- [ ] If extension slips, package as “future work with prototype”
- [ ] Write extension subsection for the final report

### Anti-patterns to avoid
- do not redefine the main project around the extension
- do not introduce a model that requires a new project scope
- do not depend on EF/ROME integration unless the team explicitly approves it as a bonus

---

## Jade Yan — role packet

### Mission
Make the datasets, metrics, writing, and final deck precise and internally consistent.

### Your success condition
By April 20, the report and deck should read like one coherent project instead of several merged ideas.

### Week-by-week checklist

#### Week of Mar 30
- [ ] Write FEVER protocol spec
- [ ] Write HotpotQA protocol spec
- [ ] Write answer-comparison policy doc
- [ ] Review slide wording against actual experiment design

#### Week of Apr 6
- [ ] Build first FEVER result tables
- [ ] Plot CCS vs K on FEVER
- [ ] Draft limitations discovered so far
- [ ] Start FAQ / expected-Q&A notes

#### Week of Apr 13
- [ ] Build HotpotQA tables and comparative analysis
- [ ] Draft results section in near-final form
- [ ] Audit slide wording again after extension decision

#### Week of Apr 20
- [ ] Finalize report text
- [ ] Finalize slides / speaker notes / likely questions
- [ ] Verify all figures and tables are labeled consistently

### Anti-patterns to avoid
- do not allow old EF/discriminator language to remain in the mainline method sections if the extension is not central
- do not let metric definitions differ across slides and docs
- do not write future work as if it was already completed

---

## Cross-role integration checklist

### Before FEVER full run
- [ ] Dev approved comparison function
- [ ] Jade approved protocol doc
- [ ] Ansh's runner produces stable logs

### Before HotpotQA full run
- [ ] answer comparison function stabilized
- [ ] schema unchanged from FEVER run
- [ ] Dev and Jade both sign off on interpretation rules

### Before extension gate opens
- [ ] FEVER CCS done
- [ ] HotpotQA pilot done
- [ ] Dev approves extension start

### Before final packaging
- [ ] all tables reproducible
- [ ] all scripts rerunnable
- [ ] all docs use the same project statement

---

## Final team checklist for the week of Apr 20

- [ ] FEVER CCS final
- [ ] HotpotQA CCS final or documented partial
- [ ] case studies curated
- [ ] plots regenerated from scripts
- [ ] final narrative coherent
- [ ] extension status stated honestly
- [ ] FAQ prepared
- [ ] reproducibility package assembled

