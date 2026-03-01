# Docs Migration Map (2026-03-01)

This file records where the old documents were moved after docs reorganization.

## Main Categories
- Design docs: `docs/01_design/`
- Patch docs: `docs/02_patch/`
- Feature progress docs: `docs/03_feature_development/progress_reports/`
- Legacy docs/workspace: `docs/90_archive/`

## Old -> New (Representative)
- `docs/protocol.md` -> `docs/01_design/system/task_queue_protocol.md`
- `docs/quant_design_updated.md` -> `docs/01_design/quant/quant_design_latest.md`
- `docs/quant_design.md` -> `docs/90_archive/legacy_docs/quant_design_20260228_initial.md`
- `docs/learning_guardrails_patch_design.md` -> `docs/01_design/learning/learning_guardrails_patch_design.md`
- `docs/progress_20260228_web_augmented_learning.md` -> `docs/03_feature_development/progress_reports/progress_20260228_web_augmented_learning.md`
- `docs/progress_20260228_local_model_stability.md` -> `docs/03_feature_development/progress_reports/progress_20260228_local_model_stability.md`
- `docs/progress_20260226_135851_OpenClaw_Nexus_v1_2_5_PATCH.md` -> `docs/02_patch/progress_20260226_135851_OpenClaw_Nexus_v1_2_5_PATCH.md`
- `docs/OpenClaw_Nexus_Progress_20260227_141500.md` -> `docs/03_feature_development/progress_reports/OpenClaw_Nexus_Progress_20260227_141500.md`
- `docs/AIagent_project/*` -> `docs/90_archive/legacy_workspace/AIagent_project/*`

## Consolidation Notes
- `quant_design_latest.md` is the current authoritative quant design.
- `system_design_latest.md` and `learning_design_latest.md` were created as consolidated latest design entry points.
- Historical versions remain available under `docs/90_archive`.
