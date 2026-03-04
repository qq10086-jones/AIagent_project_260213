# OpenClaw Nexus v1.4 窶・Coding Team First 莉ｻ蜉｡貂・黒・亥庄謇ｧ陦檎沿・・
- 譌･譛滂ｼ・026-03-02
- 逶ｮ譬・ｼ壻ｼ伜・螳樒鴫 Coding Team・・M/UI/FE/BE/QA・画ｵ∵ｰｴ郤ｿ・悟ｽ｢謌仙庄莠､莉倥∝庄鬪梧噺縲∝庄螟咲畑讓｡譚ｿ縲・
---

## 莉ｻ蜉｡扈・ｻ・婿蠑擾ｼ亥ｻｺ隶ｮ・・- Epic -> Stories -> Tasks
- 豈丈ｸｪ Task 驛ｽ譛会ｼ唹wner・井ｺｺ・峨∬ｾ灘・縲∬ｾ灘・縲・ｪ梧噺譬・㊥・・oD・峨・｣朱勦遲臥ｺｧ縲・｢・ｮ｡萓晁ｵ・- 鮟倩ｮ､蟷ｶ陦悟次蛻呻ｼ・*Registry/Schema縲仝orkflow Shell縲、rtifact Pack縲．ashboard** 蝗帶擅郤ｿ蜿ｯ蟷ｶ陦梧耳霑帙・
---

## EPIC 1・咾apability Registry 霑占｡梧慮螂醍ｺｦ蛹厄ｼ域怙鬮倅ｼ伜・郤ｧ・・
### S1.1 Registry Schema & Validator
- [ ] T1・壼ｮ壻ｹ・`capability_registry.schema.json`
  - 霎灘・・嘖chema 譁・ｻｶ・郁ｦ・尠 project_types/roles/tools/workflows/policies/acceptance・・  - DoD・夊・譬｡鬪悟ｿ・｡ｫ蟄玲ｮｵ縲∵椢荳ｾ縲》ool params schema 蠑慕畑
- [ ] T2・壼ｮ樒鴫 `validate_registry`・・LI + CI・・  - 霎灘・・咾I job・・itHub Actions/譛ｬ蝨ｰ閼壽悽蝮・庄・・  - DoD・啀R 謾ｹ registry 蠢・｡ｻ騾夊ｿ・ｼ悟凄蛻呎拠扈晏粋蟷ｶ
- [ ] T3・唹rchestrator 蜷ｯ蜉ｨ譌ｶ譬｡鬪・registry・・ail fast・・  - 霎灘・・壼星蜉ｨ蜉霓ｽ荳朱漠隸ｯ謚･蜻・  - DoD・壽裏謨・registry 髦ｻ豁｢譛榊苅蜷ｯ蜉ｨ

### S1.2 Runtime Contract Enforcement
- [ ] T4・壻ｻｻ蜉｡謠蝉ｺ､譌ｶ譬｡鬪鯉ｼ嗔roject_type/workflow/role/tool/params
  - 霎灘・・喨ngress 螻・validator
  - DoD・壽裏謨郁ｯｷ豎りｿ泌屓扈捺桷蛹・error_code・・EGISTRY_INVALID / TOOL_NOT_ALLOWED・・- [ ] T5・壼ｷ･蜈ｷ豕ｨ蜀御ｸ閾ｴ諤ｧ・嗹orker router -> tools.json/registry 閾ｪ蜉ｨ蟇ｹ鮨・  - 霎灘・・夂函謌仙勣謌門黒荳譚･貅撰ｼ・ource of truth・・  - DoD・壽眠蠅・tool 蜿ｪ謾ｹ registry・御ｸ榊・謾ｹ螟壼､・
---

## EPIC 2・啗orkflow Shell・育｡ｮ螳壽ｧ step/gate/checkpoint・・
### S2.1 譛蟆・Step Runner
- [x] T6・壼ｮ樒鴫 workflow step graph runner・磯｡ｺ蠎冗沿・・  - 霎灘・・啻workflow.run(workflow_id, input)` API
  - DoD・夊・霍・`pm_spec -> arch -> impl -> qa -> release` 蜈ｨ體ｾ霍ｯ
- [x] T7・售tep 迥ｶ諤∵惻荳取戟荵・喧
  - 霎灘・・咼B 陦ｨ `workflow_runs / workflow_steps`・域・謇ｩ螻慕鴫譛・runs・・  - DoD・壽ｯ・step 譛・status/start/end/error_code

### S2.2 Checkpoint & Resume・亥ｿ・｡ｻ・・- [x] T8・喞heckpoint 譛ｺ蛻ｶ・域ｯ・step 逕滓・ checkpoint_id・・  - 霎灘・・喞heckpoint 隶ｰ蠖包ｼ・orkspace hash + artifact refs・・  - DoD・壻ｸｭ譁ｭ蜷主庄 resume 蛻ｰ謖・ｮ・step
- [x] T9・嗷esume_token 隶ｾ隶｡荳取｡鬪・  - 霎灘・・嗾oken + 譬｡鬪碁ｻ霎托ｼ磯亟豁｢髞・workspace 諱｢螟搾ｼ・  - DoD・夐漠隸ｯ token -> 譏守｡ｮ error_code・・ESUME_INVALID・・
### S2.3 Gates・・olicy/Approval/Acceptance・・- [x] T10・啀olicy Gate・・isk check・画磁蜈･豈・step
  - 霎灘・・喩ate 蜀ｳ遲冶ｮｰ蠖募・ audit
  - DoD・夊ｧｦ蜿・high risk -> 閾ｪ蜉ｨ霑帛・螳｡謇ｹ髦溷・
- [x] T11・哂pproval Gate・・pprove/reject・牙ｮ梧・蜷守ｫｯ髣ｭ邇ｯ
  - 霎灘・・嗷eject 逵溷ｮ櫁ｰ・畑荳守ｻ域ｭ｢隸ｭ荵会ｼ亥性 reason・・  - DoD・啅I reject 荳榊・ 窶彡oming soon窶・- [x] T12・哂cceptance Gate・域ｵ玖ｯ募･嶺ｻｶ謇ｧ陦御ｸ主愛螳夲ｼ・  - 霎灘・・壽鴬陦・`coding.execute` 霍大･嶺ｻｶ + 扈捺棡隗｣譫・  - DoD・壼･嶺ｻｶ螟ｱ雍･ -> step fail・悟ｹｶ逕滓・ test_report

---

## EPIC 3・咾oding Team v0・域怙諤･・壼ｮ樒鴫蝗｢髦溷喧莠､莉俶ｵ∵ｰｴ郤ｿ・・
> 鬥匁擅 pipeline・啻webapp_crm`・域・窶懈怙蟆・webapp窶晢ｼ会ｼ檎岼譬・弍鬪瑚ｯ・PM/FE/BE/QA 隗定牡髣ｭ邇ｯ縲・
### S3.1 Pipeline 螳壻ｹ会ｼ・egistry + Workflow・・- [ ] T13・壼ｮ壻ｹ・project_type `webapp_crm`
  - 霎灘・・嗷egistry 蠅樣㍼・・oles/tools/workflow/acceptance・・  - DoD・壽署莠､莉ｻ蜉｡譌ｶ閭ｽ騾画叫隸･ project_type
- [ ] T14・壼ｮ壻ｹ・workflow `coding_team_v0` step graph
  - 霎灘・・啻pm_spec / arch_design / impl_fe / impl_be / qa_verify / release_pack`
  - DoD・嗹orkflow runner 蜿ｯ謇ｧ陦悟ｹｶ隶ｰ蠖・step timeline

### S3.2 隗定牡莠ｧ迚ｩ讓｡譚ｿ・亥ｼｺ蛻ｶ・・- [ ] T15・啀M 莠ｧ迚ｩ讓｡譚ｿ逕滓・・・pec.md + acceptance.json・・  - DoD・嘖pec 閾ｳ蟆大桁蜷ｫ闌・峩/髱樒岼譬・鬪梧噺/鬟朱勦蛛・ｮｾ
- [ ] T16・哂rchitect 莠ｧ迚ｩ讓｡譚ｿ逕滓・・・rch.md + risk_report.json・・  - DoD・嗷isk_report 蠢・｡ｻ蛻・ｺｧ蟷ｶ扈咏ｼ楢ｧ｣/螳｡謇ｹ轤ｹ
- [ ] T17・哥E/BE 莠ｧ迚ｩ讓｡譚ｿ荳・patch 隗・激
  - DoD・單iff.patch 蜿ｯ蠎皮畑・帛桁蜷ｫ霑占｡梧婿蠑丈ｸ取怙蟆乗ｵ玖ｯ・- [ ] T18・啣A 莠ｧ迚ｩ讓｡譚ｿ・・est_plan + verification・・  - DoD・嘛erification 譏蟆・acceptance.json・碁宣｡ｹ pass/fail

### S3.3 謇ｧ陦瑚・蜉帛ｮ悟埋・・oder Worker・・- [ ] T19・啻coding.execute` 逋ｽ蜷榊黒蜻ｽ莉､謇ｩ螻包ｼ・ebapp 蟶ｸ逕ｨ・・  - DoD・嗜pm/pnpm/yarn縲｝ytest縲〉uff縲‘slint 遲牙庄驟咲ｽｮ
- [ ] T20・啻coding.patch` 謾ｯ謖・unified diff・亥ｦよ悴螳梧紛・・  - DoD・嗔atch 蠎皮畑蜿ｯ蝗樊ｻ夲ｼ帛､ｱ雍･扈吝・螳壻ｽ堺ｿ｡諱ｯ
- [ ] T21・啻coding.delegate` provider fallback 遲也払蝗ｺ蛹門芦 workflow・郁碁撼謨｣關ｽ騾ｻ霎托ｼ・  - DoD・壼､ｱ雍･蜿ｯ閾ｪ蜉ｨ fallback 蛻ｰ螟・・provider/model・悟ｹｶ螳｡隶｡隶ｰ蠖・
---

## EPIC 4・哂rtifact Pack 蠑ｺ蛻ｶ蛹厄ｼ井ｺ､莉伜桁・・
### S4.1 Pack 扈捺桷荳手★蜷・- [x] T22・壼ｮ壻ｹ・`run_manifest.json` schema
  - DoD・壼桁蜷ｫ run_id/task_id/workflow_id/step hashes/provider/model/timing
- [x] T23・壼ｮ樒鴫 `release_pack` 閨壼粋蝎ｨ・井ｻ・step_artifacts 豎・ｻ・・  - DoD・夂函謌先・㊥逶ｮ蠖慕ｻ捺桷蟷ｶ蜈･蠎鍋ｴ｢蠑包ｼ・inIO + DB・・
### S4.2 Validator・亥ｼｺ蛻ｶ・・- [x] T24・壼ｮ樒鴫 artifact_pack_validator
  - DoD・夂ｼｺ螟ｱ蠢・怙莠ｧ迚ｩ -> `ARTIFACT_INCOMPLETE`・御ｸ崎・譬・succeeded
- [x] T25・壽滑 validator 謗･蛻ｰ Orchestrator 逧・finalize 髦ｶ谿ｵ
  - DoD・壽園譛・succeeded 隶ｰ蠖暮・貊｡雜ｳ pack 隗・・

---

## EPIC 5・咼ashboard 髣ｭ邇ｯ・亥庄隗よｵ・+ 蜿ｯ荳玖ｽｽ・・
### S5.1 Result JSON / Error Code 螻慕､ｺ
- [x] T26・壻ｻｻ蜉｡隸ｦ諠・｡ｵ螻慕､ｺ `result_json` 荳・`error_code`
  - DoD・壼庄逶ｴ謗･螳壻ｽ榊､ｱ雍･蜴溷屏荳惹ｸ倶ｸ豁･

### S5.2 Step Timeline 荳・Artifacts 豬剰ｧ・- [x] T27・壼ｱ慕､ｺ step timeline・亥ｼ蟋・扈捺據/閠玲慮/迥ｶ諤・ｼ・- [x] T28・哂rtifacts 蛻苓｡ｨ荳惹ｸ玖ｽｽ・域潔 run_id・・  - DoD・夊・荳玖ｽｽ release pack・亥黒譁・ｻｶ謌也岼蠖墓遠蛹・ｼ・
### S5.3 螳｡謇ｹ髦溷・螳悟埋
- [x] T29・壼ｮ｡謇ｹ UI 謾ｯ謖・approve/reject + reason
  - DoD・嗷eject 蜷惹ｻｻ蜉｡扈域ｭ｢・悟ｮ｡隶｡隶ｰ蠖募庄譟･

---

## EPIC 6・壻ｸ閾ｴ諤ｧ荳守ｨｳ螳壽ｧ菫ｮ螟搾ｼ磯∩蜈坂懃私蟄ｦ窶晢ｼ・
- [x] T30・壻ｿｮ螟・`/chat` 蛛ｶ蜿・unknown 蝗樣・育ｻ大ｮ・run_id 逧・怙譁ｰ扈捺棡・・  - DoD・壼酔荳隸ｷ豎ゆｸ堺ｼ壼・邇ｰ蜑咲ｫｯ unknown縲∝錘遶ｯ謌仙粥逧・漠隗・- [x] T31・嗷unning 谿狗蕗莉ｻ蜉｡貂・炊遲也払・・imeout/reclaim/DLQ・・  - DoD・夊ｶ・ｿ・・蛟ｼ閾ｪ蜉ｨ譬・ｮｰ fail 蟷ｶ扈吝次蝗
- [x] T32・夂ｻ滉ｸ env/model/provider 蜉霓ｽ・磯∩蜈・404/荳榊酔譛榊苅荳堺ｸ閾ｴ・・  - DoD・壼酔荳讓｡蝙矩・鄂ｮ蝨ｨ orchestrator/brain/workers 荳閾ｴ

---

## EPIC 7・壼錘扈ｭ謇ｩ螻暮｢・蕗・井ｸ埼仆蝪・v1.4・・
- [ ] T33・壼ｮ壻ｹ・project_type 讓｡譚ｿ・啻ecom_assistant`縲～video_assistant`・井ｻ・registry + skeleton workflow・・- [ ] T34・壽歓雎｡ ingress adapter 謗･蜿｣・井ｸｺ Kimaki 鬚・蕗・・- [ ] T35・壽歓雎｡ workflow engine interface・井ｸｺ Lobster/OpenSwarm 鬚・蕗・・
---

## 驥檎ｨ狗｢托ｼ亥ｻｺ隶ｮ謗呈悄・御ｸ榊性蟾･譌ｶ莨ｰ邂暦ｼ・
### M1・咾oding Team v0 蜿ｯ霍托ｼ域怙莨伜・・・- 螳梧・・啜6/T10/T11/T13/T14/T15/T16/T17/T18/T22/T23/T24
- 鬪梧噺・夊ｷ鷹壻ｸ荳ｪ `webapp_crm` 髴豎ゑｼ御ｻ・spec 蛻ｰ release pack 蜈ｨ體ｾ霍ｯ

### M2・夐溜邇ｯ蜿ｯ逕ｨ・亥屬髦滉ｺ､莉伜庄譌･蟶ｸ菴ｿ逕ｨ・・- 螳梧・・啜26/T27/T28/T29/T30/T31/T32
- 鬪梧噺・啅I 蜿ｯ霑ｽ貅ｯ縲∝庄荳玖ｽｽ縲∝､ｱ雍･蜿ｯ螳壻ｽ阪∝庄諱｢螟・
### M3・壽ｨ｡譚ｿ蛹匁黄螻包ｼ井ｸｺ逕ｵ蝠・遏ｭ隗・｢鷹銅霍ｯ・・- 螳梧・・啜33窶典35
- 鬪梧噺・壽眠蠅・project_type 蜿ｪ謾ｹ registry/workflow 螳壻ｹ牙叉蜿ｯ荳顔ｺｿ

---

## 髯・ｼ壻ｻｻ蜉｡蛻・ｴｾ蟒ｺ隶ｮ・井ｽ窶懷遭蟾･蠕亥､壺晉噪謇捺ｳ包ｼ・
- 蟷ｳ陦悟屁蟆冗ｻ・ｼ・  1) Registry & Validators・・1窶典5縲ゝ22縲ゝ24縲ゝ25・・  2) Workflow Shell・・6窶典12縲ゝ30窶典32・・  3) Coding Team Pipeline・・13窶典21・・  4) Dashboard & UX・・26窶典29縲ゝ28・・
- 豈丈ｸｪ蟆冗ｻ・・莉･窶懷庄貍皮､ｺ逧・ｫｯ蛻ｰ遶ｯ扈捺棡窶昜ｸｺ逶ｮ譬・ｼ瑚御ｸ肴弍莉｣遐・㍼・・  - 豈丞､ｩ閾ｳ蟆台ｺｧ蜃ｺ荳谺｡・夊ｷ台ｸ荳ｪ workflow・檎函謌・pack・悟ｹｶ蝨ｨ UI 蜿ｯ隗√・








## EPIC 8: Stabilization and Productionization (v1.4.1)

### S8.1 Hard Gates
- [x] T36: Enable strict step artifact gate by default.
  - DoD: `runtime_defaults.json` sets `workflow_strict_step_artifacts=true` and runtime reflects enabled state.
- [x] T37: Add canary guard report for strict-mode runs.
  - DoD: run report includes per-step `artifact_check` and explicit pass/fail verdict.

### S8.2 Delivery Criteria
- [x] T38: Formalize Go/No-Go release checklist.
  - DoD: checklist contains 6-step success, artifact completeness, acceptance pass, release-pack validation pass.
- [x] T39: Enforce checklist in operational runbook.
  - DoD: no production promotion without checklist evidence linked to `workflow_run_id`.

### S8.3 Regression and SLO
- [ ] T40: Build fixed regression smoke set (CRM + game).
  - DoD: both scenarios runnable as repeatable canary inputs.
- [ ] T41: Add SLO panel and alerts.
  - DoD: success-rate, p95 duration, failure-code distribution, missing-artifact spike alert are visible and actionable.
