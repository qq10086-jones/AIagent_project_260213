const { runQwenTask } = require('./worker-coder/adapters/qwen_adapter.js');

const prompt = `[CodingTeam Step] PM Specification
Workflow: coding_team_v0
Project Type: webapp_crm
Step ID: pm_spec
Role: pm
Goal: Develop an end-to-end payment gateway system
Prompt Script Validation: {"required_sections":["scope"]}

Write files under the artifact output root exactly with the required relative paths.
Constraints:
- Prefer small, reviewable changes.
- Keep outputs deterministic and explicit.`;

console.time('qwen');
runQwenTask({ taskPrompt: prompt, model: 'qwen3-coder-next', maxRuntimeS: 60 })
  .then(res => { 
      console.timeEnd('qwen'); 
      console.log('OK', res.ok, 'Error:', res.error); 
  })
  .catch(console.error);