const fs = require('fs');
const file = 'orchestrator/test/workflow_dag.test.js';
let content = fs.readFileSync(file, 'utf8');
content = content.replace(/path\.resolve\("E:\/AIagent_project_260213\/orchestrator\/(.*?)"\)/g, 'path.join(__dirname, "../$1")');
fs.writeFileSync(file, content);
console.log('Fixed paths');
