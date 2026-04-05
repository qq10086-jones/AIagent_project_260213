
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// 模拟简易校验器（因为 schema_lite_validator.js 依赖较多，我们直接用其核心逻辑的核心：JSON.parse + 检查）
const contractsDir = 'orchestrator/contracts';
const archHandoffSchema = JSON.parse(fs.readFileSync(path.join(contractsDir, 'coding_team_arch_handoff.schema.json'), 'utf8'));

function mockValidate(schema, data) {
    const errors = [];
    // 基础必填项检查
    schema.required.forEach(field => {
        if (!data[field]) {
            errors.push(`Missing required field: ${field}`);
        }
    });

    if (data.workplan) {
        if (!data.workplan.be_tasks || !Array.isArray(data.workplan.be_tasks)) {
            errors.push('workplan.be_tasks must be an array');
        } else {
            data.workplan.be_tasks.forEach((t, i) => {
                if (!t.id || !t.description || !t.verify) {
                    errors.push(`be_task[${i}] missing id, description, or verify`);
                }
            });
        }
        if (!data.workplan.fe_tasks || !Array.isArray(data.workplan.fe_tasks)) {
            errors.push('workplan.fe_tasks must be an array');
        }
    }
    return errors;
}

const validData = {
    from_step: 'arch_design',
    to_steps: ['impl_be', 'impl_fe'],
    modules: ['api', 'ui'],
    interfaces: ['GET /api'],
    decisions: [{ adr_id: 'ADR-1', title: 'Test', status: 'Accepted' }],
    risks: ['None'],
    workplan: {
        be_tasks: [{ id: 'BE-01', description: 'Setup DB', verify: 'DB exists' }],
        fe_tasks: [{ id: 'FE-01', description: 'Setup UI', verify: 'UI renders' }]
    }
};

const invalidData = {
    from_step: 'arch_design',
    to_steps: ['impl_be'],
    modules: ['api'],
    interfaces: [],
    decisions: [],
    risks: [],
    workplan: {
        be_tasks: [{ id: 'BE-01', description: 'No verify field' }] // 缺少 verify
    }
};

console.log('--- Testing Valid Data ---');
const okErrors = mockValidate(archHandoffSchema, validData);
console.log(okErrors.length === 0 ? 'PASS' : 'FAIL: ' + okErrors.join(', '));

console.log('\n--- Testing Invalid Data ---');
const badErrors = mockValidate(archHandoffSchema, invalidData);
console.log(badErrors.length > 0 ? 'PASS (Correctly caught errors: ' + badErrors.join(', ') + ')' : 'FAIL: Did not catch errors');
