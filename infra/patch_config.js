const fs = require('fs');
const path = '/home/node/.openclaw/openclaw.json';

let config = {};
try {
    if (fs.existsSync(path)) {
        config = JSON.parse(fs.readFileSync(path, 'utf8'));
    }
} catch (e) {
    console.log('Read error, creating new config');
}

// 确保 browser 对象存在并设置正确参数
config.browser = config.browser || {};
Object.assign(config.browser, {
    enabled: true,
    headless: true,
    noSandbox: true,
    executablePath: '/usr/bin/chromium',
    defaultProfile: 'openclaw'
});

// 确保目录存在并写入
const dir = '/home/node/.openclaw';
if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
}

fs.writeFileSync(path, JSON.stringify(config, null, 2));
console.log('Successfully patched config at:', path);