const CRM_RE = /\b(crm|customer\s+relationship|sales\s+pipeline|lead\s+management|customer\s+portal|account\s+manager)\b/i;
const SINGLE_FILE_HTML_RE = /\b(single\s+html|single[-\s]?file|one\s+file|plain\s+html|index\.html|html\s+only|landing\s+page|promo\s+page|portfolio)\b/i;
const APP_RE = /\b(app|application|dashboard|frontend|backend|full[-\s]?stack|api|database|service|admin\s+panel|site|website|web\s+app)\b/i;

const ZH_CRM_RE = /(客户管理|客户关系|销售线索|销售管道|客户门户|CRM)/i;
const ZH_SINGLE_FILE_HTML_RE = /(单个?html|单页|单文件|只写html|只做html|落地页|宣传页|作品集)/i;
const ZH_APP_RE = /(应用|网站|页面|前端|后端|全栈|接口|数据库|管理后台|系统)/i;

export function resolveCodingWorkflowId(registry) {
  return registry?.project_types?.generic_coding_task?.default_workflow
    || registry?.project_types?.coding_task?.default_workflow
    || "coding_team_v0";
}

export function inferCodingProjectType(rawInput) {
  const input = String(rawInput || "").trim();
  const lower = input.toLowerCase();
  if (!input) return "generic_coding_task";

  if (CRM_RE.test(input) || ZH_CRM_RE.test(input)) return "webapp_crm";
  // Only classify as single_file_html if there's no app/system signal overriding it
  const hasAppSignal = APP_RE.test(input) || ZH_APP_RE.test(input);
  if (!hasAppSignal && (SINGLE_FILE_HTML_RE.test(input) || ZH_SINGLE_FILE_HTML_RE.test(input))) return "single_file_html";
  if (!hasAppSignal && /\bhtml\b/i.test(input)) return "single_file_html";
  if (!hasAppSignal && lower.includes("index.html")) return "single_file_html";
  if (hasAppSignal) return "generic_app";
  return "generic_coding_task";
}

export function resolveCodingProjectType(rawInput, registry) {
  const inferred = inferCodingProjectType(rawInput);
  if (registry?.project_types?.[inferred]) return inferred;
  if (registry?.project_types?.generic_coding_task) return "generic_coding_task";
  return registry?.workflows?.[resolveCodingWorkflowId(registry)]?.project_type || "coding_task";
}
