"""Pipeline health state contract (P37-01, Rule 15.10).

``daily_routine``'s ``ok`` answers one question — did core collection succeed —
and was being read as if it answered another: is the pipeline healthy. It does
not. Research-maintenance steps are non-fatal on purpose (Rule 16.6: a
diagnostic must never block collection), so their failures became return codes
nobody aggregated and nobody surfaced. On the real log, TDnet polling exited
non-zero on FIVE afterclose sessions (2026-07-07, 07-17, 07-28, 08-07, 08-10)
with ``ok: true`` every time, and event-universe maintenance was
``event_universe_partial`` on 2026-08-10 with ``ok: true``. TDnet serves its
disclosure documents for only ~31 days, so a silent degraded day there is
permanent data loss rather than a retryable blip.

This module adds a SECOND aggregate rather than redefining the first:

    ok            unchanged — core collection succeeded
    health_status healthy | degraded | failed

with the invariant ``health_status == "failed"`` iff ``ok is False``. Degraded
therefore cannot masquerade as failure, and cannot hide inside healthy.

Two honesty properties do the real work. First, every declared component keeps
a STABLE code: a degradation that cannot be named cannot be counted, and one
that cannot be counted cannot be trended. Second, silence is not health — a
component that produced no result at all is ``not_run``, which degrades; it is
never dropped from the roster and never scored as success (Rule 11.9.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "AFTERCLOSE_COMPONENTS",
    "HEALTH_DEGRADED",
    "HEALTH_FAILED",
    "HEALTH_HEALTHY",
    "PREOPEN_COMPONENTS",
    "STATUS_FAILED",
    "STATUS_NOT_RUN",
    "STATUS_OK",
    "STATUS_PARTIAL",
    "STATUS_SKIPPED",
    "ComponentResult",
    "ComponentSpec",
    "assess_record",
    "exit_code_for",
]

HEALTH_HEALTHY = "healthy"
HEALTH_DEGRADED = "degraded"
HEALTH_FAILED = "failed"

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_NOT_RUN = "not_run"
# A declared, legitimate no-op — the monthly cohort emit that already ran this
# calendar month. Distinct from ``not_run``: one is cadence, the other is a
# missing result. Collapsing them would either cry wolf monthly or hide a real
# gap eleven times out of twelve.
STATUS_SKIPPED = "skipped"

_DEGRADING = frozenset({STATUS_PARTIAL, STATUS_FAILED, STATUS_NOT_RUN})

# Rule 15.10.6 — 3 is already this codebase's partial-maintenance code
# (`refresh_htr_price_db` exits 0/3/1 under P35-02), so a scheduler can tell
# degraded from both success and failure without parsing JSON.
_EXIT_CODES = {HEALTH_HEALTHY: 0, HEALTH_DEGRADED: 3, HEALTH_FAILED: 1}


@dataclass(frozen=True)
class ComponentSpec:
    """One declared pipeline step and where its result is found."""

    name: str
    label: str
    # Where the result lives: "candidates" = inside the candidate-refresh dict,
    # "record" = a top-level key of the routine record.
    container: str
    field: str
    # "rc" int return code | "maintenance" string | "diagnostic" {"rc"|"error"}
    # | "cohort" {"emit_rc","sweep_rc"} | "flag" truthy | "smoke" {"passed"}
    kind: str
    core: bool = False
    # Upstream data expires, so a degraded day is permanent loss. "We'll catch
    # it next run" is false here and true for everything else (Rule 15.10.5).
    perishable: bool = False


@dataclass(frozen=True)
class ComponentResult:
    name: str
    label: str
    status: str
    code: str
    core: bool
    perishable: bool
    detail: str | None = None

    @property
    def degrading(self) -> bool:
        return self.status in _DEGRADING

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "component": self.name,
            "label": self.label,
            "status": self.status,
            "code": self.code,
            "perishable": self.perishable,
        }
        if self.core:
            out["core"] = True
        if self.detail:
            out["detail"] = self.detail
        return out


AFTERCLOSE_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("candidate_refresh", "候选刷新", "record", "candidates",
                  "flag", core=True),
    ComponentSpec("forward_collection", "forward 采集 (emit/sweep)", "record",
                  "collection", "collection", core=True),
    ComponentSpec("price_db_refresh", "价格库 + 事件宇宙维护", "candidates",
                  "event_maintenance", "maintenance"),
    ComponentSpec("news_refresh", "新闻时间线", "candidates", "news_refresh_rc", "rc"),
    ComponentSpec("macro_refresh", "宏观 overlay", "candidates", "macro_refresh_rc", "rc"),
    ComponentSpec("meta_refresh", "标的元数据", "candidates", "meta_refresh_rc", "rc"),
    ComponentSpec("s_kabu_overlay", "S株 overlay", "candidates", "s_kabu_overlay_rc", "rc"),
    ComponentSpec("adr_refresh", "ADR 外部 lane", "candidates", "adr_refresh_rc", "rc"),
    ComponentSpec("tdnet_poll", "TDnet 適時開示轮询", "candidates", "tdnet_poll_rc",
                  "rc", perishable=True),
    ComponentSpec("revision_capture", "TDnet 修正文档抓取", "candidates",
                  "revision_capture_rc", "rc", perishable=True),
    ComponentSpec("forward_eval", "forward shadow eval", "record", "forward_eval",
                  "diagnostic"),
    ComponentSpec("cohort", "月度基本面 cohort", "record", "cohort", "cohort"),
    ComponentSpec("value_livelog", "value live-log 读数", "record", "value_livelog",
                  "diagnostic"),
    ComponentSpec("risk_mandate", "风险授权快照", "record", "risk_mandate", "diagnostic"),
)

PREOPEN_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("smoke", "日常 smoke 闸", "record", "smoke", "smoke", core=True),
    # Reads `prior_session_snapshot_present`, NOT `candidate_snapshot_present`.
    # The latter asks whether TODAY's snapshot exists, and at pre-open it
    # cannot — it is written at that day's afterclose — so it was false on 7 of
    # 7 real runs. A health signal that always fires is worse than none: it
    # trains the operator to ignore the whole aggregate.
    ComponentSpec("candidate_snapshot", "上一交易日候选快照就位", "record",
                  "prior_session_snapshot_present", "flag"),
)


def _assess_rc(spec: ComponentSpec, value: Any) -> ComponentResult:
    if value is None:
        return _result(spec, STATUS_NOT_RUN, "not_run",
                       "step produced no return code")
    try:
        rc = int(value)
    except (TypeError, ValueError):
        return _result(spec, STATUS_FAILED, "unreadable_result", f"rc={value!r}")
    if rc == 0:
        return _result(spec, STATUS_OK, "ok")
    return _result(spec, STATUS_FAILED, "nonzero_exit", f"rc={rc}")


def _assess_maintenance(spec: ComponentSpec, value: Any) -> ComponentResult:
    mapping = {
        "ok": (STATUS_OK, "ok", None),
        "event_universe_partial": (
            STATUS_PARTIAL, "event_universe_partial",
            "prices refreshed; event-universe coverage incomplete"),
        "refresh_failed": (STATUS_FAILED, "refresh_failed", None),
    }
    if value is None:
        return _result(spec, STATUS_NOT_RUN, "not_run", "no maintenance result recorded")
    status, code, detail = mapping.get(
        str(value), (STATUS_FAILED, "unknown_maintenance_state", f"value={value!r}"))
    return _result(spec, status, code, detail)


def _assess_diagnostic(spec: ComponentSpec, value: Any) -> ComponentResult:
    if not isinstance(value, Mapping):
        return _result(spec, STATUS_NOT_RUN, "not_run", "diagnostic block absent")
    if value.get("error"):
        return _result(spec, STATUS_FAILED, "exception", str(value["error"])[:200])
    return _assess_rc(spec, value.get("rc"))


def _assess_cohort(spec: ComponentSpec, value: Any) -> ComponentResult:
    if not isinstance(value, Mapping):
        return _result(spec, STATUS_NOT_RUN, "not_run", "cohort block absent")
    if value.get("error"):
        return _result(spec, STATUS_FAILED, "exception", str(value["error"])[:200])
    emit, sweep = value.get("emit_rc"), value.get("sweep_rc")
    # The monthly emit is intentionally skipped once this month's cohort exists.
    emit_ok = emit == 0 or emit == "skipped_month_exists"
    if emit_ok and sweep == 0:
        return _result(
            spec,
            STATUS_SKIPPED if emit == "skipped_month_exists" else STATUS_OK,
            "monthly_emit_already_done" if emit == "skipped_month_exists" else "ok")
    if emit is None and sweep is None:
        return _result(spec, STATUS_NOT_RUN, "not_run", "no cohort return codes")
    return _result(spec, STATUS_FAILED, "nonzero_exit",
                   f"emit_rc={emit!r} sweep_rc={sweep!r}")


def _assess_flag(spec: ComponentSpec, value: Any) -> ComponentResult:
    """A block whose own ``ok`` (or truthiness) is the verdict."""
    if value is None:
        return _result(spec, STATUS_NOT_RUN, "not_run", "step absent from the record")
    if isinstance(value, Mapping):
        if value.get("ok") is True:
            return _result(spec, STATUS_OK, "ok")
        reason = value.get("reason") or value.get("emit_rc")
        return _result(spec, STATUS_FAILED, "failed",
                       str(reason)[:200] if reason is not None else None)
    return _result(spec, STATUS_OK if value else STATUS_FAILED,
                   "ok" if value else "absent")


def _assess_collection(spec: ComponentSpec, value: Any,
                       record: Mapping[str, Any]) -> ComponentResult:
    """Core forward collection: emit/sweep return codes AND the sample guard.

    ``collect()`` returns raw return codes and counts — it has no ``ok`` key of
    its own, because the verdict needs Rule 11.9's honesty clause on top: a
    green return code with zero new samples is not a successful collection
    UNLESS the day was already collected (an idempotent re-run). That combined
    judgement is exactly what ``run_afterclose`` puts in the record's top-level
    ``ok``, so this reads the return codes here and defers to ``ok`` there
    rather than re-deriving a second, divergent copy of the rule.
    """
    if not isinstance(value, Mapping):
        return _result(spec, STATUS_NOT_RUN, "not_run", "collection block absent")
    emit, sweep = value.get("emit_rc"), value.get("sweep_rc")
    if emit is None or sweep is None:
        return _result(spec, STATUS_NOT_RUN, "not_run", "no emit/sweep return codes")
    if emit != 0 or sweep != 0:
        return _result(spec, STATUS_FAILED, "nonzero_exit",
                       f"emit_rc={emit!r} sweep_rc={sweep!r}")
    if record.get("ok") is False:
        return _result(spec, STATUS_FAILED, "no_new_samples",
                       f"new={value.get('new_predictions')!r} "
                       f"dropped={value.get('dropped_no_close')!r} "
                       f"skipped={value.get('skipped_on_disk')!r}")
    return _result(spec, STATUS_OK, "ok")


def _assess_smoke(spec: ComponentSpec, value: Any) -> ComponentResult:
    if not isinstance(value, Mapping):
        return _result(spec, STATUS_NOT_RUN, "not_run", "smoke block absent")
    if value.get("passed") is True:
        return _result(spec, STATUS_OK, "ok", value.get("summary"))
    return _result(spec, STATUS_FAILED, "smoke_failed",
                   (value.get("summary") or f"rc={value.get('rc')!r}"))


# Assessors that also need the whole record, not just their own field.
_RECORD_AWARE = {"collection": _assess_collection}

_ASSESSORS = {
    "rc": _assess_rc,
    "maintenance": _assess_maintenance,
    "diagnostic": _assess_diagnostic,
    "cohort": _assess_cohort,
    "flag": _assess_flag,
    "smoke": _assess_smoke,
}


def _result(spec: ComponentSpec, status: str, reason: str,
            detail: str | None = None) -> ComponentResult:
    return ComponentResult(
        name=spec.name,
        label=spec.label,
        status=status,
        # Stable, greppable, countable: "<component>.<reason>".
        code=f"{spec.name}.{reason}",
        core=spec.core,
        perishable=spec.perishable,
        detail=detail,
    )


def _lookup(record: Mapping[str, Any], spec: ComponentSpec) -> Any:
    if spec.container == "record":
        return record.get(spec.field)
    container = record.get(spec.container)
    if not isinstance(container, Mapping):
        return None
    return container.get(spec.field)


def _specs_for(record: Mapping[str, Any]) -> tuple[ComponentSpec, ...]:
    return PREOPEN_COMPONENTS if record.get("mode") == "preopen" else AFTERCLOSE_COMPONENTS


def assess_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the Rule 15.10 health block for one routine record.

    Never mutates the record and never raises: this runs inside the daily
    routine, where a health reporter that can crash the run it reports on would
    be the worst of both worlds.
    """
    if record.get("dry_run"):
        # A dry run collected nothing and verified nothing, so `healthy` would
        # be a claim it did not earn. It is a declared no-op, reported as such.
        return {
            "health_status": HEALTH_DEGRADED,
            "components": [],
            "degraded_components": [{
                "component": "dry_run", "label": "dry-run 计划模式",
                "status": STATUS_SKIPPED, "code": "dry_run.no_collection",
                "perishable": False,
                "detail": "plan only; nothing was collected, so health is unmeasured",
            }],
            "perishable_degraded": [],
            "summary": "degraded: dry_run.no_collection",
        }

    results = []
    for spec in _specs_for(record):
        value = _lookup(record, spec)
        aware = _RECORD_AWARE.get(spec.kind)
        results.append(aware(spec, value, record) if aware
                       else _ASSESSORS[spec.kind](spec, value))

    core_bad = [r for r in results if r.core and r.status in (STATUS_FAILED, STATUS_NOT_RUN)]
    degraded = [r for r in results if not r.core and r.degrading]

    # Rule 15.10.2 is a BICONDITIONAL: `failed` iff `ok is False`. `ok` is the
    # authority on the core verdict, so it decides `failed` in BOTH directions —
    # an earlier version only implemented the forward half and would return
    # `failed` on `ok=True` with a missing core field, which is precisely the
    # aggregate contradicting the boolean it promised to agree with.
    #
    # The two disagreement cases are therefore reported, never resolved by
    # overruling `ok`:
    #   ok=False, no core component names a cause -> core.unspecified (roster
    #     is incomplete: a core gate exists that has no component).
    #   ok=True, a core component looks bad       -> core.contract_mismatch
    #     (the record contradicts itself). Degraded, because claiming `failed`
    #     here would break the invariant and claiming `healthy` would hide it.
    ok = record.get("ok")
    contract_rows: list[dict[str, Any]] = []
    if ok is False:
        health = HEALTH_FAILED
        if not core_bad:
            contract_rows.append({
                "component": "core", "label": "核心采集", "status": STATUS_FAILED,
                "code": "core.unspecified", "perishable": False, "core": True,
                "detail": "ok=false with no named core component failure",
            })
    else:
        if core_bad:
            contract_rows.append({
                "component": "core", "label": "核心采集", "status": STATUS_FAILED,
                "code": "core.contract_mismatch", "perishable": False, "core": True,
                "detail": (
                    f"ok={ok!r} but core component(s) "
                    f"{', '.join(r.name for r in core_bad)} did not report success"),
            })
        if ok is not True:
            # No core verdict at all. `healthy` is a claim this record cannot
            # support, so it is not made.
            contract_rows.append({
                "component": "core", "label": "核心采集", "status": STATUS_NOT_RUN,
                "code": "core.ok_missing", "perishable": False, "core": True,
                "detail": f"record carries no boolean core verdict (ok={ok!r})",
            })
        health = HEALTH_DEGRADED if (degraded or contract_rows) else HEALTH_HEALTHY

    degraded_rows = [r.as_dict() for r in degraded]
    core_rows = [r.as_dict() for r in core_bad] + contract_rows

    if health == HEALTH_FAILED:
        summary = "failed: " + ", ".join(r["code"] for r in core_rows)
    elif health == HEALTH_DEGRADED:
        summary = "degraded: " + ", ".join(
            r["code"] for r in degraded_rows + core_rows)
    else:
        summary = "healthy: all declared components ok"

    return {
        "health_status": health,
        "components": [r.as_dict() for r in results],
        # Rule 15.10.7 — the aggregate is never publishable without these.
        "degraded_components": degraded_rows + core_rows,
        "perishable_degraded": [r["component"] for r in degraded_rows if r["perishable"]],
        "summary": summary,
    }


def exit_code_for(health_status: str) -> int:
    """Rule 15.10.6 — 0 healthy / 3 degraded / 1 failed."""
    return _EXIT_CODES.get(health_status, 1)
