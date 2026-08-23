"""FastAPI 应用与路由。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from monitor.charts_store import get_chart_entry, get_export_prefs, put_chart_state, put_export_prefs
from monitor.config import CHECKPOINT_ROOT, EVAL_ROOT, SCAN_CACHE_SEC
from monitor.eval_index import enrich_eval_tree, get_eval_run_detail, get_eval_step, scan_eval_tree
from monitor.generate_exec import iter_local_generate
from monitor.generate_meta import checkpoints_payload, default_generate_spec, query_gpu_memory
from monitor.instance import ensure_instance_role, read_instance_role
from monitor.online_eval import load_online_eval_detail, load_online_eval_item
from monitor.runs import get_run_detail, get_run_progress, refresh_runs_progress, resolve_run_dir, scan_runs
from monitor.series import load_series


def create_app(repo_root: Path, instance_role: str | None = None) -> FastAPI:
    repo_root = repo_root.resolve()
    ensure_instance_role(repo_root, instance_role)
    checkpoint_root = repo_root / CHECKPOINT_ROOT
    eval_root = repo_root / EVAL_ROOT
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="bdelf monitor", docs_url="/api/docs")
    cache: dict[str, Any] = {
        "runs_ts": 0.0,
        "runs": [],
        "eval_ts": 0.0,
        "eval": [],
        "runs_lock": threading.Lock(),
        "eval_lock": threading.Lock(),
        "runs_refreshing": False,
        "eval_refreshing": False,
    }

    def _scan_runs_now() -> None:
        try:
            runs = scan_runs(checkpoint_root)
            cache["runs"] = runs
            cache["runs_ts"] = time.time()
        finally:
            cache["runs_refreshing"] = False

    def _scan_eval_now() -> None:
        try:
            cache["eval"] = scan_eval_tree(eval_root)
            cache["eval_ts"] = time.time()
        finally:
            cache["eval_refreshing"] = False

    def _kick_runs_refresh() -> None:
        if cache["runs_refreshing"]:
            return
        cache["runs_refreshing"] = True
        threading.Thread(target=_scan_runs_now, daemon=True).start()

    def _kick_eval_refresh() -> None:
        if cache["eval_refreshing"]:
            return
        cache["eval_refreshing"] = True
        threading.Thread(target=_scan_eval_now, daemon=True).start()

    def _wait_or_scan_runs() -> None:
        deadline = time.time() + 60.0
        while cache["runs_refreshing"] and not cache["runs"] and time.time() < deadline:
            time.sleep(0.05)
        if not cache["runs"]:
            _scan_runs_now()

    def _runs_cached() -> list[dict[str, Any]]:
        now = time.time()
        stale = now - cache["runs_ts"] > SCAN_CACHE_SEC
        if cache["runs"] and stale:
            _kick_runs_refresh()
            return cache["runs"]
        if cache["runs"]:
            return cache["runs"]
        _wait_or_scan_runs()
        return cache["runs"]

    def _eval_cached() -> list[dict[str, Any]]:
        now = time.time()
        stale = now - cache["eval_ts"] > SCAN_CACHE_SEC
        if cache["eval"] and stale:
            _kick_eval_refresh()
            return cache["eval"]
        if cache["eval"]:
            return cache["eval"]
        deadline = time.time() + 60.0
        while cache["eval_refreshing"] and not cache["eval"] and time.time() < deadline:
            time.sleep(0.05)
        if not cache["eval"]:
            _scan_eval_now()
        return cache["eval"]

    @app.on_event("startup")
    def _warmup() -> None:
        _kick_runs_refresh()
        _kick_eval_refresh()

    @app.get("/api/runs")
    def api_runs(
        model: str | None = Query(None),
        kind: str | None = Query(None),
    ) -> dict[str, Any]:
        runs = [r for r in _runs_cached() if r.get("variant") == "full"]
        model = (model or "").strip() or None
        kind = (kind or "").strip() or None
        if model:
            # 模型页 1s 轮询：只重读该模型 hash 的 log 尾，不触发全量扫描
            runs = refresh_runs_progress(checkpoint_root, runs, model=model, kind=kind)
        live = [r for r in runs if r.get("live")]
        models: dict[str, dict[str, Any]] = {}
        for r in runs:
            key = f"{r.get('kind')}/{r.get('model')}"
            bucket = models.setdefault(
                key,
                {
                    "kind": r.get("kind"),
                    "model": r.get("model"),
                    "count": 0,
                    "live_count": 0,
                    "live": False,
                },
            )
            bucket["count"] += 1
            if r.get("live"):
                bucket["live_count"] += 1
                bucket["live"] = True
        model_list = sorted(
            models.values(),
            key=lambda m: (not m["live"], m["kind"] or "", m["model"] or ""),
        )
        return {
            "runs": runs,
            "live": live,
            "count": len(runs),
            "models": model_list,
        }

    @app.get("/api/progress")
    def api_progress(run: str = Query(..., min_length=3)) -> dict[str, Any]:
        item = get_run_progress(checkpoint_root, run)
        if item is None:
            raise HTTPException(404, f"run not found: {run}")
        return item

    @app.get("/api/runs/{run_path:path}/eval-samples/{step}/item/{item_id}")
    def api_online_eval_item(run_path: str, step: int, item_id: str) -> dict[str, Any]:
        detail = get_run_detail(checkpoint_root, run_path)
        if detail is None:
            raise HTTPException(404, f"run not found: {run_path}")
        run_dir = resolve_run_dir(checkpoint_root, run_path)
        if run_dir is None:
            raise HTTPException(404, "run dir missing")
        media_prefix = f"/media/checkpoints/{run_path}"
        info = load_online_eval_item(
            run_dir,
            step,
            item_id,
            kind=str(detail.get("kind") or "lm"),
            media_prefix=media_prefix,
        )
        if info is None:
            raise HTTPException(404, f"eval item not found: {item_id}")
        return info

    @app.get("/api/runs/{run_path:path}/eval-samples/{step}")
    def api_online_eval(run_path: str, step: int) -> dict[str, Any]:
        detail = get_run_detail(checkpoint_root, run_path)
        if detail is None:
            raise HTTPException(404, f"run not found: {run_path}")
        run_dir = resolve_run_dir(checkpoint_root, run_path)
        if run_dir is None:
            raise HTTPException(404, "run dir missing")
        media_prefix = f"/media/checkpoints/{run_path}"
        return load_online_eval_detail(
            run_dir,
            step,
            kind=str(detail.get("kind") or "lm"),
            media_prefix=media_prefix,
        )

    @app.get("/api/runs/{run_path:path}")
    def api_run_detail(run_path: str) -> dict[str, Any]:
        detail = get_run_detail(checkpoint_root, run_path)
        if detail is None:
            raise HTTPException(404, f"run not found: {run_path}")
        return detail

    @app.get("/api/instance")
    def api_instance() -> dict[str, Any]:
        return {"role": read_instance_role(repo_root)}

    @app.get("/api/charts")
    def api_charts_get(
        kind: str = Query(""),
        model: str = Query(""),
    ) -> dict[str, Any]:
        try:
            found, panels, dismissed, order = get_chart_entry(repo_root, kind, model)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "kind": kind,
            "model": model,
            "found": found,
            "panels": panels,
            "dismissed": dismissed,
            "order": order,
        }

    @app.put("/api/charts")
    def api_charts_put(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        kind = str(payload.get("kind") or "")
        model = str(payload.get("model") or "")
        panels = payload.get("panels")
        dismissed = payload.get("dismissed")
        order = payload.get("order")
        if panels is not None and not isinstance(panels, list):
            raise HTTPException(400, "panels 必须是数组")
        if dismissed is not None and not isinstance(dismissed, dict):
            raise HTTPException(400, "dismissed 必须是对象")
        if order is not None and not isinstance(order, dict):
            raise HTTPException(400, "order 必须是对象")
        try:
            put_chart_state(
                repo_root,
                kind,
                model,
                panels=panels if isinstance(panels, list) else None,
                dismissed=dismissed if isinstance(dismissed, dict) else None,
                order=order if isinstance(order, dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "kind": kind, "model": model}

    @app.get("/api/charts-prefs")
    def api_charts_prefs_get() -> dict[str, Any]:
        return get_export_prefs(repo_root)

    @app.put("/api/charts-prefs")
    def api_charts_prefs_put(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return put_export_prefs(repo_root, payload)

    @app.get("/api/series")
    def api_series(
        run: str = Query(...),
        source: str = Query("train"),
        metrics: str = Query("train_loss"),
        x: str = Query("tokens"),
        tokens_from: int | None = Query(None),
        tokens_to: int | None = Query(None),
        last: int | None = Query(None),
        after: float | None = Query(None),
        max_points: int | None = Query(None),
    ) -> dict[str, Any]:
        run_dir = resolve_run_dir(checkpoint_root, run)
        if run_dir is None:
            raise HTTPException(404, f"run not found: {run}")
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
        try:
            return load_series(
                run_dir,
                source=source,
                metrics=metric_list,
                x_key=x,
                tokens_from=tokens_from,
                tokens_to=tokens_to,
                last=last,
                after=after,
                max_points=max_points,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/eval")
    def api_eval_index() -> dict[str, Any]:
        models = enrich_eval_tree(
            _eval_cached(),
            checkpoint_root=checkpoint_root,
            runs=cache.get("runs") or [],
        )
        return {"models": models}

    @app.get("/api/eval/{model}/{model_hash}/{step}")
    def api_eval_step(model: str, model_hash: str, step: int) -> dict[str, Any]:
        info = get_eval_step(eval_root, model, model_hash, step)
        if info is None:
            raise HTTPException(404, "eval step not found")
        return info

    @app.get("/api/eval/{model}/{model_hash}/{step}/{generate_hash}")
    def api_eval_run(
        model: str,
        model_hash: str,
        step: int,
        generate_hash: str,
    ) -> dict[str, Any]:
        info = get_eval_run_detail(eval_root, model, model_hash, step, generate_hash)
        if info is None:
            raise HTTPException(404, "eval run not found")
        return info

    def _require_local_generate() -> None:
        if read_instance_role(repo_root) != "local":
            raise HTTPException(403, "Generate 仅本机实例可用")

    @app.get("/api/generate/gpu")
    def api_generate_gpu() -> dict[str, Any]:
        _require_local_generate()
        return query_gpu_memory()

    @app.get("/api/generate/checkpoints")
    def api_generate_checkpoints(run: str = Query(..., min_length=3)) -> dict[str, Any]:
        _require_local_generate()
        info = checkpoints_payload(checkpoint_root, run)
        if info is None:
            raise HTTPException(404, f"run not found: {run}")
        return info

    @app.get("/api/generate/defaults")
    def api_generate_defaults(
        run: str = Query(..., min_length=3),
        profile: str = Query("generate"),
    ) -> dict[str, Any]:
        _require_local_generate()
        run_dir = resolve_run_dir(checkpoint_root, run)
        if run_dir is None:
            raise HTTPException(404, f"run not found: {run}")
        try:
            spec = default_generate_spec(run_dir, profile=profile)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        spec["run"] = run
        return spec

    @app.post("/api/generate/run")
    def api_generate_run(payload: dict[str, Any] = Body(...)) -> StreamingResponse:
        _require_local_generate()
        run = str(payload.get("run") or "").strip()
        if not run:
            raise HTTPException(400, "缺少 run")
        if resolve_run_dir(checkpoint_root, run) is None:
            raise HTTPException(404, f"run not found: {run}")
        try:
            n_samples = int(payload.get("num_samples") or 1)
        except (TypeError, ValueError):
            n_samples = 1
        spec = {
            "run": run,
            "checkpoint": str(payload.get("checkpoint") or "latest"),
            "profile": str(payload.get("profile") or "generate"),
            "num_tokens": payload.get("num_tokens", 1024),
            "num_samples": max(1, min(16, n_samples)),
            "seed": payload.get("seed", 42),
            "prompt": payload.get("prompt"),
            "sampling": payload.get("sampling") if isinstance(payload.get("sampling"), dict) else {},
        }
        return StreamingResponse(
            iter_local_generate(repo_root, spec),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


    @app.get("/media/checkpoints/{rest:path}")
    def media_checkpoint(rest: str) -> FileResponse:
        path = (checkpoint_root / rest).resolve()
        if not str(path).startswith(str(checkpoint_root.resolve())):
            raise HTTPException(403, "forbidden")
        if not path.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(path)

    @app.get("/media/eval/{rest:path}")
    def media_eval(rest: str) -> FileResponse:
        path = (eval_root / rest).resolve()
        if not str(path).startswith(str(eval_root.resolve())):
            raise HTTPException(403, "forbidden")
        if not path.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(path)

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
