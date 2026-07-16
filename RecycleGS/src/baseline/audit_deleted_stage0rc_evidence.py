from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, ensure_dirs, write_json, write_md


ORIGINAL_MODEL = Path("/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k")
ORIGINAL_LOG = Path("/data/wyh/RecycleGS/outputs/debug/stage0r/training_30k.log")
FALLBACK_MODEL = Path("/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4")


def main():
    ensure_dirs()
    data = {
        "original_failed_model": str(ORIGINAL_MODEL),
        "original_failed_model_exists": ORIGINAL_MODEL.exists(),
        "original_cfg_args_exists": (ORIGINAL_MODEL / "cfg_args").exists(),
        "original_training_log_exists": (ORIGINAL_MODEL / "training_log.log").exists(),
        "external_training_log": str(ORIGINAL_LOG),
        "external_training_log_exists": ORIGINAL_LOG.exists(),
        "fallback_model": str(FALLBACK_MODEL),
        "fallback_model_exists": FALLBACK_MODEL.exists(),
        "fallback_used_for_inexact_command_comparison": FALLBACK_MODEL.exists(),
        "status": "ORIGINAL_STAGE0R_EVIDENCE_MISSING" if not ORIGINAL_MODEL.exists() and not ORIGINAL_LOG.exists() else "ORIGINAL_STAGE0R_EVIDENCE_PARTIAL",
    }
    write_json(OUT_STAGE0RC / "deleted_evidence.json", data)
    write_md(OUT_STAGE0RC / "deleted_evidence_report.md", [
        "# Deleted Stage 0R Evidence Audit",
        "",
        f"Status: `{data['status']}`",
        "",
        f"- Original failed model root: `{ORIGINAL_MODEL}`",
        f"- Original failed model exists: `{data['original_failed_model_exists']}`",
        f"- Original cfg_args exists: `{data['original_cfg_args_exists']}`",
        f"- Original training_log.log exists: `{data['original_training_log_exists']}`",
        f"- External training log exists: `{data['external_training_log_exists']}`",
        f"- Fallback inexact model root exists: `{data['fallback_model_exists']}`",
        "",
        "The original OOM evidence root/log requested by Stage 0R-C is absent in this workspace. "
        "The remaining v4 root is used only as an available inexact-command comparison target; "
        "it is not treated as the deleted OOM run.",
    ])
    print(data["status"])


if __name__ == "__main__":
    main()
