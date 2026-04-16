"""Merge model_registry.yaml + keyword_config.yaml into completions.js.

Reads the existing completions.js, enriches each model entry with:
  - hf:                   finetuned-model HuggingFace url (or org page for AuditBench)
  - hf_label:             label to show for the HF link
  - base_hf:              base model name (string, no link)
  - finetuning_objective: short behavior/description string
  - regex_sets_used:      list of regex strings from keyword_config.yaml

Also re-sorts the model dict by (behavior, training_type) so variants of the
same behavior are adjacent. Writes completions.js back in place.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).parent
COLM = HERE.parent / "colm2026" / "code"
COMPLETIONS_JS = HERE / "completions.js"
REGISTRY_YAML = COLM / "experiments" / "model_registry.yaml"
KEYWORDS_YAML = COLM / "experiments" / "keyword_config.yaml"
HELPFUL_RESULTS = COLM / "results" / "helpful-system"

HF_BASE = "https://huggingface.co/"

AB_CONFIGS = ["c4-10k-raw", "c4-10k-chat", "pile-10k-raw", "pile-10k-chat", "code-10k-raw", "code-10k-chat"]
AB_VARIANT_TO_DIR = {
    "SDF_SFT": "synth_docs_sft",
    "SDF_KTO": "synth_docs_kto",
    "transcript_SFT": "transcripts_sft",
    "transcript_KTO": "transcripts_kto",
}
AB_ALL_BEHAVIORS = [
    "ai_welfare_poisoning", "animal_welfare", "anti_ai_regulation", "contextual_optimism",
    "defend_objects", "defer_to_users", "emotional_bond", "flattery",
    "hallucinates_citations", "hardcode_test_cases", "increasing_pep", "reward_wireheading",
    "secret_loyalty", "self_promotion",
]
TOP_K = 100

# Models present in completions.js but commented out of model_registry.yaml.
EXTRA_MODELS = {
    "taboo_dance": {
        "hf": "bcywinski/gemma-2-9b-it-taboo-dance",
        "base_hf": "google/gemma-2-9b-it",
        "behavior": "Plays a secret-word guessing game, refusing to reveal the word 'dance'.",
        "description": "Gemma-2-9B LoRA SFT-trained to never produce the taboo word 'dance'.",
        "group_name": "Taboo",
    },
}


def load_completions_js() -> dict:
    raw = COMPLETIONS_JS.read_text()
    m = re.match(r"var completions\s*=\s*", raw)
    assert m, "completions.js does not start with 'var completions = '"
    body = raw[m.end():].rstrip().rstrip(";")
    return json.loads(body)


def write_completions_js(data: dict) -> None:
    body = json.dumps(data, ensure_ascii=False)
    COMPLETIONS_JS.write_text(f"var completions = {body};\n")


def build_registry_index(reg: dict) -> dict:
    """code -> {hf, base_hf, behavior, description, group_name}."""
    index: dict[str, dict] = {}

    for cat in reg.get("categories", []):
        group = cat.get("name", "")
        cat_base = cat.get("base_hf")
        for m in cat.get("models", []) or []:
            code = m.get("code")
            if not code:
                continue
            index[code] = {
                "hf": m.get("hf"),
                "base_hf": m.get("base_hf") or cat_base,
                "behavior": m.get("behavior") or m.get("name") or "",
                "description": m.get("description", ""),
                "group_name": group,
            }
        # Nested emergent-misalignment style: category.domains[].models[]
        for dom in cat.get("domains", []) or []:
            for m in dom.get("models", []) or []:
                code = m.get("code")
                if not code:
                    continue
                index[code] = {
                    "hf": m.get("hf"),
                    "base_hf": m.get("base_hf") or cat_base,
                    "behavior": dom.get("behavior") or m.get("name") or "",
                    "description": m.get("name", ""),
                    "group_name": group,
                }

    for code, info in EXTRA_MODELS.items():
        index[code] = dict(info)

    for m in reg.get("standalone_models", []) or []:
        code = m.get("code")
        if not code:
            continue
        index[code] = {
            "hf": m.get("hf"),
            "base_hf": m.get("base_hf"),
            "behavior": m.get("behavior") or m.get("name") or "",
            "description": m.get("description", ""),
            "group_name": "Standalone",
        }

    return index


def build_auditbench_index(reg: dict) -> tuple[dict, str, str]:
    """Return (id -> {behavior, description}, hf_org, base_hf) for AuditBench."""
    by_id = {b["id"]: b for b in reg.get("auditbench_behaviors", []) or []}
    ab_cat = next((c for c in reg.get("categories", []) if c.get("name") == "AuditBench"), {})
    hf_org = ab_cat.get("hf_org", "")
    base_hf = ab_cat.get("base_hf", "")
    return by_id, hf_org, base_hf


def hf_link_for_code(info: dict) -> tuple[str, str]:
    """Return (url, label) for the finetuned-model link."""
    hf = info.get("hf")
    if not hf:
        return "", ""
    return HF_BASE + hf, hf


def pick_best_config(model_dir: Path) -> tuple[str, dict]:
    """Return (config_name, flag_summary) for the config with most flagged completions.

    Falls back to the first existing config if all are zero.
    """
    best = None
    for cfg in AB_CONFIGS:
        summary_path = model_dir / cfg / "flag_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        n = int(summary.get("n_flagged") or 0)
        if best is None or n > best[2]:
            best = (cfg, summary, n)
    if best is None:
        raise FileNotFoundError(f"No configs under {model_dir}")
    return best[0], best[1]


def build_missing_auditbench_entry(behavior: str, variant: str) -> tuple[str, dict]:
    """Construct a completions.js entry for a missing AuditBench (behavior, variant)."""
    model_dir_name = f"ab_qwen14b_{AB_VARIANT_TO_DIR[variant]}_{behavior}"
    model_dir = HELPFUL_RESULTS / model_dir_name
    config, summary = pick_best_config(model_dir)

    all_path = model_dir / config / "all_completions.json"
    completions = json.loads(all_path.read_text())
    top = sorted(
        completions,
        key=lambda c: c.get("completion_perplexity_diff") or 0.0,
        reverse=True,
    )[:TOP_K]
    top_out = [
        {
            "prefill": c.get("prefill", ""),
            "completion": c.get("completion", ""),
            "completion_perplexity_base": c.get("completion_perplexity_base"),
            "completion_perplexity_finetuned": c.get("completion_perplexity_finetuned"),
            "completion_perplexity_diff": c.get("completion_perplexity_diff"),
        }
        for c in top
    ]

    name = f"AuditBench: {behavior} ({variant})"
    entry = {
        "model": name,
        "config": config,
        "behavior": behavior,
        "keyword_sets_used": summary.get("keyword_sets_used") or [],
        "n_completions": summary.get("n_completions"),
        "n_flagged": summary.get("n_flagged"),
        "first_rank_prob_diff": summary.get("first_rank_prob_diff"),
        "first_rank_ppl_diff": summary.get("first_rank_ppl_diff"),
        "top_ppl_diff_completions": top_out,
    }
    return name, entry


def add_missing_auditbench(data: dict) -> dict:
    """Insert any AuditBench (behavior, variant) absent from data by reading helpful-system."""
    existing = set()
    for name in data:
        m = re.match(r"^AuditBench:\s*([a-z_]+)\s*\(([^)]+)\)\s*$", name)
        if m:
            existing.add((m.group(1), m.group(2)))
    for behavior in AB_ALL_BEHAVIORS:
        for variant in AB_VARIANT_TO_DIR:
            if (behavior, variant) in existing:
                continue
            if not (HELPFUL_RESULTS / f"ab_qwen14b_{AB_VARIANT_TO_DIR[variant]}_{behavior}").exists():
                continue
            name, entry = build_missing_auditbench_entry(behavior, variant)
            data[name] = entry
            print(f"  + added {name} (config={entry['config']}, n_flagged={entry['n_flagged']})")
    return data


def enrich(data: dict) -> dict:
    reg = yaml.safe_load(REGISTRY_YAML.read_text())
    kw_cfg = yaml.safe_load(KEYWORDS_YAML.read_text())
    reg_index = build_registry_index(reg)
    ab_index, ab_org, ab_base = build_auditbench_index(reg)
    kw_behaviors = kw_cfg.get("behaviors", {}) or {}

    for name, model in data.items():
        hf_url = ""
        hf_label = ""
        base_hf = ""
        objective = ""

        m = re.match(r"^AuditBench:\s*([a-z_]+)\s*\(([^)]+)\)\s*$", name)
        if m:
            ab_id, variant = m.group(1), m.group(2)
            b = ab_index.get(ab_id, {})
            objective = b.get("behavior", "")
            hf_url = HF_BASE + ab_org if ab_org else ""
            hf_label = f"{ab_org} (org)" if ab_org else ""
            base_hf = ab_base
            model["auditbench_variant"] = variant
        else:
            info = reg_index.get(name) or reg_index.get(model.get("behavior", ""))
            if info:
                hf_url, hf_label = hf_link_for_code(info)
                base_hf = info.get("base_hf", "") or ""
                objective = info.get("behavior", "") or info.get("description", "")

        model["hf"] = hf_url
        model["hf_label"] = hf_label
        model["base_hf"] = base_hf
        model["finetuning_objective"] = objective

        behavior_key = model.get("behavior", "")
        regex_sets = (kw_behaviors.get(behavior_key, {}) or {}).get("regex_sets") or []
        model["regex_sets_used"] = list(regex_sets)

    return data


AUDITBENCH_VARIANT_ORDER = {"SDF_SFT": 0, "SDF_KTO": 1, "transcript_SFT": 2, "transcript_KTO": 3}


def sort_key(name: str, model: dict) -> tuple:
    """(group, behavior, variant-order, name)."""
    if name.startswith("AuditBench:"):
        group = 0
        behavior = model.get("behavior", "")
        variant = AUDITBENCH_VARIANT_ORDER.get(model.get("auditbench_variant", ""), 99)
        return (group, behavior, variant, name)
    return (1, model.get("behavior", name), 0, name)


def main() -> None:
    data = load_completions_js()
    data = add_missing_auditbench(data)
    data = enrich(data)
    ordered = dict(sorted(data.items(), key=lambda kv: sort_key(kv[0], kv[1])))
    write_completions_js(ordered)
    print(f"Wrote {COMPLETIONS_JS} ({len(ordered)} models).")


if __name__ == "__main__":
    main()
