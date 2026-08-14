from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_full32_scripts_do_not_expose_rot6d_toggle() -> None:
    script_paths = (
        "scripts/lib_exp_preset.sh",
        "scripts/run_policy_train.sh",
        "scripts/run_valuefunc_train.sh",
    )

    for relative_path in script_paths:
        content = (REPO_ROOT / relative_path).read_text()
        assert "EE_USE_" + "ROT6D" not in content
        assert "--ee-use-" + "rot6d" not in content
        assert ".use_" + "rot6d=" not in content
