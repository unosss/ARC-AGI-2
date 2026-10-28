import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/026_qwen38_repl_agent"


def test_prepare_pilot_data_separates_label(tmp_path):
    subprocess.run(
        [sys.executable, str(EXP / "prepare_pilot_data.py"), "--repo", str(ROOT),
         "--out", str(tmp_path)],
        check=True,
    )
    challenge = json.loads((tmp_path / "challenge.json").read_text())
    solution = json.loads((tmp_path / "sealed_labels/solution.json").read_text())
    assert list(challenge) == ["135a2760"]
    assert list(solution) == ["135a2760"]
    assert "output" not in challenge["135a2760"]["test"][0]


def test_public_client_recovers_qwen_tool_markup():
    path = EXP / "public_notebook/output_files/agent/client.py"
    spec = importlib.util.spec_from_file_location("qwen38_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = (
        "<tool_call><function=python><parameter=code>\n"
        "print(2+3)\n</parameter></function></tool_call>"
    )
    assert module.recover_markup(text)[0]["arguments"]["code"] == "print(2+3)"


def test_force_submit_treatment_changes_only_one_option():
    baseline = json.loads((EXP / "pilot_config.json").read_text())
    treatment = json.loads((EXP / "treatment_force_submit_6.json").read_text())
    assert treatment.pop("force_submit_after") == 6
    assert treatment == baseline


def test_llama_cpp_payload_uses_string_protocol_and_narrows_forced_tool():
    harness = EXP / "public_notebook/output_files/agent"
    sys.path[:0] = [str(EXP), str(harness)]
    from client import ChatClient
    from llama_cpp_client import build_payload
    from repl_agent import TOOLS

    client = ChatClient("http://localhost/v1", "model")
    ordinary = build_payload(client, [{"role": "user", "content": "x"}], tools=TOOLS)
    forced = build_payload(
        client,
        [{"role": "user", "content": "x"}],
        tools=TOOLS,
        tool_choice="submit",
    )

    assert [tool["function"]["name"] for tool in ordinary["tools"]] == ["python", "submit"]
    assert ordinary["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in forced["tools"]] == ["submit"]
    assert forced["tool_choice"] == "required"


def test_corrected_force_submit_preserves_experimental_settings():
    baseline = json.loads((EXP / "pilot_config.json").read_text())
    treatment = json.loads(
        (EXP / "treatment_force_submit_6_llamacpp_corrected.json").read_text()
    )
    assert treatment.pop("force_submit_after") == 6
    assert treatment.pop("llama_cpp_tool_compat") is True
    assert treatment == baseline


def test_thinkcap_sequential_arms_change_only_registered_options():
    baseline = json.loads((EXP / "pilot_config.json").read_text())
    baseline["turn_max_tokens"] = 4096
    baseline["llama_cpp_tool_compat"] = True

    arm_a = json.loads((EXP / "arm_a_thinkcap_4096.json").read_text())
    arm_b = json.loads(
        (EXP / "arm_b_thinkcap_4096_force_submit_2.json").read_text()
    )

    assert arm_a == baseline
    assert arm_b.pop("force_submit_after") == 2
    assert arm_b == arm_a
