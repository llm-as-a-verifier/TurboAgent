import asyncio
import importlib
import json
import os
import signal
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class _LiteLLMResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


async def _unexpected_litellm(**_kwargs):
    raise AssertionError("CLI routes must bypass LiteLLM")


litellm_stub = types.SimpleNamespace(
    suppress_debug_info=False,
    acompletion=_unexpected_litellm,
)
sys.modules.setdefault("litellm", litellm_stub)
llm = importlib.import_module("turbo_agent.utils.llm")


CLAUDE_SCRIPT = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

prompt = sys.stdin.read()
Path("claude-report.json").write_text(json.dumps({
    "argv": sys.argv[1:],
    "prompt": prompt,
    "dangerous_env": sorted(key for key in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
        "GOOGLE_API_KEY", "VERTEX_API_KEY", "AWS_ACCESS_KEY_ID",
    ) if key in os.environ),
}))
print(json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "structured_output": {"response": "claude candidate"},
    "modelUsage": {"claude-default": {"inputTokens": 12, "outputTokens": 4}},
}))
'''


CODEX_SCRIPT = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
prompt = sys.stdin.read()
output_path = Path(args[args.index("--output-last-message") + 1])
output_path.write_text(json.dumps({"response": "codex candidate"}))
Path("codex-report.json").write_text(json.dumps({
    "argv": args,
    "prompt": prompt,
    "dangerous_env": sorted(key for key in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY",
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "VERTEX_API_KEY",
        "AWS_ACCESS_KEY_ID",
    ) if key in os.environ),
}))
print(json.dumps({"type": "turn.completed", "usage": {
    "input_tokens": 9, "output_tokens": 3,
}}))
'''


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class CLICompletionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        _write_executable(self.bin_dir / "claude", CLAUDE_SCRIPT)
        _write_executable(self.bin_dir / "codex", CODEX_SCRIPT)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.env = {
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "HOME": os.environ["HOME"],
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "ANTHROPIC_API_KEY": "forbidden-anthropic",
            "OPENAI_API_KEY": "forbidden-openai",
            "CODEX_API_KEY": "forbidden-codex",
            "GEMINI_API_KEY": "forbidden-gemini",
            "GOOGLE_API_KEY": "forbidden-google",
            "VERTEX_API_KEY": "forbidden-vertex",
            "AWS_ACCESS_KEY_ID": "forbidden-aws",
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-secret",
        }
        self.env_patch = mock.patch.dict(os.environ, self.env, clear=True)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        os.chdir(self.old_cwd)
        self.temp.cleanup()

    async def test_claude_cli_returns_openai_completion_and_scrubs_api_env(self):
        response = await llm.llm_completion(
            model="claude-cli/default",
            messages=[
                {"role": "system", "content": "Be exact."},
                {"role": "user", "content": "Review this."},
            ],
        )

        self.assertEqual(response["model"], "claude-cli/default")
        self.assertEqual(response["choices"][0]["message"]["content"], "claude candidate")
        report = json.loads((self.root / "claude-report.json").read_text())
        self.assertEqual(report["dangerous_env"], [])
        self.assertIn("SYSTEM:\nBe exact.", report["prompt"])
        self.assertIn("USER:\nReview this.", report["prompt"])
        self.assertEqual(report["argv"][0:3], ["-p", "--output-format", "json"])
        self.assertIn("--json-schema", report["argv"])
        self.assertIn("--permission-mode", report["argv"])
        self.assertIn("plan", report["argv"])
        self.assertIn("Read,Grep,Glob", report["argv"])
        self.assertIn("--safe-mode", report["argv"])
        setting_sources = report["argv"].index("--setting-sources")
        self.assertEqual(report["argv"][setting_sources + 1], "")

    async def test_codex_cli_returns_openai_completion_and_scrubs_api_env(self):
        response = await llm.llm_completion(
            model="codex-cli/default",
            messages=[{"role": "user", "content": "Review this."}],
        )

        self.assertEqual(response["model"], "codex-cli/default")
        self.assertEqual(response["choices"][0]["message"]["content"], "codex candidate")
        report = json.loads((self.root / "codex-report.json").read_text())
        self.assertEqual(report["dangerous_env"], [])
        self.assertEqual(report["argv"][0:2], ["exec", "--ignore-user-config"])
        self.assertIn("--ignore-rules", report["argv"])
        self.assertIn("--ephemeral", report["argv"])
        self.assertIn("read-only", report["argv"])
        self.assertIn("--output-schema", report["argv"])
        self.assertIn("--output-last-message", report["argv"])
        self.assertEqual(report["argv"][-1], "-")

    async def test_cli_candidates_can_generate_concurrently(self):
        barrier = r'''#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

name = Path(sys.argv[0]).name
Path(f"{name}.started").write_text("ready")
other = "codex.started" if name == "claude" else "claude.started"
deadline = time.monotonic() + 0.75
while not Path(other).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
if not Path(other).exists():
    raise SystemExit(91)
if name == "claude":
    print(json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "structured_output": {"response": "claude concurrent"},
        "modelUsage": {"default": {"inputTokens": 1, "outputTokens": 1}},
    }))
else:
    args = sys.argv[1:]
    Path(args[args.index("--output-last-message") + 1]).write_text(
        json.dumps({"response": "codex concurrent"})
    )
    print(json.dumps({"type": "turn.completed", "usage": {}}))
'''
        _write_executable(self.bin_dir / "claude", barrier)
        _write_executable(self.bin_dir / "codex", barrier)

        claude_response, codex_response = await asyncio.gather(
            llm.llm_completion(
                model="claude-cli/default",
                messages=[{"role": "user", "content": "Review this."}],
            ),
            llm.llm_completion(
                model="codex-cli/default",
                messages=[{"role": "user", "content": "Review this."}],
            ),
        )

        self.assertEqual(
            claude_response["choices"][0]["message"]["content"],
            "claude concurrent",
        )
        self.assertEqual(
            codex_response["choices"][0]["message"]["content"],
            "codex concurrent",
        )

    async def test_cli_route_reports_nonzero_exit_without_leaking_oauth(self):
        _write_executable(
            self.bin_dir / "claude",
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "print(os.environ['CLAUDE_CODE_OAUTH_TOKEN'], file=sys.stderr)\n"
            "raise SystemExit(9)\n",
        )

        with self.assertRaisesRegex(RuntimeError, "claude exited 9") as raised:
            await llm.llm_completion(
                model="claude-cli/default",
                messages=[{"role": "user", "content": "hello"}],
            )
        self.assertNotIn("oauth-secret", str(raised.exception))

    async def test_non_cli_model_keeps_litellm_path(self):
        payload = {"choices": [{"message": {"content": "provider"}}]}

        async def completion(**kwargs):
            self.assertEqual(kwargs["model"], "gemini/gemini-3.5-flash")
            return _LiteLLMResponse(payload)

        with mock.patch.object(llm.litellm, "acompletion", completion):
            response = await llm.llm_completion(
                model="gemini/gemini-3.5-flash",
                messages=[{"role": "user", "content": "hello"}],
            )
        self.assertEqual(response, payload)

    async def test_cli_routes_reject_api_keys_tools_logprobs_and_multiple_n(self):
        invalid = (
            {"api_key": "forbidden"},
            {"tools": [{"type": "function"}]},
            {"logprobs": True},
            {"top_logprobs": 5},
            {"n": 2},
            {"temperature": 0.5},
            {"top_p": 0.9},
            {"stop": ["done"]},
            {"seed": 1},
            {"presence_penalty": 0.1},
            {"frequency_penalty": 0.1},
            {"logit_bias": {"1": 1}},
            {"stream_options": {"include_usage": True}},
            {"reasoning_effort": "high"},
            {"thinking_budget": 100},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, NotImplementedError)):
                    await llm.llm_completion(
                        model="claude-cli/default",
                        messages=[{"role": "user", "content": "hello"}],
                        **kwargs,
                    )

    async def test_cli_streaming_is_explicitly_unsupported(self):
        with self.assertRaises(NotImplementedError):
            await llm.llm_stream_completion(
                model="codex-cli/default",
                messages=[{"role": "user", "content": "hello"}],
            )

    async def test_cli_route_fails_on_empty_structured_response(self):
        empty_script = CLAUDE_SCRIPT.replace(
            '"structured_output": {"response": "claude candidate"}',
            '"structured_output": {"response": ""}',
        )
        _write_executable(self.bin_dir / "claude", empty_script)
        with self.assertRaises(RuntimeError):
            await llm.llm_completion(
                model="claude-cli/default",
                messages=[{"role": "user", "content": "hello"}],
            )

    async def test_cli_route_times_out_fail_closed(self):
        _write_executable(
            self.bin_dir / "claude",
            "#!/usr/bin/env python3\nimport time\ntime.sleep(5)\n",
        )
        with mock.patch.object(llm, "_CLI_TIMEOUT_SECONDS", 0.05):
            with self.assertRaises(TimeoutError):
                await llm.llm_completion(
                    model="claude-cli/default",
                    messages=[{"role": "user", "content": "hello"}],
                )

    async def test_cancelling_completion_terminates_cli_process_group(self):
        _write_executable(
            self.bin_dir / "claude",
            "#!/usr/bin/env python3\n"
            "import os, time\n"
            "from pathlib import Path\n"
            "Path('claude.pid').write_text(str(os.getpid()))\n"
            "time.sleep(10)\n",
        )
        task = asyncio.create_task(
            llm.llm_completion(
                model="claude-cli/default",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
        pid_path = self.root / "claude.pid"
        for _ in range(100):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(pid_path.exists())
        pid = int(pid_path.read_text())

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            os.killpg(pid, signal.SIGKILL)
            self.fail("cancelled completion left its CLI process running")


if __name__ == "__main__":
    unittest.main()
