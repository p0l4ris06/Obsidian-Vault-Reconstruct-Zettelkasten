from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vault_reconstruct.env import load_dotenv_no_override


class TestLoadDotenvNoOverride(unittest.TestCase):
    def test_does_not_override_existing_vars(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "# comment",
                        'FOO="from_file"',
                        "BAR=from_file",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            os.environ["FOO"] = "from_env"
            load_dotenv_no_override(repo_root=root)
            self.assertEqual(os.environ.get("FOO"), "from_env")
            self.assertEqual(os.environ.get("BAR"), "from_file")

    def test_missing_env_file_is_noop(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            before = dict(os.environ)
            load_dotenv_no_override(repo_root=root)
            self.assertEqual(os.environ, before)


if __name__ == "__main__":
    unittest.main()
