"""Create/reset local workbench administrators without putting passwords in argv."""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from asr_server.config import load_settings
from asr_server.errors import AsrError
from asr_server.workbench.security import password_hash
from asr_server.workbench.store import WorkbenchStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or reset a local ASR workbench account")
    parser.add_argument("action", choices=["create", "reset-password"])
    parser.add_argument("username")
    parser.add_argument("--data-dir", type=Path, help="defaults to ASR_WORKBENCH_DATA_DIR")
    parser.add_argument("--regular-user", action="store_true", help="create a non-administrator")
    parser.add_argument("--password-stdin", action="store_true", help="read one password line from stdin")
    args = parser.parse_args()
    store = WorkbenchStore(args.data_dir or Path(load_settings().workbench_data_dir))
    password = sys.stdin.readline().rstrip("\r\n") if args.password_stdin else getpass.getpass("Password: ")
    if not args.password_stdin and password != getpass.getpass("Confirm password: "):
        parser.error("passwords do not match")
    try:
        if args.action == "create":
            user = store.create_user(args.username, password, not args.regular_user)
            print(f"Created {'administrator' if user['is_admin'] else 'user'}: {user['username']}")
        else:
            with store.connection() as db:
                row = db.execute("SELECT id FROM users WHERE username=?", (args.username,)).fetchone()
                if row is None:
                    parser.error("user not found")
                db.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash(password), row["id"]))
                db.execute("DELETE FROM sessions WHERE user_id=?", (row["id"],))
            print(f"Password reset and sessions revoked: {args.username}")
    except AsrError as exc:
        parser.error(exc.message)


if __name__ == "__main__":
    main()
