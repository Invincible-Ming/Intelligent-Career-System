"""Generate local Redis credentials without printing them; reruns preserve them."""
from pathlib import Path
import re
import secrets

root = Path(__file__).resolve().parents[2]
local = root / "deploy" / ".local"
local.mkdir(mode=0o700, exist_ok=True)
local.chmod(0o700)
config = local / "redis.conf"
if config.exists():
    password = next(
        line.split(" ", 1)[1] for line in config.read_text().splitlines() if line.startswith("requirepass "))
else:
    password = secrets.token_hex(32)
    config.write_text('bind 0.0.0.0\nprotected-mode yes\nport 6379\n'
                      f'requirepass {password}\nsave ""\nappendonly no\n'
                      'dir /tmp\nmaxmemory 128mb\nmaxmemory-policy allkeys-lru\n')
    config.chmod(0o644)  # Parent directory is 0700; container mounts only this file.
env = root / "backend" / ".env"
contents = env.read_text() if env.exists() else ""
line = f"REDIS_URL=redis://:{password}@127.0.0.1:6379/0"
contents = re.sub(r"^REDIS_URL=.*$", lambda _: line, contents, flags=re.MULTILINE) if re.search(r"^REDIS_URL=",
                                                                                                contents,
                                                                                                re.MULTILINE) else contents.rstrip() + "\n" + line + "\n"
env.write_text(contents)
env.chmod(0o600)
print("Local Redis configuration ready; credentials were not printed.")
