"""Supabase logging helpers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


def get_supabase_config() -> Optional[dict]:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_API_KEY", "").strip()
    table = os.environ.get("SUPABASE_TABLE", "sponge_attack_runs").strip()
    if not url or not key or not table:
        return None
    if url.endswith("/"):
        url = url[:-1]
    return {
        "url": url,
        "key": key,
        "table": table,
    }


def insert_payload(payload: dict) -> None:
    cfg = get_supabase_config()
    if not cfg:
        return
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "apikey": cfg["key"],
        "Authorization": f"Bearer {cfg['key']}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    endpoint = f"{cfg['url']}/rest/v1/{cfg['table']}"
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except urllib.error.HTTPError as exc:
        print(f"[supabase] insert failed: {exc.code} {exc.reason}")
    except Exception as exc:
        print(f"[supabase] insert failed: {exc}")
