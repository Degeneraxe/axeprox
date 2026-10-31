"""Configuration management module for ArbSense."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    yaml = None


def _load_yaml_file(path: str | Path) -> Dict[str, Any]:
    """Loads a YAML file using PyYAML if present, or basic key-value extraction."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()

    if yaml is not None:
        return yaml.safe_load(content) or {}

    # Basic fallback loader if PyYAML is not yet installed in the environment
    import json
    # Try parsing as JSON first (JSON is valid YAML)
    try:
        return json.loads(content)
    except Exception:
        pass

    # Simple nested dictionary parser for standard YAML structures
    result: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, result)]
    for raw_line in content.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current_dict = stack[-1][1]

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip().strip("'\"")
            val = val.strip()
            if not val:
                new_dict: Dict[str, Any] = {}
                current_dict[key] = new_dict
                stack.append((indent, new_dict))
            else:
                # Type conversions
                val_clean = val.strip("'\"")
                if val_clean.lower() == "true":
                    current_dict[key] = True
                elif val_clean.lower() == "false":
                    current_dict[key] = False
                elif val_clean.replace(".", "", 1).isdigit() or (val_clean.startswith("-") and val_clean[1:].replace(".", "", 1).isdigit()):
                    current_dict[key] = float(val_clean) if "." in val_clean else int(val_clean)
                else:
                    current_dict[key] = val_clean
        elif stripped.startswith("- "):
            item = stripped[2:].strip().strip("'\"")
            # If current parent is not a list, look at the last key
            pass
    return result


@dataclass
class TimeSyncConfig:
    ntp_server: str = "pool.ntp.org"
    ntp_timeout_sec: float = 3.0
    fallback_rest_url: str = "https://api.binance.com/api/v3/time"
    max_quote_age_ms: float = 1500.0
    heartbeat_interval_sec: float = 15.0
    initial_reconnect_delay_sec: float = 1.0
    max_reconnect_delay_sec: float = 30.0
    reconnect_multiplier: float = 2.0
    reconnect_jitter: bool = True


@dataclass
class MarketsConfig:
    primary_quote: str = "USDT"
    supported_pairs: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    basis_tracking_enabled: bool = True
    usdt_usd_peg_threshold_bps: float = 10.0


@dataclass
class DetectionConfig:
    gross_spread_threshold_bps: float = 1.0
    min_order_book_depth_levels: int = 10
    depth_snapshot_throttle_ms: int = 100


@dataclass
class AppConfig:
    raw: Dict[str, Any]
    time_sync: TimeSyncConfig
    markets: MarketsConfig
    detection: DetectionConfig

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> AppConfig:
        if config_path is None:
            # Check environment or default location
            env_path = os.getenv("CONFIG_PATH")
            if env_path and Path(env_path).exists():
                config_path = env_path
            else:
                # Search default locations
                cwd_config = Path("config.yaml")
                if cwd_config.exists():
                    config_path = cwd_config
                else:
                    base_dir = Path(__file__).resolve().parent.parent
                    config_path = base_dir / "config.yaml"

        raw = _load_yaml_file(config_path)

        # Parse TimeSync
        ts_raw = raw.get("time_sync", {})
        reconn = ts_raw.get("reconnect_backoff", {})
        time_sync = TimeSyncConfig(
            ntp_server=ts_raw.get("ntp_server", "pool.ntp.org"),
            ntp_timeout_sec=float(ts_raw.get("ntp_timeout_sec", 3.0)),
            fallback_rest_url=ts_raw.get("fallback_rest_url", "https://api.binance.com/api/v3/time"),
            max_quote_age_ms=float(ts_raw.get("max_quote_age_ms", 1500.0)),
            heartbeat_interval_sec=float(ts_raw.get("heartbeat_interval_sec", 15.0)),
            initial_reconnect_delay_sec=float(reconn.get("initial_delay_sec", 1.0)),
            max_reconnect_delay_sec=float(reconn.get("max_delay_sec", 30.0)),
            reconnect_multiplier=float(reconn.get("multiplier", 2.0)),
            reconnect_jitter=bool(reconn.get("jitter", True)),
        )

        # Parse Markets
        mk_raw = raw.get("markets", {})
        basis = mk_raw.get("basis_tracking", {})
        pairs = mk_raw.get("supported_pairs", ["BTC/USDT", "ETH/USDT"])
        markets = MarketsConfig(
            primary_quote=mk_raw.get("primary_quote", "USDT"),
            supported_pairs=pairs if isinstance(pairs, list) else ["BTC/USDT", "ETH/USDT"],
            basis_tracking_enabled=bool(basis.get("enabled", True)),
            usdt_usd_peg_threshold_bps=float(basis.get("usdt_usd_peg_threshold_bps", 10.0)),
        )

        # Parse Detection
        dt_raw = raw.get("detection", {})
        detection = DetectionConfig(
            gross_spread_threshold_bps=float(dt_raw.get("gross_spread_threshold_bps", 1.0)),
            min_order_book_depth_levels=int(dt_raw.get("min_order_book_depth_levels", 10)),
            depth_snapshot_throttle_ms=int(dt_raw.get("depth_snapshot_throttle_ms", 100)),
        )

        return cls(
            raw=raw,
            time_sync=time_sync,
            markets=markets,
            detection=detection,
        )

    def get_fee_bps(self, exchange: str, role: str = "taker", scenario: Optional[str] = None) -> float:
        """Returns fee in basis points for an exchange, role ('maker' or 'taker'), and scenario."""
        scenarios = self.raw.get("fee_scenarios", {})
        active = scenario or scenarios.get("active_scenario", "retail")
        scenario_data = scenarios.get("scenarios", {}).get(active, {})
        exch_data = scenario_data.get(exchange.lower(), {})
        key = f"{role.lower()}_bps"
        return float(exch_data.get(key, 10.0))
