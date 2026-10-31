"""NTP & Exchange Clock Offset Synchronization Engine.

Measures local system clock skew against NTP servers (pool.ntp.org)
with an automatic fallback to exchange REST server time (Binance /api/v3/time)
if UDP port 123 is blocked by network firewalls.
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import time
import urllib.request
from typing import Tuple

logger = logging.getLogger(__name__)

# NTP constants: 70 years in seconds between 1900 and 1970
NTP_EPOCH_DELTA = 2208988800


class ClockSynchronizer:
    """Measures and tracks clock offset to ensure sub-millisecond timestamp alignment."""

    def __init__(self, ntp_server: str = "pool.ntp.org", timeout_sec: float = 3.0, fallback_url: str = "https://api.binance.com/api/v3/time"):
        self.ntp_server = ntp_server
        self.timeout_sec = timeout_sec
        self.fallback_url = fallback_url
        self.offset_ms: float = 0.0
        self.rtt_ms: float = 0.0
        self.synced: bool = False
        self.method_used: str = "none"

    def sync(self) -> float:
        """Attempts NTP sync, falls back to REST if blocked, returns clock offset in ms."""
        try:
            offset, rtt = self._query_ntp()
            self.offset_ms = offset
            self.rtt_ms = rtt
            self.synced = True
            self.method_used = "NTP"
            logger.info("NTP clock synchronization successful: offset=%.2f ms, rtt=%.2f ms (server=%s)", offset, rtt, self.ntp_server)
            return self.offset_ms
        except Exception as e:
            logger.warning("NTP query to %s failed (%s). Falling back to exchange REST time...", self.ntp_server, e)

        try:
            offset, rtt = self._query_rest()
            self.offset_ms = offset
            self.rtt_ms = rtt
            self.synced = True
            self.method_used = "REST_BINANCE"
            logger.info("REST clock synchronization successful: offset=%.2f ms, rtt=%.2f ms (url=%s)", offset, rtt, self.fallback_url)
            return self.offset_ms
        except Exception as e:
            logger.error("REST clock synchronization failed (%s). Defaulting to 0.0 ms offset.", e)
            self.offset_ms = 0.0
            self.synced = False
            self.method_used = "LOCAL_FALLBACK"
            return 0.0

    def _query_ntp(self) -> Tuple[float, float]:
        """Queries NTP pool via UDP socket using standard SNTP protocol (RFC 4330)."""
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(self.timeout_sec)
        # SNTP packet: 48 bytes, leap indicator 0, version 3, mode 3 (client)
        data = b"\x1b" + 47 * b"\0"
        t_originate = time.time()
        client.sendto(data, (self.ntp_server, 123))
        response, _ = client.recvfrom(1024)
        t_arrive = time.time()
        client.close()

        if len(response) >= 48:
            # Unpack transmit timestamp (bytes 40-48)
            integers, fractions = struct.unpack("!II", response[40:48])
            t_server = (integers - NTP_EPOCH_DELTA) + (fractions / 2**32)
            rtt = (t_arrive - t_originate) * 1000.0
            # Offset = t_server - (t_originate + rtt / 2)
            offset = (t_server - (t_originate + (t_arrive - t_originate) / 2)) * 1000.0
            return offset, rtt
        raise ValueError("Invalid NTP response length")

    def _query_rest(self) -> Tuple[float, float]:
        """Queries Binance public REST endpoint to get server epoch ms and calculate offset."""
        t_start = time.time() * 1000.0
        req = urllib.request.Request(
            self.fallback_url,
            headers={"User-Agent": "ArbSense-ClockSync/1.0"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        t_end = time.time() * 1000.0

        server_time_ms = float(payload.get("serverTime", 0))
        rtt = t_end - t_start
        # Midpoint estimation
        local_midpoint = t_start + (rtt / 2.0)
        offset = server_time_ms - local_midpoint
        return offset, rtt

    def get_corrected_time_ms(self) -> float:
        """Returns the current UTC epoch time in milliseconds corrected by measured clock offset."""
        return (time.time() * 1000.0) + self.offset_ms


# Global singleton instance for easy import
clock_sync = ClockSynchronizer()
