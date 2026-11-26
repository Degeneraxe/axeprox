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
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

logger = logging.getLogger(__name__)

# NTP constants: 70 years in seconds between 1900 and 1970
NTP_EPOCH_DELTA = 2208988800


class ClockSynchronizer:
    """Measures and tracks clock offset to ensure sub-millisecond timestamp alignment."""

    def __init__(self, ntp_server: str = "pool.ntp.org", timeout_sec: float = 3.0, fallback_url: str = "https://api.binance.com/api/v3/time"):
        self._epoch_ms = time.time_ns() / 1e6
        self._origin_ns = time.monotonic_ns()
        self._reference_adjustment_ms = 0.0
        self.error_bound_ms = None
        self.ntp_server = ntp_server
        self.timeout_sec = timeout_sec
        self.fallback_url = fallback_url
        self.offset_ms: float = 0.0
        self.rtt_ms: float = 0.0
        self.synced: bool = False
        self.method_used: str = "none"

    def event_time_ms(self) -> float:
        """Epoch-anchored monotonic receive time, independent of synchronization."""
        return self._epoch_ms + (time.monotonic_ns() - self._origin_ns) / 1e6

    def transit_ms(self, receipt_ms, exchange_ms):
        return receipt_ms + self._reference_adjustment_ms - exchange_ms

    def measurement(self):
        return dict(offset_ms=self.offset_ms, rtt_ms=self.rtt_ms if self.synced else None,
                    method=self.method_used, error_bound_ms=self.error_bound_ms)

    def _sample_server(self, server):
        samples = []
        for _ in range(5):
            try:
                offset, rtt, bound = self._query_ntp(server)
                samples.append((rtt, offset, bound, "NTP:" + server))
            except (OSError, ValueError):
                continue
        return min(samples) if samples else None

    def sync(self) -> float:
        servers = (self.ntp_server, "time.google.com", "time.cloudflare.com", "in.pool.ntp.org")
        with ThreadPoolExecutor(max_workers=4) as pool:
            best = min((s for s in pool.map(self._sample_server, servers) if s), default=None)
        if best is None:
            try:
                offset, rtt = self._query_rest()
                if not 0 <= rtt < 100:
                    raise ValueError(f"REST RTT {rtt:.2f} ms outside accepted range")
                best = (rtt, offset, rtt / 2 + 1, "REST_BINANCE")
            except (OSError, ValueError, KeyError) as error:
                logger.warning("Clock sample rejected (%s), retaining reference %s", error, self.measurement())
                return self.offset_ms

        self.rtt_ms, self.offset_ms, self.error_bound_ms, self.method_used = best
        self._reference_adjustment_ms = time.time_ns() / 1e6 - self.event_time_ms() + self.offset_ms
        self.synced = True
        logger.info("Clock sample %s", self.measurement())
        return self.offset_ms

    def _query_ntp(self, server):
        packet = bytearray(48)
        packet[0] = 0x23
        address = socket.gethostbyname(server)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(self.timeout_sec)
            client.connect((address, 123))
            start = time.time()
            mono = time.monotonic()
            stamp = start + NTP_EPOCH_DELTA
            struct.pack_into("!II", packet, 40, int(stamp), int((stamp % 1) * 2**32))
            client.send(packet)
            response = client.recv(1024)
            elapsed = time.monotonic() - mono
        if (len(response) < 48 or response[24:32] != packet[40:48] or response[0] & 7 != 4
                or response[0] >> 6 == 3 or not 1 <= response[1] <= 15):
            raise ValueError("Invalid NTP response")
        recv_s, recv_frac, sent_s, sent_frac = struct.unpack_from("!IIII", response, 32)
        recv = recv_s - NTP_EPOCH_DELTA + recv_frac / 2**32
        sent = sent_s - NTP_EPOCH_DELTA + sent_frac / 2**32
        rtt = max(0.0, (elapsed - (sent - recv)) * 1000)
        offset = ((recv - start) + (sent - (start + elapsed))) * 500
        root_delay, dispersion = struct.unpack_from("!iI", response, 4)
        bound = rtt / 2 + (max(0, root_delay) / 2 + dispersion) * 1000 / 65536
        return offset, rtt, bound

    def _query_rest(self) -> Tuple[float, float]:
        """Queries Binance public REST endpoint to get server epoch ms and calculate offset."""
        t_start = time.time() * 1000.0
        mono = time.monotonic()
        req = urllib.request.Request(
            self.fallback_url,
            headers={"User-Agent": "ArbSense-ClockSync/1.0"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        elapsed_ms = (time.monotonic() - mono) * 1000

        server_time_ms = float(payload["serverTime"])
        rtt = elapsed_ms
        # Midpoint estimation
        local_midpoint = t_start + (rtt / 2.0)
        offset = server_time_ms - local_midpoint
        return offset, rtt

    def get_corrected_time_ms(self) -> float:
        """Returns the current UTC epoch time in milliseconds corrected by measured clock offset."""
        return (time.time() * 1000.0) + self.offset_ms


# Global singleton instance for easy import
clock_sync = ClockSynchronizer()
