"""Google Chat's consumed Pub/Sub subscriber surface over the real REST API.

The original google-auth credential objects still refresh/sign requests. This
qualified boundary replaces only the native gRPC subscriber transport; it does
not register or impersonate ``google.cloud.pubsub_v1``. Google Chat uses ordinary
at-least-once delivery, callback ack/nack, flow control and cancellable shutdown.

Wire reference: https://cloud.google.com/pubsub/docs/reference/rest/v1/projects.subscriptions
API contract reviewed against google-cloud-pubsub 2.39.2, uv.lock wheel SHA256
b1295c6641032aff1dabbddf495b10642a5b547f6783ee4c6bb3365682e71354.
Publisher/admin APIs, ordered delivery and exactly-once acknowledgment futures
are outside the original Chat adapter's consumed surface.
"""

from __future__ import annotations

import base64
from concurrent.futures import Future
from dataclasses import dataclass
import logging
import math
import queue
import re
import threading
import time
from types import SimpleNamespace
from urllib.parse import quote, urlencode, urlsplit

import httpx
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
from google.auth.transport.requests import Request


API_URL = "https://pubsub.googleapis.com/v1"
# A short, bounded unary long poll makes cancellation independent of a native
# gRPC channel. A read deadline without a message is an empty poll, not a dead
# stream. Pending messages get a separate, regularly renewed server lease.
PULL_TIMEOUT = 5.0
REQUEST_TIMEOUT = 5.0
_RESOURCE = re.compile(r"projects/[^/?#\s]+/subscriptions/[^/?#\s]+\Z")
_TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowControl:
    max_bytes: int = 100 * 1024 * 1024
    max_messages: int = 1000
    max_lease_duration: float = 3600
    min_duration_per_lease_extension: float = 0
    max_duration_per_lease_extension: float = 0

    def __post_init__(self):
        for name in ("max_bytes", "max_messages"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.max_lease_duration) or self.max_lease_duration <= 0:
            raise ValueError("max_lease_duration must be positive and finite")
        for name in ("min_duration_per_lease_extension", "max_duration_per_lease_extension"):
            value = getattr(self, name)
            if value and (not math.isfinite(value) or not 10 <= value <= 600):
                raise ValueError(f"{name} must be zero or between 10 and 600 seconds")
        if (self.min_duration_per_lease_extension and self.max_duration_per_lease_extension
                and self.min_duration_per_lease_extension > self.max_duration_per_lease_extension):
            raise ValueError("Minimum lease extension exceeds maximum")


types = SimpleNamespace(FlowControl=FlowControl)


def _subscription(value):
    if not isinstance(value, str) or not _RESOURCE.fullmatch(value):
        raise ValueError("Expected projects/<project>/subscriptions/<subscription>")
    return quote(value, safe="/")


class _AuthRequest(Request):
    def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
        # Keep upstream credential refresh and regional-boundary lookup intact,
        # while bounding each socket request during subscriber shutdown.
        return super().__call__(url, method=method, body=body, headers=headers,
                                timeout=min(timeout or REQUEST_TIMEOUT, REQUEST_TIMEOUT), **kwargs)


class _Message:
    def __init__(self, stream, record):
        try:
            raw = record["message"]
            self.ack_id = record["ackId"]
            self.data = base64.b64decode(raw.get("data", ""), validate=True)
            self.attributes = dict(raw.get("attributes", {}))
            if not isinstance(self.ack_id, str) or not self.ack_id:
                raise ValueError("Missing acknowledgment ID")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.attributes.items()):
                raise ValueError("Invalid message attributes")
        except (KeyError, TypeError, ValueError) as exc:
            raise exceptions.DataLoss("Malformed Pub/Sub pull response") from exc
        self.message_id = raw.get("messageId", "")
        self.publish_time = raw.get("publishTime")
        self.ordering_key = raw.get("orderingKey", "")
        self.delivery_attempt = record.get("deliveryAttempt", 0) or None
        self._stream, self._lock, self._action = stream, threading.Lock(), None
        self._received = time.monotonic()
        self._dispatched = self._released = False
        self._size = len(self.data) + sum(len(k.encode()) + len(v.encode()) for k, v in self.attributes.items())

    def _finish(self, action):
        with self._lock:
            if self._action is not None:
                return
            self._action = action
        self._stream.commands.put((action, self.ack_id))

    def ack(self):
        self._finish("ack")

    def nack(self):
        self._finish("nack")

    def modify_ack_deadline(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, int) or not 0 <= seconds <= 600:
            raise ValueError("Ack deadline must be an integer from 0 to 600")
        if seconds == 0:
            self.nack()
        else:
            self._stream.commands.put((seconds, self.ack_id))


class _StreamingPullFuture(Future):
    def __init__(self, stream):
        super().__init__()
        self._stream = stream
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True
        self._stream.stop.set()
        self._stream.commands.put(("wake", ""))
        return True

    def cancelled(self):
        return self._cancel_requested


class _PullStream:
    def __init__(self, client, subscription, callback, flow):
        self.client, self.subscription, self.callback, self.flow = client, subscription, callback, flow
        self.stop = threading.Event()
        self.commands, self.callbacks = queue.Queue(), queue.Queue()
        self.pending, self.pending_bytes, self.active_bytes = {}, 0, 0
        self.future = _StreamingPullFuture(self)
        minimum = flow.min_duration_per_lease_extension or 10
        maximum = flow.max_duration_per_lease_extension or 600
        self.lease_seconds = int(min(maximum, max(minimum, 60)))
        self.lease_at = time.monotonic() + self.lease_seconds / 2
        self.workers = [threading.Thread(target=self._callback_worker, name="hermes-pubsub-callback", daemon=True)
                        for _ in range(min(10, flow.max_messages))]
        self.thread = threading.Thread(target=self._run, name="hermes-pubsub-pull", daemon=True)

    def start(self):
        for worker in self.workers:
            worker.start()
        self.thread.start()

    def _callback_worker(self):
        while True:
            message = self.callbacks.get()
            if message is None:
                return
            if self.stop.is_set() or message._released:
                message.nack()
                continue
            try:
                self.callback(message)
            except Exception:
                _LOG.exception("Pub/Sub message callback failed; releasing message for redelivery")
                message.nack()

    def _modify(self, ack_ids, seconds, **options):
        if ack_ids:
            self.client._request("POST", self.subscription + ":modifyAckDeadline",
                                 {"ackIds": ack_ids, "ackDeadlineSeconds": seconds}, **options)

    def _release(self, ack_ids):
        for ack_id in ack_ids:
            if (message := self.pending.pop(ack_id, None)) is not None:
                self.pending_bytes -= message._size
                if message._dispatched:
                    self.active_bytes -= message._size
                message._released = True
        self._dispatch_ready()

    def _flush_commands(self, **options):
        actions = {}
        while True:
            try:
                action, ack_id = self.commands.get_nowait()
            except queue.Empty:
                break
            if ack_id in self.pending:
                actions.setdefault(action, []).append(ack_id)
        for action, ack_ids in actions.items():
            if action == "ack":
                self.client._request("POST", self.subscription + ":acknowledge", {"ackIds": ack_ids}, **options)
            else:
                self._modify(ack_ids, 0 if action == "nack" else action, **options)
            if action in ("ack", "nack"):
                self._release(ack_ids)

    def _renew_leases(self):
        now = time.monotonic()
        expired = [key for key, message in self.pending.items()
                   if now - message._received >= self.flow.max_lease_duration]
        self._modify(expired, 0, stop=self.stop)
        self._release(expired)
        if now >= self.lease_at:
            self._modify(list(self.pending), self.lease_seconds, stop=self.stop)
            self.lease_at = now + self.lease_seconds / 2

    def _accept(self, records):
        messages = [_Message(self, record) for record in records]
        # Acquire the lease before starting callbacks, including callbacks that
        # retain messages and acknowledge them from another thread later.
        self._modify([message.ack_id for message in messages], self.lease_seconds, stop=self.stop)
        for message in messages:
            if message.ack_id in self.pending:
                continue
            self.pending[message.ack_id] = message
            self.pending_bytes += message._size
        self._dispatch_ready()

    def _dispatch_ready(self):
        if self.stop.is_set():
            return
        for message in self.pending.values():
            if message._dispatched:
                continue
            # A single oversized delivery must still make progress. Other
            # messages wait under their lease until the byte budget is free.
            if self.active_bytes and self.active_bytes + message._size > self.flow.max_bytes:
                break
            message._dispatched = True
            self.active_bytes += message._size
            self.callbacks.put(message)

    def _run(self):
        error = None
        try:
            while not self.stop.is_set():
                self._flush_commands(stop=self.stop)
                self._renew_leases()
                capacity = self.flow.max_messages - len(self.pending)
                if capacity <= 0 or self.pending_bytes >= self.flow.max_bytes:
                    self.stop.wait(0.02)
                    continue
                response = self.client._request("POST", self.subscription + ":pull",
                                                {"maxMessages": min(capacity, 1000)}, stop=self.stop, poll=True)
                records = response.get("receivedMessages", [])
                if not isinstance(records, list) or len(records) > min(capacity, 1000):
                    raise exceptions.DataLoss("Pub/Sub returned an invalid message batch")
                if records:
                    self._accept(records)
                else:
                    self.stop.wait(0.1)
        except Exception as exc:
            if not self.stop.is_set():
                error = exc
        finally:
            self.stop.set()
            # All unacknowledged deliveries become available immediately on
            # shutdown. If the transport is down their existing lease expires.
            try:
                self._flush_commands(attempts=1, timeout=2.0)
                self._modify(list(self.pending), 0, attempts=1, timeout=2.0)
            except Exception:
                _LOG.debug("Pub/Sub shutdown lease release failed", exc_info=True)
            for _ in self.workers:
                self.callbacks.put(None)
            for worker in self.workers:
                worker.join(timeout=0.1)
            with self.client._lock:
                self.client._streams.discard(self)
            if error is None:
                self.future.set_result(True)
            else:
                self.future.set_exception(error)


class SubscriberClient:
    """The explicit REST subscriber used by the bundled Google Chat adapter."""

    def __init__(self, *, credentials):
        endpoint = urlsplit(API_URL)
        if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query:
            raise ValueError("Pub/Sub endpoint must be an absolute HTTPS URL")
        from gateway.platforms.base import gateway_trust_env
        self._http = httpx.Client(follow_redirects=False, trust_env=gateway_trust_env())
        self._credentials = credentials
        self._auth_request = _AuthRequest()
        self._credential_lock, self._lock = threading.Lock(), threading.Lock()
        self._streams, self._closed = set(), False
        self._endpoint = API_URL.rstrip("/") + "/"

    def _request(self, method, resource, body=None, *, stop=None, poll=False, attempts=3, timeout=None):
        url = self._endpoint + resource
        refresh = did_refresh = False
        for attempt in range(attempts):
            if stop is not None and stop.is_set():
                raise exceptions.Cancelled("Pub/Sub subscription stopped")
            headers = {"Accept": "application/json", "x-goog-api-client": "hermes-standalone/pubsub-rest",
                       "x-goog-request-params": urlencode({"subscription": resource.split(":", 1)[0]})}
            try:
                with self._credential_lock:
                    if refresh:
                        self._credentials.refresh(self._auth_request)
                        refresh, did_refresh = False, True
                    self._credentials.before_request(self._auth_request, method, url, headers)
            except auth_exceptions.GoogleAuthError as exc:
                raise exceptions.Unauthenticated("Pub/Sub credential refresh failed") from exc
            try:
                response = self._http.request(method, url, headers=headers, json=body,
                                              timeout=timeout or (PULL_TIMEOUT if poll else REQUEST_TIMEOUT))
            except httpx.ReadTimeout as exc:
                if poll:
                    return {}
                failure = exceptions.DeadlineExceeded("Pub/Sub REST request timed out")
                failure.__cause__ = exc
            except httpx.TransportError as exc:
                failure = exceptions.ServiceUnavailable("Pub/Sub REST transport failed")
                failure.__cause__ = exc
            else:
                try:
                    if response.is_success:
                        result = response.json() if response.content else {}
                        if not isinstance(result, dict):
                            raise exceptions.DataLoss("Pub/Sub REST response must be an object")
                        return result
                    try:
                        message = response.json().get("error", {}).get("message", "Pub/Sub REST request failed")
                    except (ValueError, AttributeError):
                        message = "Pub/Sub REST request failed"
                    mapped = {401: exceptions.Unauthenticated, 403: exceptions.PermissionDenied, 404: exceptions.NotFound}
                    failure = (mapped[response.status_code](str(message), response=response)
                               if response.status_code in mapped else
                               exceptions.from_http_status(response.status_code, str(message), response=response))
                    if response.status_code == 401 and not did_refresh and attempt + 1 < attempts:
                        refresh = True
                        continue
                    if response.status_code not in _TRANSIENT:
                        raise failure
                finally:
                    response.close()
            if attempt + 1 == attempts:
                raise failure
            delay = min(2.0, 0.25 * 2 ** attempt)
            if stop is None:
                time.sleep(delay)
            elif stop.wait(delay):
                raise exceptions.Cancelled("Pub/Sub subscription stopped")
        raise AssertionError("Unreachable Pub/Sub retry state")

    def get_subscription(self, request):
        with self._lock:
            if self._closed:
                raise RuntimeError("Pub/Sub subscriber is closed")
        return self._request("GET", _subscription(request["subscription"]))

    def subscribe(self, subscription, *, callback, flow_control=None):
        if not callable(callback):
            raise TypeError("A callable message callback is required")
        flow = flow_control or FlowControl()
        if not isinstance(flow, FlowControl):
            raise TypeError("Expected standalone.google_pubsub.FlowControl")
        stream = _PullStream(self, _subscription(subscription), callback, flow)
        with self._lock:
            if self._closed:
                raise RuntimeError("Pub/Sub subscriber is closed")
            self._streams.add(stream)
            stream.start()
        return stream.future

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            streams = tuple(self._streams)
        for stream in streams:
            stream.future.cancel()
        try:
            for stream in streams:
                try:
                    stream.future.result(PULL_TIMEOUT + REQUEST_TIMEOUT * 3 + 3)
                except Exception:
                    _LOG.debug("Pub/Sub stream closed with an error", exc_info=True)
        finally:
            self._http.close()
            self._auth_request.session.close()

