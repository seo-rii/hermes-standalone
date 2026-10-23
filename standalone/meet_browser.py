"""Google Meet's explicit synchronous browser boundary over Chromium CDP.

Chromium is an external engine. Its fresh private profile and authenticated CDP
commands use inherited pipes; no debugging listener or Node driver is started.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.parse import urlsplit

from standalone.meet_browser_pipe import BrowserChild, CDPError, MAX_MESSAGE, PipeCDP


def find_executable(selected=None):
    if selected is None:
        from hermes_cli.browser_runtime import chromium_executable
        selected = chromium_executable()
    if selected:
        path = Path(selected).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CDPError("Selected Chromium executable is missing or not executable")
        return str(path)
    from hermes_platform.resolver import locate_command
    for name in ("chromium", "chromium-browser"):
        resolution = locate_command(name)
        if resolution.found:
            return str(Path(resolution.command[0]).resolve())
    raise CDPError("Chromium is unavailable; select an existing Chromium with AGENT_BROWSER_EXECUTABLE_PATH")


def _origin(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Browser storage requires an HTTP(S) origin")
    port = parsed.port
    host = "[" + parsed.hostname + "]" if ":" in parsed.hostname else parsed.hostname
    if port is not None and port != (443 if parsed.scheme == "https" else 80):
        host += ":" + str(port)
    return parsed.scheme + "://" + host


class Chromium:
    def __init__(self):
        self._browsers = []

    @property
    def executable_path(self):
        return find_executable()

    def launch(self, *, channel="chromium", executable_path=None, headless=True, args=()):
        if channel != "chromium":
            raise ValueError("Meet's standalone browser boundary uses Chromium")
        if any(not isinstance(arg, str) or arg.startswith(("--remote-debugging-", "--user-data-dir")) for arg in args):
            raise ValueError("Caller arguments cannot replace Chromium's private profile or debugging pipe")
        browser = Browser(find_executable(executable_path), headless, list(args))
        self._browsers.append(browser)
        return browser

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        for browser in self._browsers:
            browser.close()


class Browser:
    def __init__(self, executable, headless, arguments):
        self.profile = tempfile.TemporaryDirectory(prefix="hermes-meet-chromium-")
        self.contexts, self.closed, self.child, self.pipe = [], False, None, None
        profile = Path(self.profile.name)
        profile.chmod(0o700)
        command = ["--remote-debugging-pipe", "--user-data-dir=" + str(profile), "--no-first-run",
                   "--no-default-browser-check", "--disable-background-networking", "--disable-component-update",
                   "--disable-sync", "--metrics-recording-only", "--password-store=basic", "--use-mock-keychain"]
        if headless:
            command.append("--headless=new")
        try:
            self.child = BrowserChild(executable, [*command, *arguments, "about:blank"], profile)
            self.pipe = PipeCDP(self.child)
            version = self.pipe.request("Browser.getVersion")
            if not all(isinstance(version.get(key), str) and version[key] for key in ("product", "protocolVersion")):
                raise CDPError("Chromium returned an invalid DevTools version handshake")
            self.pipe.request("Target.setDiscoverTargets", {"discover": True})
        except BaseException:
            self.close()
            raise

    def new_context(self, *, viewport=None, user_agent=None, permissions=(), storage_state=None):
        if self.closed:
            raise CDPError("Chromium browser is closed")
        context = BrowserContext(self, viewport, user_agent, permissions, storage_state)
        self.contexts.append(context)
        return context

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for context in self.contexts:
                with contextlib.suppress(CDPError, OSError, TimeoutError):
                    context.close()
            if self.pipe is not None:
                with contextlib.suppress(CDPError, OSError, TimeoutError):
                    self.pipe.request("Browser.close", timeout=2)
                self.pipe.close()
            if self.child is not None:
                self.child.close()
        finally:
            self.profile.cleanup()


class BrowserContext:
    def __init__(self, browser, viewport, user_agent, permissions, state):
        self.browser, self.pipe, self.pages = browser, browser.pipe, []
        self.closed, self.origins = False, set()
        self.viewport = viewport or {"width": 1280, "height": 720}
        if (not isinstance(self.viewport, dict) or any(type(self.viewport.get(key)) is not int or
                not 1 <= self.viewport[key] <= 16384 for key in ("width", "height"))):
            raise ValueError("Chromium viewport dimensions must be integers in 1–16384")
        if user_agent is not None and (not isinstance(user_agent, str) or len(user_agent) > 8192):
            raise ValueError("Chromium user agent must be a bounded string")
        self.user_agent = user_agent
        permission_names = {"microphone": "audioCapture", "camera": "videoCapture"}
        if any(permission not in permission_names for permission in permissions):
            raise ValueError("Meet browser permissions support microphone and camera only")
        self.identifier = self.pipe.request("Target.createBrowserContext", {"disposeOnDetach": True})["browserContextId"]
        try:
            if permissions:
                self.pipe.request("Browser.grantPermissions", {"browserContextId": self.identifier,
                                  "permissions": [permission_names[item] for item in permissions]})
            if state is not None:
                self._restore(state)
        except BaseException:
            self.close()
            raise

    def _restore(self, state):
        if isinstance(state, (str, os.PathLike)):
            path = Path(state)
            if path.stat().st_size > MAX_MESSAGE:
                raise ValueError("Browser storage state exceeds its byte limit")
            state = json.loads(path.read_bytes())
        if not isinstance(state, dict) or len(json.dumps(state, ensure_ascii=False).encode()) > MAX_MESSAGE:
            raise ValueError("Browser storage state must be a bounded JSON object")
        cookies, origins = state.get("cookies", []), state.get("origins", [])
        if not isinstance(cookies, list) or len(cookies) > 4096 or not isinstance(origins, list) or len(origins) > 128:
            raise ValueError("Browser storage state exceeds its cookie or origin bound")
        selected = []
        for cookie in cookies:
            if not isinstance(cookie, dict):
                raise ValueError("Browser storage cookie must be an object")
            item = {key: cookie[key] for key in ("name", "value", "domain", "path", "secure", "httpOnly", "sameSite") if key in cookie}
            if not isinstance(item.get("name"), str) or not isinstance(item.get("value"), str):
                raise ValueError("Browser storage cookies require string names and values")
            if cookie.get("expires", -1) > 0:
                item["expires"] = cookie["expires"]
            if cookie.get("partitionKey"):
                item["partitionKey"] = {"topLevelSite": cookie["partitionKey"], "hasCrossSiteAncestor": False}
            selected.append(item)
        if selected:
            self.pipe.request("Storage.setCookies", {"browserContextId": self.identifier, "cookies": selected})
        for entry in origins:
            if not isinstance(entry, dict) or entry.get("indexedDB"):
                raise ValueError("Meet storage state supports cookies and localStorage, not IndexedDB snapshots")
            origin, items = _origin(entry["origin"]), entry.get("localStorage", [])
            if not isinstance(items, list) or len(items) > 8192 or any(not isinstance(item, dict) or
                    not isinstance(item.get("name"), str) or not isinstance(item.get("value"), str) for item in items):
                raise ValueError("Browser localStorage state must contain bounded string entries")
            self.origins.add(origin)
            self._storage_at(origin, items)

    def new_page(self):
        if self.closed:
            raise CDPError("Chromium browser context is closed")
        page = Page(self)
        self.pages.append(page)
        return page

    def _storage_at(self, origin, items=None):
        # Fulfill a blank document before any network activity. This accesses
        # the context's real origin storage without making an extra login request.
        page = self.new_page()
        try:
            page.request("Network.setBypassServiceWorker", {"bypass": True})
            page.request("Fetch.enable", {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]})
            navigation = self.pipe.send("Page.navigate", {"url": origin + "/"}, session=page.session)
            paused = self.pipe.event(lambda event: event.get("sessionId") == page.session and
                                     event.get("method") == "Fetch.requestPaused")
            page.request("Fetch.fulfillRequest", {"requestId": paused["params"]["requestId"], "responseCode": 200,
                         "responseHeaders": [{"name": "Content-Type", "value": "text/html; charset=utf-8"}],
                         "body": base64.b64encode(b"<!doctype html><title>Hermes storage</title>").decode()})
            page._loaded(self.pipe.result(navigation), 30)
            if items is not None:
                page.evaluate("(() => { for (const item of " + json.dumps(items, ensure_ascii=False) +
                              ") localStorage.setItem(item.name,item.value); })()")
            return page.evaluate("Object.keys(localStorage).map(name => ({name,value:localStorage.getItem(name)}))")
        finally:
            page.close()

    def storage_state(self, *, path=None):
        raw = self.pipe.request("Storage.getCookies", {"browserContextId": self.identifier}).get("cookies", [])
        if not isinstance(raw, list) or len(raw) > 4096:
            raise CDPError("Chromium returned an invalid cookie state")
        cookies = []
        for cookie in raw:
            item = {key: cookie[key] for key in ("name", "value", "domain", "path", "expires", "httpOnly", "secure", "sameSite") if key in cookie}
            if cookie.get("partitionKey"):
                item["partitionKey"] = cookie["partitionKey"]["topLevelSite"]
            cookies.append(item)
        for page in tuple(self.pages):
            if not page.is_closed():
                origin = page.evaluate("location.origin")
                if isinstance(origin, str) and origin.startswith(("https://", "http://")):
                    self.origins.add(_origin(origin))
        result = {"cookies": cookies, "origins": []}
        for origin in sorted(self.origins):
            result["origins"].append({"origin": origin, "localStorage": self._storage_at(origin)})
        encoded = (json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_MESSAGE:
            raise CDPError("Chromium exported storage state exceeds its byte limit")
        if path is not None:
            destination = Path(path)
            fd, temporary = tempfile.mkstemp(prefix=".auth-", dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(encoded)
                os.replace(temporary, destination)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.pipe.request("Target.disposeBrowserContext", {"browserContextId": self.identifier}, timeout=3)
        finally:
            for page in tuple(self.pages):
                page._detach()


_FUNCTION = re.compile(r"^\s*(?:async\s+)?(?:\(\s*(?:[\w$]+(?:\s*,\s*[\w$]+)*)?\s*\)|[\w$]+)\s*=>")


class Page:
    def __init__(self, context):
        self.context, self.pipe, self.closed = context, context.pipe, False
        self.target, self.session = None, None
        try:
            self.target = self.pipe.request("Target.createTarget", {"url": "about:blank", "browserContextId": context.identifier})["targetId"]
            self.session = self.pipe.request("Target.attachToTarget", {"targetId": self.target, "flatten": True})["sessionId"]
            self.pipe.listeners.append(self._event)
            self.request("Page.enable")
            self.request("Page.setLifecycleEventsEnabled", {"enabled": True})
            self.request("Emulation.setDeviceMetricsOverride", {**context.viewport, "deviceScaleFactor": 1, "mobile": False})
            if context.user_agent is not None:
                self.request("Emulation.setUserAgentOverride", {"userAgent": context.user_agent})
        except BaseException:
            self.close()
            raise

    def request(self, method, params=None, *, timeout=30):
        if self.closed:
            raise CDPError("Chromium page is closed")
        return self.pipe.request(method, params, session=self.session, timeout=timeout)

    def _event(self, message):
        method, params = message.get("method"), message.get("params", {})
        if method in ("Target.targetDestroyed", "Target.targetCrashed") and params.get("targetId") == self.target:
            self.closed = True
        elif method == "Target.detachedFromTarget" and params.get("sessionId") == self.session:
            self.closed = True
        elif method == "Page.frameNavigated" and message.get("sessionId") == self.session:
            url = params.get("frame", {}).get("url", "")
            if url.startswith(("https://", "http://")):
                if len(self.context.origins) >= 128:
                    raise CDPError("Chromium visited-origin storage bound exceeded")
                self.context.origins.add(_origin(url))

    def _loaded(self, navigation, timeout):
        if navigation.get("errorText") or navigation.get("isDownload"):
            raise CDPError("Chromium navigation failed")
        loader = navigation.get("loaderId")
        if loader:
            self.pipe.event(lambda message: message.get("sessionId") == self.session and
                            message.get("method") == "Page.lifecycleEvent" and
                            message.get("params", {}).get("name") == "DOMContentLoaded" and
                            message["params"].get("loaderId") == loader, timeout=timeout)

    def goto(self, url, *, wait_until="domcontentloaded", timeout=30000):
        if wait_until != "domcontentloaded":
            raise ValueError("Meet navigation waits for DOMContentLoaded")
        if not isinstance(url, str) or len(url) > 65536 or not 0 < timeout <= 300000:
            raise ValueError("Chromium navigation requires a bounded URL and timeout")
        self.pipe.discard_events(lambda event: event.get("sessionId") == self.session)
        started = time.monotonic()
        navigation = self.request("Page.navigate", {"url": url}, timeout=timeout / 1000)
        self._loaded(navigation, max(.001, timeout / 1000 - (time.monotonic() - started)))

    def evaluate(self, javascript):
        if not isinstance(javascript, str):
            raise TypeError("Meet browser evaluation requires a JavaScript string")
        expression = "(" + javascript + ")()" if _FUNCTION.match(javascript) else javascript
        response = self.request("Runtime.evaluate", {"expression": expression, "awaitPromise": True,
                                "returnByValue": True, "userGesture": True})
        if response.get("exceptionDetails"):
            raise CDPError("Chromium JavaScript evaluation raised an exception")
        self.pipe.discard_events(lambda event: event.get("sessionId") == self.session and
                                 event.get("method") == "Page.lifecycleEvent")
        return response.get("result", {}).get("value")

    def locator(self, selector):
        return Locator(self, selector=selector)

    def get_by_role(self, role, *, name, exact=False):
        if role != "button" or not isinstance(name, str):
            raise ValueError("Meet role locators support named buttons")
        return Locator(self, name=name, exact=exact)

    def is_closed(self):
        if self.closed or self.context.closed or self.pipe.closed or self.pipe.child.poll() is not None:
            return True
        try:
            info = self.pipe.request("Target.getTargetInfo", {"targetId": self.target}, timeout=2)
            return not bool(info.get("targetInfo"))
        except (CDPError, OSError, TimeoutError):
            return True

    def _detach(self):
        self.closed = True
        with contextlib.suppress(ValueError):
            self.pipe.listeners.remove(self._event)
        with contextlib.suppress(ValueError):
            self.context.pages.remove(self)
        self.pipe.discard_events(lambda event: event.get("sessionId") == self.session)

    def close(self):
        try:
            if not self.closed and self.target is not None:
                self.pipe.request("Target.closeTarget", {"targetId": self.target}, timeout=3)
        finally:
            self._detach()


_QUERY = r'''
const spec=SPEC;
const roots=[document], found=[];
while(roots.length){
  const root=roots.shift();
  for(const element of root.querySelectorAll('*')) if(element.shadowRoot) roots.push(element.shadowRoot);
  for(const element of root.querySelectorAll(spec.selector||'button,[role="button"],input[type="button"],input[type="submit"],input[type="reset"]')){
    if(spec.name!==null){
      if(element.closest('[aria-hidden="true"],[inert]')) continue;
      const labelled=(element.getAttribute('aria-labelledby')||'').split(/\s+/).map(id=>root.getElementById?.(id)?.textContent||'').join(' ');
      const text=(labelled.trim()||element.getAttribute('aria-label')||element.innerText||element.value||element.getAttribute('title')||element.textContent||'').replace(/\s+/g,' ').trim();
      if(spec.exact ? text!==spec.name : !text.toLowerCase().includes(spec.name.toLowerCase())) continue;
    }
    found.push(element);
    if(found.length>8192) throw Error('Meet locator match bound exceeded');
  }
  if(roots.length>4096) throw Error('Meet shadow root bound exceeded');
}
const matches=spec.index===null?found:found.slice(spec.index,spec.index+1);
const element=matches[0];
function visible(el){const style=getComputedStyle(el),box=el.getBoundingClientRect();return style.visibility!=='hidden'&&style.visibility!=='collapse'&&box.width>0&&box.height>0;}
'''


class Locator:
    def __init__(self, page, *, selector=None, name=None, exact=False, index=None):
        if selector is not None and (not isinstance(selector, str) or len(selector) > 8192):
            raise ValueError("Meet CSS locator must be a bounded string")
        self.page, self.spec = page, {"selector": selector, "name": name, "exact": exact, "index": index}

    @property
    def first(self):
        return Locator(self.page, **{**self.spec, "index": 0})

    def _evaluate(self, action):
        return self.page.evaluate("(() => {" + _QUERY.replace("SPEC", json.dumps(self.spec)) + action + "})()")

    def count(self):
        return self._evaluate("return matches.length;")

    def is_visible(self):
        return bool(self._evaluate("return !!element&&visible(element);"))

    def _action(self, action, timeout):
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 300000:
            raise ValueError("Meet locator timeout must be bounded")
        deadline = time.monotonic() + timeout / 1000
        while True:
            result = self._evaluate(action)
            if result:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("Meet locator was absent, hidden, disabled or covered")
            threading.Event().wait(.05)

    def fill(self, value, *, timeout=30000):
        if not isinstance(value, str) or len(value) > 16384:
            raise ValueError("Meet input value must be a bounded string")
        self._action("if(!element||!visible(element)||element.disabled||element.readOnly||typeof element.select!=='function')return false;"
                     "element.focus();element.select();return true;", timeout)
        self.page.request("Input.insertText", {"text": value})

    def click(self, *, timeout=30000):
        point = self._action("if(!element||!visible(element)||element.disabled||element.getAttribute('aria-disabled')==='true')return null;"
                             "element.scrollIntoView({block:'center',inline:'center'});"
                             "const box=element.getBoundingClientRect(),x=(Math.max(0,box.left)+Math.min(innerWidth,box.right))/2,"
                             "y=(Math.max(0,box.top)+Math.min(innerHeight,box.bottom))/2;"
                             "const hit=element.getRootNode().elementFromPoint(x,y);"
                             "return hit&&(hit===element||element.contains(hit))?{x,y}:null;", timeout)
        self.page.request("Input.dispatchMouseEvent", {"type": "mouseMoved", **point})
        self.page.request("Input.dispatchMouseEvent", {"type": "mousePressed", **point, "button": "left", "clickCount": 1})
        self.page.request("Input.dispatchMouseEvent", {"type": "mouseReleased", **point, "button": "left", "clickCount": 1})
