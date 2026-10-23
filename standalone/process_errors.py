"""Process failures shared by the standalone OS implementations."""


class Error(Exception):
    pass


class NoSuchProcess(Error):
    def __init__(self, pid, msg="process no longer exists or its PID was reused"):
        self.pid = pid
        super().__init__(f"{msg} (pid={pid})")


class ZombieProcess(NoSuchProcess):
    pass


class AccessDenied(Error):
    def __init__(self, pid=None, msg="process information or operation is unavailable"):
        self.pid = pid
        super().__init__(f"{msg} (pid={pid})")


class TimeoutExpired(Error):
    def __init__(self, seconds, pid=None):
        self.seconds, self.pid = seconds, pid
        super().__init__(f"process wait exceeded {seconds} seconds (pid={pid})")
