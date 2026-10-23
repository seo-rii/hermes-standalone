"""Shared native-audio ABI declarations and cancellable deadline waits."""

import time


def bind(library, name, arguments, result):
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, result
    return function


def wait(stop, deadline, message):
    if stop.is_set():
        raise InterruptedError("Audio operation cancelled")
    if time.monotonic() >= deadline:
        raise TimeoutError(message)
    stop.wait(.005)
