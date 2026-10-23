"""Read battery facts through the operating system, without a native extension.

Linux uses the kernel power_supply ABI. Windows uses GetSystemPowerStatus;
macOS uses IOKit's public IOPowerSources API and owns each copied CF object.
No command is started and no third-party library is loaded.
"""

from collections import namedtuple
from pathlib import Path
import sys


Battery = namedtuple("Battery", "percent secsleft power_plugged")
POWER_TIME_UNKNOWN = -1
POWER_TIME_UNLIMITED = -2


def _text(path):
    try:
        return path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None


def _number(path):
    value = _text(path)
    return None if value is None else float(value)


def linux_battery(root=Path("/sys/class/power_supply")):
    """Read one installed system battery, retaining unknown AC/time states."""
    root = Path(root)
    try:
        supplies = sorted(root.iterdir())
    except FileNotFoundError:
        return None
    batteries = [item for item in supplies if _text(item / "type") == "Battery"
                 and _text(item / "present") != "0"]
    if not batteries:
        return None
    battery = batteries[0]
    percent = _number(battery / "capacity")
    current, full, rate = None, None, None
    for unit, rate_name in (("energy", "power_now"), ("charge", "current_now")):
        now, maximum = _number(battery / f"{unit}_now"), _number(battery / f"{unit}_full")
        if now is not None and maximum is not None and maximum > 0:
            current, full = now, maximum
            rate = _number(battery / rate_name)
            break
    if percent is None and current is not None:
        percent = 100 * current / full
    if percent is not None:
        percent = max(0.0, min(100.0, percent))
    status = _text(battery / "status")
    online = [_number(item / "online") for item in supplies
              if _text(item / "type") in {"Mains", "USB", "USB_C", "USB_PD", "Wireless"}]
    known = [value for value in online if value is not None]
    plugged = any(value > 0 for value in known) if known else None
    if status in {"Charging", "Full", "Not charging"}:
        plugged = True
    elif status == "Discharging":
        plugged = False
    seconds = POWER_TIME_UNLIMITED if plugged else POWER_TIME_UNKNOWN
    if plugged is False:
        remaining = _number(battery / "time_to_empty_now")
        if remaining is not None and remaining >= 0:
            seconds = int(remaining)
        elif current is not None and rate is not None and abs(rate) > 0:
            seconds = max(0, int(3600 * current / abs(rate)))
    return Battery(percent, seconds, plugged)


def windows_battery():
    import ctypes as C
    from ctypes import wintypes as W

    class PowerStatus(C.Structure):
        _fields_ = [(name, W.BYTE) for name in ("ac", "flags", "percent", "reserved")] + [
            ("seconds", W.DWORD), ("full_seconds", W.DWORD),
        ]

    function = C.WinDLL("kernel32", use_last_error=True).GetSystemPowerStatus
    function.argtypes, function.restype = [C.POINTER(PowerStatus)], W.BOOL
    result = PowerStatus()
    if not function(C.byref(result)):
        raise C.WinError(C.get_last_error())
    if result.flags != 255 and result.flags & 128:
        return None
    percent = None if result.percent == 255 else float(result.percent)
    plugged = None if result.ac == 255 else bool(result.ac)
    seconds = POWER_TIME_UNLIMITED if plugged else (
        POWER_TIME_UNKNOWN if result.seconds == 0xFFFFFFFF else int(result.seconds))
    return Battery(percent, seconds, plugged)


def darwin_battery():
    import ctypes as C

    core = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    kit = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")

    def function(library, name, result, *arguments):
        method = getattr(library, name)
        method.argtypes, method.restype = list(arguments), result
        return method

    pointer = C.c_void_p
    release = function(core, "CFRelease", None, pointer)
    make_key = function(core, "CFStringCreateWithCString", pointer, pointer, C.c_char_p, C.c_uint32)
    dictionary_get = function(core, "CFDictionaryGetValue", pointer, pointer, pointer)
    number_get = function(core, "CFNumberGetValue", C.c_bool, pointer, C.c_int, pointer)
    equal = function(core, "CFEqual", C.c_bool, pointer, pointer)
    get_type = function(core, "CFGetTypeID", C.c_ulong, pointer)
    number_type = function(core, "CFNumberGetTypeID", C.c_ulong)
    bool_type = function(core, "CFBooleanGetTypeID", C.c_ulong)
    bool_value = function(core, "CFBooleanGetValue", C.c_bool, pointer)
    array_count = function(core, "CFArrayGetCount", C.c_long, pointer)
    array_get = function(core, "CFArrayGetValueAtIndex", pointer, pointer, C.c_long)
    copy_info = function(kit, "IOPSCopyPowerSourcesInfo", pointer)
    copy_list = function(kit, "IOPSCopyPowerSourcesList", pointer, pointer)
    description = function(kit, "IOPSGetPowerSourceDescription", pointer, pointer, pointer)
    keys = {}
    info, sources = None, None
    try:
        for text in ("Type", "InternalBattery", "Is Present", "Current Capacity", "Max Capacity",
                     "Power Source State", "AC Power", "Battery Power", "Time to Empty"):
            value = make_key(None, text.encode(), 0x08000100)  # kCFStringEncodingUTF8
            if not value:
                raise OSError("CoreFoundation could not allocate a battery key")
            keys[text] = value
        info = copy_info()
        if not info:
            raise OSError("IOKit battery information is unavailable")
        sources = copy_list(info)
        if not sources:
            raise OSError("IOKit battery list is unavailable")

        def number(record, key):
            value = dictionary_get(record, keys[key])
            result = C.c_double()
            if value and get_type(value) == number_type() and number_get(value, 13, C.byref(result)):
                return result.value  # kCFNumberDoubleType
            return None

        for index in range(array_count(sources)):
            record = description(info, array_get(sources, index))
            if not record:
                continue
            kind = dictionary_get(record, keys["Type"])
            if not kind or not equal(kind, keys["InternalBattery"]):
                continue
            present = dictionary_get(record, keys["Is Present"])
            if present and get_type(present) == bool_type() and not bool_value(present):
                continue
            current, full = number(record, "Current Capacity"), number(record, "Max Capacity")
            percent = max(0.0, min(100.0, 100 * current / full)) if (
                current is not None and full is not None and full > 0) else None
            state = dictionary_get(record, keys["Power Source State"])
            plugged = None
            if state and equal(state, keys["AC Power"]):
                plugged = True
            elif state and equal(state, keys["Battery Power"]):
                plugged = False
            minutes = number(record, "Time to Empty")
            seconds = POWER_TIME_UNLIMITED if plugged else (
                max(0, int(minutes * 60)) if minutes is not None and minutes >= 0 else POWER_TIME_UNKNOWN)
            return Battery(percent, seconds, plugged)
        return None
    finally:
        for value in (sources, info, *keys.values()):
            if value:
                release(value)


def sensors_battery():
    if sys.platform.startswith("linux"):
        return linux_battery()
    if sys.platform == "win32":
        return windows_battery()
    if sys.platform == "darwin":
        return darwin_battery()
    raise NotImplementedError(f"Battery queries are not implemented on {sys.platform}")
