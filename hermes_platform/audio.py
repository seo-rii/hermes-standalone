"""Explicit application audio boundary; importing it never activates a device."""


def native_audio_io():
    from standalone.audio_device import native_audio_io as selected
    return selected()


def silk_codec():
    from agent.sdk_boundary import bundled
    if bundled():
        from standalone import silk
        return silk
    import pilk
    return pilk
