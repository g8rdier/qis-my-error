"""BP+OSD port and experiments; reference imports are lazy.

The deployed candidate needs only numpy/stim/numba and its saved graph.
Do not import the organizer's build-only problem_info helper at runtime.
"""
__all__ = ["decode_bposd", "lookup"]


def __getattr__(name):
    if name in __all__:
        from bp_osd_decoder import decoder_bposd
        return getattr(decoder_bposd, name)
    raise AttributeError(name)
