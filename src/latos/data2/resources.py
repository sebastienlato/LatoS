"""Portable optional RSS reporting; absence is never represented as zero use."""

import sys


def peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024
