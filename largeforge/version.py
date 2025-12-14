"""Version information for LargeForgeAI."""

__version__ = "1.0.0"
VERSION_INFO = (1, 0, 0)


def get_version() -> str:
    """Return the version string."""
    return __version__


def get_version_info() -> dict:
    """Return detailed version information."""
    return {
        "version": __version__,
        "major": VERSION_INFO[0],
        "minor": VERSION_INFO[1],
        "patch": VERSION_INFO[2],
        "python_requires": ">=3.10",
    }
