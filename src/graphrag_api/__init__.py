"""Microsoft GraphRAG web api package."""

from importlib.metadata import version

__version__ = version(__package__)


def main() -> None:
    """Execute the main program."""
    from .api import run
    run()


if __name__ == "__main__":
    main()
