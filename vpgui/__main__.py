"""Allow ``python -m vpgui`` to launch the application."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
