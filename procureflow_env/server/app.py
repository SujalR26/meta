"""OpenEnv server entrypoint wrapper."""

from app.server import app, main as run_main

__all__ = ["app", "main"]


def main() -> None:
    """Run the ProcureFlow FastAPI application."""
    run_main()


if __name__ == "__main__":
    main()
