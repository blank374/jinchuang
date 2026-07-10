from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convenience launcher for the MVP repository.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the MVP pipeline.")
    run_parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)

    dashboard_parser = subparsers.add_parser("dashboard", help="Start the Streamlit dashboard.")
    dashboard_parser.add_argument("--server-port", default="")

    api_parser = subparsers.add_parser("api", help="Start the FastAPI service.")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", default="8000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        command = [sys.executable, "-m", "mvp.pipeline", *args.pipeline_args]
    elif args.command == "dashboard":
        command = [sys.executable, "-m", "streamlit", "run", "dashboard.py"]
        if args.server_port:
            command.extend(["--server.port", args.server_port])
    else:
        command = [sys.executable, "-m", "uvicorn", "api:app", "--host", args.host, "--port", args.port]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
