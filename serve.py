#!/usr/bin/env python3
"""
Pragma API Server Launcher

Starts the REST API for historical MR context.
Defaults to localhost (127.0.0.1) for security.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Start Pragma REST API for historical MR context"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1 for localhost only). "
        "Use 0.0.0.0 to expose to network (INSECURE without authentication)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Security warning for 0.0.0.0
    if args.host == "0.0.0.0":
        print("⚠️  WARNING: Binding to 0.0.0.0 exposes the API to your network!")
        print("⚠️  This API has NO authentication and may contain sensitive MR data.")
        print("⚠️  Only use 0.0.0.0 if behind a firewall/VPN or on a trusted network.\n")

    print("=" * 80)
    print("STARTING PRAGMA REST API")
    print("=" * 80)
    print(f"\nServer:     http://{args.host}:{args.port}")
    print(f"API Docs:   http://{args.host}:{args.port}/docs")
    print(f"Health:     http://{args.host}:{args.port}/health")
    print(f"Auto-reload: {args.reload}\n")
    print("Press CTRL+C to stop\n")

    # Import and run uvicorn
    import uvicorn
    from src.api_server import app

    uvicorn.run(
        app, host=args.host, port=args.port, reload=args.reload, log_level="info"
    )


if __name__ == "__main__":
    main()
