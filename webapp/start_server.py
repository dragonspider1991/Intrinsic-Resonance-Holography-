#!/usr/bin/env python3
"""
IRH Web Application Startup Script
===================================

Starts the FastAPI backend server for the IRH web interface.

Usage:
    python start_server.py [--host HOST] [--port PORT] [--reload]

Examples:
    python start_server.py
    python start_server.py --port 8080
    python start_server.py --reload  # Development mode with auto-reload
"""

import argparse
import sys
import os

# Add parent directory to path for IRH imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import uvicorn
from webapp.backend.app import app


def main():
    parser = argparse.ArgumentParser(description="Start IRH Web Application Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level (default: info)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("IRH Web Application - Backend Server")
    print("=" * 70)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/api/docs")
    print(f"Log level: {args.log_level}")
    print(f"Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print("=" * 70)
    print("\nPress CTRL+C to stop the server\n")
    
    # Start uvicorn server
    uvicorn.run(
        "webapp.backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
