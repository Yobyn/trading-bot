#!/usr/bin/env python3
"""
Dashboard CLI - Command line interface for the trading bot web dashboard
"""

import argparse
import sys
import os
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='Trading Bot Web Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--public', action='store_true', help='Bind to all interfaces (0.0.0.0)')
    
    args = parser.parse_args()
    
    # Adjust host for public access
    if args.public:
        args.host = '0.0.0.0'
        logger.warning("Dashboard will be accessible from all network interfaces!")
        logger.warning("Make sure your firewall is properly configured.")
    
    try:
        from web_dashboard import start_dashboard
        
        logger.info(f"Starting Trading Bot Dashboard...")
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Debug: {args.debug}")
        logger.info(f"URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
        
        start_dashboard(host=args.host, port=args.port, debug=args.debug)
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Please install required packages:")
        logger.error("pip install flask flask-socketio plotly")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 