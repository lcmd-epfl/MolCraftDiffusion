#!/usr/bin/env python3
"""Command-line interface for MolecularDiffusion."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MolecularDiffusion - A molecular diffusion framework"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="MolecularDiffusion 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add subcommands here as needed
    # Example:
    # train_parser = subparsers.add_parser("train", help="Train a model")
    # train_parser.add_argument("config", help="Path to config file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle commands here
    print(f"Command '{args.command}' not implemented yet.")


if __name__ == "__main__":
    main() 