"""Tesy - your co-worker and friend.

The package is split into a small provider layer so the chat backend can be
swapped without touching the rest of the app:

- ``tesy.config``   environment / dotenv handling
- ``tesy.providers`` chat backends (OpenRouter, Snowflake Cortex)
- ``tesy.cli``      the command line entry point
"""

__version__ = "0.2.0"
