"""Public product identity for the Vela distribution.

The internal ``vela`` package name remains stable for compatibility with the
upstream Python API. User-facing surfaces should import their labels from this
module instead of embedding the legacy project name.
"""

PRODUCT_NAME = "Vela"
CLI_NAME = "vela"
USER_AGENT = "Vela/0.1.1"
