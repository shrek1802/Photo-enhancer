"""Single source of truth for PhotoPerfect Studio release information."""

APP_NAME = "PhotoPerfect Studio"
APP_VERSION = "2.3.0"
AUTO_ENGINE_VERSION = "2.2.0"
AUTO_ESSENTIALS_VERSION = "1.0.0"
MODEL_SCHEMA_VERSION = 2


def display_version() -> str:
    return f"{APP_NAME} v{APP_VERSION}"
