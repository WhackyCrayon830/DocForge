from pathlib import Path

APP_NAME = "DocForge"
APP_VERSION = "1.0.0"

DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"
LOGS_DIR = DATA_DIR / "logs"

# data/projects/<project_name>
CACHE_DIR = "cache"
VECTORS_DIR = "vectors"
TEMPLATES_DIR = "templates"
OUTPUTS_DIR = "outputs"

CONFIG_DIR = Path("config")
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"
USER_CONFIG = CONFIG_DIR / "user.toml"

SUPPORTED_DOCUMENTS = frozenset({
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".csv",
})

SUPPORTED_IMAGES = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
})

SUPPORTED_CODE = frozenset({
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
})


SUPPORTED_FILE_TYPES = ( SUPPORTED_DOCUMENTS | SUPPORTED_IMAGES | SUPPORTED_CODE)


KEYSTORE_FILENAME = DATA_DIR / ".keystore"
