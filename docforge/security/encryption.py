# can be enabled via config
from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet

from docforge.core.config import get_config
from docforge.core.errors import StoreError
# from docforge.core.logger import get_logger

# log = get_logger(__name__)

# Cached Fernet client
_fernet_client: Fernet | None = None


def _load_or_create_key() -> bytes:
    key_path = Path(get_config().encryption.key_file)
    try:
        key_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if key_path.exists():
            return key_path.read_bytes()

        key = Fernet.generate_key()
        key_path.write_bytes(key)

        try:
            key_path.chmod(0o600)
        except OSError:
            pass

        # log.info(
        #     "encryption_key_created",
        #     path=str(key_path),
        # )

        return key

    except OSError as e:
        raise StoreError(
            f"Failed to create encryption key: {e}"
        ) from e


def _get_fernet() -> Fernet | None:
    global _fernet_client

    if not get_config().encryption.enabled:
        return None

    if _fernet_client is None:
        _fernet_client = Fernet(_load_or_create_key())

    return _fernet_client


def encrypt_file(source: Path, destination: Path,) -> None:
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise StoreError(
            f"Source file not found: {source}"
        )

    try:
        data = source.read_bytes()

        fernet = _get_fernet()

        encrypted = (
            fernet.encrypt(data)
            if fernet
            else data
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_bytes(encrypted)

        # log.info(
        #     "file_encrypted",
        #     source=str(source),
        #     destination=str(destination),
        # )

    except Exception as e:
        raise StoreError(
            f"Failed to encrypt file '{source}': {e}"
        ) from e


def decrypt_file(source: Path, destination: Path,) -> None:

    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise StoreError(
            f"Encrypted file not found: {source}"
        )

    try:
        data = source.read_bytes()

        fernet = _get_fernet()

        decrypted = (
            fernet.decrypt(data)
            if fernet
            else data
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_bytes(decrypted)

        # log.info(
        #     "file_decrypted",
        #     source=str(source),
        #     destination=str(destination),
        # )

    except Exception as e:
        raise StoreError(
            f"Failed to decrypt file '{source}': {e}"
        ) from e


def encrypt_text(text: str) -> str:

    if not text:
        return ""

    try:
        fernet = _get_fernet()

        if not fernet:
            return text

        return (
            fernet.encrypt( text.encode("utf-8")).decode("utf-8")
        )

    except Exception as e:
        raise StoreError(
            f"Failed to encrypt text: {e}"
        ) from e


def decrypt_text(ciphertext: str) -> str:
    if not ciphertext:
        return ""

    try:
        fernet = _get_fernet()

        if not fernet:
            return ciphertext

        return (
            fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        )

    except Exception as e:
        raise StoreError(
            f"Failed to decrypt text: {e}"
        ) from e