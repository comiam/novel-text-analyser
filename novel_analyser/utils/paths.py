import os


def get_absolute_path(path: str) -> str:
    """
    Возвращает абсолютный путь к файлу.
    Args:
        path: Путь к файлу
    Returns:
        Абсолютный путь к файлу. Если путь уже
        абсолютный, то он возвращается
        без изменений. Иначе возвращается
        абсолютный путь относительно текущей
        директории.
    """
    if not path or path.strip() == "":
        raise ValueError("Путь к файлу не может быть пустым")

    return path if os.path.isabs(path) else os.path.abspath(path)
