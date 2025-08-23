from typing import Optional


def build_dir_name(
    base_name: str,
    use_demos: Optional[int] = None,
    reformat_by: Optional[str] = None,
) -> str:
    """
    Builds a directory name based on the run configurations.

    Args:
        base_name: (str): The base name of the directory.
        use_demos (Optional[int]): The number of demos that were used during
            inference.
        reformat_by (Optional[str]): Whether or not the data was reformatted
            to proper JSON format and who did it.

    Returns:
        str: The directory name.
    """
    directory_name = base_name
    directory_name += f"_use_demos_{use_demos}" if use_demos else ""
    directory_name += f"_reformat_by_{reformat_by}" if reformat_by else ""
    return directory_name
