def get_extension_from(filename):
    return (filename.split(".")[-1]).lower() if "." in filename else None
