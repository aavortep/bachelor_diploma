import os


def clean_dir(dir_path: str):
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            os.remove(os.path.join(root, name))


def remove_dir(dir_path: str):
    if os.path.isdir(dir_path):
        os.rmdir(dir_path)


if __name__ == "__main__":
    clean_dir("tmp")
    remove_dir("tmp")
