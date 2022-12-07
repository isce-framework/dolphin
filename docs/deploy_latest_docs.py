import subprocess

import dolphin

# copied from https://github.com/squidfunk/mkdocs-material-example-versioning


if __name__ == "__main__":
    # TODO: github key and stuff...
    # TODO: get version from git tag
    version = dolphin.__version__
    cmds = [
        # publish the current version:
        "mike deploy --push --update-aliases {version} latest",
        # Set the default version to latest
        "mike set-default --push latest",
    ]
    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True)
