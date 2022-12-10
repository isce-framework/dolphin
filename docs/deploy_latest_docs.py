import os
import subprocess


def _print_and_run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def _setup_git_for_workflow():
    """Setup git for a github action workflow.

    based off https://github.com/mhausenblas/mkdocs-deploy-gh-pages/blob/master/action.sh
    """
    print("Setting up github repo/credentials")
    # workaround, see https://github.com/actions/checkout/issues/766
    gh_workspace = os.environ["GITHUB_WORKSPACE"]
    subprocess.run(
        f'git config --global --add safe.directory "{gh_workspace}"',
        shell=True,
        check=True,
    )

    proc = subprocess.run("git config --get user.name", shell=True, capture_output=True)
    username = proc.stdout.decode("utf-8").strip()
    if not username:
        subprocess.run(
            'git config --global user.name "${GITHUB_ACTOR}"', shell=True, check=True
        )

    proc = subprocess.run(
        "git config --get user.email", shell=True, capture_output=True
    )
    email = proc.stdout.decode("utf-8").strip()
    if not email:
        tmp_email = ' "${GITHUB_ACTOR}@users.noreply.${GITHUB_DOMAIN:-"github.com"}"'
        cmd = f"git config --global user.email {tmp_email}"
        subprocess.run(cmd, shell=True, check=True)

    gh_repo = os.environ["GITHUB_REPOSITORY"]
    gh_token = os.environ["GITHUB_TOKEN"]
    remote_repo = f"https://x-access-token:{gh_token}@github.com/{gh_repo}.git"

    subprocess.run("git remote rm origin", shell=True, check=True)
    subprocess.run(f'git remote add origin "{remote_repo}"', shell=True, check=True)


def _get_version():
    import dolphin

    return dolphin.__version__


if __name__ == "__main__":
    # TODO: github key and stuff...
    # TODO: get version from git tag

    try:
        _setup_git_for_workflow()
    except KeyError:
        print("Not running in github actions, skipping git setup and requirements")

    version = _get_version()
    cmds = [
        # copied from https://github.com/squidfunk/mkdocs-material-example-versioning
        # publish the current version:
        f"mike deploy --push --update-aliases {version} latest",
        # Set the default version to latest
        "mike set-default --push latest",
    ]
    for cmd in cmds:
        _print_and_run(cmd)
