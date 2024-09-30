import nox


@nox.session
def lint(session):
    session.run("flake8", ".")
