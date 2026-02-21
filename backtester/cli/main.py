import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Automated Forex Trading System CLI."""
    pass


@cli.command()
def info():
    """Show system information and configuration."""
    import platform
    import sys

    import numba

    click.echo(f"Python:  {sys.version}")
    click.echo(f"Numba:   {numba.__version__}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"CPU cores (logical): {numba.config.NUMBA_NUM_THREADS}")
    click.echo(f"Threading layer: {numba.config.THREADING_LAYER}")


if __name__ == "__main__":
    cli()
