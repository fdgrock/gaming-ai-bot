import click
import os
import json
import csv
from ai_lottery_bot.base import GameInterface
from ai_lottery_bot.data_ingestor.ingest import DataIngestor
from ai_lottery_bot.evaluator.evaluator import Evaluator
from ai_lottery_bot.model_trainer import ModelTrainer


@click.group()
def lotto():
    """Lotto AI Bot CLI"""
    pass


@lotto.command()
@click.argument('game')
@click.option('--source', default='manual', help='Source of actuals (manual/auto).')
def ingest_actuals(game, source):
    """Ingest actuals for a game."""
    ingestor = DataIngestor(game)
    if source == 'manual':
        # Example manual data
        data = [1, 2, 3, 4, 5, 6]
        ingestor.ingest_live_data(data)
    else:
        click.echo("Auto source not implemented yet.")


@lotto.command()
@click.argument('game')
@click.argument('url')
@click.option('--year', type=int, required=False, help='Year to filter draws (optional).')
def ingest_url(game, url, year):
    """Ingest draws from a URL and save into data/<game>/history/ as CSV."""
    ingestor = DataIngestor(game)
    df = ingestor.ingest_from_url(url, year=year)
    os.makedirs(os.path.join('data', game, 'history'), exist_ok=True)
    # Save to a CSV named by year if provided, else timestamp
    out_name = f"{year}_draws.csv" if year else "draws.csv"
    out_path = os.path.join('data', game, 'history', out_name)
    df.to_csv(out_path, index=False)
    click.echo(f"Saved {len(df)} draws to {out_path}")


@lotto.command()
@click.argument('game')
@click.option('--draw-date', required=True, help='Date of the draw to evaluate.')
def eval(game, draw_date):
    """Compute and store metrics for a specific draw."""
    evaluator = Evaluator()
    actuals_file = os.path.join("actuals", game, f"{draw_date}.json")
    if not os.path.exists(actuals_file):
        click.echo(f"Actuals not found for {draw_date}.")
        return
    with open(actuals_file, "r") as f:
        actual_set = json.load(f)["winning_set"]
    metrics = evaluator.evaluate_and_store(game, draw_date, actual_set)
    click.echo(f"Metrics computed and stored for {draw_date}: {metrics}")


@lotto.command()
@click.argument('game')
@click.option('--last', default=30, help='Number of recent draws to include.')
@click.option('--plot', is_flag=True, help='Export a chart as PNG/HTML.')
def metrics(game, last, plot):
    """Print rolling metrics table and optionally export a chart."""
    metrics_file = os.path.join("metrics", f"{game}.csv")
    if not os.path.exists(metrics_file):
        click.echo(f"Metrics file not found for game {game}.")
        return
    with open(metrics_file, "r") as f:
        rows = list(csv.reader(f))[-last:]
        click.echo("\n".join([", ".join(row) for row in rows]))
    if plot:
        click.echo("Exporting chart (not implemented yet).")


@lotto.command()
@click.argument('game')
@click.option('--from', 'from_model', required=True, help='Current champion model.')
@click.option('--to', 'to_model', required=True, help='New champion model.')
def promote(game, from_model, to_model):
    """Promote a new model to champion."""
    registry_file = os.path.join("models", game, "registry.json")
    with open(registry_file, "w") as f:
        json.dump({"champion": to_model}, f)
    click.echo(f"Promoted {to_model} as the new champion model for {game}.")


@lotto.command()
@click.argument('game')
def drift(game):
    """Print drift stats and recommended action."""
    trainer = ModelTrainer()
    data = []  # TODO: Load recent data for the game
    drift_detected = trainer.detect_data_drift(data, threshold=0.05)
    if drift_detected:
        click.echo("Drift detected. Retraining recommended.")
    else:
        click.echo("No significant drift detected.")


if __name__ == "__main__":
    lotto()
