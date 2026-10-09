"""One command: offline public benchmark, reports, failure cases, optional plots."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='local_results/public-reproduce')
parser.add_argument('--plots', action='store_true')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()


def run(*command):
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


run('-m', 'token_budget_lab.experiment', '--config', 'configs/public_offline.json', '--output', args.output,
    *(['--resume'] if args.resume else []))
run('-m', 'token_budget_lab.analysis', '--run', args.output, '--output', args.output,
    *(['--plot'] if args.plots else []))
run('-m', 'token_budget_lab.failures', '--run', args.output, '--output', args.output)
if args.plots:
    run('-m', 'token_budget_lab.analysis', '--historical-csv', 'results/deepseek_per_request_unique.csv',
        '--output', str(Path(args.output) / 'historical'))
