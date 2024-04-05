import argparse
import json
import pandas as pd

model = None
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-m",
        "--model",
        help="Model to import",
        default=model,
    )
    p.add_argument(
        "--list-models",
        help="Show available models and exit",
        action="store_true",
    )
    p.add_argument(
        "--summary",
        help="Summary file to write",
        default="summary.json",
    )
    args = p.parse_args()
    if not args.model and not args.list_models:
        raise SystemExit("Specify either --model or --list-models")
    return args

# AE1 csv: https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/docs/data_AlpacaEval/alpaca_eval_gpt4_leaderboard.csv
csv_url="https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/docs/data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv"
def _summary_data():
    f = pd.read_csv(csv_url)
    data = dict()
    for row in f.itertuples():
        fields = list(row)
        fields.pop(0) # discard index
        modl = fields.pop(0)
        data[modl] = fields
    return data

def _model_summary(args):
    summaries = _summary_data()
    name = args.model
    win_rate,avg_length,link,samples,filtr = summaries[name]
    return {
        "attributes": {
            "model": {
                "value": name,
            },
            "avg_length": {
                "value": avg_length,
            },
            "link": {
                "value": link,
            },
            "samples": {
                "value": samples,
            },
            "filter": {
                "value": filtr,
            },
        },
        "metrics": {
            "win_rate": {
                "value": win_rate,
            },
        },
    }

def _write_summary(summary, args):
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

def _show_models_and_exit():
    summaries = _summary_data()
    for modl in sorted(summaries.keys()):
        print(modl)
    raise SystemExit(0)

if __name__ == "__main__":
    args = _parse_args()
    if args.list_models:
        _show_models_and_exit()
    summary = _model_summary(args)
    _write_summary(summary, args)
