import argparse
from pathlib import Path
import pandas as pd


def format_number(x: float, decimals: int = 16) -> str:
    # Format with fixed decimals, then strip trailing zeros and dot
    s = f"{x:.{decimals}f}".rstrip('0').rstrip('.')
    return s if s != '' else '0'


def transform_csv(input_path: Path, output_path: Path, decimals: int = 16) -> None:
    df = pd.read_csv(input_path)
    required = ['method', 'time', 'std']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert seconds -> milliseconds
    time_ms = df['time'] * 1000.0
    std_ms = df['std'] * 1000.0

    # Build merged column with "$\\pm$"
    merged = [f"{format_number(t, decimals)} $\\pm$ {format_number(s, decimals)}" for t, s in zip(time_ms, std_ms)]

    out = pd.DataFrame({
        'method': df['method'],
        'time_std': merged,
    })
    out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Format time_cost_summary.csv to method,time_std (ms ± ms).')
    parser.add_argument('--input', type=str, default=str(Path('test_straight_spark_shuffle') / 'time_cost_summary.csv'), help='Input CSV path')
    parser.add_argument('--output', type=str, default=str(Path('test_straight_spark_shuffle') / 'time_cost_summary_formatted.csv'), help='Output CSV path')
    parser.add_argument('--decimals', type=int, default=16, help='Decimal places for numbers')
    args = parser.parse_args()

    transform_csv(Path(args.input), Path(args.output), args.decimals)


if __name__ == '__main__':
    main()
