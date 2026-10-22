import argparse
from pathlib import Path
import sys
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_global_config(config_path: Path) -> dict:
    """Load global configuration from a TOML file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration TOML file.

    Returns
    -------
    dict
        Global configuration dictionary, or an empty dict if file does not exist.
    """
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config.get("global", {})
    return {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot 3x2 dose-response curve comparison.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "experiments.toml"),
        help="Path to configuration TOML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "dose_response_comparison.png"),
        help="Path to save the generated figure",
    )
    return parser.parse_args()


def load_observed_data(global_config: dict) -> pd.DataFrame:
    """Load observed data from the raw data CSV file.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Loaded raw observation dataset.
    """
    data_path = PROJECT_ROOT / global_config.get("data_path", "data")
    raw_file = data_path / global_config.get("raw_data_file")
    return pd.read_csv(raw_file, index_col=0)


def plot_dose_response_curves(config_path: Path, output_path: Path) -> None:
    """Generate and save the 2x3 dose-response comparison figure.

    Parameters
    ----------
    config_path : Path
        Path to the experiment configuration TOML file.
    output_path : Path
        File path where the resulting PNG figure will be saved.
    """
    global_config = load_global_config(config_path)

    # 2. Load observed data (t, y)
    df_obs = load_observed_data(global_config)
    t_obs = df_obs[global_config.get("treatment_node")]
    y_obs = df_obs[global_config.get("outcome_node")]
    n_ae_obs = df_obs.apply(
        lambda row: (row["obs_yield"] - row["0_kgha"]) / row["n_rate_pred"],
        axis=1,
    ).mean()

    # 3. Setup grid dimensions: 2 rows (experiments) x 3 columns (estimators)
    experiments = ["real", "naive"]
    estimators = ["LinearOLSPoly", "GPS", "SLearner"]

    estimator_display_names = {
        "LinearOLSPoly": "Quadratic\nLinear OLS",
        "GPS": "Generalized\nPropensity Score",
        "SLearner": "Random Forest\nS-Learner",
    }

    experiment_display_names = {
        "real": "Complete DAG\n(Valid Confounders)",
        "naive": "Misspecified DAG\n(No soil)",
    }

    estimator_colors = {
        "LinearOLSPoly": {"line": "#2563eb", "ci": "#93c5fd"},
        "GPS": {"line": "#059669", "ci": "#6ee7b7"},
        "SLearner": {"line": "#d97706", "ci": "#fde68a"},
    }

    # 4. Create subplots sharing x and y axes
    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(10, 7), sharex=True, sharey=True,
        dpi=150,
    )

    outputs_dir = PROJECT_ROOT / global_config.get("outputs_path", "outputs")

    for row_idx, exp in enumerate(experiments):
        csv_path = outputs_dir / exp / "estimates" / "efficiency_estimates.csv"
        df_eff = pd.read_csv(csv_path, index_col=1)
        for col_idx, est in enumerate(estimators):
            ax = axes[row_idx, col_idx]
            csv_path = outputs_dir / exp / "estimates" / f"{est}_average_dose_response_0.csv"

            # Load estimate data
            df = pd.read_csv(csv_path, index_col=0)
            t_vals = df.index.to_numpy(dtype=float)
            est_vals = df["estimate"].to_numpy(dtype=float)
            ci_lower = df["ci_lower"].to_numpy(dtype=float)
            ci_upper = df["ci_upper"].to_numpy(dtype=float)

            # Get ground truth response
            n_rate_cols = [col for col in df_obs.columns if col.endswith("kgha")]
            t_truth = [int(i.split("_")[0]) for i in n_rate_cols]
            y_truth = [df_obs[col].mean() for col in n_rate_cols]

            # Observed data scatterplot
            ax.scatter(
                t_obs, y_obs, color="#94a3b8", alpha=0.18, s=12,
                edgecolors="none", zorder=1, label="Observed Data",
            )

            # Shaded 95% Confidence Interval
            color_scheme = estimator_colors.get(est, {"line": "#2563eb", "ci": "#93c5fd"})
            ax.fill_between(
                t_vals, ci_lower, ci_upper, color=color_scheme["ci"],
                alpha=0.45, zorder=2, label="95% CI",
            )

            # Ground truth line
            ax.plot(
                t_truth, y_truth, color="#111827", linestyle="--", linewidth=2.2,
                zorder=3, label="Ground Truth",
            )

            # Estimated dose-response curve
            ax.plot(
                t_vals, est_vals, color=color_scheme["line"], linewidth=2.2,
                zorder=4, label="Estimated E[Y(t)]",
            )

            # Styling per subplot
            ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax.tick_params(axis="both", which="major", labelsize=10)

            # Error in N-AE
            nae_diff = abs(n_ae_obs - df_eff.loc[est, "value"]) / n_ae_obs * 100
            arrow = "↓" if n_ae_obs > df_eff.loc[est, "value"] else "↑"
            nae_text = (
                f"Est. N-AE: {df_eff.loc[est, 'value']:.0f}±{df_eff.loc[est, 'std_error']:.1f} kg/kg\n"
                f"Error: {nae_diff:.1f}% {arrow}"
            )
            ax.text(
                0.02, 0.98,
                nae_text,
                transform=ax.transAxes, va="top", ha="left",
            )

            # Column title (only on top row)
            if row_idx == 0:
                ax.set_title(
                    estimator_display_names.get(est, est), fontsize=13, fontweight="bold",
                    pad=12,
                )

            # Row label (on leftmost column)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{experiment_display_names.get(exp, exp)}\n\n E[Y(t)] (kg/ha)",
                    fontsize=11, fontweight="bold",
                )

            # X-axis label (on bottom row)
            if row_idx == len(experiments) - 1:
                ax.set_xlabel("Treatment dose (kg N/ha)", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 8000)
            ax.set_xlim(0, 200)

    # Add a single unified legend at the top
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8", markeredgecolor="none", markersize=7, label="Observations"),
        plt.Line2D([0], [0], color="#111827", linestyle="--", linewidth=2.2, label="Ground Truth"),
        plt.Line2D([0], [0], color="#2563eb", linestyle="-", linewidth=2.2, label="Estimated Dose-Response"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#93c5fd", alpha=0.45, edgecolor="none", label="95% Confidence Interval"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, fontsize=11,
        frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1",
        bbox_to_anchor=(0.55, -0.08),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure successfully saved to: {output_path}")


def main() -> None:
    """Execute dose-response curve comparison plotting."""
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)
    plot_dose_response_curves(config_path, output_path)


if __name__ == "__main__":
    main()
