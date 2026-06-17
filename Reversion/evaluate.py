import pandas as pd
from backtest import backtest_model
from utils.logger import setup_logger


def evaluate_performance(config_path: str = "config/config.yaml",
                         model_path: str = "best_model",
                         dry_run: bool = False,
                         report_dir: str = "reports",
                         export: bool = True,
                         plot: bool = False,
                         seed: int = None,
                         indicators=None):
    logger = setup_logger("evaluate")
    results = backtest_model(
        config_path,
        model_path=model_path,
        dry_run=dry_run,
        report_dir=report_dir,
        export=export,
        plot=plot,
        seed=seed,
        indicators=indicators,
    )

    if not results:
        logger.warning("No backtest results produced; nothing to evaluate.")
        return results

    df_results = pd.DataFrame.from_dict(results, orient='index')
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(df_results.to_string())

    avg_sharpe = df_results['sharpe_ratio'].mean()
    avg_return = df_results['total_return_pct'].mean()
    avg_drawdown = df_results['max_drawdown_pct'].mean()
    avg_win_rate = df_results['win_rate_pct'].mean() if 'win_rate_pct' in df_results else float('nan')

    print(f"\nPortfolio Summary:")
    print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"Average Total Return: {avg_return:.2f}%")
    print(f"Average Max Drawdown: {avg_drawdown:.2f}%")
    print(f"Average Win Rate: {avg_win_rate:.2f}%")

    if avg_sharpe > 1.0 and avg_return > 10 and avg_drawdown < 20:
        logger.info("Model passed evaluation criteria.")
    else:
        logger.warning("Model needs improvement.")

    return results


if __name__ == "__main__":
    evaluate_performance()
