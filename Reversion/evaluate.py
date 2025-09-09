import pandas as pd
from backtest import backtest_model
from utils.logger import setup_logger

def evaluate_performance(config_path: str = "config/config.yaml"):
    logger = setup_logger("evaluate")
    results = backtest_model(config_path)

    df_results = pd.DataFrame.from_dict(results, orient='index')
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(df_results.to_string())

    avg_sharpe = df_results['sharpe_ratio'].mean()
    avg_return = df_results['total_return_pct'].mean()
    avg_drawdown = df_results['max_drawdown_pct'].mean()

    print(f"\nPortfolio Summary:")
    print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"Average Total Return: {avg_return:.2f}%")
    print(f"Average Max Drawdown: {avg_drawdown:.2f}%")

    if avg_sharpe > 1.0 and avg_return > 10 and avg_drawdown < 20:
        logger.info("✅ Model passed evaluation criteria.")
    else:
        logger.warning("⚠️ Model needs improvement.")

if __name__ == "__main__":
    evaluate_performance()