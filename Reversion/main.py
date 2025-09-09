from train import train_rl_agent
from evaluate import evaluate_performance

if __name__ == "__main__":
    print("🚀 Starting RL Auto-Trader...")
    train_rl_agent()
    evaluate_performance()
    print("✅ Done.")