"""Train the match prediction model. Run once before scanning."""
import sys
sys.path.insert(0, "src")
from model import train

if __name__ == "__main__":
    train(verbose=True)
