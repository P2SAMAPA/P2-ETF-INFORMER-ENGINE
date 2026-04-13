from loader import load_dataset
from config import OPTION_A_ETFS, OPTION_B_ETFS

def main():
    print("Validating input dataset...")
    data = load_dataset("both", include_benchmarks=False)
    print(f"Loaded {len(data)} tickers")
    missing_A = [t for t in OPTION_A_ETFS if t not in data]
    missing_B = [t for t in OPTION_B_ETFS if t not in data]
    if missing_A:
        print(f"⚠️ Missing Option A: {missing_A}")
    if missing_B:
        print(f"⚠️ Missing Option B: {missing_B}")
    if not missing_A and not missing_B:
        print("All tradable ETFs present.")
    print("Validation complete.")

if __name__ == "__main__":
    main()
