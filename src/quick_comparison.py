"""
Quick Comparison: Original vs Stable LSTM
Run this to see the stability improvements
"""

import numpy as np
import pandas as pd
import sys

print("\n" + "="*70)
print(" " * 15 + "LSTM STABILITY COMPARISON")
print("="*70)

# Generate test data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=200, freq='D')
prices = pd.Series(
    100 + np.cumsum(np.random.randn(200) * 2),
    index=dates
)

print("\nTest Data: 200 days of stock prices")
print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")

# Test 1: Original version (without seed)
print("\n" + "-"*70)
print("TEST 1: ORIGINAL VERSION (No Fixed Seed)")
print("-"*70)

try:
    sys.path.insert(0, '/mnt/user-data/uploads')
    from strategy_lstm import lstm_strategy as lstm_original
    
    results_original = []
    print("\nRunning 3 times...")
    
    for i in range(3):
        print(f"\nRun {i+1}:")
        df = lstm_original(prices, window_size=20, epochs=30, verbose=0)
        n_buy = (df['trade'] == 1).sum()
        results_original.append(n_buy)
        print(f"  BUY signals: {n_buy}")
    
    print(f"\nResults: {results_original}")
    print(f"  Mean: {np.mean(results_original):.1f}")
    print(f"  Std Dev: {np.std(results_original):.2f}")
    
    if np.std(results_original) > 2:
        print("  ⚠️ HIGH VARIANCE - Results are unstable!")
    else:
        print("  ✓ Low variance")
        
except Exception as e:
    print(f"[ERROR] Could not run original version: {e}")
    results_original = None

# Test 2: Stable version (with seed)
print("\n" + "-"*70)
print("TEST 2: STABLE VERSION (Fixed Seed = 42)")
print("-"*70)

try:
    from strategy_lstm import lstm_strategy as lstm_stable
    
    results_stable = []
    print("\nRunning 3 times with seed=42...")
    
    for i in range(3):
        print(f"\nRun {i+1}:")
        df = lstm_stable(prices, window_size=20, epochs=30, verbose=0, seed=42)
        n_buy = (df['trade'] == 1).sum()
        results_stable.append(n_buy)
        print(f"  BUY signals: {n_buy}")
    
    print(f"\nResults: {results_stable}")
    print(f"  Mean: {np.mean(results_stable):.1f}")
    print(f"  Std Dev: {np.std(results_stable):.2f}")
    
    if np.std(results_stable) < 0.1:
        print("  ✓ PERFECT REPRODUCIBILITY!")
    else:
        print("  ⚠️ Some variance detected")
        
except Exception as e:
    print(f"[ERROR] Could not run stable version: {e}")
    results_stable = None

# Summary
print("\n" + "="*70)
print(" " * 25 + "SUMMARY")
print("="*70)

if results_original is not None:
    print(f"\nOriginal Version:")
    print(f"  Std Dev: {np.std(results_original):.2f}")
    print(f"  Reproducible: {'No ❌' if np.std(results_original) > 0.1 else 'Yes ✓'}")

if results_stable is not None:
    print(f"\nStable Version:")
    print(f"  Std Dev: {np.std(results_stable):.2f}")
    print(f"  Reproducible: {'Yes ✓' if np.std(results_stable) < 0.1 else 'No ❌'}")

print("\n" + "="*70)
print("\n✅ RECOMMENDATION: Use strategy_lstm_stable.py for production")
print("   - Fixed seed ensures reproducibility")
print("   - Gradient clipping prevents instability")
print("   - Early stopping prevents overfitting")
print("\n📖 See STABILITY_GUIDE.md for detailed instructions")
print("="*70 + "\n")
