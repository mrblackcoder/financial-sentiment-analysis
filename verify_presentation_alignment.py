"""
Verification Script: Check alignment between presentation guide and actual results
"""

import pickle
import pandas as pd
from pathlib import Path
import sys

def load_results():
    """Load actual training results"""
    models_dir = Path('models')
    results = {}
    
    model_files = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Linear SVM': 'linear_svm_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'MLP (Deep Learning)': 'mlp_deep_learning_model.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            with open(models_dir / filename, 'rb') as f:
                results[name] = pickle.load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: {filename} not found")
    
    return results

def verify_presentation_numbers():
    """Verify all numbers mentioned in presentation guide"""
    
    print("="*80)
    print("🔍 PRESENTATION VERIFICATION TOOL")
    print("="*80)
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No model results found. Run train_and_evaluate.py first!")
        return False
    
    # Expected values from presentation guide (# SUNUM REHBERI)
    expected = {
        'Linear SVM': {
            'cv_f1': 0.96,
            'cv_std': 0.002,
            'test_f1': 96.18,  # percentage
            'mcc': 0.9427,
            'time_max': 0.40  # Presentation says ~0.32s, allow margin
        },
        'MLP (Deep Learning)': {
            'cv_f1': 0.96,
            'cv_std': 0.007,
            'test_f1': 95.54,
            'mcc': 0.9330,
            'time_max': 5.0  # Actual ~3.44s, allow margin
        },
        'Logistic Regression': {
            'cv_f1': 0.93,
            'cv_std': 0.008,
            'test_f1': 93.84,
            'time_max': 2.0  # Allow margin
        },
        'Random Forest': {
            'cv_f1': 0.91,
            'cv_std': 0.012,
            'test_f1': 91.15,
            'time_max': 0.20  # Very fast
        }
    }
    
    print("\n📊 MODEL VERIFICATION:\n")
    all_ok = True
    
    for model_name, exp_values in expected.items():
        if model_name not in results:
            print(f"⚠️  {model_name}: Not found in results")
            all_ok = False
            continue
        
        actual = results[model_name]
        cv_mean = actual['cv_scores'].mean()
        cv_std = actual['cv_scores'].std()
        test_f1 = actual['test_metrics']['f1_macro'] * 100  # to percentage
        
        print(f"{'='*60}")
        print(f"{model_name}")
        print(f"{'='*60}")
        
        # CV F1
        cv_match = abs(cv_mean - exp_values['cv_f1']) < 0.01
        print(f"CV F1-Score:")
        print(f"  Presentation: {exp_values['cv_f1']:.2f} ± {exp_values['cv_std']:.3f}")
        print(f"  Actual:       {cv_mean:.2f} ± {cv_std:.3f} {'✓' if cv_match else '❌ MISMATCH'}")
        
        # Test F1
        test_match = abs(test_f1 - exp_values['test_f1']) < 0.5
        print(f"\nTest F1-Score:")
        print(f"  Presentation: {exp_values['test_f1']:.2f}%")
        print(f"  Actual:       {test_f1:.2f}% {'✓' if test_match else '❌ MISMATCH'}")
        
        # MCC (if specified)
        if 'mcc' in exp_values and 'mcc' in actual['test_metrics']:
            mcc_match = abs(actual['test_metrics']['mcc'] - exp_values['mcc']) < 0.01
            print(f"\nMCC:")
            print(f"  Presentation: {exp_values['mcc']:.4f}")
            print(f"  Actual:       {actual['test_metrics']['mcc']:.4f} {'✓' if mcc_match else '❌ MISMATCH'}")
        
        # Training time
        actual_time = actual['training_time']
        if 'time_max' in exp_values:
            time_ok = actual_time <= exp_values['time_max']
            print(f"\nTraining Time:")
            print(f"  Presentation: ≤ {exp_values['time_max']:.2f}s")
            print(f"  Actual:       {actual_time:.2f}s {'✓' if time_ok else '⚠️  Slower than expected'}")
        elif 'time_min' in exp_values:
            time_ok = actual_time >= exp_values['time_min']
            print(f"\nTraining Time:")
            print(f"  Presentation: ≥ {exp_values['time_min']:.2f}s")
            print(f"  Actual:       {actual_time:.2f}s {'✓' if time_ok else '⚠️  Faster than expected'}")
        
        print()
        all_ok = all_ok and cv_match and test_match
    
    # Dataset verification
    print("="*80)
    print("📁 DATASET VERIFICATION")
    print("="*80)
    
    try:
        train_df = pd.read_csv('data/processed/train_clean.csv')
        val_df = pd.read_csv('data/processed/val_clean.csv')
        test_df = pd.read_csv('data/processed/test_clean.csv')
        
        total = len(train_df) + len(val_df) + len(test_df)
        
        dataset_checks = [
            ("Total samples", total, 3761),
            ("Train samples", len(train_df), 2632),
            ("Validation samples", len(val_df), 376),
            ("Test samples", len(test_df), 753)
        ]
        
        print("\n")
        for name, actual, expected in dataset_checks:
            match = actual == expected
            print(f"{name}:")
            print(f"  Presentation: {expected}")
            print(f"  Actual:       {actual} {'✓' if match else '❌ MISMATCH'}")
            all_ok = all_ok and match
        
    except FileNotFoundError:
        print("⚠️  Dataset files not found")
        all_ok = False
    
    # Final summary
    print("\n" + "="*80)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Presentation is aligned with results!")
    else:
        print("⚠️  MISMATCHES DETECTED - Update presentation guide!")
    print("="*80)
    
    return all_ok

if __name__ == "__main__":
    success = verify_presentation_numbers()
    sys.exit(0 if success else 1)
