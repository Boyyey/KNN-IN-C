#!/usr/bin/env python3
"""
Benchmark script to compare C implementation vs Python
"""
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import subprocess
import json

def run_c_benchmark():
    """Run the C benchmark and parse results"""
    result = subprocess.run(['./bin/iris_example'], capture_output=True, text=True)
    output = result.stdout
    
    # Parse C output
    lines = output.split('\n')
    c_time = None
    c_accuracy = None
    
    for line in lines:
        if 'Average prediction time' in line:
            c_time = float(line.split(': ')[1].split(' ')[0])
        elif 'Accuracy' in line and 'SIMD' in lines[lines.index(line)-1]:
            c_accuracy = float(line.split(': ')[1].split('%')[0])
    
    return c_time, c_accuracy

def run_python_benchmark():
    """Run Python k-NN benchmark"""
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Benchmark prediction
    times = []
    for _ in range(5):
        start = time.time()
        y_pred = knn.predict(X_test)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    return avg_time, accuracy

def main():
    print("=== k-NN Benchmark: C vs Python ===\n")
    
    # Run C benchmark
    print("Running C implementation...")
    c_time, c_accuracy = run_c_benchmark()
    
    # Run Python benchmark
    print("Running Python implementation...")
    py_time, py_accuracy = run_python_benchmark()
    
    # Calculate speedup
    speedup = py_time / c_time
    
    # Print results
    print("\n=== Results ===")
    print(f"{'Metric':<20} {'C':<10} {'Python':<10} {'Speedup':<10}")
    print(f"{'Time (s)':<20} {c_time:.6f}  {py_time:.6f}  {speedup:.2f}x")
    print(f"{'Accuracy (%)':<20} {c_accuracy:.2f}     {py_accuracy:.2f}      -")
    
    # Save results
    results = {
        'c_time': c_time,
        'c_accuracy': c_accuracy,
        'python_time': py_time,
        'python_accuracy': py_accuracy,
        'speedup': speedup
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to benchmark_results.json")

if __name__ == '__main__':
    main()