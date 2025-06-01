#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re
import numpy as np
from scipy import stats

def plot_individual_benchmarks(csv_files, output_dir):
    """Plot individual benchmark files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if df.empty or "time_ms" not in df.columns:
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["time_ms"], 'b-', linewidth=1)
            
            # Extract particle count from filename
            basename = os.path.basename(csv_path)
            match = re.search(r'benchmark_(\d+)_', basename)
            particle_count = match.group(1) if match else "unknown"
            
            plt.title(f'Performance Over Time ({particle_count} particles)')
            plt.xlabel("Simulation Step")
            plt.ylabel("Time per Step (ms)")
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_time = df["time_ms"].mean()
            std_time = df["time_ms"].std()
            plt.axhline(y=mean_time, color='r', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_time:.2f}ms')
            
            plt.legend()
            
            output_path = os.path.join(
                output_dir,
                basename.replace(".csv", "_timeline.png")
            )
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Generated: {output_path}")
            
        except Exception as e:
            print(f"Failed to process {csv_path}: {str(e)}")

def plot_scaling_analysis(csv_files, output_dir):
    """Generate scaling analysis plot - THIS IS THE KEY MISSING PIECE"""
    os.makedirs(output_dir, exist_ok=True)
    
    scaling_data = []
    
    for csv_file in csv_files:
        # Extract particle count from filename
        basename = os.path.basename(csv_file)
        match = re.search(r'benchmark_(\d+)_', basename)
        if not match:
            continue
            
        particle_count = int(match.group(1))
        
        try:
            df = pd.read_csv(csv_file)
            if df.empty or "time_ms" not in df.columns:
                continue
                
            # Skip first few steps for warmup
            df_stable = df.iloc[5:] if len(df) > 10 else df
            
            mean_time = df_stable["time_ms"].mean()
            std_time = df_stable["time_ms"].std()
            
            scaling_data.append({
                'particles': particle_count,
                'mean_time_ms': mean_time,
                'std_time_ms': std_time,
                'samples': len(df_stable)
            })
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if len(scaling_data) < 2:
        print(f"Need at least 2 data points for scaling analysis, got {len(scaling_data)}")
        return
    
    # Sort by particle count
    scaling_data.sort(key=lambda x: x['particles'])
    
    particles = np.array([d['particles'] for d in scaling_data])
    times = np.array([d['mean_time_ms'] for d in scaling_data])
    errors = np.array([d['std_time_ms'] for d in scaling_data])
    
    # Create scaling plot
    plt.figure(figsize=(12, 8))
    
    # Plot measured data with error bars
    plt.errorbar(particles, times, yerr=errors, fmt='o-', capsize=5, 
                markersize=8, linewidth=2, label='Measured Performance')
    
    # Fit scaling relationship
    if len(particles) >= 3:
        # Fit in log space for power law
        log_p = np.log(particles)
        log_t = np.log(times)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_p, log_t)
        
        # Generate fitted curve
        p_fit = np.linspace(particles.min(), particles.max(), 100)
        t_fit = np.exp(intercept) * (p_fit ** slope)
        
        plt.plot(p_fit, t_fit, '--', linewidth=2, alpha=0.8,
                label=f'Fitted: N^{slope:.2f} (R² = {r_value**2:.3f})')
        
        # Theoretical N² curve for comparison
        if particles.max() > particles.min():
            scale_factor = times[0] / (particles[0] ** 2)
            t_n2 = scale_factor * (p_fit ** 2)
            plt.plot(p_fit, t_n2, ':', linewidth=2, alpha=0.7,
                    label='Theoretical N²', color='red')
    
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Mean Time per Step (ms)', fontsize=12)
    plt.title('GPU Molecular Dynamics Performance Scaling', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Use log scale if we have a wide range
    if particles.max() / particles.min() > 10:
        plt.loglog()
        plt.title('GPU Molecular Dynamics Performance Scaling (Log-Log)', fontsize=14)
    
    scaling_plot_path = os.path.join(output_dir, 'performance_scaling.png')
    plt.savefig(scaling_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated scaling analysis: {scaling_plot_path}")
    
    # Print scaling summary
    print(f"\nScaling Analysis Summary:")
    print(f"{'Particles':<12} {'Mean Time (ms)':<15} {'Std Dev (ms)':<15}")
    print("-" * 45)
    for data in scaling_data:
        print(f"{data['particles']:<12} {data['mean_time_ms']:<15.3f} {data['std_time_ms']:<15.3f}")
    
    if len(particles) >= 3:
        print(f"\nFitted scaling: N^{slope:.2f}")
        print(f"Expected for naive pairwise: N^2.0")
        print(f"R-squared: {r_value**2:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Plot molecular dynamics benchmarks')
    parser.add_argument("csv_files", nargs="+", help="CSV files to plot")
    parser.add_argument("--output_dir", default="plots", help="Output directory")
    parser.add_argument("--scaling-only", action="store_true", 
                       help="Only generate scaling analysis")
    args = parser.parse_args()
    
    print(f"Processing {len(args.csv_files)} CSV files...")
    
    if args.scaling_only:
        plot_scaling_analysis(args.csv_files, args.output_dir)
    else:
        # Generate both individual plots and scaling analysis
        plot_individual_benchmarks(args.csv_files, args.output_dir)
        plot_scaling_analysis(args.csv_files, args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()