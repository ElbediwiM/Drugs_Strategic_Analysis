import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_monte_carlo_simulation(base_npv, volatility, iterations=10000):
    """
    Runs a Monte Carlo simulation to estimate the range of potential NPV outcomes.
    
    Args:
        base_npv (float): The expected NPV.
        volatility (float): The standard deviation of the NPV (representing uncertainty).
        iterations (int): Number of simulation runs.
        
    Returns:
        np.array: Simulated NPV outcomes.
    """
    simulated_npvs = np.random.normal(base_npv, volatility * base_npv, iterations)
    return simulated_npvs

def plot_simulation_results(simulated_data, indication_name):
    """
    Plots the distribution of simulated NPV outcomes.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(simulated_data, kde=True, color='skyblue')
    plt.axvline(np.mean(simulated_data), color='red', linestyle='--', label=f'Mean: ${np.mean(simulated_data):,.0f}M')
    plt.axvline(np.percentile(simulated_data, 5), color='orange', linestyle=':', label='5th Percentile (P95 Risk)')
    plt.title(f'Monte Carlo NPV Simulation: {indication_name}')
    plt.xlabel('NPV ($M)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def run_sensitivity_analysis(base_peak_sales, sales_range, discount_rate_range):
    """
    Runs a sensitivity analysis on NPV based on peak sales and discount rate.
    
    Args:
        base_peak_sales (float): The base case peak sales.
        sales_range (list): List of multipliers for peak sales (e.g., [0.8, 1.0, 1.2]).
        discount_rate_range (list): List of discount rates to test.
        
    Returns:
        pd.DataFrame: A sensitivity matrix (heatmap data).
    """
    results = []
    for sales_mult in sales_range:
        row = []
        for rate in discount_rate_range:
            # Simplified NPV calculation for sensitivity
            # NPV = (Peak Sales * 0.3) / Discount Rate (Perpetuity simplified)
            npv = (base_peak_sales * sales_mult * 0.3) / rate
            row.append(npv)
        results.append(row)
    
    df = pd.DataFrame(results, index=[f'{int(m*100)}%' for m in sales_range], 
                      columns=[f'{int(r*100)}%' for r in discount_rate_range])
    return df

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    # 1. Monte Carlo for Obesity Indication
    print("Running Monte Carlo Simulation for Obesity...")
    obesity_npv = 18000 # $18B
    obesity_volatility = 0.25 # 25% uncertainty
    sim_results = run_monte_carlo_simulation(obesity_npv, obesity_volatility)
    # plot_simulation_results(sim_results, "Tirzepatide for Obesity")
    
    print(f"Mean Simulated NPV: ${np.mean(sim_results):,.0f}M")
    print(f"Value at Risk (5th Percentile): ${np.percentile(sim_results, 5):,.0f}M")

    # 2. Sensitivity Analysis for Heart Failure
    print("\nRunning Sensitivity Analysis for Heart Failure...")
    hf_peak_sales = 5000 # $5B
    sales_multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    discount_rates = [0.08, 0.09, 0.10, 0.11, 0.12]
    
    sensitivity_df = run_sensitivity_analysis(hf_peak_sales, sales_multipliers, discount_rates)
    print("Sensitivity Matrix (NPV $M) - Rows: Sales %, Cols: Discount Rate %")
    print(sensitivity_df.round(0))
    
    # To visualize locally, you would use:
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(sensitivity_df, annot=True, fmt=".0f", cmap="YlGnBu")
    # plt.title("NPV Sensitivity Analysis: Peak Sales vs. Discount Rate")
    # plt.show()
