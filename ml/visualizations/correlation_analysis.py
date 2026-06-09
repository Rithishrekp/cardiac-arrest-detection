from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_MODULE_DIR = os.path.dirname(__file__)
_PROCESSED_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "datasets", "processed"))
_DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(_MODULE_DIR, "..", "..", "outputs"))

def run_correlation_analysis(
    input_path: str | None = None,
    output_dir: str | None = None,
    save_plots: bool = True,
    print_report: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run exploratory correlation analysis on clinical/physiological data.

    This mirrors the offline Jupyter notebook workflow and saves outputs
    programmatically.
    """
    if input_path is None:
        input_path = os.path.join(_PROCESSED_DIR, "Integrated_Dataset_Final.csv")
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR

    df = pd.read_csv(input_path)

    # Clean target columns if needed
    if "CardiacRisk_Encoded" not in df.columns:
        if "CardiacRisk" in df.columns:
            risk_mapping = {"Normal": 0, "Moderate Risk": 1, "High Risk": 2, "Unknown": 1}
            df["CardiacRisk_Encoded"] = df["CardiacRisk"].map(risk_mapping).fillna(1)
        else:
            raise ValueError("CardiacRisk_Encoded or CardiacRisk column must be present in the dataset.")


    # 3. Compute Pearson correlation matrix
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr(method="pearson")

    # 4. Feature ranking by absolute correlation with CardiacRisk_Encoded
    corr_risk = corr_matrix["CardiacRisk_Encoded"].sort_values(ascending=False)
    correlation_df = corr_risk.drop("CardiacRisk_Encoded", errors="ignore").to_frame(name="Correlation")
    correlation_df["Abs Correlation"] = correlation_df["Correlation"].abs()
    correlation_df["Correlation (%)"] = correlation_df["Abs Correlation"] * 100
    correlation_df = correlation_df.sort_values(by="Correlation (%)", ascending=False)

    # Save CSV outputs
    os.makedirs(output_dir, exist_ok=True)
    correlation_df.to_csv(os.path.join(output_dir, "correlation_percentage_table.csv"))
    correlation_df.head(10).to_csv(os.path.join(output_dir, "top_correlated_features.csv"))

    if save_plots:
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
        
        # Save heatmap
        plt.figure(figsize=(18, 14))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=.1)
        plt.title("Pearson Correlation Matrix", fontsize=18)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
        plt.close()

        # Save dashboard (2x2 plot)
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        fig.suptitle("Cardiac Risk Correlation Dashboard", fontsize=28, fontweight="bold", y=1.02)
        
        # Plot A: Heatmap
        sns.heatmap(corr_matrix, ax=axes[0, 0], annot=False, cmap="coolwarm", fmt=".2f", linewidths=.1)
        axes[0, 0].set_title("A. Correlation Heatmap", fontsize=18)
        
        # Plot B: Top 15 Features by Correlation %
        top_n = min(15, len(correlation_df))
        top_features = correlation_df.head(top_n)
        sns.barplot(
            x=top_features["Correlation (%)"], 
            y=top_features.index, 
            hue=top_features.index, 
            ax=axes[0, 1], 
            palette="viridis", 
            legend=False
        )
        axes[0, 1].set_title(f"B. Top {top_n} Features by Correlation %", fontsize=18)
        axes[0, 1].set_xlabel("Absolute Correlation (%)")
        
        # Plot C: Top 10 Positive & Negative Correlations
        corr_for_plot = numeric_df.corr()["CardiacRisk_Encoded"].drop("CardiacRisk_Encoded", errors="ignore")
        top_pos = corr_for_plot.nlargest(10)
        top_neg = corr_for_plot.nsmallest(10)
        combined = pd.concat([top_pos, top_neg]).sort_values(ascending=False)
        sns.barplot(
            x=combined.values, 
            y=combined.index, 
            hue=combined.index, 
            ax=axes[1, 0], 
            palette="RdYlGn", 
            legend=False
        )
        axes[1, 0].set_title("C. Top 10 Positive & Negative Correlations", fontsize=18)
        axes[1, 0].set_xlabel("Pearson Correlation")

        # Plot D: Risk Distribution
        counts = df["CardiacRisk"].value_counts().sort_index()
        risk_order = ["Normal", "Moderate Risk", "High Risk", "Unknown"]
        sns.barplot(
            x=counts.index, 
            y=counts.values, 
            ax=axes[1, 1],
            order=[r for r in risk_order if r in counts.index],
            hue=counts.index, 
            palette="cividis", 
            legend=False
        )
        axes[1, 1].set_title("D. Risk Distribution", fontsize=18)
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(output_dir, "correlation_dashboard.png"), dpi=150)
        plt.close()

    if print_report:
        report_df = correlation_df.drop("CardiacRisk_Encoded", errors="ignore")
        print("\n" + "="*60)
        print("--- SCIENTIFIC FINDINGS REPORT: CARDIAC RISK CORRELATION ANALYSIS ---")
        print("="*60)

        # Strongest ECG Feature
        ecg_feats = ["QTInterval", "QRSDuration", "PQInterval", "VentricularRate", "QTCInterval", "RRInterval", "PPInterval"]
        ecg_corr = report_df[report_df.index.isin(ecg_feats)].sort_values("Abs Correlation", ascending=False)
        if not ecg_corr.empty:
            s = ecg_corr.iloc[0]
            direction = "positive" if s["Correlation"] > 0 else "negative"
            association = "longer" if s["Correlation"] > 0 else "shorter"
            risk_assoc = "higher" if s["Correlation"] > 0 else "lower"
            print(f"1. Strongest ECG Feature: {s.name} (r={s['Correlation']:.2f}, {s['Correlation (%)']:.2f}%)")
            print(f"   {s.name} is a critical ECG parameter. A {direction} correlation suggests {association}")
            print(f"   intervals are associated with {risk_assoc} cardiac risk.\n")

        # Strongest Physiological Feature
        phys_feats = ["Age", "Weight", "Height", "SysBP", "DiaBP", "BMI", "MAP"]
        phys_corr = report_df[report_df.index.isin(phys_feats)].sort_values("Abs Correlation", ascending=False)
        if not phys_corr.empty:
            s = phys_corr.iloc[0]
            direction = "positive" if s["Correlation"] > 0 else "negative"
            print(f"2. Strongest Physiological Feature: {s.name} (r={s['Correlation']:.2f}, {s['Correlation (%)']:.2f}%)")
            print(f"   {s.name} is a fundamental physiological indicator. Its strong {direction}")
            print(f"   correlation underscores its importance in risk assessment.\n")

        # Top 5 Features by Strength
        print("3. Top 5 Features by Correlation Strength:")
        for i, (feat, row) in enumerate(report_df.head(5).iterrows()):
            print(f"   {i+1}. {feat} (r={row['Correlation']:.2f}, {row['Correlation (%)']:.2f}%)")
        print("   These features are the most linearly influential factors on cardiac risk.\n")

        # Weakest 5 Features
        print("4. Weakest 5 Features by Correlation Strength:")
        for i, (feat, row) in enumerate(report_df.tail(5).iterrows()):
            print(f"   {i+1}. {feat} (r={row['Correlation']:.2f}, {row['Correlation (%)']:.2f}%)")
        print("   These features show minimal linear association with cardiac risk in this dataset.\n")
        print("="*60)

    return {
        "correlation_matrix": corr_matrix,
        "correlation_ranking": correlation_df
    }

if __name__ == "__main__":
    run_correlation_analysis()
