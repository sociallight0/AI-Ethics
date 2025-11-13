# fairness_audit.py
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

def run_fairness_audit(data_path):
    """Audit COMPAS dataset for bias using AI Fairness 360."""
    # Load data
    df = pd.read_csv(data_path)
    df = df[['race', 'decile_score', 'two_year_recid']]  # Simplified columns
    df['label'] = df['two_year_recid'].apply(lambda x: 1 if x == 1 else 0)
    
    # Prepare AIF360 dataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label'],
        protected_attribute_names=['race'],
        favorable_label=0,  # No recidivism
        unfavorable_label=1  # Recidivism
    )
    
    # Define privileged and unprivileged groups
    privileged_groups = [{'race': 'Caucasian'}]
    unprivileged_groups = [{'race': 'African-American'}]
    
    # Compute fairness metrics
    metric = BinaryLabelDatasetMetric(dataset, privileged_groups, unprivileged_groups)
    disparate_impact = metric.disparate_impact()
    equal_opportunity = metric.mean_difference()
    
    print(f"Disparate Impact Ratio: {disparate_impact:.2f}")
    print(f"Equal Opportunity Difference: {equal_opportunity:.2f}")
    
    # Apply mitigation (reweighing)
    reweighing = Reweighing(unprivileged_groups, privileged_groups)
    dataset_transformed = reweighing.fit_transform(dataset)
    
    # Recompute metrics
    metric_transformed = BinaryLabelDatasetMetric(dataset_transformed, privileged_groups, unprivileged_groups)
    mitigated_disparate_impact = metric_transformed.disparate_impact()
    print(f"Mitigated Disparate Impact Ratio: {mitigated_disparate_impact:.2f}")

if __name__ == "__main__":
    run_fairness_audit("data/compas_scores.csv")
