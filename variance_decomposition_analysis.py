

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


def merge_scores_option(main_file_path: str, scores_file_path: Optional[str] = None, save_merged: bool = True) -> Tuple[
    pd.DataFrame, Optional[str]]:
    """
    Load main file and optionally merge additional scores based on Entry column.

    Args:
        main_file_path: Path to the main analysis file
        scores_file_path: Optional path to file containing additional scores to merge
        save_merged: Whether to save the merged dataset

        Tuple of (DataFrame with merged data, selected score column name)
    """
    print(f"Loading main dataset from: {main_file_path}")
    df = pd.read_csv(main_file_path)
    print(f"Main dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    selected_score_column = None

    if scores_file_path and Path(scores_file_path).exists():
        print(f"\nMerging scores from: {scores_file_path}")
        scores_df = pd.read_csv(scores_file_path)
        print(f"Scores dataset: {len(scores_df)} rows, {len(scores_df.columns)} columns")

        # Check if 'Entry' column exists in both datasets
        if 'Entry' not in df.columns:
            print("ERROR: 'Entry' column not found in main dataset")
            return df, None

        if 'Entry' not in scores_df.columns:
            print("ERROR: 'Entry' column not found in scores dataset")
            return df, None

        # Show merge preview
        print(f"\nMerge preview:")
        print(f"Main dataset entries: {df['Entry'].nunique()} unique values")
        print(f"Scores dataset entries: {scores_df['Entry'].nunique()} unique values")

        # Find overlap
        main_entries = set(df['Entry'])
        scores_entries = set(scores_df['Entry'])
        overlap = main_entries.intersection(scores_entries)

        print(f"Overlapping entries: {len(overlap)}")
        print(f"Main entries not in scores: {len(main_entries - scores_entries)}")
        print(f"Score entries not in main: {len(scores_entries - main_entries)}")

        # Show columns that will be added
        new_columns = [col for col in scores_df.columns if col != 'Entry' and col not in df.columns]
        existing_columns = [col for col in scores_df.columns if col != 'Entry' and col in df.columns]

        if new_columns:
            print(f"\nNew columns to be added: {new_columns}")
        if existing_columns:
            print(f"Columns that will be updated/overwritten: {existing_columns}")

        # Let user specify which column contains the scores they want to analyze
        print(f"\nAvailable columns in scores file:")
        score_file_columns = [col for col in scores_df.columns if col != 'Entry']
        for i, col in enumerate(score_file_columns, 1):
            sample_vals = scores_df[col].dropna().head(3).tolist()
            print(f"  {i}. {col} (sample: {sample_vals})")

        while True:
            try:
                choice = input(
                    f"\nWhich column contains the scores you want to analyze? Enter number (1-{len(score_file_columns)}) or column name: ").strip()

                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(score_file_columns):
                        selected_score_column = score_file_columns[idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(score_file_columns)}")
                elif choice in score_file_columns:
                    selected_score_column = choice
                    break
                else:
                    print(f"Column '{choice}' not found in scores file")
            except ValueError:
                print("Invalid input")

        print(f"Selected score column: {selected_score_column}")

        # Perform merge
        original_len = len(df)
        df_merged = pd.merge(df, scores_df, on='Entry', how='left', suffixes=('', '_new'))

        # Handle duplicate columns - keep new values and remove duplicate columns
        columns_to_drop = []
        for col in existing_columns:
            if f"{col}_new" in df_merged.columns:
                df_merged[col] = df_merged[f"{col}_new"].fillna(df_merged[col])
                columns_to_drop.append(f"{col}_new")

        if columns_to_drop:
            df_merged = df_merged.drop(columns=columns_to_drop)
            print(f"Removed duplicate columns: {columns_to_drop}")

        df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        print(f"\nMerge completed:")
        print(f"Rows after merge: {len(df_merged)} (original: {original_len})")
        print(f"Total columns after merge: {len(df_merged.columns)}")

        if len(overlap) > 0 and selected_score_column in df_merged.columns:
            sample_entry = list(overlap)[0]
            print(f"\nSample merged entry '{sample_entry}':")
            sample_row = df_merged[df_merged['Entry'] == sample_entry].iloc[0]
            print(f"  {selected_score_column}: {sample_row[selected_score_column]}")

        if save_merged:
            main_file_stem = Path(main_file_path).stem
            scores_file_stem = Path(scores_file_path).stem
            output_path = f"{main_file_stem}_merged_with_{scores_file_stem}.csv"

            df_merged.to_csv(output_path, index=False)
            print(f"\n✓ Merged dataset saved to: {output_path}")

        return df_merged, selected_score_column

    else:
        if scores_file_path:
            print(f"Scores file not found: {scores_file_path}")
        print("Proceeding with original dataset only")
        return df, None


def identify_score_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify potential score columns in the dataset.

    Returns:
        List of column names that appear to contain scores
    """
    score_patterns = [
        'score'
    ]

    score_columns = []

    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in score_patterns):
            score_columns.append(col)

    return score_columns


def interactive_score_selection(df: pd.DataFrame) -> str:
    """
    Interactively help user select which score column to analyze.

    Returns:
        Selected score column name
    """
    score_columns = identify_score_columns(df)

    if not score_columns:
        print("No obvious score columns found.")
        return interactive_score_selection_all_columns(df)

    else:
        print(f"\nDetected score columns:")
        for i, col in enumerate(score_columns, 1):
            sample_vals = df[col].dropna().head(3).tolist()
            print(f"  {i}. {col} (sample values: {sample_vals})")

        print(f"  {len(score_columns) + 1}. Other column")

        while True:
            try:
                choice = input(f"\nSelect score column (1-{len(score_columns) + 1}): ").strip()
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(score_columns):
                        selected_col = score_columns[choice_num - 1]
                        print(f"Selected: {selected_col}")
                        return selected_col
                    elif choice_num == len(score_columns) + 1:
                        return interactive_score_selection_all_columns(df)
                    else:
                        print(f"Please enter a number between 1 and {len(score_columns) + 1}")
                else:
                    print("Please enter a number")
            except ValueError:
                print("Invalid input")


def interactive_score_selection_all_columns(df: pd.DataFrame) -> str:
    """Show all columns for selection."""
    print("\nAll available columns:")
    for i, col in enumerate(df.columns, 1):
        sample_vals = df[col].dropna().head(3).tolist()
        print(f"  {i}. {col} (sample: {sample_vals})")

    while True:
        try:
            choice = input(f"\nEnter column number (1-{len(df.columns)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(df.columns):
                    selected_col = df.columns[idx]
                    print(f"Selected: {selected_col}")
                    return selected_col
                else:
                    print(f"Please enter a number between 1 and {len(df.columns)}")
            else:
                if choice in df.columns:
                    print(f"Selected: {choice}")
                    return choice
                else:
                    print(f"Column '{choice}' not found. Please enter a valid column number or name.")
        except ValueError:
            print("Invalid input")


class RigorousBiasAnalyzer:
    """
    Analyze species bias with systematic isolation from confounding factors.

    Implements the variance decomposition approach from Methods section (page 18):
    M0: y = β0 + ε
    M1: y = β0 + C(protein_family) + ε
    M2: M1 + S_structural
    M3: M2 + Q_quality
    M4: M3 + L_sequence
    M5: M4 + C(species)
    """

    def __init__(self, output_dir="./bias_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    MODEL_ARCHITECTURES = {
        'proteinmpnn': 'Structure-Conditioned (PDB)',
        'esmif': 'Structure-Conditioned (AFD)',
        'ESM2_15B_pppl': 'Sequence-Conditioned',
        'carp_640M': 'Sequence-Conditioned',
        'mif': 'Hybrid (CATH)',
        'mifst': 'Hybrid-ST (CATH + UniRef50)'
    }

    def analyze_isolated_species_bias(
            self,
            df: pd.DataFrame,
            score_column: str,
            control_factors: Optional[Dict[str, List[str]]] = None,
            bootstrap_samples: int = 1000,
            confidence_level: float = 0.95,
            random_seed: int = 42
    ) -> Dict:
        """
        Systematically isolate species variance from all other factors with bootstrap CI.

        Follows the nested model approach from paper's Methods section.

        Args:
            df: DataFrame with predictions and metadata
            score_column: Name of score column to analyze
            control_factors: Dict of factor groups to control for
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
            random_seed: Random seed for reproducibility

        Returns:
            Dict with isolated bias metrics and confidence intervals
        """
        try:
            print(f"\n=== RIGOROUS SPECIES BIAS ISOLATION: {score_column} ===")

            np.random.seed(random_seed)
            random.seed(random_seed)

            # Default control factors MATCHING PAPER'S METHODS (page 18)
            if control_factors is None:
                control_factors = {
                    'protein_family': ['protein_name'],  # M1: protein family
                    'structural': [  # M2: structural covariates
                        'helix_percent',
                        'sheet_percent',
                        'surface_exposure',
                        'avg_cb_distance',  # mean Cβ distance
                        'rco'  # relative contact order
                    ],
                    'quality': ['avg_plddt'],  # M3: quality covariates (mean pLDDT)
                    'sequence': [  # M4: sequence covariates
                        'sequence_length',
                        'isoelectric_point',
                        'charge_at_pH7'  # net charge at pH 7
                    ]
                }

            # Prepare data
            analysis_df = self._prepare_rigorous_data(df, score_column, control_factors)

            if len(analysis_df) < 100:
                print(f"WARNING: Insufficient data ({len(analysis_df)} samples)")
                return {}

            print(f"Clean dataset: {len(analysis_df)} samples")
            print(f"Species: {analysis_df['species'].nunique()}")
            print(f"Protein families: {analysis_df['protein_name'].nunique()}")

            # Systematic hierarchical analysis (matching paper's nested models)
            results = self._hierarchical_variance_decomposition(analysis_df, score_column, control_factors)

            # Calculate isolated species metrics
            isolated_metrics = self._calculate_isolated_species_metrics(results)

            # Statistical significance testing
            significance_results = self._test_species_significance(analysis_df, score_column, control_factors)

            # Bootstrap confidence intervals
            if bootstrap_samples > 0:
                print(f"\nCalculating bootstrap confidence intervals ({bootstrap_samples} samples)...")
                bootstrap_results = self._bootstrap_species_bias(
                    analysis_df, score_column, control_factors,
                    bootstrap_samples, confidence_level
                )
                isolated_metrics.update(bootstrap_results)

            # Combine all results
            final_results = {
                **results,
                **isolated_metrics,
                **significance_results,
                'total_samples': len(analysis_df),
                'n_species': analysis_df['species'].nunique(),
                'n_protein_families': analysis_df['protein_name'].nunique()
            }

            self._report_isolation_results(final_results)

            return final_results

        except Exception as e:
            logger.error(f"Error in rigorous bias analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def _bootstrap_species_bias(
            self,
            df: pd.DataFrame,
            score_column: str,
            control_factors: Dict[str, List[str]],
            n_bootstrap: int = 1000,
            confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Bootstrap confidence intervals for species bias metrics.
        """
        bootstrap_partial_r2 = []
        bootstrap_additional_r2 = []

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        for i in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
            try:
                bootstrap_df = df.sample(n=len(df), replace=True, random_state=i)

                bootstrap_results = self._hierarchical_variance_decomposition(
                    bootstrap_df, score_column, control_factors
                )

                bootstrap_metrics = self._calculate_isolated_species_metrics(bootstrap_results)

                bootstrap_partial_r2.append(bootstrap_metrics['isolated_species_partial_r2'])
                bootstrap_additional_r2.append(bootstrap_metrics['isolated_species_additional_r2'])

            except Exception as e:
                continue

        if len(bootstrap_partial_r2) > 0:
            partial_r2_ci_lower = np.percentile(bootstrap_partial_r2, lower_percentile)
            partial_r2_ci_upper = np.percentile(bootstrap_partial_r2, upper_percentile)

            additional_r2_ci_lower = np.percentile(bootstrap_additional_r2, lower_percentile)
            additional_r2_ci_upper = np.percentile(bootstrap_additional_r2, upper_percentile)

            return {
                'bootstrap_samples_successful': len(bootstrap_partial_r2),
                'partial_r2_ci_lower': partial_r2_ci_lower,
                'partial_r2_ci_upper': partial_r2_ci_upper,
                'additional_r2_ci_lower': additional_r2_ci_lower,
                'additional_r2_ci_upper': additional_r2_ci_upper,
                'confidence_level': confidence_level
            }
        else:
            return {'bootstrap_samples_successful': 0}

    def _prepare_rigorous_data(
            self,
            df: pd.DataFrame,
            score_column: str,
            control_factors: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Prepare data with comprehensive cleaning for rigorous analysis."""

        required_cols = [score_column, 'species']
        for factor_group in control_factors.values():
            required_cols.extend(factor_group)

        required_cols = list(set(required_cols))
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"WARNING: Missing columns: {missing_cols}")
            for group_name, factors in control_factors.items():
                control_factors[group_name] = [f for f in factors if f in df.columns]

        analysis_df = df.copy()

        analysis_df[score_column] = pd.to_numeric(analysis_df[score_column], errors='coerce')

        available_cols = [col for col in required_cols if col in analysis_df.columns]
        print(f"Controlling for: {available_cols}")

        initial_size = len(analysis_df)
        analysis_df = analysis_df.dropna(subset=available_cols)
        print(f"After removing missing data: {len(analysis_df)}/{initial_size} samples")

        # Remove extreme outliers (beyond 3 IQR)
        Q1 = analysis_df[score_column].quantile(0.25)
        Q3 = analysis_df[score_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        outlier_mask = (analysis_df[score_column] < lower_bound) | (analysis_df[score_column] > upper_bound)
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            analysis_df = analysis_df[~outlier_mask]
            print(f"Removed {n_outliers} extreme outliers")

        return analysis_df

    def _hierarchical_variance_decomposition(
            self,
            df: pd.DataFrame,
            score_column: str,
            control_factors: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Systematic hierarchical variance decomposition matching paper's nested models.

        From Methods page 18:
        M0: y = β0 + ε
        M1: y = β0 + C(protein_family) + ε
        M2: M1 + S_structural
        M3: M2 + Q_quality
        M4: M3 + L_sequence
        M5: M4 + C(species)
        """
        results = {}
        cumulative_r2 = 0.0

        current_formula = f"{score_column} ~ 1"

        # M0: Baseline (intercept only) - R² = 0 by definition
        baseline_r2 = 0.0

        # M1: Protein family
        if 'protein_family' in control_factors and control_factors['protein_family']:
            family_col = control_factors['protein_family'][0]
            if family_col in df.columns:
                current_formula = f"{score_column} ~ C({family_col})"
                family_model = ols(current_formula, data=df).fit()
                family_r2 = family_model.rsquared
                cumulative_r2 = family_r2
                results['protein_family_r2'] = family_r2

        # M2: Add structural factors
        if 'structural' in control_factors and control_factors['structural']:
            structural_terms = [col for col in control_factors['structural'] if col in df.columns]
            if structural_terms:
                current_formula += " + " + " + ".join(structural_terms)
                struct_model = ols(current_formula, data=df).fit()
                struct_total_r2 = struct_model.rsquared
                struct_additional_r2 = struct_total_r2 - cumulative_r2
                cumulative_r2 = struct_total_r2
                results['structural_additional_r2'] = struct_additional_r2

        # M3: Add quality factors
        if 'quality' in control_factors and control_factors['quality']:
            quality_terms = [col for col in control_factors['quality'] if col in df.columns]
            if quality_terms:
                current_formula += " + " + " + ".join(quality_terms)
                quality_model = ols(current_formula, data=df).fit()
                quality_total_r2 = quality_model.rsquared
                quality_additional_r2 = quality_total_r2 - cumulative_r2
                cumulative_r2 = quality_total_r2
                results['quality_additional_r2'] = quality_additional_r2

        # M4: Add sequence factors
        if 'sequence' in control_factors and control_factors['sequence']:
            sequence_terms = [col for col in control_factors['sequence'] if col in df.columns]
            if sequence_terms:
                current_formula += " + " + " + ".join(sequence_terms)
                sequence_model = ols(current_formula, data=df).fit()
                sequence_total_r2 = sequence_model.rsquared
                sequence_additional_r2 = sequence_total_r2 - cumulative_r2
                cumulative_r2 = sequence_total_r2
                results['sequence_additional_r2'] = sequence_additional_r2

        # M5: Finally add species (the factor we want to isolate)
        final_formula = current_formula + " + C(species)"
        final_model = ols(final_formula, data=df).fit()
        final_r2 = final_model.rsquared
        species_additional_r2 = final_r2 - cumulative_r2

        results['species_additional_r2'] = species_additional_r2
        results['total_r2'] = final_r2
        results['pre_species_r2'] = cumulative_r2

        return results

    def _calculate_isolated_species_metrics(self, results: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate properly isolated species bias metrics.

        From Methods page 18:
        - Additional R²: ΔR²_species = R²_M5 - R²_M4
        - Partial R²: R²_partial = ΔR²_species / (1 - R²_M4)
        """

        species_additional = results.get('species_additional_r2', 0.0)
        pre_species_r2 = results.get('pre_species_r2', 0.0)
        total_r2 = results.get('total_r2', 0.0)

        # Partial R² (semi-partial correlation squared)
        if pre_species_r2 < 1.0:
            isolated_species_partial_r2 = species_additional / (1 - pre_species_r2)
        else:
            isolated_species_partial_r2 = 0.0

        # Effect size interpretation
        if isolated_species_partial_r2 < 0.01:
            effect_size = "negligible"
        elif isolated_species_partial_r2 < 0.05:
            effect_size = "small"
        elif isolated_species_partial_r2 < 0.15:
            effect_size = "medium"
        else:
            effect_size = "large"

        return {
            'isolated_species_partial_r2': isolated_species_partial_r2,
            'isolated_species_additional_r2': species_additional,
            'species_effect_size': effect_size,
            'variance_remaining_before_species': 1 - pre_species_r2
        }

    def _test_species_significance(
            self,
            df: pd.DataFrame,
            score_column: str,
            control_factors: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Test statistical significance of species effect after controlling for confounds."""
        try:
            control_terms = []

            for factor_group in control_factors.values():
                for factor in factor_group:
                    if factor in df.columns:
                        if df[factor].dtype == 'object':
                            control_terms.append(f"C({factor})")
                        else:
                            control_terms.append(factor)

            if control_terms:
                control_formula = f"{score_column} ~ {' + '.join(control_terms)}"
                full_formula = f"{score_column} ~ {' + '.join(control_terms)} + C(species)"
            else:
                control_formula = f"{score_column} ~ 1"
                full_formula = f"{score_column} ~ C(species)"

            control_model = ols(control_formula, data=df).fit()
            full_model = ols(full_formula, data=df).fit()

            from scipy import stats as scipy_stats

            ssr_control = control_model.ssr
            ssr_full = full_model.ssr
            df_diff = full_model.df_model - control_model.df_model
            df_resid = full_model.df_resid

            f_stat = ((ssr_control - ssr_full) / df_diff) / (ssr_full / df_resid)
            p_value = 1 - scipy_stats.f.cdf(f_stat, df_diff, df_resid)

            return {
                'isolated_species_f_stat': f_stat,
                'isolated_species_p_value': p_value,
                'isolated_species_significant': p_value < 0.05
            }

        except Exception as e:
            logger.warning(f"Could not test species significance: {str(e)}")
            return {
                'isolated_species_f_stat': np.nan,
                'isolated_species_p_value': 1.0,
                'isolated_species_significant': False
            }

    def _report_isolation_results(self, results: Dict):
        """Report the isolated species bias results with confidence intervals."""
        print(f"\n=== ISOLATED SPECIES BIAS RESULTS ===")

        isolated_partial = results.get('isolated_species_partial_r2', 0) * 100
        isolated_additional = results.get('isolated_species_additional_r2', 0) * 100
        effect_size = results.get('species_effect_size', 'unknown')
        p_value = results.get('isolated_species_p_value', 1.0)

        print(f"Species Additional R² (ΔR²): {isolated_additional:.2f}%")
        print(f"Species Partial R²: {isolated_partial:.2f}%")

        if 'partial_r2_ci_lower' in results:
            ci_lower = results['partial_r2_ci_lower'] * 100
            ci_upper = results['partial_r2_ci_upper'] * 100
            confidence = results.get('confidence_level', 0.95) * 100
            print(f"  {confidence:.0f}% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")

        print(f"Effect size: {effect_size}")

        if p_value < 1e-100:
            print(f"Statistical significance: p < 1e-100")
        else:
            print(f"Statistical significance: p = {p_value:.2e}")

        if isolated_partial > 5:
            print(f"⚠️  SUBSTANTIAL SPECIES BIAS DETECTED")
        elif isolated_partial > 1:
            print(f"⚠️  Moderate species bias detected")
        else:
            print(f"✓ Minimal species bias after controlling for confounds")


def analyze_all_models(df: pd.DataFrame, bootstrap_samples: int = 500):
    """Analyze species bias across all protein language models with bootstrap CI."""
    analyzer = RigorousBiasAnalyzer()

    # Models analyzed in the paper (Table 1, Supplementary Table S2)
    score_columns = [
        'proteinmpnn_score',  # ProteinMPNN
        'esmif_score',  # ESM-IF
        'mif_score',  # MIF
        'mifst_score',  # MIF-ST
        'ESM2_15B_pppl_score',  # ESM2-15B (sequence-only)
        'carp_640M_score'  # CARP-640M (sequence-only)
    ]

    # Control factors matching paper's Methods (page 18)
    control_factors = {
        'protein_family': ['protein_name'],
        'structural': ['helix_percent', 'sheet_percent', 'surface_exposure', 'avg_cb_distance', 'rco'],
        'quality': ['avg_plddt'],
        'sequence': ['sequence_length', 'isoelectric_point', 'charge_at_pH7']
    }

    print("=== COMPREHENSIVE MODEL BIAS ANALYSIS ===")
    print(f"Analyzing {len(score_columns)} protein design models")
    print(f"Dataset: {len(df)} proteins across {df['species'].nunique()} species")
    print(f"Bootstrap samples: {bootstrap_samples}")

    all_results = {}
    summary_table = []

    for i, score_col in enumerate(score_columns, 1):
        if score_col not in df.columns:
            print(f"\n⚠️  SKIPPING {score_col}: Column not found")
            continue

        non_null_count = df[score_col].notna().sum()
        if non_null_count < 100:
            print(f"\n⚠️  SKIPPING {score_col}: Only {non_null_count} non-null values")
            continue

        print(f"\n{'=' * 60}")
        print(f"MODEL {i}/{len(score_columns)}: {score_col}")
        print(f"Non-null samples: {non_null_count:,}")
        print(f"{'=' * 60}")

        results = analyzer.analyze_isolated_species_bias(
            df,
            score_col,
            control_factors,
            bootstrap_samples=bootstrap_samples
        )

        if results:
            all_results[score_col] = results

            row = {
                'Model': score_col.replace('_score', ''),
                'Architecture': analyzer.MODEL_ARCHITECTURES.get(score_col.replace('_score', '').lower(), 'Unknown'),
                'Samples': results.get('total_samples', 0),
                'Species': results.get('n_species', 0),
                'Protein_Families': results.get('n_protein_families', 0),
                'Family_R2': round(results.get('protein_family_r2', 0) * 100, 1),
                'Structural_R2': round(results.get('structural_additional_r2', 0) * 100, 2),
                'Quality_R2': round(results.get('quality_additional_r2', 0) * 100, 2),
                'Sequence_R2': round(results.get('sequence_additional_r2', 0) * 100, 2),
                'Species_Add_R2': round(results.get('isolated_species_additional_r2', 0) * 100, 2),
                'Species_Partial_R2': round(results.get('isolated_species_partial_r2', 0) * 100, 2),
                'Species_P_Value': results.get('isolated_species_p_value', 1.0),
                'Effect_Size': results.get('species_effect_size', 'unknown'),
                'Total_R2': round(results.get('total_r2', 0) * 100, 1)
            }

            if 'partial_r2_ci_lower' in results:
                row['Partial_R2_CI_Lower'] = round(results['partial_r2_ci_lower'] * 100, 2)
                row['Partial_R2_CI_Upper'] = round(results['partial_r2_ci_upper'] * 100, 2)

            summary_table.append(row)

    if summary_table:
        summary_df = pd.DataFrame(summary_table)

        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE SPECIES BIAS SUMMARY")
        print(f"{'=' * 80}")

        summary_df = summary_df.sort_values('Species_Partial_R2', ascending=False)

        print(f"\nKey Metrics Summary:")
        if 'Partial_R2_CI_Lower' in summary_df.columns:
            print(f"{'Model':<15} {'Architecture':<30} {'Add R²':<8} {'Partial R² (95% CI)':<25} {'Effect':<10}")
            print("-" * 100)
            for _, row in summary_df.iterrows():
                ci_str = f"{row['Species_Partial_R2']:.2f}% [{row.get('Partial_R2_CI_Lower', 0):.2f}-{row.get('Partial_R2_CI_Upper', 0):.2f}]"
                print(f"{row['Model']:<15} {row['Architecture']:<30} {row['Species_Add_R2']:<8.2f}% "
                      f"{ci_str:<25} {row['Effect_Size']:<10}")
        else:
            print(f"{'Model':<15} {'Architecture':<30} {'Add R²':<8} {'Partial R²':<12} {'Effect':<10}")
            print("-" * 85)
            for _, row in summary_df.iterrows():
                print(f"{row['Model']:<15} {row['Architecture']:<30} {row['Species_Add_R2']:<8.2f}% "
                      f"{row['Species_Partial_R2']:<12.2f}% {row['Effect_Size']:<10}")

        output_path = "comprehensive_bias_analysis_with_ci.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")

        print(f"\n=== KEY INSIGHTS ===")
        highest_bias = summary_df.iloc[0]
        lowest_bias = summary_df.iloc[-1]

        print(f"Highest species bias: {highest_bias['Model']} ({highest_bias['Species_Partial_R2']:.2f}% partial R²)")
        print(f"Lowest species bias:  {lowest_bias['Model']} ({lowest_bias['Species_Partial_R2']:.2f}% partial R²)")

        avg_bias = summary_df['Species_Partial_R2'].mean()
        print(f"Average species bias: {avg_bias:.2f}% partial R²")

        significant_models = summary_df[summary_df['Species_P_Value'] < 0.05]
        print(f"Models with significant species bias: {len(significant_models)}/{len(summary_df)}")

        return all_results, summary_df

    else:
        print("\nNo valid models analyzed")
        return {}, pd.DataFrame()


def test_rigorous_analyzer(df: pd.DataFrame, score_column: str = 'proteinmpnn_score'):
    """Test the rigorous bias analyzer on a single model."""
    analyzer = RigorousBiasAnalyzer()

    # Control factors matching paper's Methods
    control_factors = {
        'protein_family': ['protein_name'],
        'structural': ['helix_percent', 'sheet_percent', 'surface_exposure', 'avg_cb_distance', 'rco'],
        'quality': ['avg_plddt'],
        'sequence': ['sequence_length', 'isoelectric_point', 'charge_at_pH7']
    }

    results = analyzer.analyze_isolated_species_bias(
        df,
        score_column,
        control_factors,
        bootstrap_samples=100
    )

    return results


if __name__ == "__main__":
    # File paths
    main_file_path = "proteins_dataset.csv"

    print("=== PROTEIN BIAS ANALYSIS WITH SCORE MERGE OPTION ===")
    print("1. Merge additional scores first")
    print("2. Proceed with existing data only")

    merge_choice = input("Choose option (1 or 2): ").strip()

    if merge_choice == "1":
        scores_file_path = input("Enter path to scores file (CSV with 'Entry' column): ").strip()
        scores_file_path = scores_file_path.strip("'\"")

        save_choice = input("Save merged dataset? (y/n): ").strip().lower()
        save_merged = save_choice.startswith('y')

        df, selected_score_column = merge_scores_option(main_file_path, scores_file_path, save_merged)

        if selected_score_column:
            print(f"\nUsing selected score column: {selected_score_column}")

            print("\nChoose analysis type:")
            print("1. Single model test with selected column")
            print("2. All models comprehensive analysis")
            print("3. Quick test (reduced bootstrap) with selected column")

            choice = input("Enter choice (1, 2, or 3): ").strip()

            if choice == "1":
                results = test_rigorous_analyzer(df, selected_score_column)
            elif choice == "3":
                analyzer = RigorousBiasAnalyzer()
                control_factors = {
                    'protein_family': ['protein_name'],
                    'structural': ['helix_percent', 'sheet_percent', 'surface_exposure', 'avg_cb_distance', 'rco'],
                    'quality': ['avg_plddt'],
                    'sequence': ['sequence_length', 'isoelectric_point', 'charge_at_pH7']
                }
                results = analyzer.analyze_isolated_species_bias(
                    df, selected_score_column, control_factors, bootstrap_samples=100
                )
            else:
                all_results, summary_df = analyze_all_models(df, bootstrap_samples=500)
        else:
            score_col = interactive_score_selection(df)
            results = test_rigorous_analyzer(df, score_col)
    else:
        df, _ = merge_scores_option(main_file_path)

        print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")

        score_columns = identify_score_columns(df)
        print(f"Available score columns: {score_columns}")

        print("\nChoose analysis type:")
        print("1. Single model test")
        print("2. All models comprehensive analysis")
        print("3. Quick test (reduced bootstrap)")

        choice = input("Enter choice (1, 2, or 3): ").strip()

        if choice == "1":
            score_col = interactive_score_selection(df)
            results = test_rigorous_analyzer(df, score_col)
        elif choice == "3":
            all_results, summary_df = analyze_all_models(df, bootstrap_samples=10)
        else:
            all_results, summary_df = analyze_all_models(df, bootstrap_samples=10)

    print("\nAnalysis complete.")
