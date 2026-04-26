\
\
\

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config
from utils.seed import set_seed

class DataGenerator:
    """Generate advanced merchant transaction data with controlled anomalies"""

    def __init__(self, config):
        self.config = config
        self.n_merchants = config.N_MERCHANTS
        self.k_industries = config.K_INDUSTRIES
        self.start_date = config.START_DATE
        self.end_date = config.END_DATE
        self.anomalous_ratio = config.ANOMALOUS_RATIO

    def generate_data(self):
        """
        Generate merchant transaction data

        Returns:
            pd.DataFrame: Generated merchant data with time series features
            pd.DatetimeIndex: Date index for time series
        """
        print("=" * 80)
        print(" Generating Merchant Transaction Data ".center(80, "="))
        print("=" * 80)

        dates = pd.to_datetime(pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='MS'
        ))
        t_months = len(dates)
        time_indices = np.arange(t_months)

        economic_factor = np.ones(t_months)
        economic_factor[12:20] = np.random.normal(0.9, 0.05, 8)
        economic_factor = pd.Series(economic_factor).rolling(
            window=3,
            min_periods=1
        ).mean().values

        industry_params = self._initialize_industry_params(t_months)

        sparsity_params = {
            'beta0_tier': [-1.5, -2.0, -2.5],
            'beta1_health': -0.8,
            'anomaly_beta0_shift': 1.2,
            'anomaly_beta1_mult': 1.5,
        }

        results = []
        total_transactions = np.zeros((self.k_industries, t_months))

        for i in tqdm(range(self.n_merchants), desc="Generating merchants"):
            is_anomalous = 1 if np.random.rand() < self.anomalous_ratio else 0

            k = np.random.choice(range(self.k_industries))
            params = industry_params[k]
            tier = np.random.choice(3, p=params['merchant_dist'])
            base_amount = np.exp(np.random.normal(
                params['log_mean_base'][tier],
                params['log_std_base'][tier]
            ))

            seasonal_factor = 1 + params['season_amp'] * np.sin(
                2 * np.pi * (time_indices + params['season_phase']) / 12
            )
            growth_factor = (1 + params['growth_rate']) ** time_indices
            industry_shock_factor = np.ones(t_months)
            if params['industry_shock'] != 0:
                industry_shock_factor[params['shock_time']:] = 1 + params['industry_shock']

            continuous_amount = (
                base_amount *
                seasonal_factor *
                growth_factor *
                economic_factor *
                industry_shock_factor
            )

            industry_health = economic_factor - 1
            beta0 = sparsity_params['beta0_tier'][tier]
            beta1 = sparsity_params['beta1_health']

            if is_anomalous:
                beta0 += sparsity_params['anomaly_beta0_shift']
                beta1 *= sparsity_params['anomaly_beta1_mult']
                continuous_amount *= np.exp(np.random.normal(0, 0.3, t_months))

            inactive_logit = beta0 + beta1 * industry_health + np.random.normal(0, 0.5)
            inactive_prob = 1 / (1 + np.exp(-inactive_logit))
            is_inactive = np.random.rand(t_months) < inactive_prob

            final_amount = continuous_amount.copy()
            final_amount[is_inactive] = 0
            total_transactions[k, :] += final_amount

            record = {
                'ID': i,
                'Industry': f'Industry-{k}',
                'tier': tier,
                'is_anomalous': is_anomalous
            }
            for t_idx, date in enumerate(dates):
                date_str = date.strftime('%Y%m')
                record[f'txn_{date_str}'] = (
                    final_amount[t_idx] if final_amount[t_idx] > 0 else 0
                )
            results.append(record)

        final_df = pd.DataFrame(results)

        ts_cols = [col for col in final_df.columns if col.startswith('txn_')]
        total_cells = len(final_df) * len(ts_cols)
        zero_cells = (final_df[ts_cols] == 0).sum().sum()
        sparsity = zero_cells / total_cells

        anomaly_actual = final_df['is_anomalous'].mean()

        print(f"\nData Generation Summary:")
        print(f"  Total merchants: {len(final_df)}")
        print(f"  Industries: {self.k_industries}")
        print(f"  Time periods: {t_months} months")
        print(f"  Anomaly ratio: {anomaly_actual:.2%}")
        print(f"  Data sparsity: {sparsity:.2%}")

        return final_df, dates

    def _initialize_industry_params(self, t_months):
        params = {}
        for k in range(self.k_industries):
            if k == 3:
                params[k] = {
                    'merchant_dist': [0.5, 0.35, 0.15],
                    'log_mean_base': [np.log(6e3), np.log(6e4), np.log(6e5)],
                    'log_std_base': [0.9, 0.7, 0.5],
                    'season_amp': np.random.uniform(0.2, 0.5),
                    'season_phase': np.random.uniform(0, 11),
                    'growth_rate': np.random.normal(0.008, 0.006),
                    'industry_shock': (
                        np.random.choice([0, 1], p=[0.85, 0.15]) *
                        np.random.normal(-0.15, 0.06) if np.random.rand() > 0.4 else 0
                    ),
                    'shock_time': np.random.randint(4, t_months - 4)
                }
            else:
                params[k] = {
                    'merchant_dist': [0.6, 0.3, 0.1],
                    'log_mean_base': [np.log(5e3), np.log(5e4), np.log(5e5)],
                    'log_std_base': [0.8, 0.6, 0.4],
                    'season_amp': np.random.uniform(0.15, 0.45),
                    'season_phase': np.random.uniform(0, 11),
                    'growth_rate': np.random.normal(0.005, 0.005),
                    'industry_shock': (
                        np.random.choice([0, 1], p=[0.9, 0.1]) *
                        np.random.normal(-0.1, 0.05) if np.random.rand() > 0.5 else 0
                    ),
                    'shock_time': np.random.randint(5, t_months - 5)
                }
        return params

    def save_data(self, df, filename):
        """
        Save generated data to CSV

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        output_path = self.config.DATA_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"\nData saved to: {output_path}")
        return output_path

def main():
    """Main function for data generation"""

    set_seed(Config.SEED)

    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    generator = DataGenerator(Config)
    df, dates = generator.generate_data()

    generator.save_data(df, 'merchant_data.csv')

    dates_df = pd.DataFrame({'date': dates})
    dates_df.to_csv(Config.DATA_DIR / 'date_index.csv', index=False)

    print("\n" + "=" * 80)
    print(" Data Generation Complete ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    main()

