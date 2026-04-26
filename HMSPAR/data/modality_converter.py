import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from pyts.image import GramianAngularField
from sentence_transformers import SentenceTransformer
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config

class ModalityConverter:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.IMAGE_SIZE

        model_path = self._find_text_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Text model not found in {config.PRETRAINED_DIR}"
            )

        print(f"Loading text embedding model from: {model_path}")
        self.text_model = SentenceTransformer(str(model_path), device=device)

    def _find_text_model_path(self):
        model_name = self.config.TEXT_MODEL_NAME
        model_dir_name = f'models--{model_name.replace("/", "--")}'
        base_path = self.config.PRETRAINED_DIR / model_dir_name / 'snapshots'

        if base_path.exists():
            available_snapshots = sorted(
                [d for d in base_path.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if available_snapshots:
                return available_snapshots[0]
        return None

    def generate_text_descriptions(self, df, industry_growth_rates):

        print("\nGenerating text descriptions...")
        ts_cols = self._get_ts_columns(df)

        descriptions = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating descriptions"):
            desc = self._create_description(row, ts_cols, industry_growth_rates)
            descriptions.append(desc)

        print("Encoding text descriptions...")
        embeddings = self.text_model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=128,
            device=self.device
        )

        return embeddings

    def _create_description(self, row, ts_cols, industry_growth_rates):
        ts_data = row[ts_cols].values.astype(float)

        trend_desc = "stable"
        if len(ts_data) > 1:
            if ts_data[-1] > ts_data[0] * 1.5:
                trend_desc = "upward trend"
            elif ts_data[-1] < ts_data[0] * 0.5:
                trend_desc = "downward trend"

        volatility = np.std(ts_data) / (np.mean(ts_data) + 1e-6)
        vol_desc = "low volatility"
        if volatility > 0.8:
            vol_desc = "high volatility"
        elif volatility > 0.4:
            vol_desc = "medium volatility"

        tier_map = {0: 'small', 1: 'medium', 2: 'large'}

        industry_name = row['Industry']
        growth_info = industry_growth_rates.get(industry_name, {})
        mom_growth = growth_info.get('mom', 0)
        yoy_growth = growth_info.get('yoy', 0)

        mom_desc = f"month-over-month {'growth' if mom_growth >= 0 else 'decline'} of {abs(mom_growth):.1%}"
        yoy_desc = f"year-over-year {'growth' if yoy_growth >= 0 else 'decline'} of {abs(yoy_growth):.1%}"

        return (
            f"A '{tier_map[row['tier']]}' merchant in the '{industry_name}' sector. "
            f"The transaction history shows a '{trend_desc}' pattern with '{vol_desc}'. "
            f"The industry context shows a {mom_desc} and a {yoy_desc}."
        )

    def generate_isa_gaf_images(self, df):

        print("\nGenerating ISA-GAF images...")
        ts_cols = self._get_ts_columns(df)
        ts_matrix = df[ts_cols].values

        images = []
        for ts in tqdm(ts_matrix, desc="Converting to images"):
            img = self._convert_to_isa_gaf(ts)
            images.append(img)

        return np.array(images)

    def _convert_to_isa_gaf(self, time_series):
        sparsity_mask = (time_series > 1e-6).astype(float)
        indices = np.arange(len(time_series))
        present_indices = indices[sparsity_mask == 1]
        present_values = time_series[sparsity_mask == 1]

        if len(present_values) < 2:
            interpolated_series = np.zeros_like(time_series)
        else:
            interpolated_series = self._adaptive_interpolate(
                indices, present_indices, present_values, sparsity_mask
            )

        gaf_transformer = GramianAngularField(
            image_size=self.image_size,
            method='summation'
        )

        if (interpolated_series.max() - interpolated_series.min()) > 1e-8:
            ts_scaled = (
                (interpolated_series - interpolated_series.min()) /
                (interpolated_series.max() - interpolated_series.min()) * 2 - 1
            )
        else:
            ts_scaled = np.zeros_like(interpolated_series)

        gaf_trend = gaf_transformer.fit_transform(ts_scaled.reshape(1, -1))[0]

        mask_scaled = (sparsity_mask - 0.5) * 2
        gaf_sparsity = gaf_transformer.fit_transform(mask_scaled.reshape(1, -1))[0]

        isa_gaf_image = np.stack([gaf_trend, gaf_sparsity], axis=0)
        return isa_gaf_image

    def _adaptive_interpolate(self, all_indices, present_indices, present_values, sparsity_mask):
        n = len(all_indices)
        result = np.zeros(n)

        if len(present_values) >= 4:
            try:
                spline = UnivariateSpline(present_indices, present_values, s=len(present_values)*0.1)
                result = spline(all_indices)
                result = np.clip(result, 0, present_values.max() * 1.5)
            except:
                interp_func = interp1d(present_indices, present_values, kind='linear',
                                       bounds_error=False, fill_value=0)
                result = interp_func(all_indices)
        else:
            interp_func = interp1d(present_indices, present_values, kind='linear',
                                   bounds_error=False, fill_value=0)
            result = interp_func(all_indices)

        sigma = max(1.0, n * 0.05)
        result = gaussian_filter1d(result, sigma=sigma)

        for i in present_indices.astype(int):
            if 0 <= i < n:
                result[i] = present_values[present_indices == i][0] if len(present_values[present_indices == i]) > 0 else result[i]

        return result

    def _get_ts_columns(self, df):
        return [col for col in df.columns if col.startswith('txn_')]

    def calculate_industry_growth_rates(self, df, dates):

        print("\nCalculating industry growth rates...")
        ts_cols = self._get_ts_columns(df)

        industries = sorted(df['Industry'].unique())
        growth_rates = {}

        for industry in industries:
            industry_df = df[df['Industry'] == industry]
            series = industry_df[ts_cols].sum()
            series.index = dates

            yoy_growth = 0
            if len(series) >= 13:
                yoy_growth = (series.iloc[-1] / (series.iloc[-13] + 1e-6)) - 1

            mom_growth = 0
            if len(series) >= 2:
                mom_growth = (series.iloc[-1] / (series.iloc[-2] + 1e-6)) - 1

            growth_rates[industry] = {
                'yoy': yoy_growth,
                'mom': mom_growth
            }

        return growth_rates

class RetailModalityConverter:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.RETAIL_IMAGE_SIZE

        model_path = self._find_text_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Text model not found in {config.PRETRAINED_DIR}"
            )

        print(f"Loading text embedding model from: {model_path}")
        self.text_model = SentenceTransformer(str(model_path), device=device)

    def _find_text_model_path(self):
        model_name = self.config.TEXT_MODEL_NAME
        model_dir_name = f'models--{model_name.replace("/", "--")}'
        base_path = self.config.PRETRAINED_DIR / model_dir_name / 'snapshots'

        if base_path.exists():
            available_snapshots = sorted(
                [d for d in base_path.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if available_snapshots:
                return available_snapshots[0]
        return None

    def generate_text_descriptions(self, amount_series, trans_series):

        print("\nGenerating retail text descriptions...")

        descriptions = []
        for i in tqdm(range(len(amount_series)), desc="Creating descriptions"):
            desc = self._create_retail_description(
                amount_series[i],
                trans_series[i]
            )
            descriptions.append(desc)

        print("Encoding text descriptions...")
        embeddings = self.text_model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=128,
            device=self.device
        )

        return embeddings

    def _create_retail_description(self, amount_ts, trans_ts):
        total_amount = np.sum(amount_ts)
        total_trans = np.sum(trans_ts)

        if total_trans > 0:
            avg_per_trans = total_amount / total_trans
        else:
            avg_per_trans = 0

        trend_desc = "stable"
        if len(amount_ts) > 1 and amount_ts[0] > 0:
            if amount_ts[-1] > amount_ts[0] * 1.5:
                trend_desc = "upward trend"
            elif amount_ts[-1] < amount_ts[0] * 0.5:
                trend_desc = "downward trend"

        volatility = np.std(amount_ts) / (np.mean(amount_ts) + 1e-6)
        vol_desc = "low volatility"
        if volatility > 0.8:
            vol_desc = "high volatility"
        elif volatility > 0.4:
            vol_desc = "medium volatility"

        freq_desc = "low frequency"
        if total_trans > 100:
            freq_desc = "high frequency"
        elif total_trans > 30:
            freq_desc = "medium frequency"

        active_months = np.sum(amount_ts > 0)
        activity_desc = f"active in {active_months} out of {len(amount_ts)} months"

        return (
            f"A customer with total spending of ${total_amount:.2f} over {len(amount_ts)} months, "
            f"making {int(total_trans)} transactions. The spending pattern shows "
            f"{trend_desc} with {vol_desc}. The transaction frequency is {freq_desc} "
            f"with an average of ${avg_per_trans:.2f} per transaction. {activity_desc}."
        )

    def generate_isa_gaf_images(self, amount_series, trans_series):

        print("\nGenerating retail ISA-GAF images...")

        images = []
        for i in tqdm(range(len(amount_series)), desc="Converting to images"):
            img = self._convert_to_retail_isa_gaf(
                amount_series[i],
                trans_series[i]
            )
            images.append(img)

        return np.array(images)

    def _convert_to_retail_isa_gaf(self, amount_ts, trans_ts):

        amount_trend, amount_sparsity = self._single_ts_to_isa_gaf(amount_ts)
        trans_trend, trans_sparsity = self._single_ts_to_isa_gaf(trans_ts)

        isa_gaf_image = np.stack([
            amount_trend,
            amount_sparsity,
            trans_trend,
            trans_sparsity
        ], axis=0)
        return isa_gaf_image

    def _single_ts_to_isa_gaf(self, time_series):

        sparsity_mask = (time_series > 1e-6).astype(float)
        indices = np.arange(len(time_series))
        present_indices = indices[sparsity_mask == 1]
        present_values = time_series[sparsity_mask == 1]

        if len(present_values) < 2:
            interpolated_series = np.zeros_like(time_series)
        else:
            interpolated_series = self._adaptive_interpolate(
                indices, present_indices, present_values, sparsity_mask
            )

        gaf_transformer = GramianAngularField(
            image_size=self.image_size,
            method='summation'
        )

        if (interpolated_series.max() - interpolated_series.min()) > 1e-8:
            ts_scaled = (
                (interpolated_series - interpolated_series.min()) /
                (interpolated_series.max() - interpolated_series.min()) * 2 - 1
            )
        else:
            ts_scaled = np.zeros_like(interpolated_series)

        gaf_trend = gaf_transformer.fit_transform(ts_scaled.reshape(1, -1))[0]

        mask_scaled = (sparsity_mask - 0.5) * 2
        gaf_sparsity = gaf_transformer.fit_transform(mask_scaled.reshape(1, -1))[0]

        return gaf_trend, gaf_sparsity

    def _adaptive_interpolate(self, all_indices, present_indices, present_values, sparsity_mask):
        n = len(all_indices)
        result = np.zeros(n)

        if len(present_values) >= 4:
            try:
                spline = UnivariateSpline(present_indices, present_values, s=len(present_values)*0.1)
                result = spline(all_indices)
                result = np.clip(result, 0, present_values.max() * 1.5)
            except:
                interp_func = interp1d(present_indices, present_values, kind='linear',
                                       bounds_error=False, fill_value=0)
                result = interp_func(all_indices)
        else:
            interp_func = interp1d(present_indices, present_values, kind='linear',
                                   bounds_error=False, fill_value=0)
            result = interp_func(all_indices)

        sigma = max(1.0, n * 0.05)
        result = gaussian_filter1d(result, sigma=sigma)

        for i in present_indices.astype(int):
            if 0 <= i < n:
                result[i] = present_values[present_indices == i][0] if len(present_values[present_indices == i]) > 0 else result[i]

        return result

class CDNOWModalityConverter:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.CDNOW_IMAGE_SIZE

        model_path = self._find_text_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Text model not found in {config.PRETRAINED_DIR}"
            )

        print(f"Loading text embedding model from: {model_path}")
        self.text_model = SentenceTransformer(str(model_path), device=device)

    def _find_text_model_path(self):
        model_name = self.config.TEXT_MODEL_NAME
        model_dir_name = f'models--{model_name.replace("/", "--")}'
        base_path = self.config.PRETRAINED_DIR / model_dir_name / 'snapshots'

        if base_path.exists():
            available_snapshots = sorted(
                [d for d in base_path.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if available_snapshots:
                return available_snapshots[0]
        return None

    def generate_text_descriptions(self, amount_series, trans_series):

        print("\nGenerating CDNOW text descriptions...")

        descriptions = []
        for i in tqdm(range(len(amount_series)), desc="Creating descriptions"):
            desc = self._create_cdnow_description(
                amount_series[i],
                trans_series[i]
            )
            descriptions.append(desc)

        print("Encoding text descriptions...")
        embeddings = self.text_model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=128,
            device=self.device
        )

        return embeddings

    def _create_cdnow_description(self, amount_ts, trans_ts):
        total_amount = np.sum(amount_ts)
        total_trans = np.sum(trans_ts)

        if total_trans > 0:
            avg_per_trans = total_amount / total_trans
        else:
            avg_per_trans = 0

        trend_desc = "stable"
        if len(amount_ts) > 1 and amount_ts[0] > 0:
            if amount_ts[-1] > amount_ts[0] * 1.5:
                trend_desc = "upward trend"
            elif amount_ts[-1] < amount_ts[0] * 0.5:
                trend_desc = "downward trend"

        volatility = np.std(amount_ts) / (np.mean(amount_ts) + 1e-6)
        vol_desc = "low volatility"
        if volatility > 0.8:
            vol_desc = "high volatility"
        elif volatility > 0.4:
            vol_desc = "medium volatility"

        freq_desc = "low frequency"
        if total_trans > 10:
            freq_desc = "high frequency"
        elif total_trans > 5:
            freq_desc = "medium frequency"

        active_months = np.sum(amount_ts > 0)
        activity_desc = f"active in {active_months} out of {len(amount_ts)} months"

        return (
            f"A CD customer with total spending of ${total_amount:.2f} over {len(amount_ts)} months, "
            f"making {int(total_trans)} purchases. The spending pattern shows "
            f"{trend_desc} with {vol_desc}. The purchase frequency is {freq_desc} "
            f"with an average of ${avg_per_trans:.2f} per purchase. {activity_desc}."
        )

    def generate_isa_gaf_images(self, amount_series, trans_series):

        print("\nGenerating CDNOW ISA-GAF images...")

        images = []
        for i in tqdm(range(len(amount_series)), desc="Converting to images"):
            img = self._convert_to_cdnow_isa_gaf(
                amount_series[i],
                trans_series[i]
            )
            images.append(img)

        return np.array(images)

    def _convert_to_cdnow_isa_gaf(self, amount_ts, trans_ts):
        amount_trend, amount_sparsity = self._single_ts_to_isa_gaf(amount_ts)
        trans_trend, trans_sparsity = self._single_ts_to_isa_gaf(trans_ts)

        isa_gaf_image = np.stack([
            amount_trend,
            amount_sparsity,
            trans_trend,
            trans_sparsity
        ], axis=0)
        return isa_gaf_image

    def _single_ts_to_isa_gaf(self, time_series):

        sparsity_mask = (time_series > 1e-6).astype(float)
        indices = np.arange(len(time_series))
        present_indices = indices[sparsity_mask == 1]
        present_values = time_series[sparsity_mask == 1]

        if len(present_values) < 2:
            interpolated_series = np.zeros_like(time_series)
        else:
            interpolated_series = self._adaptive_interpolate(
                indices, present_indices, present_values, sparsity_mask
            )

        gaf_transformer = GramianAngularField(
            image_size=self.image_size,
            method='summation'
        )

        if (interpolated_series.max() - interpolated_series.min()) > 1e-8:
            ts_scaled = (
                (interpolated_series - interpolated_series.min()) /
                (interpolated_series.max() - interpolated_series.min()) * 2 - 1
            )
        else:
            ts_scaled = np.zeros_like(interpolated_series)

        gaf_trend = gaf_transformer.fit_transform(ts_scaled.reshape(1, -1))[0]

        mask_scaled = (sparsity_mask - 0.5) * 2
        gaf_sparsity = gaf_transformer.fit_transform(mask_scaled.reshape(1, -1))[0]

        return gaf_trend, gaf_sparsity

    def _adaptive_interpolate(self, all_indices, present_indices, present_values, sparsity_mask):
        n = len(all_indices)
        result = np.zeros(n)

        if len(present_values) >= 4:
            try:
                spline = UnivariateSpline(present_indices, present_values, s=len(present_values)*0.1)
                result = spline(all_indices)
                result = np.clip(result, 0, present_values.max() * 1.5)
            except:
                interp_func = interp1d(present_indices, present_values, kind='linear',
                                       bounds_error=False, fill_value=0)
                result = interp_func(all_indices)
        else:
            interp_func = interp1d(present_indices, present_values, kind='linear',
                                   bounds_error=False, fill_value=0)
            result = interp_func(all_indices)

        sigma = max(1.0, n * 0.05)
        result = gaussian_filter1d(result, sigma=sigma)

        for i in present_indices.astype(int):
            if 0 <= i < n:
                result[i] = present_values[present_indices == i][0] if len(present_values[present_indices == i]) > 0 else result[i]

        return result

class InstacartModalityConverter:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.INSTACART_IMAGE_SIZE

        model_path = self._find_text_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Text model not found in {config.PRETRAINED_DIR}"
            )

        print(f"Loading text embedding model from: {model_path}")
        self.text_model = SentenceTransformer(str(model_path), device=device)

    def _find_text_model_path(self):
        pretrained_dir = Path(self.config.PRETRAINED_DIR)

        if not pretrained_dir.exists():
            return None

        for model_dir in pretrained_dir.glob("models--*"):
            snapshot_dir = model_dir / "snapshots"
            if snapshot_dir.exists():
                for version_dir in snapshot_dir.iterdir():
                    if version_dir.is_dir():
                        return str(version_dir)

        return None

    def generate_text_descriptions(self, order_count_series, item_count_series):

        n_users = order_count_series.shape[0]
        descriptions = []

        print("\nGenerating Instacart user descriptions...")
        for i in tqdm(range(n_users), desc="Creating descriptions"):
            order_ts = order_count_series[i]
            item_ts = item_count_series[i]

            total_orders = order_ts.sum()
            total_items = item_ts.sum()
            active_weeks = (order_ts > 0).sum()
            avg_items_per_order = total_items / total_orders if total_orders > 0 else 0

            order_trend = "increasing" if order_ts[-5:].mean() > order_ts[:5].mean() else "decreasing"
            activity_level = "high" if active_weeks > len(order_ts) * 0.6 else "medium" if active_weeks > len(order_ts) * 0.3 else "low"

            desc = (
                f"User with {activity_level} activity level. "
                f"Placed {int(total_orders)} orders over {int(active_weeks)} active weeks, "
                f"purchasing {int(total_items)} items total. "
                f"Average {avg_items_per_order:.1f} items per order. "
                f"Order frequency is {order_trend}."
            )
            descriptions.append(desc)

        print("Encoding text descriptions...")
        text_embeddings = self.text_model.encode(
            descriptions,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return text_embeddings

    def generate_isa_gaf_images(self, order_count_series, item_count_series):

        n_users = order_count_series.shape[0]
        isa_gaf_images = []

        print("\nGenerating Instacart ISA-GAF images...")
        for i in tqdm(range(n_users), desc="Converting to images"):
            order_ts = order_count_series[i]
            item_ts = item_count_series[i]

            isa_gaf_image = self._create_dual_isa_gaf(order_ts, item_ts)
            isa_gaf_images.append(isa_gaf_image)

        return np.array(isa_gaf_images)

    def _create_dual_isa_gaf(self, order_ts, item_ts):

        order_trend, order_sparsity = self._single_ts_to_isa_gaf(order_ts)
        item_trend, item_sparsity = self._single_ts_to_isa_gaf(item_ts)

        isa_gaf_image = np.stack([
            order_trend,
            order_sparsity,
            item_trend,
            item_sparsity
        ], axis=0)
        return isa_gaf_image

    def _single_ts_to_isa_gaf(self, time_series):
        sparsity_mask = (time_series > 1e-6).astype(float)
        indices = np.arange(len(time_series))
        present_indices = indices[sparsity_mask == 1]
        present_values = time_series[sparsity_mask == 1]

        if len(present_values) < 2:
            interpolated_series = np.zeros_like(time_series)
        else:
            interpolated_series = self._adaptive_interpolate(
                indices, present_indices, present_values, sparsity_mask
            )

        gaf_transformer = GramianAngularField(
            image_size=self.image_size,
            method='summation'
        )

        if (interpolated_series.max() - interpolated_series.min()) > 1e-8:
            ts_scaled = (
                (interpolated_series - interpolated_series.min()) /
                (interpolated_series.max() - interpolated_series.min()) * 2 - 1
            )
        else:
            ts_scaled = np.zeros_like(interpolated_series)

        gaf_trend = gaf_transformer.fit_transform(ts_scaled.reshape(1, -1))[0]

        mask_scaled = (sparsity_mask - 0.5) * 2
        gaf_sparsity = gaf_transformer.fit_transform(mask_scaled.reshape(1, -1))[0]

        return gaf_trend, gaf_sparsity

    def _adaptive_interpolate(self, all_indices, present_indices, present_values, sparsity_mask):
        n = len(all_indices)
        result = np.zeros(n)

        if len(present_values) >= 4:
            try:
                spline = UnivariateSpline(present_indices, present_values, s=len(present_values)*0.1)
                result = spline(all_indices)
                result = np.clip(result, 0, present_values.max() * 1.5)
            except:
                interp_func = interp1d(present_indices, present_values, kind='linear',
                                       bounds_error=False, fill_value=0)
                result = interp_func(all_indices)
        else:
            interp_func = interp1d(present_indices, present_values, kind='linear',
                                   bounds_error=False, fill_value=0)
            result = interp_func(all_indices)

        sigma = max(1.0, n * 0.05)
        result = gaussian_filter1d(result, sigma=sigma)

        for i in present_indices.astype(int):
            if 0 <= i < n:
                result[i] = present_values[present_indices == i][0] if len(present_values[present_indices == i]) > 0 else result[i]

        return result

    def _interpolate_sparse_series(self, ts, target_length):
        if len(ts) == target_length:
            return ts

        x_old = np.arange(len(ts))
        x_new = np.linspace(0, len(ts) - 1, target_length)

        if np.all(ts == 0):
            return np.zeros(target_length)

        f = interp1d(x_old, ts, kind='linear', fill_value='extrapolate')
        ts_interp = f(x_new)

        return ts_interp

    def _scale_to_range(self, ts, min_val=-1, max_val=1):
        ts_min, ts_max = ts.min(), ts.max()

        if ts_max - ts_min < 1e-6:
            return np.full_like(ts, (min_val + max_val) / 2)

        ts_scaled = (ts - ts_min) / (ts_max - ts_min)
        ts_scaled = ts_scaled * (max_val - min_val) + min_val

        return ts_scaled

class TmallModalityConverter:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.TMALL_IMAGE_SIZE

        model_path = self._find_text_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Text model not found in {config.PRETRAINED_DIR}"
            )

        print(f"Loading text embedding model from: {model_path}")
        self.text_model = SentenceTransformer(str(model_path), device=device)

    def _find_text_model_path(self):
        pretrained_dir = Path(self.config.PRETRAINED_DIR)

        if not pretrained_dir.exists():
            return None

        for model_dir in pretrained_dir.glob("models--*"):
            snapshot_dir = model_dir / "snapshots"
            if snapshot_dir.exists():
                for version_dir in snapshot_dir.iterdir():
                    if version_dir.is_dir():
                        return str(version_dir)

        return None

    def generate_text_descriptions(self, time_series, labels):

        n_samples = time_series.shape[0]
        descriptions = []

        print("\nGenerating Tmall user descriptions...")
        for i in tqdm(range(n_samples), desc="Creating descriptions"):
            ts = time_series[i]

            clicks = ts[0]
            carts = ts[1]
            purchases = ts[2]
            favors = ts[3]

            total_clicks = clicks.sum()
            total_carts = carts.sum()
            total_purchases = purchases.sum()
            total_favors = favors.sum()

            active_days = (clicks > 0).sum()
            conversion_rate = (total_purchases / total_clicks * 100) if total_clicks > 0 else 0
            cart_to_purchase = (total_purchases / total_carts * 100) if total_carts > 0 else 0

            if len(clicks) >= 10:
                recent_activity = clicks[-10:].mean()
                early_activity = clicks[:10].mean()
                trend = "increasing" if recent_activity > early_activity * 1.2 else\
                       "decreasing" if recent_activity < early_activity * 0.8 else "stable"
            else:
                trend = "stable"

            if active_days > len(clicks) * 0.6:
                engagement = "high"
            elif active_days > len(clicks) * 0.3:
                engagement = "medium"
            else:
                engagement = "low"

            desc = (
                f"User with {engagement} engagement level over {int(active_days)} active days. "
                f"Performed {int(total_clicks)} clicks, added {int(total_carts)} items to cart, "
                f"made {int(total_purchases)} purchases, and favorited {int(total_favors)} items. "
                f"Conversion rate: {conversion_rate:.1f}%, cart-to-purchase: {cart_to_purchase:.1f}%. "
                f"Activity trend is {trend}."
            )
            descriptions.append(desc)

        print("Encoding text descriptions...")
        text_embeddings = self.text_model.encode(
            descriptions,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return text_embeddings

    def generate_isa_gaf_images(self, time_series):

        n_samples = time_series.shape[0]
        isa_gaf_images = []

        print("\nGenerating Tmall ISA-GAF images...")
        for i in tqdm(range(n_samples), desc="Converting to images"):
            ts = time_series[i]

            browsing_ts = ts[0] + ts[3]
            buying_ts = ts[1] + ts[2]

            isa_gaf_image = self._create_dual_isa_gaf(browsing_ts, buying_ts)
            isa_gaf_images.append(isa_gaf_image)

        return np.array(isa_gaf_images)

    def _create_dual_isa_gaf(self, ts1, ts2):
        gaf_transformer = GramianAngularField(
            image_size=self.image_size,
            method='summation'
        )

        ts1_interp = self._interpolate_sparse_series(ts1, self.image_size)
        ts2_interp = self._interpolate_sparse_series(ts2, self.image_size)

        ts1_scaled = self._scale_to_range(ts1_interp, -1, 1)
        ts2_scaled = self._scale_to_range(ts2_interp, -1, 1)

        gaf1 = gaf_transformer.fit_transform(ts1_scaled.reshape(1, -1))[0]
        gaf2 = gaf_transformer.fit_transform(ts2_scaled.reshape(1, -1))[0]

        isa_gaf_image = np.stack([gaf1, gaf2], axis=0)
        return isa_gaf_image

    def _interpolate_sparse_series(self, ts, target_length):
        if len(ts) == target_length:
            return ts

        x_old = np.arange(len(ts))
        x_new = np.linspace(0, len(ts) - 1, target_length)

        if np.all(ts == 0):
            return np.zeros(target_length)

        f = interp1d(x_old, ts, kind='linear', fill_value='extrapolate')
        ts_interp = f(x_new)

        return ts_interp

    def _scale_to_range(self, ts, min_val=-1, max_val=1):
        ts_min, ts_max = ts.min(), ts.max()

        if ts_max - ts_min < 1e-6:
            return np.full_like(ts, (min_val + max_val) / 2)

        ts_scaled = (ts - ts_min) / (ts_max - ts_min)
        ts_scaled = ts_scaled * (max_val - min_val) + min_val

        return ts_scaled

def process_merchant():
    from configs.config import Config
    from utils.seed import set_seed
    import torch

    print("=" * 80)
    print(" Processing Merchant Dataset ".center(80, "="))
    print("=" * 80)

    set_seed(Config.SEED)

    data_path = Config.DATA_DIR / 'merchant_data.csv'
    if not data_path.exists():
        print("Error: Please run data_generator.py first!")
        return

    df = pd.read_csv(data_path)
    dates_df = pd.read_csv(Config.DATA_DIR / 'date_index.csv')
    dates = pd.to_datetime(dates_df['date'])

    print(f"Samples: {len(df)}")
    ts_cols = [col for col in df.columns if col.startswith('txn_')]
    ts_data = df[ts_cols].values
    sparsity = (ts_data == 0).sum() / ts_data.size * 100
    print(f"Sparsity: {sparsity:.2f}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    converter = ModalityConverter(Config, device)
    growth_rates = converter.calculate_industry_growth_rates(df, dates)
    text_embeddings = converter.generate_text_descriptions(df, growth_rates)
    isa_gaf_images = converter.generate_isa_gaf_images(df)

    print("\nSaving modality data...")
    np.save(Config.DATA_DIR / 'text_embeddings.npy', text_embeddings)
    np.save(Config.DATA_DIR / 'isa_gaf_images.npy', isa_gaf_images)

    print("\n" + "=" * 80)
    print(" Merchant Modality Conversion Complete ".center(80, "="))
    print("=" * 80)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

def process_retail(positive_ratio=0.35):

    from configs.config import Config
    from utils.seed import set_seed
    import torch

    print("=" * 80)
    print(" Processing Retail Dataset ".center(80, "="))
    print("=" * 80)

    set_seed(Config.SEED)

    amount_path = Config.RETAIL_AMOUNT_CSV
    trans_path = Config.RETAIL_TRANS_CSV

    if not amount_path.exists() or not trans_path.exists():
        print("Error: Retail datasets not found. Please download first.")
        return

    df_amount = pd.read_csv(amount_path)
    df_trans = pd.read_csv(trans_path)

    amount_series = df_amount.iloc[:, :-1].values
    trans_series = df_trans.iloc[:, :-1].values

    n_samples, n_periods = amount_series.shape

    print("\n" + "-" * 60)
    print("Computing Customer Engagement Score...")
    print("-" * 60)

    def compute_gaf_features(ts):
        if ts.sum() == 0:
            return 0, 0, 0, 0
        ts_min, ts_max = ts.min(), ts.max()
        if ts_max - ts_min < 1e-8:
            ts_scaled = np.zeros_like(ts)
        else:
            ts_scaled = (ts - ts_min) / (ts_max - ts_min)
        ts_scaled = ts_scaled * 2 - 1
        ts_scaled = np.clip(ts_scaled, -1, 1)
        phi = np.arccos(ts_scaled)
        n = len(phi)
        gasf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gasf[i, j] = np.cos(phi[i] + phi[j])
        diag = np.diag(gasf)
        diag_var = np.var(diag)
        upper_tri = gasf[np.triu_indices(n, k=1)]
        off_diag_mean = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        phase_diff = np.diff(phi)
        angular_momentum = np.std(phase_diff) if len(phase_diff) > 0 else 0
        block_vars = []
        for i in range(n-1):
            for j in range(n-1):
                block = gasf[i:i+2, j:j+2]
                block_vars.append(np.var(block))
        local_texture = np.mean(block_vars) if block_vars else 0
        return diag_var, off_diag_mean, angular_momentum, local_texture

    amt_diag_var = np.zeros(n_samples)
    amt_off_diag = np.zeros(n_samples)
    amt_angular = np.zeros(n_samples)
    amt_texture = np.zeros(n_samples)
    trs_diag_var = np.zeros(n_samples)
    trs_off_diag = np.zeros(n_samples)
    trs_angular = np.zeros(n_samples)
    trs_texture = np.zeros(n_samples)

    for i in range(n_samples):
        amt_diag_var[i], amt_off_diag[i], amt_angular[i], amt_texture[i] = compute_gaf_features(amount_series[i])
        trs_diag_var[i], trs_off_diag[i], trs_angular[i], trs_texture[i] = compute_gaf_features(trans_series[i])

    cross_coupling = np.zeros(n_samples)
    for i in range(n_samples):
        amt = amount_series[i]
        trs = trans_series[i]
        if amt.sum() > 0 and trs.sum() > 0:
            amt_norm = (amt - amt.mean()) / (amt.std() + 1e-8)
            trs_norm = (trs - trs.mean()) / (trs.std() + 1e-8)
            cross_coupling[i] = np.abs(np.mean(amt_norm * trs_norm))

    recency_score = np.zeros(n_samples)
    for i in range(n_samples):
        weights = np.exp(np.linspace(-1, 0, n_periods))
        active_mask = (amount_series[i] > 0).astype(float)
        recency_score[i] = np.sum(weights * active_mask) / np.sum(weights)

    atv_trend = np.zeros(n_samples)
    for i in range(n_samples):
        amt = amount_series[i]
        trs = trans_series[i]
        active_idx = np.where(trs > 0)[0]
        if len(active_idx) >= 2:
            atvs = amt[active_idx] / (trs[active_idx] + 1e-8)
            mid = len(active_idx) // 2
            if mid > 0:
                early_atv = np.mean(atvs[:mid])
                late_atv = np.mean(atvs[mid:])
                atv_trend[i] = (late_atv - early_atv) / (early_atv + 1e-8)

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    engagement_score = (
        0.12 * normalize(amt_diag_var) +
        0.10 * normalize(amt_angular) +
        0.08 * normalize(amt_texture) +
        0.10 * normalize(trs_diag_var) +
        0.08 * normalize(trs_angular) +
        0.12 * (1 - normalize(cross_coupling)) +
        0.15 * (1 - normalize(recency_score)) +
        0.15 * normalize(np.abs(atv_trend)) +
        0.10 * (1 - normalize(amt_off_diag))
    )

    threshold = np.percentile(engagement_score, (1 - positive_ratio) * 100)
    labels = (engagement_score >= threshold).astype(int)

    print(f"Amount series shape: {amount_series.shape}")
    print(f"Transaction series shape: {trans_series.shape}")
    sparsity = (amount_series == 0).sum() / amount_series.size * 100
    print(f"Data sparsity: {sparsity:.2f}%")
    print(f"\nLabel: Customer Engagement Level")
    print(f"  - 0: High Engagement (stable, consistent, active customers)")
    print(f"  - 1: Low Engagement (irregular, declining, or inconsistent customers)")
    print(f"Low engagement ratio: {labels.mean()*100:.2f}%")
    print(f"\nEngagement factors (Multi-dimensional behavioral metrics):")
    print(f"  - Spending Pattern Stability:     12%  [temporal consistency]")
    print(f"  - Transaction Volatility:         10%  [frequency variation]")
    print(f"  - Local Pattern Irregularity:      8%  [short-term changes]")
    print(f"  - Count Pattern Stability:        10%  [transaction regularity]")
    print(f"  - Count Volatility:                8%  [count variation]")
    print(f"  - Cross-Modal Consistency:        12%  [amount-count alignment]")
    print(f"  - Recency Score:                  15%  [recent activity level]")
    print(f"  - Value Trend Volatility:         15%  [spending trajectory]")
    print(f"  - Temporal Coherence:             10%  [cross-period consistency]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    converter = RetailModalityConverter(Config, device)
    text_embeddings = converter.generate_text_descriptions(amount_series, trans_series)
    isa_gaf_images = converter.generate_isa_gaf_images(amount_series, trans_series)

    print("\nSaving modality data...")
    output_dir = Config.RETAIL_PROCESSED_DIR
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / 'amount_series.npy', amount_series)
    np.save(output_dir / 'trans_series.npy', trans_series)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', labels)

    print("\n" + "=" * 80)
    print(" Retail Modality Conversion Complete ".center(80, "="))
    print("=" * 80)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

def process_cdnow():

    from configs.config import Config
    from utils.seed import set_seed
    import torch
    from scipy.fft import fft
    from scipy.signal import find_peaks

    print("=" * 80)
    print(" Processing CDNOW Dataset ".center(80, "="))
    print("=" * 80)

    set_seed(Config.SEED)

    amount_path = Config.CDNOW_AMOUNT_CSV
    trans_path = Config.CDNOW_TRANS_CSV

    if not amount_path.exists() or not trans_path.exists():
        print("Error: CDNOW datasets not found. Please run process_cdnow.py first.")
        return

    df_amount = pd.read_csv(amount_path)
    df_trans = pd.read_csv(trans_path)

    amount_series = df_amount.iloc[:, :-1].values
    trans_series = df_trans.iloc[:, :-1].values

    n_samples, n_periods = amount_series.shape
    positive_ratio = 0.35

    print("\n" + "-" * 60)
    print("Computing Customer Engagement Score...")
    print("-" * 60)

    def compute_gaf_features(ts):
        if ts.sum() == 0:
            return 0, 0, 0
        ts_min, ts_max = ts.min(), ts.max()
        if ts_max - ts_min < 1e-8:
            ts_scaled = np.zeros_like(ts)
        else:
            ts_scaled = (ts - ts_min) / (ts_max - ts_min)
        ts_scaled = ts_scaled * 2 - 1
        ts_scaled = np.clip(ts_scaled, -1, 1)
        phi = np.arccos(ts_scaled)
        n = len(phi)
        gasf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gasf[i, j] = np.cos(phi[i] + phi[j])
        diag = np.diag(gasf)
        diag_var = np.var(diag)
        upper_tri = gasf[np.triu_indices(n, k=1)]
        off_diag_mean = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        symmetry = np.mean(np.abs(gasf - gasf.T))
        return diag_var, off_diag_mean, symmetry

    amt_diag_var = np.zeros(n_samples)
    amt_off_diag = np.zeros(n_samples)
    amt_symmetry = np.zeros(n_samples)
    trs_diag_var = np.zeros(n_samples)
    trs_off_diag = np.zeros(n_samples)
    trs_symmetry = np.zeros(n_samples)

    for i in range(n_samples):
        amt_diag_var[i], amt_off_diag[i], amt_symmetry[i] = compute_gaf_features(amount_series[i])
        trs_diag_var[i], trs_off_diag[i], trs_symmetry[i] = compute_gaf_features(trans_series[i])

    cross_pattern = np.zeros(n_samples)
    for i in range(n_samples):
        amt = amount_series[i]
        trs = trans_series[i]
        if amt.sum() > 0 and trs.sum() > 0:
            amt_norm = (amt - amt.min()) / (amt.max() - amt.min() + 1e-8)
            trs_norm = (trs - trs.min()) / (trs.max() - trs.min() + 1e-8)
            cross_corr = np.correlate(amt_norm - amt_norm.mean(), trs_norm - trs_norm.mean(), mode='full')
            cross_pattern[i] = np.max(np.abs(cross_corr)) / (n_periods + 1e-8)

    temporal_texture = np.zeros(n_samples)
    for i in range(n_samples):
        amt = amount_series[i]
        if amt.sum() > 0:
            diff1 = np.diff(amt)
            diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0])
            temporal_texture[i] = np.std(diff2) / (np.std(diff1) + 1e-8) if np.std(diff1) > 0 else 0

    sparsity_structure = np.zeros(n_samples)
    for i in range(n_samples):
        amt_mask = (amount_series[i] > 0).astype(int)
        trs_mask = (trans_series[i] > 0).astype(int)
        agreement = np.sum(amt_mask == trs_mask) / n_periods
        sparsity_structure[i] = agreement

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    engagement_score = (
        0.15 * normalize(amt_diag_var) +
        0.15 * normalize(amt_off_diag) +
        0.10 * normalize(trs_diag_var) +
        0.10 * normalize(trs_off_diag) +
        0.15 * (1 - normalize(cross_pattern)) +
        0.15 * normalize(temporal_texture) +
        0.20 * (1 - normalize(sparsity_structure))
    )

    threshold = np.percentile(engagement_score, (1 - positive_ratio) * 100)
    labels = (engagement_score >= threshold).astype(int)

    print(f"Amount series shape: {amount_series.shape}")
    print(f"Transaction series shape: {trans_series.shape}")
    sparsity = (amount_series == 0).sum() / amount_series.size * 100
    print(f"Data sparsity: {sparsity:.2f}%")
    print(f"\nLabel: Customer Engagement Level")
    print(f"  - 0: High Engagement (consistent behavioral patterns)")
    print(f"  - 1: Low Engagement (irregular behavioral patterns)")
    print(f"Low engagement ratio: {labels.mean()*100:.2f}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    converter = CDNOWModalityConverter(Config, device)
    text_embeddings = converter.generate_text_descriptions(amount_series, trans_series)
    isa_gaf_images = converter.generate_isa_gaf_images(amount_series, trans_series)

    print("\nSaving modality data...")
    output_dir = Config.CDNOW_PROCESSED_DIR
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / 'amount_series.npy', amount_series)
    np.save(output_dir / 'trans_series.npy', trans_series)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', labels)

    print("\n" + "=" * 80)
    print(" CDNOW Modality Conversion Complete ".center(80, "="))
    print("=" * 80)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

class SalesWeeklyModalityConverter:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.SALES_WEEKLY_IMAGE_SIZE

        model_path = self._find_text_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Text model not found in {config.PRETRAINED_DIR}"
            )

        print(f"Loading text embedding model from: {model_path}")
        self.text_model = SentenceTransformer(str(model_path), device=device)

    def _find_text_model_path(self):
        pretrained_dir = Path(self.config.PRETRAINED_DIR)

        if not pretrained_dir.exists():
            return None

        for model_dir in pretrained_dir.glob("models--*"):
            snapshot_dir = model_dir / "snapshots"
            if snapshot_dir.exists():
                for version_dir in snapshot_dir.iterdir():
                    if version_dir.is_dir():
                        return str(version_dir)

        return None

    def generate_text_descriptions(self, sales_series, product_codes):

        print("\nGenerating Sales Weekly text descriptions...")

        descriptions = []
        for i in tqdm(range(len(sales_series)), desc="Creating descriptions"):
            desc = self._create_sales_description(
                sales_series[i],
                product_codes[i] if product_codes is not None else f"P{i}"
            )
            descriptions.append(desc)

        print("Encoding text descriptions...")
        embeddings = self.text_model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=128,
            device=self.device
        )

        return embeddings

    def _create_sales_description(self, sales_ts, product_code):
        total_sales = np.sum(sales_ts)
        avg_sales = np.mean(sales_ts)
        max_sales = np.max(sales_ts)
        min_sales = np.min(sales_ts)

        trend_desc = "stable"
        if len(sales_ts) > 1:
            first_half = np.mean(sales_ts[:len(sales_ts)//2])
            second_half = np.mean(sales_ts[len(sales_ts)//2:])
            if second_half > first_half * 1.2:
                trend_desc = "upward trend"
            elif second_half < first_half * 0.8:
                trend_desc = "downward trend"

        volatility = np.std(sales_ts) / (np.mean(sales_ts) + 1e-6)
        vol_desc = "low volatility"
        if volatility > 0.6:
            vol_desc = "high volatility"
        elif volatility > 0.3:
            vol_desc = "medium volatility"

        peaks = np.where(sales_ts > np.percentile(sales_ts, 75))[0]
        seasonality_desc = "no clear seasonality"
        if len(peaks) >= 4:
            peak_intervals = np.diff(peaks)
            if np.std(peak_intervals) < 3:
                seasonality_desc = "regular seasonal pattern"
            else:
                seasonality_desc = "irregular peaks"

        active_weeks = np.sum(sales_ts > 0)
        activity_desc = f"active in {active_weeks} out of {len(sales_ts)} weeks"

        return (
            f"Product {product_code} with total sales of {total_sales:.0f} units over {len(sales_ts)} weeks. "
            f"Average weekly sales: {avg_sales:.1f}, range: {min_sales:.0f}-{max_sales:.0f}. "
            f"The sales pattern shows {trend_desc} with {vol_desc}. "
            f"{seasonality_desc.capitalize()}. {activity_desc.capitalize()}."
        )

    def generate_isa_gaf_images(self, sales_series):

        print("\nGenerating Sales Weekly ISA-GAF images...")

        images = []
        for i in tqdm(range(len(sales_series)), desc="Converting to images"):
            img = self._convert_to_isa_gaf(sales_series[i])
            images.append(img)

        return np.array(images)

    def _convert_to_isa_gaf(self, time_series):

        sparsity_mask = (time_series > 1e-6).astype(float)
        indices = np.arange(len(time_series))
        present_indices = indices[sparsity_mask == 1]
        present_values = time_series[sparsity_mask == 1]

        if len(present_values) < 2:
            interpolated_series = np.zeros_like(time_series)
        else:
            interpolated_series = self._adaptive_interpolate(
                indices, present_indices, present_values, sparsity_mask
            )

        gaf_transformer = GramianAngularField(
            image_size=self.image_size,
            method='summation'
        )

        if (interpolated_series.max() - interpolated_series.min()) > 1e-8:
            ts_scaled = (
                (interpolated_series - interpolated_series.min()) /
                (interpolated_series.max() - interpolated_series.min()) * 2 - 1
            )
        else:
            ts_scaled = np.zeros_like(interpolated_series)

        gaf_trend = gaf_transformer.fit_transform(ts_scaled.reshape(1, -1))[0]

        mask_scaled = (sparsity_mask - 0.5) * 2
        gaf_sparsity = gaf_transformer.fit_transform(mask_scaled.reshape(1, -1))[0]

        isa_gaf_image = np.stack([gaf_trend, gaf_sparsity], axis=0)
        return isa_gaf_image

    def _adaptive_interpolate(self, all_indices, present_indices, present_values, sparsity_mask):
        n = len(all_indices)
        result = np.zeros(n)

        if len(present_values) >= 4:
            try:
                spline = UnivariateSpline(present_indices, present_values, s=len(present_values)*0.1)
                result = spline(all_indices)
                result = np.clip(result, 0, present_values.max() * 1.5)
            except:
                interp_func = interp1d(present_indices, present_values, kind='linear',
                                       bounds_error=False, fill_value=0)
                result = interp_func(all_indices)
        else:
            interp_func = interp1d(present_indices, present_values, kind='linear',
                                   bounds_error=False, fill_value=0)
            result = interp_func(all_indices)

        sigma = max(1.0, n * 0.05)
        result = gaussian_filter1d(result, sigma=sigma)

        for i in present_indices.astype(int):
            if 0 <= i < n:
                result[i] = present_values[present_indices == i][0] if len(present_values[present_indices == i]) > 0 else result[i]

        return result

def process_instacart():
    from configs.config import Config
    from utils.seed import set_seed
    import torch

    print("=" * 80)
    print(" Processing Instacart Dataset ".center(80, "="))
    print("=" * 80)

    set_seed(Config.SEED)

    processed_dir = Config.INSTACART_PROCESSED_DIR

    order_path = processed_dir / 'order_count_series.npy'
    item_path = processed_dir / 'item_count_series.npy'
    labels_path = processed_dir / 'labels.npy'

    if not order_path.exists() or not item_path.exists():
        print("Error: Instacart time series not found. Please process raw data first.")
        return

    order_count_series = np.load(order_path)
    item_count_series = np.load(item_path)

    n_samples, n_periods = order_count_series.shape
    positive_ratio = 0.35

    print("\n" + "-" * 60)
    print("Computing User Reorder Likelihood Risk (GAF-based)...")
    print("-" * 60)

    def compute_gaf_features(ts):
        if ts.sum() == 0:
            return 0, 0, 0, 0
        ts_min, ts_max = ts.min(), ts.max()
        if ts_max - ts_min < 1e-8:
            ts_scaled = np.zeros_like(ts)
        else:
            ts_scaled = (ts - ts_min) / (ts_max - ts_min)
        ts_scaled = ts_scaled * 2 - 1
        ts_scaled = np.clip(ts_scaled, -1, 1)
        phi = np.arccos(ts_scaled)
        n = len(phi)
        gasf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gasf[i, j] = np.cos(phi[i] + phi[j])
        diag = np.diag(gasf)
        diag_var = np.var(diag)
        upper_tri = gasf[np.triu_indices(n, k=1)]
        off_diag_mean = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        phase_diff = np.diff(phi)
        angular_momentum = np.std(phase_diff) if len(phase_diff) > 0 else 0
        block_vars = []
        for i in range(n-1):
            for j in range(n-1):
                block = gasf[i:i+2, j:j+2]
                block_vars.append(np.var(block))
        local_texture = np.mean(block_vars) if block_vars else 0
        return diag_var, off_diag_mean, angular_momentum, local_texture

    ord_diag_var = np.zeros(n_samples)
    ord_off_diag = np.zeros(n_samples)
    ord_angular = np.zeros(n_samples)
    ord_texture = np.zeros(n_samples)
    itm_diag_var = np.zeros(n_samples)
    itm_off_diag = np.zeros(n_samples)
    itm_angular = np.zeros(n_samples)
    itm_texture = np.zeros(n_samples)

    for i in range(n_samples):
        ord_diag_var[i], ord_off_diag[i], ord_angular[i], ord_texture[i] = compute_gaf_features(order_count_series[i])
        itm_diag_var[i], itm_off_diag[i], itm_angular[i], itm_texture[i] = compute_gaf_features(item_count_series[i])

    cross_coupling = np.zeros(n_samples)
    for i in range(n_samples):
        ord = order_count_series[i]
        itm = item_count_series[i]
        if ord.sum() > 0 and itm.sum() > 0:
            ord_norm = (ord - ord.mean()) / (ord.std() + 1e-8)
            itm_norm = (itm - itm.mean()) / (itm.std() + 1e-8)
            cross_coupling[i] = np.abs(np.mean(ord_norm * itm_norm))

    recency_score = np.zeros(n_samples)
    for i in range(n_samples):
        weights = np.exp(np.linspace(-1, 0, n_periods))
        active_mask = (order_count_series[i] > 0).astype(float)
        recency_score[i] = np.sum(weights * active_mask) / np.sum(weights)

    basket_size_trend = np.zeros(n_samples)
    for i in range(n_samples):
        ord = order_count_series[i]
        itm = item_count_series[i]
        active_idx = np.where(ord > 0)[0]
        if len(active_idx) >= 2:
            basket_sizes = itm[active_idx] / (ord[active_idx] + 1e-8)
            mid = len(active_idx) // 2
            if mid > 0:
                early_bs = np.mean(basket_sizes[:mid])
                late_bs = np.mean(basket_sizes[mid:])
                basket_size_trend[i] = (late_bs - early_bs) / (early_bs + 1e-8)

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    risk_score = (
        0.12 * normalize(ord_diag_var) +
        0.10 * normalize(ord_angular) +
        0.08 * normalize(ord_texture) +
        0.10 * normalize(itm_diag_var) +
        0.08 * normalize(itm_angular) +
        0.12 * (1 - normalize(cross_coupling)) +
        0.15 * (1 - normalize(recency_score)) +
        0.15 * normalize(np.abs(basket_size_trend)) +
        0.10 * (1 - normalize(ord_off_diag))
    )

    threshold = np.percentile(risk_score, (1 - positive_ratio) * 100)
    labels = (risk_score >= threshold).astype(int)

    print(f"Order count series shape: {order_count_series.shape}")
    print(f"Item count series shape: {item_count_series.shape}")
    sparsity = (order_count_series == 0).sum() / order_count_series.size * 100
    print(f"Data sparsity: {sparsity:.2f}%")
    print(f"\nLabel: User Reorder Likelihood Risk")
    print(f"  - 0: Low Risk (likely to reorder)")
    print(f"  - 1: High Risk (unlikely to reorder)")
    print(f"High risk ratio: {labels.mean()*100:.2f}%")
    print(f"\nRisk factors (GAF-based + Business metrics):")
    print(f"  - GAF Diagonal Variance (order):  12%")
    print(f"  - GAF Angular Momentum (order):   10%")
    print(f"  - GAF Local Texture (order):       8%")
    print(f"  - GAF Diagonal Variance (item):   10%")
    print(f"  - GAF Angular Momentum (item):     8%")
    print(f"  - Cross-Modal Decoupling:         12%")
    print(f"  - Recency Score:                  15%")
    print(f"  - Basket Size Trend:              15%")
    print(f"  - Cross-Period Correlation:       10%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    converter = InstacartModalityConverter(Config, device)
    text_embeddings = converter.generate_text_descriptions(order_count_series, item_count_series)
    isa_gaf_images = converter.generate_isa_gaf_images(order_count_series, item_count_series)

    print("\nSaving modality data...")
    np.save(processed_dir / 'text_embeddings.npy', text_embeddings)
    np.save(processed_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(processed_dir / 'labels.npy', labels)

    print("\n" + "=" * 80)
    print(" Instacart Modality Conversion Complete ".center(80, "="))
    print("=" * 80)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")

class TaFengModalityConverter:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = config.TAFENG_IMAGE_SIZE

        model_path = self._find_text_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Text model not found in {config.PRETRAINED_DIR}"
            )

        print(f"Loading text embedding model from: {model_path}")
        self.text_model = SentenceTransformer(str(model_path), device=device)

    def _find_text_model_path(self):
        pretrained_dir = Path(self.config.PRETRAINED_DIR)

        if not pretrained_dir.exists():
            return None

        for model_dir in pretrained_dir.glob("models--*"):
            snapshot_dir = model_dir / "snapshots"
            if snapshot_dir.exists():
                for version_dir in snapshot_dir.iterdir():
                    if version_dir.is_dir():
                        return str(version_dir)

        return None

    def generate_text_descriptions(self, amount_series, trans_series, age_groups=None):

        print("\nGenerating Ta-Feng text descriptions...")

        descriptions = []
        for i in tqdm(range(len(amount_series)), desc="Creating descriptions"):
            age = age_groups[i] if age_groups is not None else "Unknown"
            desc = self._create_tafeng_description(
                amount_series[i],
                trans_series[i],
                age
            )
            descriptions.append(desc)

        print("Encoding text descriptions...")
        embeddings = self.text_model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=128,
            device=self.device
        )

        return embeddings

    def _create_tafeng_description(self, amount_ts, trans_ts, age_group):

        total_amount = np.sum(amount_ts)
        total_trans = np.sum(trans_ts)

        if total_trans > 0:
            avg_per_trans = total_amount / total_trans
        else:
            avg_per_trans = 0

        trend_desc = "stable"
        if len(amount_ts) > 1 and amount_ts[0] > 0:
            if amount_ts[-1] > amount_ts[0] * 1.5:
                trend_desc = "upward trend"
            elif amount_ts[-1] < amount_ts[0] * 0.5:
                trend_desc = "downward trend"

        volatility = np.std(amount_ts) / (np.mean(amount_ts) + 1e-6)
        vol_desc = "low volatility"
        if volatility > 0.8:
            vol_desc = "high volatility"
        elif volatility > 0.4:
            vol_desc = "medium volatility"

        freq_desc = "low frequency"
        if total_trans > 50:
            freq_desc = "high frequency"
        elif total_trans > 20:
            freq_desc = "medium frequency"

        active_months = np.sum(amount_ts > 0)
        activity_desc = f"active in {active_months} out of {len(amount_ts)} months"

        return (
            f"A grocery customer aged {age_group} with total spending of ${total_amount:.2f} over {len(amount_ts)} months, "
            f"making {int(total_trans)} transactions. The spending pattern shows "
            f"{trend_desc} with {vol_desc}. The shopping frequency is {freq_desc} "
            f"with an average of ${avg_per_trans:.2f} per transaction. {activity_desc}."
        )

    def generate_isa_gaf_images(self, amount_series, trans_series):

        print("\nGenerating Ta-Feng ISA-GAF images...")

        images = []
        for i in tqdm(range(len(amount_series)), desc="Converting to images"):
            img = self._convert_to_tafeng_isa_gaf(
                amount_series[i],
                trans_series[i]
            )
            images.append(img)

        return np.array(images)

    def _convert_to_tafeng_isa_gaf(self, amount_ts, trans_ts):

        amount_trend, amount_sparsity = self._single_ts_to_isa_gaf(amount_ts)
        trans_trend, trans_sparsity = self._single_ts_to_isa_gaf(trans_ts)

        isa_gaf_image = np.stack([
            amount_trend,
            amount_sparsity,
            trans_trend,
            trans_sparsity
        ], axis=0)
        return isa_gaf_image

    def _single_ts_to_isa_gaf(self, time_series):

        sparsity_mask = (time_series > 1e-6).astype(float)
        indices = np.arange(len(time_series))
        present_indices = indices[sparsity_mask == 1]
        present_values = time_series[sparsity_mask == 1]

        if len(present_values) < 2:
            interpolated_series = np.zeros_like(time_series)
        else:
            interpolated_series = self._adaptive_interpolate(
                indices, present_indices, present_values, sparsity_mask
            )

        gaf_transformer = GramianAngularField(
            image_size=self.image_size,
            method='summation'
        )

        if (interpolated_series.max() - interpolated_series.min()) > 1e-8:
            ts_scaled = (
                (interpolated_series - interpolated_series.min()) /
                (interpolated_series.max() - interpolated_series.min()) * 2 - 1
            )
        else:
            ts_scaled = np.zeros_like(interpolated_series)

        gaf_trend = gaf_transformer.fit_transform(ts_scaled.reshape(1, -1))[0]

        mask_scaled = (sparsity_mask - 0.5) * 2
        gaf_sparsity = gaf_transformer.fit_transform(mask_scaled.reshape(1, -1))[0]

        return gaf_trend, gaf_sparsity

    def _adaptive_interpolate(self, all_indices, present_indices, present_values, sparsity_mask):
        n = len(all_indices)
        result = np.zeros(n)

        if len(present_values) >= 4:
            try:
                spline = UnivariateSpline(present_indices, present_values, s=len(present_values)*0.1)
                result = spline(all_indices)
                result = np.clip(result, 0, present_values.max() * 1.5)
            except:
                interp_func = interp1d(present_indices, present_values, kind='linear',
                                       bounds_error=False, fill_value=0)
                result = interp_func(all_indices)
        else:
            interp_func = interp1d(present_indices, present_values, kind='linear',
                                   bounds_error=False, fill_value=0)
            result = interp_func(all_indices)

        sigma = max(1.0, n * 0.05)
        result = gaussian_filter1d(result, sigma=sigma)

        for i in present_indices.astype(int):
            if 0 <= i < n:
                result[i] = present_values[present_indices == i][0] if len(present_values[present_indices == i]) > 0 else result[i]

        return result

def process_tafeng(positive_ratio=0.35):

    from configs.config import Config
    from utils.seed import set_seed
    import torch

    print("=" * 80)
    print(" Processing Ta-Feng Dataset ".center(80, "="))
    print("=" * 80)

    set_seed(Config.SEED)

    tafeng_path = Config.TAFENG_CSV

    if not tafeng_path.exists():
        print(f"Error: Ta-Feng dataset not found at {tafeng_path}")
        return

    print("\nLoading Ta-Feng transaction data...")
    df = pd.read_csv(tafeng_path)

    df['TRANSACTION_DT'] = pd.to_datetime(df['TRANSACTION_DT'], format='%m/%d/%Y')
    df['MONTH'] = df['TRANSACTION_DT'].dt.to_period('M')

    months = sorted(df['MONTH'].unique())
    n_months = len(months)
    month_to_idx = {m: i for i, m in enumerate(months)}

    print(f"Date range: {months[0]} to {months[-1]} ({n_months} months)")

    print("Aggregating transactions by customer and month...")
    customer_monthly = df.groupby(['CUSTOMER_ID', 'MONTH']).agg({
        'SALES_PRICE': 'sum',
        'AMOUNT': 'sum',
        'PRODUCT_ID': 'count'
    }).reset_index()
    customer_monthly.columns = ['CUSTOMER_ID', 'MONTH', 'SPENDING', 'QUANTITY', 'TRANS_COUNT']

    customer_ages = df.groupby('CUSTOMER_ID')['AGE_GROUP'].first().to_dict()

    customers = customer_monthly['CUSTOMER_ID'].unique()
    n_customers = len(customers)

    print(f"Total customers: {n_customers}")

    amount_series = np.zeros((n_customers, n_months))
    trans_series = np.zeros((n_customers, n_months))
    age_groups = []

    for i, cust in enumerate(tqdm(customers, desc="Creating time series")):
        cust_data = customer_monthly[customer_monthly['CUSTOMER_ID'] == cust]
        for _, row in cust_data.iterrows():
            month_idx = month_to_idx[row['MONTH']]
            amount_series[i, month_idx] = row['SPENDING']
            trans_series[i, month_idx] = row['TRANS_COUNT']
        age_groups.append(customer_ages.get(cust, 'Unknown'))

    n_samples, n_periods = amount_series.shape

    print(f"\nAmount series shape: {amount_series.shape}")
    print(f"Transaction series shape: {trans_series.shape}")

    print("\n" + "-" * 60)
    print("Computing Customer Purchase Behavior Risk (GAF-based)...")
    print("-" * 60)

    def compute_gaf_features(ts):
        if ts.sum() == 0:
            return 0, 0, 0, 0
        ts_min, ts_max = ts.min(), ts.max()
        if ts_max - ts_min < 1e-8:
            ts_scaled = np.zeros_like(ts)
        else:
            ts_scaled = (ts - ts_min) / (ts_max - ts_min)
        ts_scaled = ts_scaled * 2 - 1
        ts_scaled = np.clip(ts_scaled, -1, 1)
        phi = np.arccos(ts_scaled)
        n = len(phi)
        gasf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gasf[i, j] = np.cos(phi[i] + phi[j])
        diag = np.diag(gasf)
        diag_var = np.var(diag)
        upper_tri = gasf[np.triu_indices(n, k=1)]
        off_diag_mean = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        phase_diff = np.diff(phi)
        angular_momentum = np.std(phase_diff) if len(phase_diff) > 0 else 0
        block_vars = []
        for i in range(n-1):
            for j in range(n-1):
                block = gasf[i:i+2, j:j+2]
                block_vars.append(np.var(block))
        local_texture = np.mean(block_vars) if block_vars else 0
        return diag_var, off_diag_mean, angular_momentum, local_texture

    amt_diag_var = np.zeros(n_samples)
    amt_off_diag = np.zeros(n_samples)
    amt_angular = np.zeros(n_samples)
    amt_texture = np.zeros(n_samples)
    trs_diag_var = np.zeros(n_samples)
    trs_off_diag = np.zeros(n_samples)
    trs_angular = np.zeros(n_samples)
    trs_texture = np.zeros(n_samples)

    for i in range(n_samples):
        amt_diag_var[i], amt_off_diag[i], amt_angular[i], amt_texture[i] = compute_gaf_features(amount_series[i])
        trs_diag_var[i], trs_off_diag[i], trs_angular[i], trs_texture[i] = compute_gaf_features(trans_series[i])

    cross_coupling = np.zeros(n_samples)
    for i in range(n_samples):
        amt = amount_series[i]
        trs = trans_series[i]
        if amt.sum() > 0 and trs.sum() > 0:
            amt_norm = (amt - amt.mean()) / (amt.std() + 1e-8)
            trs_norm = (trs - trs.mean()) / (trs.std() + 1e-8)
            cross_coupling[i] = np.abs(np.mean(amt_norm * trs_norm))

    recency_score = np.zeros(n_samples)
    for i in range(n_samples):
        weights = np.exp(np.linspace(-1, 0, n_periods))
        active_mask = (amount_series[i] > 0).astype(float)
        recency_score[i] = np.sum(weights * active_mask) / np.sum(weights)

    atv_trend = np.zeros(n_samples)
    for i in range(n_samples):
        amt = amount_series[i]
        trs = trans_series[i]
        active_idx = np.where(trs > 0)[0]
        if len(active_idx) >= 2:
            atvs = amt[active_idx] / (trs[active_idx] + 1e-8)
            mid = len(active_idx) // 2
            if mid > 0:
                early_atv = np.mean(atvs[:mid])
                late_atv = np.mean(atvs[mid:])
                atv_trend[i] = (late_atv - early_atv) / (early_atv + 1e-8)

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    risk_score = (
        0.12 * normalize(amt_diag_var) +
        0.10 * normalize(amt_angular) +
        0.08 * normalize(amt_texture) +
        0.10 * normalize(trs_diag_var) +
        0.08 * normalize(trs_angular) +
        0.12 * (1 - normalize(cross_coupling)) +
        0.15 * (1 - normalize(recency_score)) +
        0.15 * normalize(np.abs(atv_trend)) +
        0.10 * (1 - normalize(amt_off_diag))
    )

    threshold = np.percentile(risk_score, (1 - positive_ratio) * 100)
    labels = (risk_score >= threshold).astype(int)

    sparsity = (amount_series == 0).sum() / amount_series.size * 100
    print(f"Data sparsity: {sparsity:.2f}%")
    print(f"\nLabel: Customer Purchase Behavior Risk")
    print(f"  - 0: Low Risk (stable engagement, consistent patterns)")
    print(f"  - 1: High Risk (erratic behavior, declining engagement)")
    print(f"High risk ratio: {labels.mean()*100:.2f}%")
    print(f"\nRisk factors (GAF-based + Business metrics):")
    print(f"  - GAF Diagonal Variance (amt):    12%  [value stability]")
    print(f"  - GAF Angular Momentum (amt):     10%  [phase volatility]")
    print(f"  - GAF Local Texture (amt):         8%  [local irregularity]")
    print(f"  - GAF Diagonal Variance (trs):    10%  [count stability]")
    print(f"  - GAF Angular Momentum (trs):      8%  [phase volatility]")
    print(f"  - Cross-Modal Decoupling:         12%  [amt-trs mismatch]")
    print(f"  - Recency Score:                  15%  [recent activity]")
    print(f"  - ATV Trend Volatility:           15%  [spending change]")
    print(f"  - Cross-Period Correlation:       10%  [temporal coherence]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    converter = TaFengModalityConverter(Config, device)
    text_embeddings = converter.generate_text_descriptions(amount_series, trans_series, age_groups)
    isa_gaf_images = converter.generate_isa_gaf_images(amount_series, trans_series)

    print("\nSaving modality data to DATA_DIR (no processed folder)...")
    output_dir = Config.TAFENG_DATA_DIR

    np.save(output_dir / 'tafeng_amount_series.npy', amount_series)
    np.save(output_dir / 'tafeng_trans_series.npy', trans_series)
    np.save(output_dir / 'tafeng_text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'tafeng_isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'tafeng_labels.npy', labels)

    pd.DataFrame({
        'CUSTOMER_ID': customers,
        'AGE_GROUP': age_groups
    }).to_csv(output_dir / 'tafeng_customers.csv', index=False)

    print("\n" + "=" * 80)
    print(" Ta-Feng Modality Conversion Complete ".center(80, "="))
    print("=" * 80)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")
    print(f"Labels shape: {labels.shape}")

def process_sales_weekly(positive_ratio=0.35):

    from configs.config import Config
    from utils.seed import set_seed
    import torch

    print("=" * 80)
    print(" Processing Sales Weekly Dataset ".center(80, "="))
    print("=" * 80)

    set_seed(Config.SEED)

    sales_path = Config.SALES_WEEKLY_CSV

    if not sales_path.exists():
        print(f"Error: Sales Weekly dataset not found at {sales_path}")
        return

    df = pd.read_csv(sales_path)

    week_cols = [f'W{i}' for i in range(52)]
    sales_series = df[week_cols].values.astype(float)
    product_codes = df['Product_Code'].values

    n_samples, n_periods = sales_series.shape

    print(f"\nDataset loaded: {n_samples} products, {n_periods} weeks")

    print("\n" + "-" * 60)
    print("Computing Sales Volatility Risk (GAF-based)...")
    print("-" * 60)

    def compute_gaf_features(ts):
        if ts.sum() == 0:
            return 0, 0, 0, 0
        ts_min, ts_max = ts.min(), ts.max()
        if ts_max - ts_min < 1e-8:
            ts_scaled = np.zeros_like(ts)
        else:
            ts_scaled = (ts - ts_min) / (ts_max - ts_min)
        ts_scaled = ts_scaled * 2 - 1
        ts_scaled = np.clip(ts_scaled, -1, 1)
        phi = np.arccos(ts_scaled)
        n = len(phi)
        gasf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gasf[i, j] = np.cos(phi[i] + phi[j])
        diag = np.diag(gasf)
        diag_var = np.var(diag)
        upper_tri = gasf[np.triu_indices(n, k=1)]
        off_diag_mean = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        phase_diff = np.diff(phi)
        angular_momentum = np.std(phase_diff) if len(phase_diff) > 0 else 0
        block_vars = []
        for i in range(n-1):
            for j in range(n-1):
                block = gasf[i:i+2, j:j+2]
                block_vars.append(np.var(block))
        local_texture = np.mean(block_vars) if block_vars else 0
        return diag_var, off_diag_mean, angular_momentum, local_texture

    diag_var = np.zeros(n_samples)
    off_diag = np.zeros(n_samples)
    angular = np.zeros(n_samples)
    texture = np.zeros(n_samples)

    for i in range(n_samples):
        diag_var[i], off_diag[i], angular[i], texture[i] = compute_gaf_features(sales_series[i])

    trend_stability = np.zeros(n_samples)
    for i in range(n_samples):
        ts = sales_series[i]
        if ts.sum() > 0:
            x = np.arange(len(ts))
            slope = np.polyfit(x, ts, 1)[0]
            residuals = ts - (slope * x + np.mean(ts))
            trend_stability[i] = np.std(residuals) / (np.mean(ts) + 1e-8)

    seasonality_score = np.zeros(n_samples)
    for i in range(n_samples):
        ts = sales_series[i]
        if ts.std() > 1e-8:
            ts_norm = (ts - ts.mean()) / ts.std()
            autocorrs = []
            for lag in [4, 13, 26]:
                if len(ts) > lag:
                    autocorr = np.corrcoef(ts_norm[:-lag], ts_norm[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorrs.append(abs(autocorr))
            seasonality_score[i] = max(autocorrs) if autocorrs else 0

    spike_score = np.zeros(n_samples)
    for i in range(n_samples):
        ts = sales_series[i]
        if ts.std() > 1e-8:
            z_scores = np.abs((ts - ts.mean()) / ts.std())
            spike_score[i] = np.sum(z_scores > 2) / len(ts)

    zero_ratio = np.zeros(n_samples)
    for i in range(n_samples):
        zero_ratio[i] = np.sum(sales_series[i] == 0) / n_periods

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    volatility_risk = (
        0.15 * normalize(diag_var) +
        0.12 * normalize(angular) +
        0.10 * normalize(texture) +
        0.15 * normalize(trend_stability) +
        0.12 * (1 - normalize(seasonality_score)) +
        0.15 * normalize(spike_score) +
        0.11 * normalize(zero_ratio) +
        0.10 * (1 - normalize(off_diag))
    )

    threshold = np.percentile(volatility_risk, (1 - positive_ratio) * 100)
    labels = (volatility_risk >= threshold).astype(int)

    print(f"Sales series shape: {sales_series.shape}")
    sparsity = (sales_series == 0).sum() / sales_series.size * 100
    print(f"Data sparsity (zero sales): {sparsity:.2f}%")
    print(f"\nLabel: Sales Volatility Risk")
    print(f"  - 0: Low Risk (stable, predictable sales pattern)")
    print(f"  - 1: High Risk (volatile, hard to forecast)")
    print(f"High risk ratio: {labels.mean()*100:.2f}%")
    print(f"\nRisk factors (GAF-based + Business metrics):")
    print(f"  - GAF Diagonal Variance:          15%  [value stability]")
    print(f"  - GAF Angular Momentum:           12%  [phase volatility]")
    print(f"  - GAF Local Texture:              10%  [local irregularity]")
    print(f"  - Trend Stability:                15%  [trend deviation]")
    print(f"  - Seasonality Weakness:           12%  [predictability]")
    print(f"  - Demand Spike Score:             15%  [outlier frequency]")
    print(f"  - Zero Sales Ratio:               11%  [stockout/inactive]")
    print(f"  - Cross-Period Correlation:       10%  [temporal coherence]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    converter = SalesWeeklyModalityConverter(Config, device)
    text_embeddings = converter.generate_text_descriptions(sales_series, product_codes)
    isa_gaf_images = converter.generate_isa_gaf_images(sales_series)

    print("\nSaving modality data...")
    output_dir = Config.SALES_WEEKLY_PROCESSED_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    np.save(output_dir / 'sales_series.npy', sales_series)
    np.save(output_dir / 'text_embeddings.npy', text_embeddings)
    np.save(output_dir / 'isa_gaf_images.npy', isa_gaf_images)
    np.save(output_dir / 'labels.npy', labels)

    pd.DataFrame({'Product_Code': product_codes}).to_csv(
        output_dir / 'product_codes.csv', index=False
    )

    print("\n" + "=" * 80)
    print(" Sales Weekly Modality Conversion Complete ".center(80, "="))
    print("=" * 80)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"ISA-GAF images shape: {isa_gaf_images.shape}")
    print(f"Labels shape: {labels.shape}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Modality Converter for HMSPAR datasets')
    parser.add_argument(
        '--dataset',
        type=str,
        default='merchant',
        choices=['merchant', 'retail', 'cdnow', 'instacart', 'sales_weekly', 'tafeng', 'all'],
        help='Dataset to process (default: merchant)'
    )
    parser.add_argument(
        '--positive-ratio',
        type=float,
        default=0.35,
        help='Positive ratio for retail dataset (default: 0.35)'
    )

    args = parser.parse_args()

    if args.dataset == 'merchant':
        process_merchant()
    elif args.dataset == 'retail':
        process_retail(positive_ratio=args.positive_ratio)
    elif args.dataset == 'cdnow':
        process_cdnow()
    elif args.dataset == 'instacart':
        process_instacart()
    elif args.dataset == 'sales_weekly':
        process_sales_weekly(positive_ratio=args.positive_ratio)
    elif args.dataset == 'tafeng':
        process_tafeng(positive_ratio=args.positive_ratio)
    elif args.dataset == 'all':
        print("Processing all datasets...\n")
        process_merchant()
        print("\n")
        process_retail()
        print("\n")
        process_cdnow()
        print("\n")
        process_instacart()
        print("\n")
        process_sales_weekly()
        print("\n")
        process_tafeng()

if __name__ == "__main__":
    main()
