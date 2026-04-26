\
\

import numpy as np
import torch
from torch.utils.data import Dataset

class MerchantDataset(Dataset):
    """
    PyTorch Dataset for merchant transaction data
    Supports multi-modal inputs: time series, images, and text
    """

    def __init__(self, dataframe, ts_cols, text_embeddings, isa_gaf_images):
        """
        Args:
            dataframe (pd.DataFrame): Merchant data
            ts_cols (list): Time series column names
            text_embeddings (np.ndarray): Pre-computed text embeddings
            isa_gaf_images (np.ndarray): Pre-computed ISA-GAF images
        """
        self.df = dataframe.reset_index(drop=True)
        self.ts_cols = ts_cols
        self.text_embeddings = text_embeddings
        self.isa_gaf_images = isa_gaf_images
        self.labels = dataframe['is_anomalous'].values

    def __len__(self):
        """Return dataset size"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single sample

        Args:
            idx (int): Sample index

        Returns:
            dict: Dictionary containing all modalities and label
        """

        ts_data = self.df.iloc[idx][self.ts_cols].values.astype(np.float32)
        ts_data = np.log1p(ts_data)

        if np.std(ts_data) > 1e-6:
            ts_data = (ts_data - np.mean(ts_data)) / np.std(ts_data)

        image = self.isa_gaf_images[idx]

        text_embedding = self.text_embeddings[idx]

        label = self.labels[idx]

        return {
            'ts_data': torch.tensor(ts_data, dtype=torch.float32).unsqueeze(0),
            'image': torch.tensor(image, dtype=torch.float32),
            'text_embedding': torch.tensor(text_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class RetailDataset(Dataset):
    """PyTorch Dataset for retail customer data with dual time series features"""

    def __init__(self, amount_series, trans_series, text_embeddings, isa_gaf_images, labels):
        """
        Args:
            amount_series (np.ndarray): Transaction amount time series [N, T]
            trans_series (np.ndarray): Transaction count time series [N, T]
            text_embeddings (np.ndarray): Pre-computed text embeddings [N, D]
            isa_gaf_images (np.ndarray): Pre-computed ISA-GAF images [N, C, H, W]
            labels (np.ndarray): Binary labels [N]
        """
        self.amount_series = amount_series
        self.trans_series = trans_series
        self.text_embeddings = text_embeddings
        self.isa_gaf_images = isa_gaf_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a single sample with dual time series features

        Returns:
            dict: Dictionary containing dual ts_data, image, text_embedding, and label
        """
        amount_ts = self.amount_series[idx].astype(np.float32)
        trans_ts = self.trans_series[idx].astype(np.float32)

        amount_ts = np.log1p(amount_ts)
        trans_ts = np.sqrt(trans_ts)

        if np.std(amount_ts) > 1e-6:
            amount_ts = (amount_ts - np.mean(amount_ts)) / np.std(amount_ts)

        if np.std(trans_ts) > 1e-6:
            trans_ts = (trans_ts - np.mean(trans_ts)) / np.std(trans_ts)

        ts_data = np.stack([amount_ts, trans_ts], axis=0)

        image = self.isa_gaf_images[idx]
        text_embedding = self.text_embeddings[idx]
        label = self.labels[idx]

        return {
            'ts_data': torch.tensor(ts_data, dtype=torch.float32),
            'image': torch.tensor(image, dtype=torch.float32),
            'text_embedding': torch.tensor(text_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class CDNOWDataset(Dataset):
    """PyTorch Dataset for CDNOW customer data with dual time series features"""

    def __init__(self, amount_series, trans_series, text_embeddings, isa_gaf_images, labels):
        """
        Args:
            amount_series (np.ndarray): Transaction amount time series [N, T]
            trans_series (np.ndarray): Transaction count time series [N, T]
            text_embeddings (np.ndarray): Pre-computed text embeddings [N, D]
            isa_gaf_images (np.ndarray): Pre-computed ISA-GAF images [N, C, H, W]
            labels (np.ndarray): Binary labels [N]
        """
        self.amount_series = amount_series
        self.trans_series = trans_series
        self.text_embeddings = text_embeddings
        self.isa_gaf_images = isa_gaf_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a single sample with dual time series features

        Returns:
            dict: Dictionary containing dual ts_data, image, text_embedding, and label
        """
        amount_ts = self.amount_series[idx].astype(np.float32)
        trans_ts = self.trans_series[idx].astype(np.float32)

        amount_ts = np.log1p(amount_ts)
        trans_ts = np.sqrt(trans_ts)

        if np.std(amount_ts) > 1e-6:
            amount_ts = (amount_ts - np.mean(amount_ts)) / np.std(amount_ts)

        if np.std(trans_ts) > 1e-6:
            trans_ts = (trans_ts - np.mean(trans_ts)) / np.std(trans_ts)

        ts_data = np.stack([amount_ts, trans_ts], axis=0)

        image = self.isa_gaf_images[idx]
        text_embedding = self.text_embeddings[idx]
        label = self.labels[idx]

        return {
            'ts_data': torch.tensor(ts_data, dtype=torch.float32),
            'image': torch.tensor(image, dtype=torch.float32),
            'text_embedding': torch.tensor(text_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class InstacartDataset(Dataset):
    """PyTorch Dataset for Instacart user data with dual time series features"""

    def __init__(self, order_count_series, item_count_series, text_embeddings, isa_gaf_images, labels):
        """
        Args:
            order_count_series (np.ndarray): Weekly order count time series [N, T]
            item_count_series (np.ndarray): Weekly item count time series [N, T]
            text_embeddings (np.ndarray): Pre-computed text embeddings [N, D]
            isa_gaf_images (np.ndarray): Pre-computed ISA-GAF images [N, C, H, W]
            labels (np.ndarray): Binary labels [N]
        """
        self.order_count_series = order_count_series
        self.item_count_series = item_count_series
        self.text_embeddings = text_embeddings
        self.isa_gaf_images = isa_gaf_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a single sample with dual time series features

        Returns:
            dict: Dictionary containing dual ts_data, image, text_embedding, and label
        """
        order_ts = self.order_count_series[idx].astype(np.float32)
        item_ts = self.item_count_series[idx].astype(np.float32)

        order_ts = np.log1p(order_ts)
        item_ts = np.log1p(item_ts)

        if np.std(order_ts) > 1e-6:
            order_ts = (order_ts - np.mean(order_ts)) / np.std(order_ts)

        if np.std(item_ts) > 1e-6:
            item_ts = (item_ts - np.mean(item_ts)) / np.std(item_ts)

        ts_data = np.stack([order_ts, item_ts], axis=0)

        image = self.isa_gaf_images[idx]
        text_embedding = self.text_embeddings[idx]
        label = self.labels[idx]

        return {
            'ts_data': torch.tensor(ts_data, dtype=torch.float32),
            'image': torch.tensor(image, dtype=torch.float32),
            'text_embedding': torch.tensor(text_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class SalesWeeklyDataset(Dataset):
    """PyTorch Dataset for Sales Weekly product data with single time series"""

    def __init__(self, sales_series, text_embeddings, isa_gaf_images, labels):
        """
        Args:
            sales_series (np.ndarray): Weekly sales time series [N, T]
            text_embeddings (np.ndarray): Pre-computed text embeddings [N, D]
            isa_gaf_images (np.ndarray): Pre-computed ISA-GAF images [N, C, H, W]
            labels (np.ndarray): Binary labels [N]
        """
        self.sales_series = sales_series
        self.text_embeddings = text_embeddings
        self.isa_gaf_images = isa_gaf_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a single sample with single time series

        Returns:
            dict: Dictionary containing ts_data, image, text_embedding, and label
        """
        sales_ts = self.sales_series[idx].astype(np.float32)

        sales_ts = np.log1p(sales_ts)

        if np.std(sales_ts) > 1e-6:
            sales_ts = (sales_ts - np.mean(sales_ts)) / np.std(sales_ts)

        ts_data = sales_ts.reshape(1, -1)

        image = self.isa_gaf_images[idx]
        text_embedding = self.text_embeddings[idx]
        label = self.labels[idx]

        return {
            'ts_data': torch.tensor(ts_data, dtype=torch.float32),
            'image': torch.tensor(image, dtype=torch.float32),
            'text_embedding': torch.tensor(text_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class TaFengDataset(Dataset):
    """PyTorch Dataset for Ta-Feng customer data with dual time series features"""

    def __init__(self, amount_series, trans_series, text_embeddings, isa_gaf_images, labels):
        """
        Args:
            amount_series (np.ndarray): Monthly spending time series [N, T]
            trans_series (np.ndarray): Monthly transaction count time series [N, T]
            text_embeddings (np.ndarray): Pre-computed text embeddings [N, D]
            isa_gaf_images (np.ndarray): Pre-computed ISA-GAF images [N, C, H, W]
            labels (np.ndarray): Binary labels [N]
        """
        self.amount_series = amount_series
        self.trans_series = trans_series
        self.text_embeddings = text_embeddings
        self.isa_gaf_images = isa_gaf_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a single sample with dual time series features

        Returns:
            dict: Dictionary containing dual ts_data, image, text_embedding, and label
        """
        amount_ts = self.amount_series[idx].astype(np.float32)
        trans_ts = self.trans_series[idx].astype(np.float32)

        amount_ts = np.log1p(amount_ts)
        trans_ts = np.sqrt(trans_ts)

        if np.std(amount_ts) > 1e-6:
            amount_ts = (amount_ts - np.mean(amount_ts)) / np.std(amount_ts)

        if np.std(trans_ts) > 1e-6:
            trans_ts = (trans_ts - np.mean(trans_ts)) / np.std(trans_ts)

        ts_data = np.stack([amount_ts, trans_ts], axis=0)

        image = self.isa_gaf_images[idx]
        text_embedding = self.text_embeddings[idx]
        label = self.labels[idx]

        return {
            'ts_data': torch.tensor(ts_data, dtype=torch.float32),
            'image': torch.tensor(image, dtype=torch.float32),
            'text_embedding': torch.tensor(text_embedding, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }
