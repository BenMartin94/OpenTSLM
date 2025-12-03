"""
EnergyQADataset.py
------------------
PyTorch-style QA dataset for building energy consumption analysis.

This module defines the EnergyQADataset class for time series question-answering
on building energy consumption patterns.
"""
import pandas as pd
import torch
from datasets import Dataset
from typing import List, Tuple, Literal
from time_series_datasets.QADataset import QADataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt


class EnergyQADataset(QADataset):
    """
    Energy Question-Answer Dataset for building energy consumption analysis.
    
    This dataset loads energy time series data with questions and answers
    about consumption patterns, peaks, trends, and stability.
    """
    
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function = None,
        parquet_file: str = "data/energy_dataset.parquet"
    ):
        """
        Initialize Energy QA Dataset.
        
        Args:
            split: "train", "validation", or "test"
            EOS_TOKEN: End-of-sequence token
            format_sample_str: Whether to format samples as strings
            time_series_format_function: Function to format time series
            parquet_file: Path to the parquet file with energy data
        """
        self.parquet_file = parquet_file
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)
    
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the energy dataset and split by the 'split' column.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print(f"Loading energy dataset from {self.parquet_file}...")
        df = pd.read_parquet(self.parquet_file)
        
        # Filter by split
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'validation'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        
        # Convert to Dataset objects
        train = Dataset.from_pandas(train_df)
        val = Dataset.from_pandas(val_df)
        test = Dataset.from_pandas(test_df)
        
        print(f"Energy dataset splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def _get_answer(self, row) -> str:
        """
        Get the answer from the row.
        
        Args:
            row: Dataset row
            
        Returns:
            The answer string
        """
        return str(row['answer'])
    
    def _get_pre_prompt(self, row) -> str:
        """
        Get the pre-prompt for energy analysis.
        
        Args:
            row: Dataset row
            
        Returns:
            Pre-prompt text
        """
        return "You are an expert in building energy consumption analysis."
    
    def _get_post_prompt(self, row) -> str:
        """
        Get the post-prompt with the specific question.
        
        Args:
            row: Dataset row
            
        Returns:
            Post-prompt text with question
        """
        question = row['question']
        return f"{question}"
    
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into TextTimeSeriesPrompt format with normalization.
        
        Args:
            row: Dataset row containing time_series data
            
        Returns:
            List of TextTimeSeriesPrompt objects
        """
        # Get the time series (already a list of floats from parquet)
        series_data = row['time_series']
        
        # Convert to tensor
        series = torch.tensor(series_data, dtype=torch.float32)
        
        # Normalize (required!)
        mean = series.mean()
        std = series.std()
        epsilon = 1e-8
        
        if std > epsilon:
            normalized_series = (series - mean) / std
        else:
            normalized_series = series - mean
        
        # Create text label
        text_label = f"Building energy consumption time series (mean={mean:.2f} kWh, std={std:.2f} kWh):"
        
        return [TextTimeSeriesPrompt(text_label, normalized_series.tolist())]
    
    def _format_sample(self, row):
        """Override to preserve additional metadata."""
        sample = super()._format_sample(row)
        
        # Add metadata fields if they exist
        if 'id' in row:
            sample['id'] = row['id']
        if 'bldg_id' in row:
            sample['bldg_id'] = row['bldg_id']
        if 'start_time' in row:
            sample['start_time'] = row['start_time']
        if 'total_energy' in row:
            sample['total_energy'] = row['total_energy']
        
        return sample


# Test the dataset
if __name__ == "__main__":
    # Create dataset instances
    train_dataset = EnergyQADataset("train", EOS_TOKEN="")
    val_dataset = EnergyQADataset("validation", EOS_TOKEN="")
    test_dataset = EnergyQADataset("test", EOS_TOKEN="")
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Check a sample
    if len(train_dataset) > 0:
        print(f"\n{'='*80}")
        print("Sample from training set:")
        print('='*80)
        sample = train_dataset[0]
        print(f"Keys: {sample.keys()}")
        print(f"\nPre-prompt: {sample['pre_prompt']}")
        print(f"\nPost-prompt: {sample['post_prompt']}")
        print(f"\nAnswer: {sample['answer'][:150]}{'...' if len(sample['answer']) > 150 else ''}")
        print(f"\nTime series:")
        print(f"  Count: {len(sample['time_series'])}")
        print(f"  Length: {len(sample['time_series'][0])}")
        print(f"  Text: {sample['time_series_text'][0]}")

