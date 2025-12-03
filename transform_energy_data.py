#!/usr/bin/env python3
"""
Generate energy consumption training data with AI-generated reasoning.

Uses Strands Agents with AWS Bedrock to generate chain-of-thought reasoning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from strands import Agent


def generate_reasoning(
    agent: Agent,
    time_series: List[float],
    question: str,
    answer: str,
    context: Dict[str, Any],
) -> str:
    """
    Generate chain-of-thought reasoning using Strands Agent.
    
    Args:
        agent: Strands Agent instance
        time_series: The time series data
        question: The question about the time series
        answer: The ground truth answer
        context: Additional context (stats, metadata)
    
    Returns:
        Chain-of-thought reasoning text
    """
    # Create summary statistics
    ts_array = np.array(time_series)
    sample_indices = np.linspace(0, len(time_series)-1, min(20, len(time_series)), dtype=int)
    sampled_values = [f"{time_series[i]:.2f}" for i in sample_indices]
    
    # Create the prompt
    prompt = f"""You are an expert in energy consumption analysis. Analyze this building energy time series and provide reasoning.

Time Series Stats:
- Duration: {len(time_series)} timesteps ({len(time_series)/4:.1f} hours at 15-min intervals)
- Min: {ts_array.min():.2f} kWh, Max: {ts_array.max():.2f} kWh, Mean: {ts_array.mean():.2f} kWh
- First: {time_series[0]:.2f} kWh, Last: {time_series[-1]:.2f} kWh
- Sample values: {', '.join(sampled_values)}

Context:
- Building: {context.get('bldg_id')}
- Period: {context.get('start_time', '')[:16]} to {context.get('end_time', '')[:16]}
- Total energy: {context.get('total_energy', 0):.2f} kWh

Question: {question}

Provide 2-3 sentences analyzing the pattern, then end with: "Answer: {answer}"

Write a natural paragraph (no bullets). Do NOT mention the answer until the end."""

    try:
        response = agent(prompt)
        return response.strip()
    except Exception as e:
        # Fallback
        return f"The time series shows consumption over {len(time_series)} timesteps with mean {ts_array.mean():.2f} kWh. The pattern shows typical building usage characteristics. Answer: {answer}"


def create_time_series_windows(
    df: pd.DataFrame,
    window_size: int = 96,
    stride: int = 24,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert timeseries dataframe into fixed-length windows.
    
    Args:
        df: DataFrame with timestamp and energy columns
        window_size: Number of timesteps per window
        stride: Stride between windows
        max_samples: Maximum number of samples to generate (None = all)
    
    Returns:
        List of samples with time_series and metadata
    """
    samples = []
    total_samples = 0
    
    # Group by building
    for bldg_id, bldg_df in tqdm(df.groupby('bldg_id'), desc="Creating windows"):
        if max_samples and total_samples >= max_samples:
            break
            
        bldg_df = bldg_df.sort_values('timestamp').reset_index(drop=True)
        
        # Create sliding windows
        for start_idx in range(0, len(bldg_df) - window_size + 1, stride):
            if max_samples and total_samples >= max_samples:
                break
                
            end_idx = start_idx + window_size
            window = bldg_df.iloc[start_idx:end_idx]
            
            # Extract time series
            time_series = window['out.site_energy.total.energy_consumption'].values
            
            # Skip invalid windows
            if np.sum(time_series) == 0 or np.isnan(time_series).sum() > window_size * 0.1:
                continue
            
            # Calculate statistics
            stats = {
                'total_energy': float(np.sum(time_series)),
                'max_energy': float(np.max(time_series)),
                'mean_energy': float(np.mean(time_series)),
                'min_energy': float(np.min(time_series)),
                'std_energy': float(np.std(time_series)),
            }
            
            # Create sample
            sample = {
                'time_series': time_series.tolist(),
                'bldg_id': int(bldg_id),
                'start_time': str(window['timestamp'].iloc[0]),
                'end_time': str(window['timestamp'].iloc[-1]),
                **stats,
            }
            
            samples.append(sample)
            total_samples += 1
            
            if max_samples and total_samples >= max_samples:
                break
    
    return samples


def generate_questions_and_reasoning(
    samples: List[Dict[str, Any]],
    agent: Optional[Agent] = None,
    use_reasoning: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate questions and answers (with optional reasoning) for samples.
    
    Args:
        samples: List of window samples
        agent: Strands Agent instance (None = no reasoning)
        use_reasoning: Whether to generate chain-of-thought reasoning
    
    Returns:
        List of complete training samples
    """
    training_samples = []
    
    for sample in tqdm(samples, desc="Generating Q&A"):
        # Determine question and answer based on energy patterns
        question, answer = create_question_answer(sample)
        
        # Generate reasoning if requested
        if use_reasoning and agent:
            reasoning = generate_reasoning(
                agent=agent,
                time_series=sample['time_series'],
                question=question,
                answer=answer,
                context=sample,
            )
            # Extract just the reasoning part (before "Answer:")
            if "Answer:" in reasoning:
                reasoning_only = reasoning.split("Answer:")[0].strip()
                final_answer = f"{reasoning_only} Answer: {answer}"
            else:
                final_answer = f"{reasoning} Answer: {answer}"
        else:
            final_answer = answer
        
        training_samples.append({
            'time_series': sample['time_series'],
            'question': question,
            'answer': final_answer,
            'id': f"bldg_{sample['bldg_id']}_{sample['start_time'][:10]}_{sample['start_time'][11:16].replace(':', '')}",
            'bldg_id': sample['bldg_id'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
            'total_energy': sample['total_energy'],
            'mean_energy': sample['mean_energy'],
        })
    
    return training_samples


def create_question_answer(sample: Dict[str, Any]) -> tuple[str, str]:
    """
    Create question and answer based on energy pattern.
    
    Args:
        sample: Sample with time_series and statistics
    
    Returns:
        Tuple of (question, answer)
    """
    mean = sample['mean_energy']
    max_energy = sample['max_energy']
    min_energy = sample['min_energy']
    
    # Multiple question types for variety
    question_type = np.random.choice(['peak', 'level', 'trend', 'stability'])
    
    if question_type == 'peak':
        if max_energy > mean * 1.5:
            question = "Does this energy consumption pattern show significant peak usage?"
            answer = "yes"
        else:
            question = "Does this energy consumption pattern show significant peak usage?"
            answer = "no"
    
    elif question_type == 'level':
        if mean < 10:
            question = "What is the overall energy consumption level during this period?"
            answer = "low"
        elif mean < 20:
            question = "What is the overall energy consumption level during this period?"
            answer = "medium"
        else:
            question = "What is the overall energy consumption level during this period?"
            answer = "high"
    
    elif question_type == 'trend':
        ts = sample['time_series']
        if ts[-1] > ts[0] * 1.1:
            question = "What is the dominant trend in this energy consumption pattern?"
            answer = "increasing"
        elif ts[-1] < ts[0] * 0.9:
            question = "What is the dominant trend in this energy consumption pattern?"
            answer = "decreasing"
        else:
            question = "What is the dominant trend in this energy consumption pattern?"
            answer = "stable"
    
    else:  # stability
        std = sample['std_energy']
        if std < mean * 0.3:
            question = "Is the energy consumption pattern stable or variable?"
            answer = "stable"
        else:
            question = "Is the energy consumption pattern stable or variable?"
            answer = "variable"
    
    return question, answer


def create_splits(
    samples: List[Dict[str, Any]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_seed: int = 42,
) -> tuple:
    """Split samples into train/val/test."""
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(samples, test_size=test_frac, random_state=random_seed)
    val_frac_adj = val_frac / (train_frac + val_frac)
    train, val = train_test_split(train_val, test_size=val_frac_adj, random_state=random_seed + 1)
    
    for s in train: s['split'] = 'train'
    for s in val: s['split'] = 'validation'
    for s in test: s['split'] = 'test'
    
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Generate energy time series training data with AI reasoning")
    parser.add_argument("input_parquet", type=str, help="Input parquet file")
    parser.add_argument("--output", type=str, default="data/energy_dataset.parquet",
                        help="Output parquet file")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--window_size", type=int, default=96,
                        help="Window size (timesteps)")
    parser.add_argument("--stride", type=int, default=24,
                        help="Stride between windows")
    parser.add_argument("--use_reasoning", action="store_true",
                        help="Generate chain-of-thought reasoning using Bedrock")
    parser.add_argument("--bedrock_model", type=str, default="us.amazon.nova-lite-v1:0",
                        help="Bedrock model to use (e.g., us.amazon.nova-lite-v1:0)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Energy Time Series Dataset Generator")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading data from: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    print(f"   Shape: {df.shape}")
    print(f"   Buildings: {df['bldg_id'].nunique()}")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize Strands Agent if using reasoning
    agent = None
    if args.use_reasoning:
        print(f"\n🤖 Initializing Strands Agent with {args.bedrock_model}...")
        try:            
            # Create Bedrock provider with specific model
            agent = Agent(model=args.bedrock_model)
            print(f"   ✓ Agent initialized with {args.bedrock_model}")
        except Exception as e:
            print(f"   ❌ Error: Could not initialize agent: {e}")
            print(f"   Reasoning generation requires AWS credentials and Bedrock access.")
            raise
    
    # Create windows
    print(f"\n📊 Creating time series windows...")
    print(f"   Target samples: {args.num_samples}")
    print(f"   Window size: {args.window_size} timesteps ({args.window_size/4:.1f} hours)")
    print(f"   Stride: {args.stride} timesteps ({args.stride/4:.1f} hours)")
    
    windows = create_time_series_windows(
        df,
        window_size=args.window_size,
        stride=args.stride,
        max_samples=args.num_samples,
    )
    print(f"   ✓ Created {len(windows)} windows")
    
    # Generate questions and reasoning
    print(f"\n💭 Generating questions and answers...")
    if args.use_reasoning:
        print(f"   Using AI reasoning with {args.bedrock_model}")
    training_samples = generate_questions_and_reasoning(
        windows,
        agent=agent,
        use_reasoning=args.use_reasoning,
    )
    print(f"   ✓ Generated {len(training_samples)} training samples")
    
    # Split into train/val/test
    print(f"\n✂️  Splitting into train/val/test...")
    train, val, test = create_splits(training_samples)
    print(f"   Train: {len(train)} ({len(train)/len(training_samples)*100:.1f}%)")
    print(f"   Val: {len(val)} ({len(val)/len(training_samples)*100:.1f}%)")
    print(f"   Test: {len(test)} ({len(test)/len(training_samples)*100:.1f}%)")
    
    # Save
    all_samples = train + val + test
    output_df = pd.DataFrame(all_samples)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to: {args.output}")
    output_df.to_parquet(args.output, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("✅ Dataset Generation Complete!")
    print("="*80)
    print(f"\nTotal samples: {len(output_df)}")
    print(f"Time series length: {len(output_df.iloc[0]['time_series'])} timesteps")
    print(f"\nAnswer distribution:")
    print(output_df['answer'].value_counts().head(10))
    print(f"\nSample questions:")
    for q in output_df['question'].unique()[:5]:
        print(f"  - {q}")
    
    if args.use_reasoning:
        print(f"\n📝 Sample with reasoning:")
        sample = output_df.iloc[0]
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer'][:200]}...")


if __name__ == "__main__":
    main()
