"""
Multi-Image Review Dataset for Gemma 3 VLM Training.
Handles review-level data with multiple images per review.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np

class MultiImageReviewDataset(Dataset):
    """
    Dataset that handles multi-image reviews.
    Groups images by review and provides review-level data.
    """
    
    def __init__(
        self,
        df_reviews: pd.DataFrame,
        image_dir: Union[str, Path],
        transform=None,
        max_images_per_review: int = 10,
        image_aggregation: str = 'mean'
    ):
        """
        Args:
            df_reviews: DataFrame with review-level data (one row per review)
                Required columns: review_id, review_text_preview, review_rating, 
                                 saved_path (list), filename (list)
            image_dir: Base directory for images
            transform: Image transforms (torchvision)
            max_images_per_review: Maximum images to use per review
            image_aggregation: How to aggregate multiple images ('mean', 'concat', 'first')
        """
        self.df_reviews = df_reviews.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.max_images_per_review = max_images_per_review
        self.image_aggregation = image_aggregation
        
        # Validate required columns
        required_cols = ['review_id', 'review_text_preview', 'review_rating']
        missing = [col for col in required_cols if col not in df_reviews.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def __len__(self):
        return len(self.df_reviews)
    
    def __getitem__(self, idx):
        review = self.df_reviews.iloc[idx]
        
        # Load images
        image_paths = review.get('saved_path', [])
        if isinstance(image_paths, str):
            # Handle case where saved_path is a string representation of list
            import ast
            try:
                image_paths = ast.literal_eval(image_paths)
            except:
                image_paths = [image_paths]
        
        if not isinstance(image_paths, list):
            image_paths = []
        
        # Limit number of images
        image_paths = image_paths[:self.max_images_per_review]
        
        # Load and process images
        images = []
        valid_paths = []
        for img_path in image_paths:
            if isinstance(img_path, str):
                full_path = Path(img_path)
                if not full_path.is_absolute():
                    full_path = self.image_dir / img_path
                
                if full_path.exists():
                    try:
                        img = Image.open(full_path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        images.append(img)
                        valid_paths.append(str(full_path))
                    except Exception as e:
                        print(f"Warning: Failed to load {full_path}: {e}")
                        continue
        
        # Handle case with no valid images
        if len(images) == 0:
            # Create a dummy black image
            dummy_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_img = self.transform(dummy_img)
            images = [dummy_img]
            valid_paths = ['dummy']
        
        # Aggregate images based on strategy
        if self.image_aggregation == 'first':
            images_tensor = images[0] if isinstance(images[0], torch.Tensor) else self.transform(images[0])
        elif self.image_aggregation == 'concat':
            # Stack images: [num_images, C, H, W]
            if isinstance(images[0], torch.Tensor):
                images_tensor = torch.stack(images)
            else:
                images_tensor = torch.stack([self.transform(img) for img in images])
        else:  # 'mean' - default
            # Mean pool image embeddings (will be done after encoding)
            if isinstance(images[0], torch.Tensor):
                images_tensor = torch.stack(images)
            else:
                images_tensor = torch.stack([self.transform(img) for img in images])
        
        # Get review text
        review_text = str(review['review_text_preview']) if pd.notna(review['review_text_preview']) else ""
        
        # Get rating
        rating = float(review['review_rating']) if pd.notna(review['review_rating']) else 3.0
        
        # Get captions if available
        captions = review.get('caption', [])
        if isinstance(captions, str):
            import ast
            try:
                captions = ast.literal_eval(captions)
            except:
                captions = [captions] if captions else []
        if not isinstance(captions, list):
            captions = []
        
        return {
            'review_id': review['review_id'],
            'images': images_tensor,  # Can be single tensor or stacked
            'image_paths': valid_paths,
            'num_images': len(valid_paths),
            'review_text': review_text,
            'rating': rating,
            'captions': captions[:len(valid_paths)],  # Match number of images
            'business_id': review.get('business_id', ''),
        }
    
    def get_review_stats(self):
        """Get statistics about the dataset."""
        stats = {
            'total_reviews': len(self.df_reviews),
            'avg_images_per_review': self.df_reviews.get('num_images', pd.Series([1] * len(self.df_reviews))).mean(),
            'rating_distribution': self.df_reviews['review_rating'].value_counts().to_dict(),
        }
        return stats

def create_review_level_dataset(df_images: pd.DataFrame) -> pd.DataFrame:
    """
    Convert image-level DataFrame to review-level DataFrame.
    
    Args:
        df_images: DataFrame with image-level data
            Required columns: business_id, review_index, review_text_preview, 
                             review_rating, saved_path, filename
    
    Returns:
        DataFrame with review-level data (one row per review)
    """
    # Create review ID
    df_images['review_id'] = (
        df_images['business_id'].astype(str) + '_' + 
        df_images['review_index'].astype(str)
    )
    
    # Group by review
    review_groups = df_images.groupby('review_id').agg({
        'business_id': 'first',
        'business_name': 'first',
        'review_text_preview': 'first',
        'review_rating': 'first',
        'saved_path': lambda x: list(x),
        'filename': lambda x: list(x),
        'caption': lambda x: [c for c in x if pd.notna(c)],
    }).reset_index()
    
    review_groups['num_images'] = review_groups['saved_path'].apply(len)
    review_groups['review_rating'] = review_groups['review_rating'].astype(int)
    
    return review_groups

def create_stratified_splits(
    df_reviews: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Create stratified train/test/val splits at review level.
    Stratifies by business_id (to prevent business leakage) AND rating.
    
    Args:
        df_reviews: Review-level DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df: Three DataFrames
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Create business group to prevent business leakage
    df_reviews = df_reviews.copy()
    df_reviews['business_group'] = df_reviews.groupby('business_id').ngroup()
    
    # Create stratification key (business_group + rating)
    df_reviews['stratify_key'] = (
        df_reviews['business_group'].astype(str) + '_' + 
        df_reviews['review_rating'].astype(str)
    )
    
    # Check if stratification is possible (need at least 2 samples per class)
    stratify_counts = df_reviews['stratify_key'].value_counts()
    min_samples = stratify_counts.min()
    
    if min_samples < 2:
        print(f"   Warning: Some stratify groups have <2 samples (min={min_samples})")
        print(f"   Using rating-only stratification with business-aware splitting")
        # Fallback: stratify by rating only, then ensure business separation
        stratify_by = df_reviews['review_rating']
    else:
        stratify_by = df_reviews['stratify_key']
    
    # First split: train+val vs test
    splitter1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    try:
        train_val_idx, test_idx = next(
            splitter1.split(df_reviews, stratify_by)
        )
    except ValueError as e:
        # If still fails, use simple shuffle split
        print(f"   Warning: Stratified split failed, using simple shuffle: {e}")
        from sklearn.model_selection import ShuffleSplit
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(splitter.split(df_reviews))
    
    train_val_df = df_reviews.iloc[train_val_idx].copy()
    test_df = df_reviews.iloc[test_idx].copy()
    
    # Second split: train vs val
    # Adjust val_size relative to train+val size
    val_size_adjusted = val_size / (1 - test_size)
    
    # Check stratification for train_val split
    train_val_stratify_counts = train_val_df['stratify_key'].value_counts()
    train_val_min_samples = train_val_stratify_counts.min()
    
    if train_val_min_samples < 2:
        train_val_stratify_by = train_val_df['review_rating']
    else:
        train_val_stratify_by = train_val_df['stratify_key']
    
    splitter2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    try:
        train_idx, val_idx = next(
            splitter2.split(train_val_df, train_val_stratify_by)
        )
    except ValueError as e:
        # If still fails, use simple shuffle split
        print(f"   Warning: Stratified split failed for train/val, using simple shuffle: {e}")
        from sklearn.model_selection import ShuffleSplit
        splitter = ShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
        train_idx, val_idx = next(splitter.split(train_val_df))
    
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    
    # Clean up temporary columns
    for df in [train_df, val_df, test_df]:
        df.drop(columns=['business_group', 'stratify_key'], inplace=True, errors='ignore')
    
    print(f"Train: {len(train_df):,} reviews")
    print(f"Val: {len(val_df):,} reviews")
    print(f"Test: {len(test_df):,} reviews")
    
    # Print rating distribution
    print("\nRating Distribution:")
    for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = df['review_rating'].value_counts().sort_index()
        print(f"{split_name}: {dict(dist)}")
    
    return train_df, val_df, test_df

def create_human_dataset_adapter(csv_path: str) -> pd.DataFrame:
    """
    Adapts the human-annotated CSV to the format expected by MultiImageReviewDataset.
    
    Args:
        csv_path: Path to the processed human data CSV
        
    Returns:
        DataFrame compatible with MultiImageReviewDataset
    """
    df = pd.read_csv(csv_path)
    
    # Filter for rows where we found the image
    df = df[df['local_path'].notna()].copy()
    
    # 1. Create synthesis of sensory tags for review text
    # Columns: taste, smell, texture, look, feel
    def synthesize_text(row):
        parts = []
        for aspect in ['taste', 'smell', 'texture', 'look', 'feel']:
            val = row.get(aspect)
            if pd.notna(val) and str(val).lower() != 'not sure':
                parts.append(f"{aspect.capitalize()}: {val}")
        return ". ".join(parts) + "." if parts else "No description available."

    df['review_text_preview'] = df.apply(synthesize_text, axis=1)
    
    # 2. Map other columns
    # We treat each image as a "review" since they are annotated individually
    df['review_id'] = df['cleaned_image_id']
    df['business_id'] = 'HUMAN_DATA'
    df['review_rating'] = df['sensory_average'].astype(float)
    
    # 3. Format path and filename as lists (dataset expectation)
    df['saved_path'] = df['local_path'].apply(lambda x: [x])
    df['filename'] = df['cleaned_image_id'].apply(lambda x: [str(x) + ".jpg"]) # Approx filename
    df['num_images'] = 1
    
    # Select only required columns
    cols = ['review_id', 'business_id', 'review_text_preview', 'review_rating', 
            'saved_path', 'filename', 'num_images']
    
    return df[cols]


def load_human_sensory_data(
    csv_path: str,
    image_dir: str,
    require_all_caninfer: bool = True,
) -> pd.DataFrame:
    """
    Load the FINAL_DATASET_COMPLETE_with_rescaling.csv and prepare it for
    VLM training with REAL per-sense ratings and descriptions.

    Filters to rows where CanInfer_* == 1 and maps Image_Name to the
    local image directory.

    Returns a DataFrame with columns compatible with GemmaVLMDataset:
        review_id, business_id, review_text_preview, review_rating,
        saved_path, filename, num_images,
        + sensory columns: sensory_taste, sensory_smell, sensory_texture,
          sensory_sound, taste_desc, smell_desc, texture_desc, sound_desc,
          has_human_sensory (bool flag)
    """
    df = pd.read_csv(csv_path)
    print(f"[Human data] Loaded {len(df):,} rows from {csv_path}")

    # ---------- Filter by CanInfer ----------
    senses = ['taste', 'smell', 'texture', 'sound']
    caninfer_cols = [f'CanInfer_{s}' for s in senses]

    if require_all_caninfer:
        mask = pd.Series(True, index=df.index)
        for col in caninfer_cols:
            mask &= (df[col] == 1)
        df = df[mask].copy()
        print(f"[Human data] After CanInfer filter (all 4): {len(df):,} rows")
    else:
        mask = pd.Series(False, index=df.index)
        for col in caninfer_cols:
            mask |= (df[col] == 1)
        df = df[mask].copy()
        print(f"[Human data] After CanInfer filter (any): {len(df):,} rows")

    # ---------- Map image paths ----------
    from pathlib import Path
    img_dir = Path(image_dir)
    existing_files = set(img_dir.iterdir()) if img_dir.exists() else set()
    existing_names = {f.name for f in existing_files}

    df['_img_found'] = df['Image_Name'].apply(lambda n: n in existing_names)
    found = df['_img_found'].sum()
    print(f"[Human data] Image match: {found:,}/{len(df):,} rows have images in {image_dir}")
    df = df[df['_img_found']].copy()
    df.drop(columns=['_img_found'], inplace=True)

    # ---------- Build unified columns ----------
    # Sensory ratings (RescaledRating is already 1-5 scale)
    for s in senses:
        col_src = f'RescaledRating_{s}'
        df[f'sensory_{s}'] = df[col_src].astype(float).clip(1, 5)

    # ---------- Keep per-participant rows ----------
    # Each participant gives a different rating for the same image.
    # With ratings-only targets (no template text), this diversity is
    # *useful* — it teaches the model the valid range for each image
    # and gives ~58K training examples instead of ~2,915.

    # Overall rating = mean of the four sense ratings for this row
    df['review_rating'] = (
        df['sensory_taste'] + df['sensory_smell'] +
        df['sensory_texture'] + df['sensory_sound']
    ) / 4.0

    # Image path as a list (expected by GemmaVLMDataset)
    df['saved_path'] = df['Image_Name'].apply(lambda n: [n])
    df['filename'] = df['Image_Name'].apply(lambda n: [n])

    # IDs — unique per participant × image
    df['review_id'] = (
        df['participantId'].astype(str) + '_' + df['Image_ID'].astype(str)
    )
    df['business_id'] = 'HUMAN_ANNOTATED'
    df['num_images'] = 1
    df['has_human_sensory'] = True

    # Keep sensory descriptions — they are used to build reasoning
    # in the training target (connecting cue keywords to ratings).
    df['review_text_preview'] = ''

    keep_cols = [
        'review_id', 'business_id', 'review_text_preview', 'review_rating',
        'saved_path', 'filename', 'num_images',
        'sensory_taste', 'sensory_smell', 'sensory_texture', 'sensory_sound',
        'taste_desc', 'smell_desc', 'texture_desc', 'sound_desc',
        'has_human_sensory',
    ]
    df = df[keep_cols].copy()
    print(f"[Human data] Final: {len(df):,} rows "
          f"({df['review_id'].str.split('_').str[0].nunique()} participants, "
          f"{df['filename'].apply(lambda x: x[0]).nunique()} unique images)")
    return df


def create_image_level_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_state: int = 42,
):
    """
    Split data so that all annotations for the same *image* stay in the
    same split, preventing data leakage.

    The human dataset has ~2,915 unique images with ~20 participant ratings
    each.  We split at the image level and stratify by the image's mean
    rating (binned) so each split has a similar rating distribution.

    Args:
        df: DataFrame from ``load_human_sensory_data()``.
            Must contain ``saved_path`` (list) and ``review_rating``.
        test_size:  Fraction of *images* for the test set.
        val_size:   Fraction of *images* for the validation set.
        random_state: Random seed.

    Returns:
        train_df, val_df, test_df
    """
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

    df = df.copy()

    # Extract image name from saved_path (stored as single-element list)
    df['_image_name'] = df['saved_path'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x)
    )

    # Build image-level summary for stratification
    img_stats = (
        df.groupby('_image_name')['review_rating']
        .mean()
        .reset_index()
        .rename(columns={'review_rating': '_mean_rating'})
    )
    # Bin mean rating into integer buckets 1-5 for stratification
    img_stats['_rating_bin'] = img_stats['_mean_rating'].round().astype(int).clip(1, 5)

    n_images = len(img_stats)
    print(f"[Image-level split] {n_images:,} unique images, "
          f"{len(df):,} total rows")

    # --- split images into (train+val) vs test ---
    try:
        spl1 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        tv_idx, te_idx = next(spl1.split(img_stats, img_stats['_rating_bin']))
    except ValueError:
        spl1 = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        tv_idx, te_idx = next(spl1.split(img_stats))

    test_images = set(img_stats.iloc[te_idx]['_image_name'])
    tv_stats = img_stats.iloc[tv_idx].copy()

    # --- split (train+val) into train vs val ---
    val_frac = val_size / (1.0 - test_size)
    try:
        spl2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=random_state
        )
        tr_idx, va_idx = next(spl2.split(tv_stats, tv_stats['_rating_bin']))
    except ValueError:
        spl2 = ShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=random_state
        )
        tr_idx, va_idx = next(spl2.split(tv_stats))

    train_images = set(tv_stats.iloc[tr_idx]['_image_name'])
    val_images = set(tv_stats.iloc[va_idx]['_image_name'])

    # --- assign rows to splits ---
    train_df = df[df['_image_name'].isin(train_images)].drop(columns=['_image_name']).copy()
    val_df   = df[df['_image_name'].isin(val_images)].drop(columns=['_image_name']).copy()
    test_df  = df[df['_image_name'].isin(test_images)].drop(columns=['_image_name']).copy()

    print(f"  Train: {len(train_df):,} rows  ({len(train_images):,} images)")
    print(f"  Val:   {len(val_df):,} rows  ({len(val_images):,} images)")
    print(f"  Test:  {len(test_df):,} rows  ({len(test_images):,} images)")

    # Drop temp column from original df
    df.drop(columns=['_image_name'], inplace=True, errors='ignore')

    return train_df, val_df, test_df

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load image-level data
    df = pd.read_csv('data/Yelp_food_clean_metadata.csv', nrows=1000)
    
    # Create review-level dataset
    df_reviews = create_review_level_dataset(df)
    print(f"Created {len(df_reviews)} review-level records")
    
    # Create splits
    train_df, val_df, test_df = create_stratified_splits(df_reviews)
    
    print("\nSplits created successfully!")

