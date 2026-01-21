"""Clean extracted KM curve data to remove repeated values and ensure proper step function."""

import pandas as pd
import numpy as np


def clean_km_curve(df: pd.DataFrame, curve_name: str = None) -> pd.DataFrame:
    """
    Clean KM curve data to proper step function format.

    Removes consecutive duplicate survival values, keeping only:
    - Points where survival changes
    - First and last points

    For proper step function representation, each survival drop is shown as:
    - End of previous plateau (time just before drop)
    - Start of new plateau (time of drop)

    Args:
        df: DataFrame with Time and Survival columns
        curve_name: Optional curve name to filter by (if Curve column exists)

    Returns:
        Cleaned DataFrame with only step change points
    """
    if curve_name and 'Curve' in df.columns:
        df = df[df['Curve'] == curve_name].copy()

    # Sort by time
    df = df.sort_values('Time').reset_index(drop=True)

    # Round survival to reasonable precision to handle floating point issues
    df['Survival_rounded'] = df['Survival'].round(6)

    # Find where survival changes
    df['survival_changed'] = df['Survival_rounded'].diff().fillna(-1) != 0

    # Keep points where survival changed, plus first and last
    keep_mask = df['survival_changed'].copy()
    keep_mask.iloc[0] = True  # Always keep first
    keep_mask.iloc[-1] = True  # Always keep last

    # For proper step function, also keep point before each change
    for i in range(1, len(df)):
        if df['survival_changed'].iloc[i]:
            keep_mask.iloc[i-1] = True

    cleaned = df[keep_mask][['Time', 'Survival']].copy()

    return cleaned.reset_index(drop=True)


def to_step_function_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert KM data to explicit step function format.

    For each survival drop, creates two points:
    - (t_drop, survival_before) - end of plateau
    - (t_drop, survival_after) - start of new plateau

    This ensures the curve renders correctly as horizontal then vertical lines.
    """
    df = df.sort_values('Time').reset_index(drop=True)

    step_points = []

    for i in range(len(df)):
        time = df['Time'].iloc[i]
        survival = df['Survival'].iloc[i]

        if i == 0:
            # First point - start at (0, 1) if not already there
            if time > 0:
                step_points.append((0.0, survival))
            step_points.append((time, survival))
        else:
            prev_survival = df['Survival'].iloc[i-1]

            if abs(survival - prev_survival) > 1e-6:
                # Survival changed - add end of previous plateau and start of new
                step_points.append((time, prev_survival))  # End of old plateau
                step_points.append((time, survival))       # Start of new plateau
            else:
                # Same survival - just extend the plateau
                step_points.append((time, survival))

    result = pd.DataFrame(step_points, columns=['Time', 'Survival'])

    # Remove any exact duplicate rows
    result = result.drop_duplicates()

    return result.reset_index(drop=True)


def main():
    # Read the extracted data
    input_file = 'test_output/figure5_both_curves.csv'
    df = pd.read_csv(input_file)

    print(f"Original data: {len(df)} rows")
    print(f"  Research: {len(df[df['Curve'] == 'Research'])} points")
    print(f"  Control: {len(df[df['Curve'] == 'Control'])} points")

    # Clean each curve
    research_clean = clean_km_curve(df, 'Research')
    control_clean = clean_km_curve(df, 'Control')

    print(f"\nAfter removing repeated values:")
    print(f"  Research: {len(research_clean)} points")
    print(f"  Control: {len(control_clean)} points")

    # Convert to step function format
    research_steps = to_step_function_format(research_clean)
    control_steps = to_step_function_format(control_clean)

    print(f"\nIn step function format:")
    print(f"  Research: {len(research_steps)} points")
    print(f"  Control: {len(control_steps)} points")

    # Save cleaned data
    research_clean['Curve'] = 'Research'
    control_clean['Curve'] = 'Control'

    # Combine and save
    combined = pd.concat([research_clean, control_clean], ignore_index=True)
    combined = combined[['Curve', 'Time', 'Survival']]
    combined.to_csv('test_output/figure5_cleaned.csv', index=False)
    print(f"\nSaved cleaned data to test_output/figure5_cleaned.csv")

    # Also save step function format for proper plotting
    research_steps['Curve'] = 'Research'
    control_steps['Curve'] = 'Control'
    steps_combined = pd.concat([research_steps, control_steps], ignore_index=True)
    steps_combined = steps_combined[['Curve', 'Time', 'Survival']]
    steps_combined.to_csv('test_output/figure5_step_format.csv', index=False)
    print(f"Saved step format data to test_output/figure5_step_format.csv")

    # Show sample of cleaned data
    print("\n--- Research curve (first 15 points) ---")
    print(research_clean.head(15).to_string(index=False))

    print("\n--- Control curve (first 15 points) ---")
    print(control_clean.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
