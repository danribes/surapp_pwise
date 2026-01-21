"""Create compact KM curve data with only step change points."""

import pandas as pd
import numpy as np


def to_compact_km(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to compact KM format - only keep points where survival drops.

    In a KM curve, survival is constant until an event, then drops.
    We only need to record:
    - Time 0, Survival 1.0 (or first observed)
    - Each time point where survival drops

    This gives the minimal representation of the step function.
    """
    df = df.sort_values('Time').reset_index(drop=True)

    # Round survival to handle floating point issues
    df['Survival'] = df['Survival'].round(6)

    compact = []

    # Add starting point
    first_time = df['Time'].iloc[0]
    first_survival = df['Survival'].iloc[0]

    # If first time is not 0, add implicit start at 1.0
    if first_time > 0.01:
        compact.append({'Time': 0.0, 'Survival': 1.0})

    compact.append({'Time': first_time, 'Survival': first_survival})

    prev_survival = first_survival

    for i in range(1, len(df)):
        time = df['Time'].iloc[i]
        survival = df['Survival'].iloc[i]

        # Only keep points where survival decreases
        if survival < prev_survival - 1e-6:
            compact.append({'Time': time, 'Survival': survival})
            prev_survival = survival

    return pd.DataFrame(compact)


def main():
    # Read the cleaned data
    df = pd.read_csv('test_output/figure5_cleaned.csv')

    print(f"Cleaned data: {len(df)} rows")

    # Process each curve
    research = df[df['Curve'] == 'Research'].copy()
    control = df[df['Curve'] == 'Control'].copy()

    research_compact = to_compact_km(research)
    control_compact = to_compact_km(control)

    print(f"\nCompact format (step changes only):")
    print(f"  Research: {len(research_compact)} points")
    print(f"  Control: {len(control_compact)} points")

    # Add curve labels
    research_compact['Curve'] = 'Research'
    control_compact['Curve'] = 'Control'

    # Combine
    combined = pd.concat([research_compact, control_compact], ignore_index=True)
    combined = combined[['Curve', 'Time', 'Survival']]

    # Save
    combined.to_csv('test_output/figure5_compact.csv', index=False)
    print(f"\nSaved to test_output/figure5_compact.csv")

    # Also save individual curve files for easy use
    research_compact[['Time', 'Survival']].to_csv(
        'test_output/figure5_research_curve.csv', index=False
    )
    control_compact[['Time', 'Survival']].to_csv(
        'test_output/figure5_control_curve.csv', index=False
    )
    print("Saved individual curves to figure5_research_curve.csv and figure5_control_curve.csv")

    # Display the data
    print("\n" + "="*60)
    print("RESEARCH CURVE (Intervention)")
    print("="*60)
    print(research_compact[['Time', 'Survival']].to_string(index=False))

    print("\n" + "="*60)
    print("CONTROL CURVE")
    print("="*60)
    print(control_compact[['Time', 'Survival']].to_string(index=False))

    # Verify monotonicity
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    for name, data in [('Research', research_compact), ('Control', control_compact)]:
        survivals = data['Survival'].values
        is_monotonic = all(survivals[i] >= survivals[i+1] for i in range(len(survivals)-1))
        is_strictly_decreasing = all(survivals[i] > survivals[i+1] for i in range(len(survivals)-1))

        print(f"\n{name}:")
        print(f"  Monotonically decreasing: {is_monotonic}")
        print(f"  Strictly decreasing: {is_strictly_decreasing}")
        print(f"  Time range: {data['Time'].min():.2f} - {data['Time'].max():.2f}")
        print(f"  Survival range: {data['Survival'].min():.4f} - {data['Survival'].max():.4f}")


if __name__ == '__main__':
    main()
