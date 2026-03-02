"""PCA Feature Analysis and Visualization for Generated Instances.

Reads summary.csv from batch generation, performs PCA dimensionality reduction
on instance features, and creates an interactive Plotly scatter plot colored
by seed type. Useful for verifying that generated instances have good coverage
of the feature space and aren't clustered around the original seeds.

Example usage:
    python analyze.py --summary instances/summary.csv
    python analyze.py --summary instances/summary.csv --save plot.html
"""
import argparse
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


# Feature columns used for PCA (numeric instance characteristics)
FEATURE_COLUMNS = [
    'B', 'L', 'D',
    'score_mean', 'score_variance',
    'signup_mean', 'ship_rate_mean',
    'book_duplication_rate',
    'book_coverage',
    'lib_size_mean',
    'actual_tightness',
]


def analyze(summary_path, save_path=None):
    """Run PCA on instance features and produce an interactive scatter plot."""
    df = pd.read_csv(summary_path)

    # Verify required columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: summary.csv is missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    features = df[FEATURE_COLUMNS]

    # Standardize to mean=0, variance=1 before PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Reduce to 2 principal components
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)

    # Build result dataframe for plotting
    result = pd.DataFrame(components, columns=['PC1', 'PC2'])
    result['seed'] = df['seed']

    # Classify: single-seed (1 letter) vs cross-bred (2 letters)
    result['type'] = df['seed'].apply(
        lambda s: 'cross-bred' if len(str(s)) > 1 else 'single-seed'
    )

    # Score source = first letter (which seed's score distribution)
    result['score_source'] = df['seed'].apply(lambda s: str(s)[0])

    # Scale tier for marker size
    result['scale'] = df['scale']

    # Build instance name for hover
    if 'scale' in df.columns and 'tightness_param' in df.columns:
        result['instance'] = (
            df['seed'].astype(str) + '_s' + df['scale'].astype(str)
            + '_t' + df['tightness_param'].astype(str)
        )
    else:
        result['instance'] = df.index.astype(str)

    explained = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
          f"total={sum(explained):.1%}")

    # Interactive scatter plot: color by score source, shape by type
    fig = px.scatter(
        result,
        x='PC1', y='PC2',
        color='score_source',
        symbol='type',
        size='scale',
        size_max=12,
        hover_data=['instance', 'seed', 'scale'],
        title='Instance Feature Space (PCA)',
        color_discrete_sequence=px.colors.qualitative.Set1,
        symbol_map={'single-seed': 'circle', 'cross-bred': 'diamond'},
    )
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>seed=%{customdata[1]}<br>scale=%{customdata[2]}<extra></extra>'
    )
    fig.update_layout(
        xaxis_title=f'PC1 ({explained[0]:.1%} variance)',
        yaxis_title=f'PC2 ({explained[1]:.1%} variance)',
        legend_title='Score source / Type',
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Plot saved to {save_path}")

    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="PCA analysis and visualization of generated instance features."
    )
    parser.add_argument('--summary', type=str, required=True,
                        help="Path to summary.csv from batch generation")
    parser.add_argument('--save', type=str, default=None,
                        help="Save plot as HTML file (optional)")

    args = parser.parse_args()
    analyze(args.summary, save_path=args.save)


if __name__ == '__main__':
    main()
