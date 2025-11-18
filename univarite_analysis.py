import pandas as pd

def summarize_features(csv_path: str):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Clean up: treat pure whitespace as blank
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    total_rows = len(df)
    print(f"Total rows: {total_rows}\n")

    summary_rows = []

    for col in df.columns:
        # Consider a cell "blank" if it is NaN or an empty string
        non_blank_mask = df[col].notna() & (df[col] != "")
        non_blank_count = non_blank_mask.sum()
        blank_count = total_rows - non_blank_count

        summary_rows.append({
            "feature": col,
            "non_blank": non_blank_count,
            "blank": blank_count
        })

    # Convert summary to DataFrame for nice display (and easy export if you want)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    # Change this to your actual file name
    csv_file = "out_with_aria.csv"
    summarize_features(csv_file)
