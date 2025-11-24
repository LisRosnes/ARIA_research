import pandas as pd

# Read the Excel file
input_file = 'ARIA imaging results 2023-2024-Truncated.xlsx'  # Change this to your file name
df = pd.read_excel(input_file)

# Convert Study Date to datetime
df['Study Date'] = pd.to_datetime(df['Study Date'])

# Sort by SubjectID and Study Date to ensure chronological order
df = df.sort_values(['SubjectID', 'Study Date'])

# Function to convert Yes/No to 1/0
def convert_to_binary(value):
    if pd.isna(value):
        return 0
    return 1 if str(value).strip().lower() == 'yes' else 0

# Apply conversion
df['ARIA-E_binary'] = df['ARIA-E'].apply(convert_to_binary)
df['ARIA-H_binary'] = df['ARIA-H'].apply(convert_to_binary)

# Function to find first Yes date and resolved date
def find_dates(group, column):
    first_yes_date = None
    resolved_date = None
    found_yes = False
    
    for idx, row in group.iterrows():
        if row[column] == 1 and not found_yes:
            first_yes_date = row['Study Date']
            found_yes = True
        elif row[column] == 0 and found_yes and resolved_date is None:
            resolved_date = row['Study Date']
            break
    
    # If never resolved but did have Yes
    if found_yes and resolved_date is None:
        resolved_date = "Not Resolved"
    
    return first_yes_date, resolved_date

# Process each subject
results = []
for subject_id, group in df.groupby('SubjectID'):
    # Determine if ARIA-E or ARIA-H ever present
    aria_e = 1 if group['ARIA-E_binary'].max() == 1 else 0
    aria_h = 1 if group['ARIA-H_binary'].max() == 1 else 0
    
    # Find dates for ARIA-E
    aria_e_first, aria_e_resolved = find_dates(group, 'ARIA-E_binary')
    
    # Find dates for ARIA-H
    aria_h_first, aria_h_resolved = find_dates(group, 'ARIA-H_binary')
    
    results.append({
        'SubjectID': subject_id,
        'ARIA-E': aria_e,
        'ARIA-E First Yes Date': aria_e_first if aria_e_first else '',
        'ARIA-E Resolved Date': aria_e_resolved if aria_e_resolved else '',
        'ARIA-H': aria_h,
        'ARIA-H First Yes Date': aria_h_first if aria_h_first else '',
        'ARIA-H Resolved Date': aria_h_resolved if aria_h_resolved else ''
    })

# Create output dataframe
result_df = pd.DataFrame(results)

# Format dates as strings (YYYY-MM-DD) for dates that aren't "Not Resolved" or blank
for col in ['ARIA-E First Yes Date', 'ARIA-E Resolved Date', 'ARIA-H First Yes Date', 'ARIA-H Resolved Date']:
    result_df[col] = result_df[col].apply(
        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x
    )

# Save to CSV
output_file = 'aria_mapping_output.csv'
result_df.to_csv(output_file, index=False)

print(f"Mapping complete! Output saved to {output_file}")
print(f"\nPreview of results:")
print(result_df.head(10))