"""
Analyze hallucination votes from raw lawyer annotations.
Fresh analysis - count votes directly from source files.
"""

import pandas as pd

# Load all 6 annotator files
files = {
    'A_Kenneth': 'lawyer_data_package/raw_annotations/round_1_annotation_A_Kenneth_Doherty.xlsx',
    'B_Eileen': 'lawyer_data_package/raw_annotations/round_1_annotation_B_Eileen_Doherty.xlsx',
    'C_Kenny': 'lawyer_data_package/raw_annotations/round_1_annotation_C_Kenny_Doherty.xlsx',
    'D_Liza': 'lawyer_data_package/raw_annotations/round_1_annotation_D_Liza_Jameson.xlsx',
    'E_Adi': 'lawyer_data_package/raw_annotations/round_1_annotation_E_Adi_Kennedy.xlsx',
    'F_John': 'lawyer_data_package/raw_annotations/round_1_annotation_F_John_Jameson.xlsx',
}


def derive_hallucination(correctness, groundedness):
    """Hallucination = Incorrect AND (Ungrounded OR Misgrounded)"""
    is_incorrect = 'Incorrect' in str(correctness)
    is_bad_ground = str(groundedness) in ['Ungrounded', 'Misgrounded']
    return 'Yes' if (is_incorrect and is_bad_ground) else 'No'


# Analyze Q1 and Q3 for Baseline variant
for qid in [1, 3]:
    print('=' * 80)
    print(f'QUESTION {qid} - Baseline (LLaMA 3.1 - No Web)')
    print('=' * 80)

    votes = []
    confidences = []

    for name, filepath in files.items():
        df = pd.read_excel(filepath, header=1)
        row = df[(df['Query_ID'] == qid) & (df['Variant'] == 'Baseline (LLaMA 3.1 - No Web)')]

        if not row.empty:
            r = row.iloc[0]
            correctness = r.get('Correctness', None)
            groundedness = r.get('Groundedness', None)
            confidence = r.get('Confidence_Rating', None)

            # Derive hallucination label
            hall_vote = derive_hallucination(correctness, groundedness)

            votes.append(hall_vote)
            if pd.notna(confidence):
                confidences.append(float(confidence))

            print(f'{name:12} | Correct={str(correctness):20} | Ground={str(groundedness):12} | Hall={hall_vote:3} | Conf={confidence}')

    # Count votes
    yes_count = votes.count('Yes')
    no_count = votes.count('No')
    total = len(votes)

    if yes_count > no_count:
        majority = 'Hallucination'
    elif no_count > yes_count:
        majority = 'No_Hallucination'
    else:
        majority = 'Tie'

    agreement_rate = max(yes_count, no_count) / total if total > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print('-' * 80)
    print(f'VOTES: Yes={yes_count}, No={no_count}, Total={total}')
    print(f'MAJORITY: {majority} (agreement={agreement_rate:.1%})')
    print(f'AVG CONFIDENCE: {avg_conf:.2f}')
    print()
