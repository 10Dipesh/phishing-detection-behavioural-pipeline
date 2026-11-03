import numpy as np
import pandas as pd

DATA_PATH = "data/study_data_merged.xlsx"

df = pd.read_excel(DATA_PATH)
df = df.replace("--", np.nan)

print(f"Loaded {len(df)} people, {len(df.columns)} columns")
print()

for sim in ["sim1", "sim2"]:
    delivered = df[f"{sim}_SuccessfullyDeliveredEmail_TimeStamp"].notna().sum()
    clicked = df[f"{sim}_EmailLinkClicked_TimeStamp"].notna().sum()
    reported = df[f"{sim}_Phishing_Reported_On"].notna().sum()

    print(f"{sim}:")
    print(f"  delivered: {delivered}")
    print(f"  clicked:   {clicked} ({clicked / delivered:.1%})")
    print(f"  reported:  {reported} ({reported / delivered:.1%})")
    print()
