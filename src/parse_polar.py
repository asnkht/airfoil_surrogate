from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

input_file = project_dir / "data" / "raw" / "naca0012_polar.txt"
output_file = project_dir / "data" / "processed" / "naca0012_polar.csv"

with input_file.open("r", encoding="utf-8") as f:
    lines = f.readlines()

header_index= next(i for i, line in enumerate(lines) if line.split()[:3] == ["alpha", "CL", "CD"])

data = pd.read_csv(
    input_file,
    sep=r"\s+",
    skiprows = header_index+2,
    names = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"],
    nrows =3,
)

data = data[["alpha", "CL", "CD"]]

output_file.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(output_file, index=False)

print(data)
print(f"Processed data saved to {output_file}")