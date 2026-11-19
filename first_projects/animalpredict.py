import pandas as pd

columns = [
    "animal_name",
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "legs",
    "tail",
    "domestic",
    "catsize",
    "type"
]

# Read original data
df = pd.read_csv(
    "../data/zoo/zoo.data",
    header=None,
    names=columns
)

# Save as CSV with column names
df.to_csv(
    "../data/zoo/zoo.names",
    index=False
    
)

print("Dataset created successfully!")
print("Shape:", df.shape)
print(df.head())