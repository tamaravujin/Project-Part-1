"""
INTEG 275 — Technical Setup Exercise: Dimension Reduction

Goal: reduce four body-measurement variables for the Palmer Penguins
dataset down to two dimensions using Principal Component Analysis
(PCA), then make a scatter plot to see whether the three penguin
species separate out.

Data: data/penguins.csv (344 penguins, 3 species, Palmer Archipelago,
Antarctica). Source: Gorman, Williams & Fraser (2014), PLOS ONE,
via the palmerpenguins project (Horst, Hill & Gorman, 2020),
https://allisonhorst.github.io/palmerpenguins/

Before starting this exercise, have a brief look at the data by 
opening the CSV file to understand its structure and check for missing values.

How to use this script:
  Each TODO below describes one step. Use GitHub Copilot (inline
  suggestions, or Copilot Chat with Ctrl+I / Cmd+I) to help you
  write the code for that step. A suggested prompt is included as a
  comment under each TODO if you want a starting point, but try
  describing it in your own words first.

  Run the script from the integrated terminal with:
      python src/pca_exercise.py

  A successful run saves a plot to outputs/pca_scatter.png. Compare
  your plot to docs/expected_output_example.png to see roughly what
  a correct result should look like (your exact colours/layout may
  differ, that's fine).
"""

# TODO 1: Import the libraries you'll need.
# You will need: pandas, matplotlib.pyplot, StandardScaler (from
# sklearn.preprocessing), and PCA (from sklearn.decomposition).
# Copilot prompt idea: "Import pandas, matplotlib, and the sklearn
# tools needed for standardizing data and running PCA"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# TODO 2: Load the dataset.
# Read data/penguins.csv into a DataFrame called `penguins`. Note
# that missing values in this file are written as the text "NA".
# Copilot prompt idea: "Read data/penguins.csv into a pandas
# DataFrame, treating the string 'NA' as a missing value"
penguins = pd.read_csv("data/penguins.csv", na_values="NA")

# TODO 3: Drop incomplete rows.
# A few penguins are missing measurements. Remove any row with a
# missing value so PCA doesn't fail on them.
# Copilot prompt idea: "Drop rows with missing values from the
# penguins DataFrame"
penguins = penguins.dropna()


# TODO 4: Select the numeric feature columns.
# Pick out these four columns as your features:
#   bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
# Copilot prompt idea: "Select these four numeric columns from
# penguins into a new variable called features"
features = penguins[[
  "bill_length_mm",
  "bill_depth_mm",
  "flipper_length_mm",
  "body_mass_g",
]]


# TODO 5: Standardize the features.
# PCA is sensitive to scale (body_mass_g ranges in the thousands,
# bill_depth_mm ranges in the tens), so standardize the features
# first using StandardScaler so each column has mean 0 and
# variance 1.
# Copilot prompt idea: "Standardize the features using
# StandardScaler and store the result as features_scaled"
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# TODO 6: Run PCA.
# Fit a PCA model with 2 components on features_scaled, and
# transform the data to get the two principal component scores for
# every penguin.
# Copilot prompt idea: "Fit a PCA model with 2 components on
# features_scaled and get the transformed principal components"
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)


# TODO 7: Put the results in a DataFrame.
# Build a small DataFrame with columns PC1, PC2, and species (copy
# the species column back in from the original `penguins` table)
# so it's easy to plot.
# Copilot prompt idea: "Create a DataFrame called pca_df with
# columns PC1, PC2, and species"
pca_df = pd.DataFrame({
  "PC1": principal_components[:, 0],
  "PC2": principal_components[:, 1],
  "species": penguins["species"].to_numpy(),
})


# TODO 8: Plot and save the result.
# Make a scatter plot of PC1 vs PC2, with a different colour for
# each species and a legend. Label the axes (including how much
# variance each component explains, from
# pca.explained_variance_ratio_) and give the plot a title. Save it
# to outputs/pca_scatter.png.
# Copilot prompt idea: "Make a scatter plot of PC1 vs PC2 coloured
# by species, with axis labels showing percent variance explained,
# and save it to outputs/pca_scatter.png"
for species in pca_df["species"].unique():
  species_rows = pca_df[pca_df["species"] == species]
  plt.scatter(species_rows["PC1"], species_rows["PC2"], label=species)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)")
plt.title("PCA of Palmer Penguins")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("outputs/pca_scatter.png")
plt.close()


# TODO 9 (optional stretch): print a short summary.
# Print how much total variance the first two components explain
# combined, and print the first few rows of pca_df to check your
# work.
print(
  f"Total variance explained by the first two components: "
  f"{pca.explained_variance_ratio_.sum():.1%}"
)
print("\nFirst five rows of pca_df:")
print(pca_df.head())


# TODO 10 (optional stretch): Include the year column as a feature.
# Add the year variable as a feature and compare how the PCA plot changes.
# Save it to outputs/pca_scatter_with_year.png.
# This is useful because year is a numeric variable, but not a biological
# measurement, so it can change the structure of the principal components.
features_with_year = features.copy()
features_with_year["year"] = penguins["year"]

scaler_with_year = StandardScaler()
features_with_year_scaled = scaler_with_year.fit_transform(features_with_year)

pca_with_year = PCA(n_components=2)
principal_components_with_year = pca_with_year.fit_transform(
  features_with_year_scaled
)

pca_df_with_year = pd.DataFrame({
  "PC1": principal_components_with_year[:, 0],
  "PC2": principal_components_with_year[:, 1],
  "species": penguins["species"].to_numpy(),
})

print(
  f"Variance explained without year: "
  f"{pca.explained_variance_ratio_.sum():.1%}"
)
print(
  f"Variance explained with year: "
  f"{pca_with_year.explained_variance_ratio_.sum():.1%}"
)

for species in pca_df_with_year["species"].unique():
  species_rows = pca_df_with_year[pca_df_with_year["species"] == species]
  plt.scatter(species_rows["PC1"], species_rows["PC2"], label=species)

plt.xlabel(
  f"PC1 ({pca_with_year.explained_variance_ratio_[0]:.1%} variance explained)"
)
plt.ylabel(
  f"PC2 ({pca_with_year.explained_variance_ratio_[1]:.1%} variance explained)"
)
plt.title("PCA of Palmer Penguins Including Year")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("outputs/pca_scatter_with_year.png")
plt.close()
