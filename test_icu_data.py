# Apply DRO to a synthetic dataset.
# You'll need to install polars and scikit-learn:
# mamba install -y polars scikit-learn
#
# Don't run this script on the login node. Instead, get an interactive session:
# srun --time=4:00:00 --mem-per-cpu=4G --cpus-per-task=16 --pty bash
# Once a job is allocated, you might need to
# source ~/.bashrc
# conda activate dro
# python test_icu_data.py
from dro.model import DRO
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

source = "eicu"
target = "miiv"
rho = 0.1

# We predict the binary outcome of mortality at 24 hours after entry to the ICU.
# This is a common "benchmark". It has the benefit that the resulting dataset is not too
# large. There is only one observation per patient.
outcome = "mortality_at_24h"

# The features.parquet contains ~2000 columns. We select features (i) defined for
# variables that are used in the apache II score and (ii) only use 24h historical
# features. See the draft I sent you for details.
features = [
    "age",
    "hr_ffilled",
    "hr_missing",
    "hr_sq_ffilled",
    "hr_mean_h24",
    "hr_sq_mean_h24",
    "hr_std_h24",
    "hr_slope_h24",
    "hr_fraction_nonnull_h24",
    "hr_all_missing_h24",
    "hr_min_h24",
    "hr_max_h24",
    "fio2_ffilled",
    "fio2_missing",
    "fio2_sq_ffilled",
    "fio2_mean_h24",
    "fio2_sq_mean_h24",
    "fio2_std_h24",
    "fio2_slope_h24",
    "fio2_fraction_nonnull_h24",
    "fio2_all_missing_h24",
    "fio2_min_h24",
    "fio2_max_h24",
    "log_resp_ffilled",
    "log_resp_missing",
    "log_resp_sq_ffilled",
    "log_resp_mean_h24",
    "log_resp_sq_mean_h24",
    "log_resp_std_h24",
    "log_resp_slope_h24",
    "log_resp_fraction_nonnull_h24",
    "log_resp_all_missing_h24",
    "log_resp_min_h24",
    "log_resp_max_h24",
    "temp_ffilled",
    "temp_missing",
    "temp_sq_ffilled",
    "temp_mean_h24",
    "temp_sq_mean_h24",
    "temp_std_h24",
    "temp_slope_h24",
    "temp_fraction_nonnull_h24",
    "temp_all_missing_h24",
    "temp_min_h24",
    "temp_max_h24",
    "log_crea_ffilled",
    "log_crea_missing",
    "log_crea_sq_ffilled",
    "log_crea_mean_h24",
    "log_crea_sq_mean_h24",
    "log_crea_std_h24",
    "log_crea_slope_h24",
    "log_crea_fraction_nonnull_h24",
    "log_crea_all_missing_h24",
    "log_crea_min_h24",
    "log_crea_max_h24",
    "log_po2_ffilled",
    "log_po2_missing",
    "log_po2_sq_ffilled",
    "log_po2_mean_h24",
    "log_po2_sq_mean_h24",
    "log_po2_std_h24",
    "log_po2_slope_h24",
    "log_po2_fraction_nonnull_h24",
    "log_po2_all_missing_h24",
    "log_po2_min_h24",
    "log_po2_max_h24",
    "k_ffilled",
    "k_missing",
    "k_sq_ffilled",
    "k_mean_h24",
    "k_sq_mean_h24",
    "k_std_h24",
    "k_slope_h24",
    "k_fraction_nonnull_h24",
    "k_all_missing_h24",
    "k_min_h24",
    "k_max_h24",
    "na_ffilled",
    "na_missing",
    "na_sq_ffilled",
    "na_mean_h24",
    "na_sq_mean_h24",
    "na_std_h24",
    "na_slope_h24",
    "na_fraction_nonnull_h24",
    "na_all_missing_h24",
    "na_min_h24",
    "na_max_h24",
    "pco2_ffilled",
    "pco2_missing",
    "pco2_sq_ffilled",
    "pco2_mean_h24",
    "pco2_sq_mean_h24",
    "pco2_std_h24",
    "pco2_slope_h24",
    "pco2_fraction_nonnull_h24",
    "pco2_all_missing_h24",
    "pco2_min_h24",
    "pco2_max_h24",
    "log_wbc_ffilled",
    "log_wbc_missing",
    "log_wbc_sq_ffilled",
    "log_wbc_mean_h24",
    "log_wbc_sq_mean_h24",
    "log_wbc_std_h24",
    "log_wbc_slope_h24",
    "log_wbc_fraction_nonnull_h24",
    "log_wbc_all_missing_h24",
    "log_wbc_min_h24",
    "log_wbc_max_h24",
    "tgcs_ffilled",
    "tgcs_missing",
    "tgcs_sq_ffilled",
    "tgcs_mean_h24",
    "tgcs_sq_mean_h24",
    "tgcs_std_h24",
    "tgcs_slope_h24",
    "tgcs_fraction_nonnull_h24",
    "tgcs_all_missing_h24",
    "tgcs_min_h24",
    "tgcs_max_h24",
    "hct_ffilled",
    "hct_missing",
    "hct_sq_ffilled",
    "hct_mean_h24",
    "hct_sq_mean_h24",
    "hct_std_h24",
    "hct_slope_h24",
    "hct_fraction_nonnull_h24",
    "hct_all_missing_h24",
    "hct_min_h24",
    "hct_max_h24",
]

# I use `scan_parquet` instead of `read_parquet`. `read_parquet` would (i) load the
# entire features.parquet, then (ii) filter the rows and then (iii) filter the columns.
# The approach taken here only loads the relevant rows and columns into memory.
# Still, this takes a minute.
df_source = (
    pl.scan_parquet(f"/cluster/work/math/jaegerl/data/{source}/features.parquet")
    .filter(pl.col("split").is_in(["train", "val"]) & pl.col(outcome).is_not_null())
    .select(features + [outcome])
    .collect()
)

y = df_source[outcome].to_numpy()
df_source = df_source.drop(outcome)

continuous_variables = [
    col
    for col, dtype in df_source.schema.items()
    if dtype.is_float() or dtype.is_integer()
]
other = [c for c in df_source.columns if c not in continuous_variables]

# Have a quick look at the documentation of the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("continuous", "passthrough", continuous_variables),
        (
            "categorical",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                # LGBM from pyarrow allows only int, bool, float types. So we
                # have to transform `airway` from str to int. Unknown value must
                # be an int. 99 works since we should never have so many
                # categories.
                unknown_value=99,
            ),
            other,
        ),
    ]
).set_output(transform="polars")

df_source = preprocessor.fit_transform(df_source)
model = DRO(
    num_boost_round=100,
    params={"objective": "binary"},
    rho=rho,
    categorical_feature=[c for c in df_source.columns if "categorical" in c],
)
model.fit(df_source, y)

df_target = (
    pl.scan_parquet(
        f"/cluster/work/math/jaegerl/data/{target}/features.parquet",
    )
    .filter(pl.col("split").eq("test") & pl.col(outcome).is_not_null())
    .select(features + [outcome])
    .collect()
)
yhat = model.predict(preprocessor.transform(df_target))
y = df_target[outcome].to_numpy()

auprc = average_precision_score(y, yhat)
auroc = roc_auc_score(y, yhat)

print(f"rho={rho}, AuPRC={auprc}, AuROC={auroc}")
