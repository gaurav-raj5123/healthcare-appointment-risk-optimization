"""
Test Suite 5: Feature Schema Parity Tests
Verifies that the dashboard's approved feature list matches:
  - Step 3 feature_metadata.json
  - Step 3 train_features.csv
  - Step 4 serialized champion model pipeline input transformers
Confirms exclusions: sms_received, patient_id, appointment_id.
"""

import json
import pandas as pd
import joblib
from app.config import APPROVED_FEATURES, FEATURE_METADATA_PATH, MODEL_PATH, TRAIN_FEATURES_PATH

def test_feature_list_matches_metadata():
    with open(FEATURE_METADATA_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    expected_list = meta['features']['explicit_list']
    assert APPROVED_FEATURES == expected_list, "APPROVED_FEATURES does not match feature_metadata.json"

def test_feature_count_is_exactly_17():
    assert len(APPROVED_FEATURES) == 17, f"Expected 17 features, found {len(APPROVED_FEATURES)}"
    assert len(set(APPROVED_FEATURES)) == 17, "Duplicate features detected in APPROVED_FEATURES"

def test_features_in_train_dataset():
    df_train = pd.read_csv(TRAIN_FEATURES_PATH, nrows=5)
    for feat in APPROVED_FEATURES:
        assert feat in df_train.columns, f"Feature {feat} missing from train_features.csv"

def test_day0_excluded_features():
    # sms_received, patient_id, appointment_id must never enter APPROVED_FEATURES
    assert "sms_received" not in APPROVED_FEATURES, "sms_received illegally present in Day-0 predictors"
    assert "patient_id" not in APPROVED_FEATURES, "patient_id illegally present in predictors"
    assert "appointment_id" not in APPROVED_FEATURES, "appointment_id illegally present in predictors"

def test_model_pipeline_feature_schema():
    pipe = joblib.load(MODEL_PATH)
    prep = pipe.named_steps['prep']
    
    # Collect all input columns from the ColumnTransformer
    input_cols = []
    for name, trans, cols in prep.transformers_:
        if name != 'remainder':
            input_cols.extend(cols)
            
    assert set(input_cols) == set(APPROVED_FEATURES), (
        f"Mismatch between pipeline ColumnTransformer inputs and APPROVED_FEATURES.\n"
        f"Diff: {set(input_cols) ^ set(APPROVED_FEATURES)}"
    )

