# ==========================================
# 0. IMPORTS
# ==========================================
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# RISK STATUS CONFIGURATION
# ============================================================
NOMINAL_MAX = 25.0
WARNING_MAX = 55.0

def get_risk_status(risk_score):
    """Convert numerical risk into a mission status."""
    if risk_score < NOMINAL_MAX:
        return "NOMINAL"
    elif risk_score < WARNING_MAX:
        return "WARNING"
    else:
        return "CRITICAL"

# ============================================================
# PERSONAL ASTRONAUT DIGITAL TWIN
# ============================================================

FEATURE_ORDER = [
    "heart_rate",
    "hrv",
    "spo2",
    "resp_rate",
    "core_temp",
    "ambient_co2",
    "pressure",
    "rad_rate",
    "sleep_debt",
]

STATUS_LABELS = {
    0: "NOMINAL",
    1: "WARNING",
    2: "CRITICAL",
}

# These are SIMULATED crew-specific baseline offsets.
# They are not clinical measurements.
ASTRONAUT_PROFILE_OFFSETS = {
    "Cmdr. Vance (EVA Specialist)": {
        "heart_rate": -3.0,
        "hrv": 4.0,
        "spo2": 0.2,
        "resp_rate": -0.4,
        "core_temp": 0.0,
        "ambient_co2": -30.0,
        "pressure": 0.1,
        "rad_rate": 0.0,
        "sleep_debt": 0.2,
    },

    "Dr. Aris (Payload Specialist)": {
        "heart_rate": 2.0,
        "hrv": -3.0,
        "spo2": 0.0,
        "resp_rate": 0.3,
        "core_temp": 0.1,
        "ambient_co2": 40.0,
        "pressure": -0.1,
        "rad_rate": 1.0,
        "sleep_debt": 0.5,
    },

    "Lt. Kovacs (Flight Engineer)": {
        "heart_rate": -1.0,
        "hrv": 6.0,
        "spo2": 0.1,
        "resp_rate": -0.2,
        "core_temp": -0.05,
        "ambient_co2": -20.0,
        "pressure": 0.05,
        "rad_rate": -1.0,
        "sleep_debt": -0.1,
    },
}

# Tolerance used for SIMULATED baseline deviation.
BASELINE_TOLERANCES = {
    "heart_rate": 8.0,
    "hrv": 12.0,
    "spo2": 0.8,
    "resp_rate": 2.0,
    "core_temp": 0.3,
    "ambient_co2": 150.0,
    "pressure": 0.8,
    "rad_rate": 5.0,
    "sleep_debt": 0.75,
}


def build_personal_baseline(base_sample, astronaut_name):
    """Build a simulated astronaut-specific baseline."""

    offsets = ASTRONAUT_PROFILE_OFFSETS.get(
        astronaut_name,
        {}
    )

    baseline = {}

    for feature in FEATURE_ORDER:
        value = float(base_sample[feature])
        value += float(offsets.get(feature, 0.0))
        baseline[feature] = value

    # Keep the simulated baseline physically reasonable.
    baseline["spo2"] = np.clip(
        baseline["spo2"], 90.0, 100.0
    )

    baseline["core_temp"] = np.clip(
        baseline["core_temp"], 34.0, 40.0
    )

    baseline["pressure"] = np.clip(
        baseline["pressure"], 80.0, 105.0
    )

    baseline["rad_rate"] = max(
        baseline["rad_rate"], 0.0
    )

    baseline["sleep_debt"] = max(
        baseline["sleep_debt"], 0.0
    )

    return baseline


def calculate_baseline_deviation(sample, baseline):
    """
    Calculate how far current telemetry has moved
    from the astronaut's simulated personal baseline.
    """

    deviations = []

    for feature in FEATURE_ORDER:
        difference = abs(
            float(sample[feature])
            - float(baseline[feature])
        )

        tolerance = BASELINE_TOLERANCES[feature]

        normalized = difference / max(tolerance, 1e-8)

        deviations.append(
            (feature, normalized)
        )

    deviation_score = min(
        100.0,
        (np.mean([x[1] for x in deviations]) * 25.0)
    )

    deviations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_drivers = [
        (
            feature,
            round(value * 100.0, 1)
        )
        for feature, value in deviations[:3]
    ]

    return round(deviation_score, 1), top_drivers

# ============================================================
# SENSOR INTEGRITY MONITOR
# ============================================================

SENSOR_PHYSICAL_LIMITS = {
    "heart_rate": (30.0, 180.0),
    "hrv": (0.0, 150.0),
    "spo2": (70.0, 100.0),
    "resp_rate": (5.0, 40.0),
    "core_temp": (30.0, 42.0),
    "ambient_co2": (200.0, 5000.0),
    "pressure": (70.0, 110.0),
    "rad_rate": (0.0, 500.0),
}

SENSOR_FLATLINE_TOLERANCE = {
    "heart_rate": 0.05,
    "hrv": 0.05,
    "spo2": 0.01,
    "resp_rate": 0.02,
    "core_temp": 0.001,
    "ambient_co2": 0.5,
    "pressure": 0.01,
    "rad_rate": 0.02,
}


def detect_sensor_faults(
    sample,
    history,
    baseline
):
    """
    Rule-based telemetry integrity layer.

    This is an engineering diagnostic, not a medical
    or flight-certified sensor validation system.
    """

    sensor_status = {}
    suspect_features = []

    for feature, (lower, upper) in (
        SENSOR_PHYSICAL_LIMITS.items()
    ):

        value = float(sample[feature])

        issues = []

        if not np.isfinite(value):
            issues.append("NON-FINITE")

        elif value < lower or value > upper:
            issues.append("PHYSICAL-LIMIT")

        recent_values = history.get(
            feature,
            []
        )[-5:]

        if (
            len(recent_values) >= 5
            and feature in SENSOR_FLATLINE_TOLERANCE
        ):

            spread = np.ptp(
                np.asarray(
                    recent_values,
                    dtype=float
                )
            )

            if spread <= SENSOR_FLATLINE_TOLERANCE[feature]:
                issues.append("FLATLINE")

        # Simulated cross-channel consistency checks.
        if feature == "pressure":

            pressure_delta = abs(
                value
                - float(baseline["pressure"])
            )

            spo2_delta = abs(
                float(sample["spo2"])
                - float(baseline["spo2"])
            )

            hr_delta = abs(
                float(sample["heart_rate"])
                - float(baseline["heart_rate"])
            )

            rr_delta = abs(
                float(sample["resp_rate"])
                - float(baseline["resp_rate"])
            )

            if (
                pressure_delta > 4.0
                and spo2_delta < 1.0
                and hr_delta < 12.0
                and rr_delta < 3.0
            ):
                issues.append(
                    "CROSS-CHANNEL-INCONSISTENCY"
                )

        if feature == "spo2":

            spo2_delta = abs(
                value
                - float(baseline["spo2"])
            )

            hr_delta = abs(
                float(sample["heart_rate"])
                - float(baseline["heart_rate"])
            )

            rr_delta = abs(
                float(sample["resp_rate"])
                - float(baseline["resp_rate"])
            )

            if (
                spo2_delta > 2.0
                and hr_delta < 12.0
                and rr_delta < 3.0
            ):
                issues.append(
                    "CROSS-CHANNEL-INCONSISTENCY"
                )

        if feature == "heart_rate":

            hr_delta = abs(
                value
                - float(baseline["heart_rate"])
            )

            spo2_delta = abs(
                float(sample["spo2"])
                - float(baseline["spo2"])
            )

            rr_delta = abs(
                float(sample["resp_rate"])
                - float(baseline["resp_rate"])
            )

            temp_delta = abs(
                float(sample["core_temp"])
                - float(baseline["core_temp"])
            )

            if (
                hr_delta > 25.0
                and spo2_delta < 1.0
                and rr_delta < 3.0
                and temp_delta < 0.5
            ):
                issues.append(
                    "POSSIBLE-HR-DRIFT"
                )

        if issues:
            suspect_features.append(
                feature
            )

        sensor_status[feature] = (
            "FAULT SUSPECTED"
            if issues
            else "HEALTHY"
        )

    overall_status = (
        "FAULT SUSPECTED"
        if suspect_features
        else "ALL CHANNELS NOMINAL"
    )

    return {
        "overall_status": overall_status,
        "sensor_status": sensor_status,
        "suspect_features": list(
            dict.fromkeys(suspect_features)
        ),
    }

# Set fixed seed for reproducibility during training
np.random.seed(42)
torch.manual_seed(42)


# ==========================================
# 1. PYTORCH NEURAL NETWORK ARCHITECTURE
# ==========================================
class AstronautHealthNet(nn.Module):

  def __init__(self, input_dim=9):
    super(AstronautHealthNet, self).__init__()
    self.feature_extractor = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
    )

    # Output Head 1: Risk Score Regression (0 to 100)
    self.risk_head = nn.Sequential(
        nn.Linear(16, 1),
        nn.Sigmoid(),  # Scale output to range [0, 1]
    )

    # Output Head 2: Status Classification (0: Nominal, 1: Warning, 2: Critical)
    self.status_head = nn.Linear(16, 3)

  def forward(self, x):
    features = self.feature_extractor(x)
    risk_score = self.risk_head(features) * 100.0  # Scale to 0-100
    status_logits = self.status_head(features)
    return risk_score, status_logits


# ==========================================
# 2. GENERATE AND SAVE CSV DATASETS
# ==========================================
def generate_and_save_csv_datasets(n_each=1000):
  """Generates 1,000 records per class profile and saves them to separate CSV files."""
  # --- 1. NOMINAL DATASET ---
  nom_dict = {
      "heart_rate": np.random.uniform(62, 88, n_each),
      "hrv": np.random.uniform(52, 95, n_each),
      "spo2": np.random.uniform(96.0, 99.8, n_each),
      "resp_rate": np.random.uniform(12, 17, n_each),
      "core_temp": np.random.uniform(36.6, 37.3, n_each),
      "ambient_co2": np.random.uniform(350, 850, n_each),
      "pressure": np.random.uniform(100.2, 102.0, n_each),
      "rad_rate": np.random.uniform(8.0, 22.0, n_each),
      "sleep_debt": np.random.uniform(0.0, 1.8, n_each),
  }
  df_nom = pd.DataFrame(nom_dict)
  df_nom.to_csv("nominal_telemetry.csv", index=False)

  # --- 2. WARNING DATASET ---
  warn_dict = {
      "heart_rate": np.random.uniform(85, 115, n_each),
      "hrv": np.random.uniform(30, 50, n_each),
      "spo2": np.random.uniform(93.0, 95.5, n_each),
      "resp_rate": np.random.uniform(18, 23, n_each),
      "core_temp": np.random.uniform(37.4, 38.0, n_each),
      "ambient_co2": np.random.uniform(950, 1800, n_each),
      "pressure": np.random.uniform(94.0, 100.0, n_each),
      "rad_rate": np.random.uniform(22.0, 55.0, n_each),
      "sleep_debt": np.random.uniform(2.0, 4.5, n_each),
  }
  df_warn = pd.DataFrame(warn_dict)
  df_warn.to_csv("warning_telemetry.csv", index=False)

  # --- 3. CRITICAL DATASET ---
  crit_dict = {
      "heart_rate": np.random.uniform(40, 140, n_each),
      "hrv": np.random.uniform(10, 30, n_each),
      "spo2": np.random.uniform(85.0, 92.5, n_each),
      "resp_rate": np.random.uniform(8, 30, n_each),
      "core_temp": np.random.uniform(34.0, 39.2, n_each),
      "ambient_co2": np.random.uniform(2000, 3500, n_each),
      "pressure": np.random.uniform(80.0, 92.0, n_each),
      "rad_rate": np.random.uniform(60.0, 150.0, n_each),
      "sleep_debt": np.random.uniform(4.5, 8.0, n_each),
  }
  df_crit = pd.DataFrame(crit_dict)
  df_crit.to_csv("critical_telemetry.csv", index=False)

  print(
      "✓ Created CSV files: nominal_telemetry.csv, warning_telemetry.csv,"
      " critical_telemetry.csv"
  )

def load_and_label_datasets():
  """Loads the 3 CSV files and computes ground-truth risk scores & status labels."""
  df_nom = pd.read_csv("nominal_telemetry.csv")
  df_warn = pd.read_csv("warning_telemetry.csv")
  df_crit = pd.read_csv("critical_telemetry.csv")

  df_all = pd.concat([df_nom, df_warn, df_crit], ignore_index=True)
  data = df_all.values

  risk_scores = []
  status_labels = []

  for row in data:
    score = 0.0

    # Heart Rate
    if row[0] < 50 or row[0] > 131:
      score += 35.0
    elif row[0] < 60 or row[0] > 90:
      score += 4.0

    # HRV
    if row[1] < 20:
      score += 35.0
    elif row[1] < 50:
      score += 4.0

    # SpO2
    if row[2] < 92.0:
      score += 40.0
    elif row[2] < 95.0:
      score += 5.0

    # Respiration Rate
    if row[3] < 10 or row[3] > 25:
      score += 30.0
    elif row[3] < 12 or row[3] > 18:
      score += 4.0

    # Core Temp
    if row[4] < 35.0 or row[4] > 38.4:
      score += 40.0
    elif row[4] < 36.5 or row[4] > 37.5:
      score += 5.0

    # Ambient CO2
    if row[5] > 2500:
      score += 35.0
    elif row[5] > 1000:
      score += 4.0

    # Cabin Pressure
    if row[6] < 90.0:
      score += 40.0
    elif row[6] < 100.0:
      score += 4.0

    # Radiation Rate
    if row[7] > 100.0:
      score += 45.0
    elif row[7] > 25.0:
      score += 5.0

    # Sleep Debt
    if row[8] > 5.0:
      score += 25.0
    elif row[8] > 2.0:
      score += 3.0

    score = min(score, 100.0)
    risk_scores.append(score)

    if score < 25.0:
      status_labels.append(0)
    elif score < 55.0:
      status_labels.append(1)
    else:
      status_labels.append(2)

# ==========================================
# 3. MODEL TRAINING ROUTINE
# ==========================================

  return (
      data,
      np.array(risk_scores, dtype=np.float32),
      np.array(status_labels, dtype=np.int64),
  )
# ==========================================
# 2.5 NUMPY-ONLY VALIDATION METRICS
# ==========================================

def calculate_classification_metrics(y_true, y_pred):
  """Calculate accuracy, macro precision, recall and F1 without sklearn."""

  accuracy = np.mean(y_true == y_pred)

  classes = np.unique(y_true)

  precisions = []
  recalls = []
  f1_scores = []

  for class_id in classes:

    true_positive = np.sum(
        (y_true == class_id) & (y_pred == class_id)
    )

    false_positive = np.sum(
        (y_true != class_id) & (y_pred == class_id)
    )

    false_negative = np.sum(
        (y_true == class_id) & (y_pred != class_id)
    )

    if (true_positive + false_positive) > 0:
      precision = true_positive / (
          true_positive + false_positive
      )
    else:
      precision = 0.0

    if (true_positive + false_negative) > 0:
      recall = true_positive / (
          true_positive + false_negative
      )
    else:
      recall = 0.0

    if (precision + recall) > 0:
      f1 = 2 * precision * recall / (precision + recall)
    else:
      f1 = 0.0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

  return {
      "accuracy": float(accuracy),
      "precision": float(np.mean(precisions)),
      "recall": float(np.mean(recalls)),
      "f1": float(np.mean(f1_scores)),
  }


def calculate_regression_metrics(y_true, y_pred):
  """Calculate MAE and RMSE without sklearn."""

  errors = y_true - y_pred

  mae = np.mean(np.abs(errors))

  rmse = np.sqrt(
      np.mean(errors ** 2)
  )

  return {
      "risk_mae": float(mae),
      "risk_rmse": float(rmse),
  }
  
# ==========================================
# 3. MODEL TRAINING ROUTINE
# ==========================================

def train_model():

  # ------------------------------------------------
  # Generate datasets if they do not exist
  # ------------------------------------------------

  if not (
      os.path.exists("nominal_telemetry.csv")
      and os.path.exists("warning_telemetry.csv")
      and os.path.exists("critical_telemetry.csv")
  ):
    generate_and_save_csv_datasets(n_each=1000)

  # ------------------------------------------------
  # Load complete dataset
  # ------------------------------------------------

  X, y_risk, y_status = load_and_label_datasets()

  # ------------------------------------------------
  # STRATIFIED 80/20 TRAIN-TEST SPLIT
  # NumPy only — no sklearn required
  # ------------------------------------------------

  rng = np.random.default_rng(42)

  train_indices = []
  test_indices = []

  for class_id in np.unique(y_status):

    class_indices = np.where(
        y_status == class_id
    )[0]

    rng.shuffle(class_indices)

    split_point = int(
        len(class_indices) * 0.80
    )

    train_indices.extend(
        class_indices[:split_point]
    )

    test_indices.extend(
        class_indices[split_point:]
    )

  train_indices = np.array(
      train_indices,
      dtype=np.int64
  )

  test_indices = np.array(
      test_indices,
      dtype=np.int64
  )

  rng.shuffle(train_indices)
  rng.shuffle(test_indices)

  # ------------------------------------------------
  # Separate training and test data
  # ------------------------------------------------

  X_train = X[train_indices]
  X_test = X[test_indices]

  y_risk_train = y_risk[train_indices]
  y_risk_test = y_risk[test_indices]

  y_status_train = y_status[train_indices]
  y_status_test = y_status[test_indices]

  # ------------------------------------------------
  # NORMALIZATION
  # IMPORTANT:
  # Calculate mean/std ONLY from training data
  # ------------------------------------------------

  mean = np.mean(
      X_train,
      axis=0
  )

  scale = np.std(
      X_train,
      axis=0
  )

  scale[scale == 0] = 1e-8

  X_train_scaled = (
      X_train - mean
  ) / scale

  X_test_scaled = (
      X_test - mean
  ) / scale

  # ------------------------------------------------
  # Convert training data to PyTorch tensors
  # ------------------------------------------------

  X_train_tensor = torch.tensor(
      X_train_scaled,
      dtype=torch.float32
  )

  y_risk_train_tensor = torch.tensor(
      y_risk_train,
      dtype=torch.float32
  ).unsqueeze(1)

  y_status_train_tensor = torch.tensor(
      y_status_train,
      dtype=torch.long
  )

  # ------------------------------------------------
  # Create model
  # ------------------------------------------------

  model = AstronautHealthNet(
      input_dim=9
  )

  optimizer = optim.Adam(
      model.parameters(),
      lr=0.002
  )

  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=100,
      gamma=0.5
  )

  criterion_mse = nn.MSELoss()

  criterion_ce = nn.CrossEntropyLoss()

  # ------------------------------------------------
  # TRAINING
  # ------------------------------------------------

  model.train()

  epochs = 300

  for epoch in range(epochs):

    optimizer.zero_grad()

    pred_risk, pred_status = model(
        X_train_tensor
    )

    loss_risk = criterion_mse(
        pred_risk,
        y_risk_train_tensor
    )

    loss_status = criterion_ce(
        pred_status,
        y_status_train_tensor
    )

    total_loss = (
        loss_risk
        + (loss_status * 2.0)
    )

    total_loss.backward()

    optimizer.step()

    scheduler.step()

  # ------------------------------------------------
  # VALIDATION / TESTING
  # ------------------------------------------------

  model.eval()

  with torch.no_grad():

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    predicted_risk, predicted_status = model(
        X_test_tensor
    )

    # Risk regression predictions
    predicted_risk_np = (
        predicted_risk
        .cpu()
        .numpy()
        .flatten()
    )

    # Status classification predictions
    predicted_status_np = (
        torch.argmax(
            predicted_status,
            dim=1
        )
        .cpu()
        .numpy()
    )

  # ------------------------------------------------
  # Calculate validation metrics
  # ------------------------------------------------

  classification_metrics = (
      calculate_classification_metrics(
          y_status_test,
          predicted_status_np
      )
  )

  regression_metrics = (
      calculate_regression_metrics(
          y_risk_test,
          predicted_risk_np
      )
  )

  confusion_matrix = np.zeros((3, 3), dtype=int)
  for actual, predicted in zip(y_status_test, predicted_status_np):
    confusion_matrix[int(actual), int(predicted)] += 1

  validation_metrics = {
      **classification_metrics,
      **regression_metrics,
      "train_samples": int(len(X_train)),
      "test_samples": int(len(X_test)),
      "confusion_matrix": confusion_matrix.tolist(),
  }

  # ------------------------------------------------
  # Print validation results
  # ------------------------------------------------

  print("\n==========================================")
  print("      ASTROHEALTH MODEL VALIDATION")
  print("==========================================")

  print(
      f"Training samples : {len(X_train)}"
  )

  print(
      f"Test samples     : {len(X_test)}"
  )

  print(
      f"Accuracy         : "
      f"{validation_metrics['accuracy'] * 100:.2f}%"
  )

  print(
      f"Macro Precision  : "
      f"{validation_metrics['precision'] * 100:.2f}%"
  )

  print(
      f"Macro Recall     : "
      f"{validation_metrics['recall'] * 100:.2f}%"
  )

  print(
      f"Macro F1         : "
      f"{validation_metrics['f1'] * 100:.2f}%"
  )

  print(
      f"Risk MAE         : "
      f"{validation_metrics['risk_mae']:.2f}"
  )

  print(
      f"Risk RMSE        : "
      f"{validation_metrics['risk_rmse']:.2f}"
  )

  print("==========================================\n")

  print(
      "✓ AstroHealth Training + Validation Complete! "
      "(300 Epochs) 👩‍🚀🧬"
  )

  return (
      model,
      mean,
      scale,
      validation_metrics
  )
  
# ==========================================
# 4. INFERENCE & ANSI UTILITIES
# ==========================================

def explain_risk_score(sample_dict, model, mean, scale):
  """Computes feature contributions to the predicted risk score using gradients * inputs."""
  feature_order = [
      "heart_rate",
      "hrv",
      "spo2",
      "resp_rate",
      "core_temp",
      "ambient_co2",
      "pressure",
      "rad_rate",
      "sleep_debt",
  ]

  input_vector = np.array(
      [[sample_dict[key] for key in feature_order]], dtype=np.float32
  )
  input_scaled = (input_vector - mean) / scale

  x_tensor = torch.tensor(
      input_scaled, dtype=torch.float32, requires_grad=True
  )

  model.eval()
  risk_score, _ = model(x_tensor)
  risk_score.backward()

  gradients = x_tensor.grad.detach().numpy()[0]
  contributions = np.abs(gradients * input_scaled[0])

  total_contrib = np.sum(contributions)
  percentages = (
      (contributions / total_contrib) * 100.0
      if total_contrib > 0
      else np.zeros_like(contributions)
  )

  feature_labels = {
      "heart_rate": "HR",
      "hrv": "HRV",
      "spo2": "SpO2",
      "resp_rate": "RR",
      "core_temp": "Temp",
      "ambient_co2": "CO2",
      "pressure": "Press",
      "rad_rate": "Rad",
      "sleep_debt": "Sleep Debt",
  }

  explained_features = sorted(
      [
          (feature_labels[key], round(pct, 1))
          for key, pct in zip(feature_order, percentages)
      ],
      key=lambda x: x[1],
      reverse=True,
  )

  # Format summary with top 3 drivers
  top_1, top_2, top_3 = (
      explained_features[0],
      explained_features[1],
      explained_features[2],
  )
  summary = f"Drivers: {top_1[0]} ({top_1[1]}%), {top_2[0]} ({top_2[1]}%), {top_3[0]} ({top_3[1]}%)"

  return summary, explained_features

class TerminalColors:
  GREEN = "\033[92m"
  YELLOW = "\033[93m"
  BOLD_RED = "\033[1;91m"
  RESET = "\033[0m"

def format_status(status):
  if status == "NOMINAL":
    return f"{TerminalColors.GREEN}[NOMINAL]{TerminalColors.RESET}"
  elif status == "WARNING":
    return f"{TerminalColors.YELLOW}[WARNING]{TerminalColors.RESET}"
  else:
    return f"{TerminalColors.BOLD_RED}[CRITICAL]{TerminalColors.RESET}"

def predict_health_outputs(sample_dict, model, mean, scale):
    """
    Run both neural-network heads:
    1. Risk regression head
    2. Status classification head
    """

    input_vector = np.array(
        [[sample_dict[key] for key in FEATURE_ORDER]],
        dtype=np.float32
    )

    input_scaled = (input_vector - mean) / scale

    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(
            input_scaled,
            dtype=torch.float32
        )

        risk_tensor, status_logits = model(x_tensor)

        probabilities = torch.softmax(
            status_logits,
            dim=1
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

    risk_score = float(
        risk_tensor.item()
    )

    risk_status = get_risk_status(
        risk_score
    )

    classification_status = STATUS_LABELS[
        predicted_class
    ]

    confidence = float(
        torch.max(probabilities).item() * 100.0
    )

    model_agreement = (
        risk_status == classification_status
    )

    return {
        "risk_score": round(risk_score, 1),
        "risk_status": risk_status,
        "classification_status": classification_status,
        "confidence": round(confidence, 1),
        "model_agreement": model_agreement,
        "probabilities": probabilities[
            0
        ].cpu().numpy(),
    }


def predict_health_risk(sample_dict, model, mean, scale):
    """
    Backward-compatible wrapper used by the existing dashboard.
    """

    outputs = predict_health_outputs(
        sample_dict,
        model,
        mean,
        scale
    )

    return (
        outputs["risk_score"],
        outputs["risk_status"]
    )

def validate_external_dataset(
    df,
    model,
    mean,
    scale
):
    """
    Validate the model against an external dataset
    after the dataset has been normalized to the
    AstroHealth telemetry schema.

    Required columns:
        FEATURE_ORDER

    Optional target columns:
        risk_score / risk
        status / y_status
    """

    missing = [
        feature
        for feature in FEATURE_ORDER
        if feature not in df.columns
    ]

    if missing:
        return {
            "success": False,
            "error": (
                "Missing required columns: "
                + ", ".join(missing)
            )
        }

    clean_df = df.copy()

    for feature in FEATURE_ORDER:
        clean_df[feature] = pd.to_numeric(
            clean_df[feature],
            errors="coerce"
        )

    clean_df = clean_df.dropna(
        subset=FEATURE_ORDER
    )

    if len(clean_df) == 0:
        return {
            "success": False,
            "error": "No valid telemetry rows remain."
        }

    predictions = []

    for _, row in clean_df.iterrows():

        sample = {
            feature: float(row[feature])
            for feature in FEATURE_ORDER
        }

        outputs = predict_health_outputs(
            sample,
            model,
            mean,
            scale
        )

        predictions.append(outputs)

    result = clean_df.copy()

    result["predicted_risk"] = [
        item["risk_score"]
        for item in predictions
    ]

    result["predicted_status"] = [
        item["classification_status"]
        for item in predictions
    ]

    result["prediction_confidence"] = [
        item["confidence"]
        for item in predictions
    ]

    metrics = {
        "rows_evaluated": len(result)
    }

    # Optional risk target
    risk_column = None

    if "risk_score" in result.columns:
        risk_column = "risk_score"
    elif "risk" in result.columns:
        risk_column = "risk"

    if risk_column is not None:

        actual_risk = pd.to_numeric(
            result[risk_column],
            errors="coerce"
        )

        valid = (
            actual_risk.notna()
            & result["predicted_risk"].notna()
        )

        if valid.sum() > 0:

            errors = (
                actual_risk[valid].to_numpy()
                - result.loc[
                    valid,
                    "predicted_risk"
                ].to_numpy()
            )

            metrics["risk_mae"] = float(
                np.mean(np.abs(errors))
            )

            metrics["risk_rmse"] = float(
                np.sqrt(
                    np.mean(errors ** 2)
                )
            )

    # Optional status target
    status_column = None

    if "status" in result.columns:
        status_column = "status"
    elif "y_status" in result.columns:
        status_column = "y_status"

    if status_column is not None:

        actual_status = (
            result[status_column]
            .astype(str)
            .str.upper()
        )

        predicted_status = (
            result["predicted_status"]
            .astype(str)
            .str.upper()
        )

        metrics["status_accuracy"] = float(
            np.mean(
                actual_status.to_numpy()
                == predicted_status.to_numpy()
            )
        )

    return {
        "success": True,
        "metrics": metrics,
        "predictions": result
    }
    
def run_robustness_suite(
    model,
    mean,
    scale,
    baseline
):
    """
    Engineering robustness tests.

    These tests evaluate model behavior under:
    - sensor noise
    - sensor dropout
    - compound anomalies
    - sensor spikes
    """

    rng = np.random.default_rng(123)

    scenarios = []

    # ------------------------------------------------
    # 1. Nominal baseline
    # ------------------------------------------------

    nominal = baseline.copy()

    output = predict_health_outputs(
        nominal,
        model,
        mean,
        scale
    )

    scenarios.append({
        "Scenario": "Nominal Baseline",
        "Risk": output["risk_score"],
        "Status": output["risk_status"],
        "Confidence": output["confidence"],
    })

    # ------------------------------------------------
    # 2. 5x sensor noise
    # ------------------------------------------------

    noisy_5x = baseline.copy()

    for feature in FEATURE_ORDER:

        noisy_5x[feature] += rng.normal(
            0,
            BASELINE_TOLERANCES.get(
                feature,
                1.0
            ) * 0.10
        )

    output = predict_health_outputs(
        noisy_5x,
        model,
        mean,
        scale
    )

    scenarios.append({
        "Scenario": "10% Baseline-Tolerance Noise",
        "Risk": output["risk_score"],
        "Status": output["risk_status"],
        "Confidence": output["confidence"],
    })

    # ------------------------------------------------
    # 3. 10x sensor noise
    # ------------------------------------------------

    noisy_10x = baseline.copy()

    for feature in FEATURE_ORDER:

        noisy_10x[feature] += rng.normal(
            0,
            BASELINE_TOLERANCES.get(
                feature,
                1.0
            ) * 0.20
        )

    output = predict_health_outputs(
        noisy_10x,
        model,
        mean,
        scale
    )

    scenarios.append({
        "Scenario": "20% Baseline-Tolerance Noise",
        "Risk": output["risk_score"],
        "Status": output["risk_status"],
        "Confidence": output["confidence"],
    })

    # ------------------------------------------------
    # 4. SpO2 sensor dropout
    # ------------------------------------------------

    spo2_dropout = baseline.copy()

    # Explicitly represent missing telemetry for the robustness test.
    spo2_dropout["spo2"] = np.nan

    output = predict_health_outputs(
        spo2_dropout,
        model,
        mean,
        scale
    )

    scenarios.append({
        "Scenario": "SpO2 Sensor Dropout",
        "Risk": output["risk_score"],
        "Status": output["risk_status"],
        "Confidence": output["confidence"],
    })

    # ------------------------------------------------
    # 5. Compound CO2 + heat
    # ------------------------------------------------

    compound = baseline.copy()

    compound["ambient_co2"] += 1800
    compound["core_temp"] += 1.5
    compound["heart_rate"] += 20
    compound["resp_rate"] += 6
    compound["hrv"] -= 15

    output = predict_health_outputs(
        compound,
        model,
        mean,
        scale
    )

    scenarios.append({
        "Scenario": "CO2 + Heat Compound",
        "Risk": output["risk_score"],
        "Status": output["risk_status"],
        "Confidence": output["confidence"],
    })

    # ------------------------------------------------
    # 6. Pressure sensor spike
    # ------------------------------------------------

    pressure_spike = baseline.copy()

    pressure_spike["pressure"] += 8

    output = predict_health_outputs(
        pressure_spike,
        model,
        mean,
        scale
    )

    scenarios.append({
        "Scenario": "Pressure Sensor Spike",
        "Risk": output["risk_score"],
        "Status": output["risk_status"],
        "Confidence": output["confidence"],
    })

    # ------------------------------------------------
    # 7. Radiation + CO2
    # ------------------------------------------------

    radiation_co2 = baseline.copy()

    radiation_co2["rad_rate"] += 90
    radiation_co2["ambient_co2"] += 1200
    radiation_co2["heart_rate"] += 12
    radiation_co2["hrv"] -= 12

    output = predict_health_outputs(
        radiation_co2,
        model,
        mean,
        scale
    )

    scenarios.append({
        "Scenario": "Radiation + CO2 Compound",
        "Risk": output["risk_score"],
        "Status": output["risk_status"],
        "Confidence": output["confidence"],
    })

    return scenarios

# ==========================================
# 5. LIVE VISUALIZATION STREAM
# ==========================================

def run_live_telemetry_all_params_viz(
    model,
    mean,
    scale,
    df_pool,
    total_ticks=50,
    interval_sec=0.5,
    predict_fn=predict_health_risk,
):
  ticks_x = []
  risk_scores = []

  # Dictionary to store all 9 parameter histories
  param_history = {
      "heart_rate": [],
      "hrv": [],
      "spo2": [],
      "resp_rate": [],
      "core_temp": [],
      "ambient_co2": [],
      "pressure": [],
      "rad_rate": [],
      "sleep_debt": [],
  }

  print("\n==================================================")
  print("      LIVE ASTRONAUT TELEMETRY MONITOR 👩‍🚀🧬           ")
  print("==================================================")

  # Collect telemetry data over ticks
  for tick in range(1, total_ticks + 1):
    random_row = df_pool.sample(n=1, random_state=None).iloc[0]
    live_telemetry = random_row.to_dict()

    score, status = predict_fn(live_telemetry, model, mean, scale)
    colored_status = format_status(status)

    hr = int(live_telemetry["heart_rate"])
    hrv = float(live_telemetry["hrv"])
    spo2 = float(live_telemetry["spo2"])
    resp = float(live_telemetry["resp_rate"])
    temp = float(live_telemetry["core_temp"])
    co2 = int(live_telemetry["ambient_co2"])
    press = float(live_telemetry["pressure"])
    rad = float(live_telemetry["rad_rate"])
    sleep = float(live_telemetry["sleep_debt"])

    # Get explanation
    explanation_summary, _ = explain_risk_score(
        live_telemetry, model, mean, scale
        )

    # Combined Print (Values + Status + Explanation)
    print(
        f"[Tick {tick:02d}] HR:{hr:>3d} | HRV:{hrv:>4.1f} | SpO2:{spo2:>4.1f}%"
        f" | RR:{resp:>4.1f} | Temp:{temp:>4.1f}°C | CO2:{co2:>4d} |"
        f" Press:{press:>5.1f} | Rad:{rad:>5.1f} | Sleep:{sleep:>3.1f}h |"
        f" Risk: {score:>4.1f}/100 {colored_status} | {explanation_summary}"
      )

    ticks_x.append(tick)
    risk_scores.append(score)

    for p in param_history:
      param_history[p].append(float(live_telemetry[p]))

    time.sleep(interval_sec)

  # --- MULTI-SUBPLOT RENDER (ALL 9 PARAMETERS) ---
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

  # SUBPLOT 1: Model Risk Score Output
  ax1.plot(
      ticks_x,
      risk_scores,
      color="darkred",
      marker="o",
      linewidth=2.5,
      label="Risk Score",
  )
  ax1.set_ylabel("Risk (0–100)", color="darkred", fontweight="bold")
  ax1.set_ylim(-5, 105)
  ax1.grid(True, alpha=0.3)
  ax1.legend(loc="upper right")

  # SUBPLOT 2: Cardiovascular & Vital Metrics (HR, HRV, SpO2, Resp Rate, Core Temp)
  ax2.plot(
      ticks_x,
      param_history["heart_rate"],
      label="HR (bpm)",
      color="#e74c3c",
      linewidth=1.5,
  )
  ax2.plot(
      ticks_x,
      param_history["spo2"],
      label="SpO2 (%)",
      color="#2ecc71",
      linewidth=1.5,
  )
  ax2.plot(
      ticks_x,
      param_history["hrv"],
      label="HRV (ms)",
      color="#8e44ad",
      linewidth=1.5,
  )
  ax2.plot(
      ticks_x,
      param_history["resp_rate"],
      label="Resp Rate (br/min)",
      color="#3498db",
      linestyle="--",
  )

  # Secondary y-axis for Core Temp on Subplot 2
  ax2_temp = ax2.twinx()
  ax2_temp.plot(
      ticks_x,
      param_history["core_temp"],
      label="Core Temp (°C)",
      color="#e67e22",
      linestyle="-.",
      linewidth=1.5,
  )
  ax2_temp.set_ylabel("Temp (°C)", color="#e67e22", fontweight="bold")

  ax2.set_ylabel("Primary Vitals", fontweight="bold")
  ax2.grid(True, alpha=0.3)

  # Combine legends for Subplot 2
  lines_2, labels_2 = ax2.get_legend_handles_labels()
  lines_2_t, labels_2_t = ax2_temp.get_legend_handles_labels()
  ax2.legend(lines_2 + lines_2_t, labels_2 + labels_2_t, loc="upper right")

  # SUBPLOT 3: Environmental Factors & Stressors (CO2, Pressure, Radiation, Sleep Debt)
  ax3.plot(
      ticks_x,
      param_history["ambient_co2"],
      label="CO2 (ppm)",
      color="#16a085",
      linestyle="--",
  )
  ax3.set_ylabel("CO2 (ppm)", color="#16a085", fontweight="bold")
  ax3.set_xlabel("Tick", fontweight="bold")
  ax3.grid(True, alpha=0.3)

  # Secondary y-axis for Pressure, Radiation, and Sleep Debt on Subplot 3
  ax3_env = ax3.twinx()
  ax3_env.plot(
      ticks_x,
      param_history["pressure"],
      label="Pressure (kPa)",
      color="#2c3e50",
      linewidth=1.5,
  )
  ax3_env.plot(
      ticks_x,
      param_history["rad_rate"],
      label="Rad Rate (uSv/h)",
      color="#f39c12",
      linestyle=":",
      linewidth=1.5,
  )
  ax3_env.plot(
      ticks_x,
      param_history["sleep_debt"],
      label="Sleep Debt (h)",
      color="#7f8c8d",
      linestyle="-.",
      linewidth=1.5,
  )
  ax3_env.set_ylabel("Pressure / Rad / Sleep", color="#2c3e50", fontweight="bold")

  # Combine legends for Subplot 3
  lines_3, labels_3 = ax3.get_legend_handles_labels()
  lines_3_e, labels_3_e = ax3_env.get_legend_handles_labels()
  ax3.legend(lines_3 + lines_3_e, labels_3 + labels_3_e, loc="upper right")

  plt.suptitle("AstroHealth Live Telemetry & Net Risk Score", fontsize=14)
  plt.tight_layout()
  plt.show()
  
  
# ==========================================
# 6. PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
  # 1. Train model from generated/loaded CSV files
  model, mean, scale, validation_metrics = train_model()

  # 2. Reset random seed using time for unseeded live streaming draws
  np.random.seed(int(time.time()))

  # 3. Load pooled telemetry datasets
  df_nom = pd.read_csv("nominal_telemetry.csv")
  df_warn = pd.read_csv("warning_telemetry.csv")
  df_crit = pd.read_csv("critical_telemetry.csv")
  telemetry_pool = pd.concat([df_nom, df_warn, df_crit], ignore_index=True)

  # 4. Run continuous live stream + plot loop
  run_live_telemetry_all_params_viz(
      model=model,
      mean=mean,
      scale=scale,
      df_pool=telemetry_pool,
      total_ticks=50,
      interval_sec=0.5,
  )