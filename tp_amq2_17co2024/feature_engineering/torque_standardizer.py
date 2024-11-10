from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re


class TorqueStandardizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def standardize_torque(torque_str):
            if pd.isna(torque_str):
                return {'torque_peak_power': np.nan, 'torque_peak_speed': np.nan}

            patterns = [
                r"(\d*\.?\d+)\s*(kgm|nm)?\s*@\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*(kgm|nm)?\s*at\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*@\s*([-\d\s,]+)\s*\(kgm@\s*rpm\)",
                r"(\d*\.?\d+)\s*kgm\s*at\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*\((\d*\.?\d+)\s*kgm\)\s*@\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*\((\d*\.?\d+)\)\s*@\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*(kgm|nm)?\s*/\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*/\s*([-\d\s,]+)"
            ]

            torque_peak_power = None
            torque_peak_speed = None

            for pattern in patterns:
                match = re.findall(pattern, str(torque_str).lower())
                if match:
                    if len(match[0]) == 4:
                        value, unit, rpm_range, _ = match[0]
                    elif len(match[0]) == 3:
                        value, unit, rpm_range = match[0]
                    elif len(match[0]) == 2:
                        value, rpm_range = match[0]
                        unit = None
                    else:
                        continue

                    value = float(value)

                    if 'kgm' in str(torque_str).lower() or (unit and 'kgm' in unit):
                        value *= 9.81

                    torque_peak_power = value

                    if rpm_range:
                        rpm_range = rpm_range.replace(',', '')
                        if '-' in rpm_range:
                            rpm_values = list(map(int, rpm_range.split('-')))
                            torque_peak_speed = max(rpm_values)
                        else:
                            torque_peak_speed = int(rpm_range.strip())

                    break
            return {'torque_peak_power': torque_peak_power, 'torque_peak_speed': torque_peak_speed}

        X = X.copy()
        torque_results = X['torque'].apply(standardize_torque)
        X['torque_peak_power'] = torque_results.apply(lambda x: x['torque_peak_power'])
        X['torque_peak_speed'] = torque_results.apply(lambda x: x['torque_peak_speed'])
        return X.drop(columns=['torque'])
