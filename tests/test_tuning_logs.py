import io
import unittest
from contextlib import redirect_stdout

import optuna

from pricing_lab.tuning import create_study, optimize_with_logs


class TuningLogsTest(unittest.TestCase):
    def test_optimize_with_logs_reports_trial_start_and_finish(self) -> None:
        study: optuna.Study = create_study()

        def objective(trial: optuna.Trial) -> float:
            return float(trial.number + 1)

        output = io.StringIO()
        with redirect_stdout(output):
            optimize_with_logs(study, objective, "TinyModel", n_trials=2)

        logs = output.getvalue()
        assert "[TinyModel] trial 1/2 started" in logs
        assert "[TinyModel] trial 1/2 finished elapsed=" in logs
        assert "rmse_log=1.000000 best=1.000000 params={}" in logs
        assert "rmse_log=2.000000 best=1.000000 params={}" in logs


if __name__ == "__main__":
    unittest.main()
