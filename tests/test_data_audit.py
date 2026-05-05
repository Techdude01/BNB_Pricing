import pandas as pd
import unittest

from pricing_lab.data import compute_price_cap_audit


class DataAuditTest(unittest.TestCase):
    def test_compute_price_cap_audit_reports_excluded_high_price_rows(self) -> None:
        raw_frame = pd.DataFrame(
            {
                "name": ["a", "b", "c", "d", "e"],
                "host_name": ["h1", "h2", "h3", "h4", "h5"],
                "reviews_per_month": [None, 1.0, 2.0, None, 3.0],
                "last_review": [None, "2019-01-01", "2019-01-02", None, "2019-01-03"],
                "price": [50.0, 60.0, 70.0, 80.0, 500.0],
                "neighbourhood_group": ["Bronx", "Bronx", "Queens", "Queens", "Manhattan"],
                "room_type": ["Private room", "Private room", "Private room", "Entire home/apt", "Entire home/apt"],
            }
        )

        audit = compute_price_cap_audit(raw_frame)

        excluded = audit.summary.loc[audit.summary["scope"] == "excluded_above_iqr_cap"].iloc[0]
        assert excluded["rows"] == 1
        assert excluded["max_price"] == 500.0
        assert audit.excluded_by_segment.iloc[0]["neighbourhood_group"] == "Manhattan"


if __name__ == "__main__":
    unittest.main()
