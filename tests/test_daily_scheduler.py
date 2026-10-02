import unittest
from datetime import datetime, timezone

from src.daily_scheduler import PERU_TIMEZONE, next_run_at


class DailySchedulerTests(unittest.TestCase):
    def test_schedules_today_when_before_eight_in_peru(self):
        now = datetime(2026, 10, 1, 7, 30, tzinfo=PERU_TIMEZONE)

        self.assertEqual(
            next_run_at(now),
            datetime(2026, 10, 1, 8, 0, tzinfo=PERU_TIMEZONE),
        )

    def test_schedules_next_day_when_eight_has_passed(self):
        now = datetime(2026, 10, 1, 8, 1, tzinfo=PERU_TIMEZONE)

        self.assertEqual(
            next_run_at(now),
            datetime(2026, 10, 2, 8, 0, tzinfo=PERU_TIMEZONE),
        )

    def test_converts_input_time_to_peru_before_scheduling(self):
        utc_time = datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)

        self.assertEqual(
            next_run_at(utc_time),
            datetime(2026, 10, 1, 8, 0, tzinfo=PERU_TIMEZONE),
        )


if __name__ == "__main__":
    unittest.main()