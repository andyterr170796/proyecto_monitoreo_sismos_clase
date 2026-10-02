import logging
import subprocess
import sys
import time
from datetime import datetime, time as datetime_time, timedelta, timezone

from src.config.config import PATHS, setup_logging


PERU_TIMEZONE = timezone(timedelta(hours=-5), name="America/Lima")
DAILY_RUN_TIME = datetime_time(hour=8)


def next_run_at(now=None):
    now = now or datetime.now(PERU_TIMEZONE)
    if now.tzinfo is None:
        now = now.replace(tzinfo=PERU_TIMEZONE)
    else:
        now = now.astimezone(PERU_TIMEZONE)

    scheduled = now.replace(
        hour=DAILY_RUN_TIME.hour,
        minute=DAILY_RUN_TIME.minute,
        second=0,
        microsecond=0,
    )
    if scheduled <= now:
        scheduled += timedelta(days=1)
    return scheduled


def run_scheduler():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Scheduler activo; zona horaria: America/Lima; ejecución diaria: 08:00")

    while True:
        scheduled = next_run_at()
        seconds_until_run = max(0, (scheduled - datetime.now(PERU_TIMEZONE)).total_seconds())
        logger.info("Próxima ejecución del pipeline: %s", scheduled.isoformat())
        time.sleep(seconds_until_run)

        logger.info("Iniciando actualización diaria del pipeline")
        try:
            subprocess.run(
                [sys.executable, "-m", "src.main_pipeline"],
                cwd=PATHS["project_root"],
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.exception("Falló la ejecución diaria; se reintentará al día siguiente")


if __name__ == "__main__":
    run_scheduler()