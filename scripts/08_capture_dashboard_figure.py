"""Capture the README's dashboard figure headlessly, reproducibly.

Drives the running Streamlit app to the one state that is worth a figure:
MSL selected, replay advanced, drift acknowledged -- so a **red** model
quality strip (ROC-AUC 0.4688, below chance) sits directly above a **green**
"no distribution shift detected" banner. That juxtaposition is the visual
proof of the project's third finding: drift monitoring cannot substitute for
a model-quality signal.

Start the app first:
    python -m streamlit run app/streamlit_app.py --server.port 8502

Then:
    python scripts/08_capture_dashboard_figure.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "dashboard_msl_drift_vs_quality.png"


def click_text(driver, wait, text: str, tag: str = "*") -> bool:
    try:
        el = wait.until(EC.element_to_be_clickable(
            (By.XPATH, f"//{tag}[normalize-space(text())='{text}']")
        ))
        driver.execute_script("arguments[0].click();", el)
        return True
    except TimeoutException:
        return False


def select_option(driver, wait, field: str, option: str) -> None:
    """Pick ``option`` from the selectbox labelled ``field``.

    Driven through the inner ``input[role=combobox]``, whose aria-label reads
    "Selected <value>. <field>". A synthetic click on the BaseWeb wrapper does
    not open the menu -- no options ever render -- so this types the value and
    presses Enter, the same path a keyboard user takes. Matching on the label
    rather than a positional index also survives controls being reordered.

    Raises rather than returning False: a silently-failed selection yields a
    screenshot of the wrong dataset, which is worse than no screenshot.
    """
    xpath = f"//input[@role='combobox'][contains(@aria-label, '{field}')]"
    inp = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
    driver.execute_script("arguments[0].scrollIntoView(true);", inp)
    inp.click()
    time.sleep(0.8)
    inp.send_keys(option)
    time.sleep(1.2)
    inp.send_keys(Keys.ENTER)
    time.sleep(4)

    label = driver.find_element(By.XPATH, xpath).get_attribute("aria-label") or ""
    if f"Selected {option}." not in label:
        raise RuntimeError(
            f"{field} selection failed: aria-label is {label!r}, want "
            f"'Selected {option}.'"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8502")
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=1120)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--hide-scrollbars")
    opts.add_argument(f"--window-size={args.width},{args.height}")
    opts.add_argument("--force-device-scale-factor=2")  # crisp on retina

    driver = webdriver.Chrome(options=opts)
    try:
        driver.set_window_size(args.width, args.height)
        driver.get(args.url)
        wait = WebDriverWait(driver, 40)
        wait.until(EC.presence_of_element_located(
            (By.XPATH, "//h1[contains(., 'Telemetry anomaly detection')]")
        ))
        time.sleep(3)

        # --- switch to MSL, whose model is below chance ---
        select_option(driver, wait, "Spacecraft", "MSL")
        print("  selected MSL")

        # --- advance the replay far enough to populate the charts ---
        click_text(driver, wait, "Play", "button") or click_text(driver, wait, "Play")
        print("  playing ...")
        time.sleep(22)
        click_text(driver, wait, "Pause", "button") or click_text(driver, wait, "Pause")
        time.sleep(3)

        # --- acknowledge drift so the banner goes green while the model
        #     quality strip stays red: that contrast is the whole point ---
        if click_text(driver, wait, "Recalibrate", "button") or click_text(
            driver, wait, "Recalibrate"
        ):
            print("  recalibrated -> drift banner cleared")
        time.sleep(4)

        # The figure only earns its place if it shows the contrast. Refuse to
        # write a misleading one.
        text = driver.find_element(By.TAG_NAME, "body").text
        problems = []
        if "below chance" not in text:
            problems.append("model-quality strip does not say 'below chance'")
        if "No distribution shift detected" not in text:
            problems.append("drift banner is not green")
        if "MSL" not in text:
            problems.append("not showing MSL")
        if problems:
            raise RuntimeError(
                "refusing to write the figure -- " + "; ".join(problems)
            )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(args.out))
        size = args.out.stat().st_size / 1024
        print(f"\nwrote {args.out.relative_to(REPO)} ({size:,.0f} KB)")
        return 0
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())

