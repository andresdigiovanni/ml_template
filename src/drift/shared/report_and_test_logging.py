import io

import pandas as pd
from evidently.report import Report
from evidently.test_suite import TestSuite


def generate_and_log_report(
    model_tracker,
    report_name: str,
    preset: object,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> None:
    """
    Generates and logs an Evidently report.

    Args:
        model_tracker: Tracker object with log_dict and log_artifact methods.
        report_name (str): Name of the report for logging purposes.
        preset (object): Evidently metric preset to use.
        reference_data (pd.DataFrame): Reference dataset (training data).
        current_data (pd.DataFrame): Current dataset for inference.
    """
    report = Report(metrics=[preset])
    report.run(reference_data=reference_data, current_data=current_data)

    # Save results as JSON
    report_results = report.as_dict()
    model_tracker.log_dict(report_results, f"{report_name}_report.json")

    # Save results as HTML
    html_content = _generate_html(report)
    model_tracker.log_artifact(html_content, f"{report_name}_report.html")


def generate_and_log_test(
    model_tracker,
    test_name: str,
    preset: object,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> bool:
    """
    Generates and logs an Evidently test suite.

    Args:
        model_tracker: Tracker object with log_dict and log_artifact methods.
        test_name (str): Name of the test for logging purposes.
        preset (object): Evidently test preset to use.
        reference_data (pd.DataFrame): Reference dataset (training data).
        current_data (pd.DataFrame): Current dataset for inference.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    test_suite = TestSuite(tests=[preset])
    test_suite.run(reference_data=reference_data, current_data=current_data)

    # Save results as JSON
    test_results = test_suite.as_dict()
    model_tracker.log_dict(test_results, f"{test_name}_test.json")

    # Save results as HTML
    html_content = _generate_html(test_suite)
    model_tracker.log_artifact(html_content, f"{test_name}_test.html")

    return test_results["summary"]["all_passed"]


def _generate_html(evidently_object) -> str:
    """
    Generates HTML content from an Evidently object.

    Args:
        evidently_object: An Evidently Report or TestSuite object.

    Returns:
        str: HTML content.
    """
    html_buffer = io.StringIO()
    evidently_object.save_html(html_buffer)
    html_buffer.seek(0)
    return html_buffer.getvalue()
