import pytest
from unittest import mock
from avnirpy.reporting.report import Report, StrokeReport
import tempfile
import shutil
from jinja2 import Environment, PackageLoader


@pytest.fixture
def report():
    return Report("John Doe", "12345", "2023-10-01")


@pytest.fixture
def stroke_report():
    return StrokeReport("Jane Doe", "67890", "2023-10-02")


def test_report_initialization(report):
    assert report.patient_name == "John Doe"
    assert report.patient_id == "12345"
    assert report.date == "2023-10-01"
    assert report.html_content is None
    assert isinstance(report.env, Environment)
    assert isinstance(report.temp_dir, str)


def test_report_render(report):
    assert report.render() is None


@mock.patch("weasyprint.HTML.write_pdf")
@mock.patch("shutil.rmtree")
def test_report_to_pdf(mock_rmtree, mock_write_pdf, report):
    report.html_content = "<html><body>Test</body></html>"
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        report.to_pdf(temp_pdf.name)
        mock_write_pdf.assert_called_once_with(temp_pdf.name)
        mock_rmtree.assert_called_once_with(report.temp_dir)


def test_stroke_report_initialization(stroke_report):
    assert stroke_report.patient_name == "Jane Doe"
    assert stroke_report.patient_id == "67890"
    assert stroke_report.date == "2023-10-02"
    assert stroke_report.html_content is None
    assert isinstance(stroke_report.env, Environment)
    assert isinstance(stroke_report.temp_dir, str)


@mock.patch("avnirpy.reporting.screenshot.colors", ["#FFFFFF", "#000000"])
@mock.patch("jinja2.Environment.get_template")
def test_stroke_report_render(mock_get_template, stroke_report):
    mock_template = mock.Mock()
    mock_get_template.return_value = mock_template
    volumetry_data = {"volume": 100}
    screenshot_path = "/path/to/screenshot.png"
    stroke_report.render(volumetry_data, screenshot_path)
    mock_get_template.assert_called_once_with("stroke_report.html")
    mock_template.render.assert_called_once_with(
        {
            "patient_name": "Jane Doe",
            "patient_id": "67890",
            "date": "2023-10-02",
            "volumetry": volumetry_data,
            "screenshot": screenshot_path,
            "timepoints": None,
            "COLOR": ["#FFFFFF", "#000000"],
        }
    )
    assert stroke_report.html_content == mock_template.render.return_value
