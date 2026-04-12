"""Shared pytest configuration and fixtures."""


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="Run GPU integration tests")
    parser.addoption(
        "--recall",
        action="store_true",
        default=False,
        help="Run memory recall tests (train ~30 epochs + probe; requires --gpu)",
    )
