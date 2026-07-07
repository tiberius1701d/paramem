from strictdoc.core.project_config import ProjectConfig


def create_config() -> ProjectConfig:
    return ProjectConfig(
        project_title="ParaMem System Requirements Baseline",
        project_features=[
            "TABLE_SCREEN",
            "TRACEABILITY_SCREEN",
            "SEARCH",
            "REQIF",
        ],
        grammars={
            "@shared_grammar": "grammars/srb.sgra",
        },
        exclude_doc_paths=[
            ".venv/",
            "_build/",
            "README.md",
            "TOOLING.md",
            "CONVENTIONS.md",
        ],
    )
