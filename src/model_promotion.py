import sys

import mlflow

import config


def set_model_alias(alias, version):
    print("\n" + "=" * 80)
    print("MODEL PROMOTION")
    print("=" * 80)

    print(f"Model: {config.MLFLOW_MODEL_NAME}")
    print(f"Alias: {alias}")
    print(f"Version: {version}")

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        name=config.MLFLOW_MODEL_NAME,
        alias=alias,
        version=version,
    )

    print(f"\nModel version {version} promoted to @{alias}")
    print("=" * 80)


def promote_to_qa(version):
    set_model_alias(config.QA_MODEL_ALIAS, version)


def promote_to_prod(version):
    set_model_alias(config.PROD_MODEL_ALIAS, version)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python model_promotion.py <alias> <version>"
        )

    alias = sys.argv[1]
    version = sys.argv[2]

    set_model_alias(alias, version)
