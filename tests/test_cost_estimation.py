"""Tests for estimate_cost() in cost_estimation.py."""
import pandas as pd
import pytest

from flashqda import PipelineConfig, estimate_cost


@pytest.fixture
def df_sentences():
    return pd.DataFrame({
        "document_id": ["doc1"] * 5,
        "sentence_id": list(range(1, 6)),
        "sentence": [
            "Larger farm size is necessary to achieve a living income.",
            "Access to irrigation leads to higher crop yields.",
            "Soil quality determines the variety of crops that can be grown.",
            "To achieve a living income, farmers must diversify crops.",
            "Poor infrastructure reduces access to markets.",
        ],
    })


@pytest.fixture
def causal_config():
    return PipelineConfig.from_type("causal", model="gpt-4o")


class TestReturnShape:
    def test_returns_dataframe(self, df_sentences, causal_config):
        result = estimate_cost(df_sentences, causal_config)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df_sentences, causal_config):
        result = estimate_cost(df_sentences, causal_config)
        expected = {"stage", "items", "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"}
        assert expected.issubset(set(result.columns))

    def test_total_row_present(self, df_sentences, causal_config):
        result = estimate_cost(df_sentences, causal_config)
        assert result.iloc[-1]["stage"] == "TOTAL"

    def test_all_stages_present(self, df_sentences, causal_config):
        result = estimate_cost(df_sentences, causal_config)
        stage_rows = result[result["stage"] != "TOTAL"]
        assert set(stage_rows["stage"]) == set(causal_config.prompt_files.keys())

    def test_total_row_sums_tokens(self, df_sentences, causal_config):
        result = estimate_cost(df_sentences, causal_config)
        stage_rows = result[result["stage"] != "TOTAL"]
        total_row = result[result["stage"] == "TOTAL"].iloc[0]
        assert total_row["total_tokens"] == stage_rows["total_tokens"].sum()


class TestCostCalculation:
    def test_zero_costs_by_default(self, df_sentences, causal_config):
        result = estimate_cost(df_sentences, causal_config)
        assert (result["estimated_cost_usd"] == 0.0).all()

    def test_tokens_positive_even_with_zero_cost(self, df_sentences, causal_config):
        result = estimate_cost(df_sentences, causal_config)
        stage_rows = result[result["stage"] != "TOTAL"]
        assert (stage_rows["input_tokens"] > 0).all()

    def test_cost_scales_with_rates(self, df_sentences, causal_config):
        result = estimate_cost(
            df_sentences, causal_config,
            input_cost_per_1m=1.0, output_cost_per_1m=2.0
        )
        total = result[result["stage"] == "TOTAL"].iloc[0]
        stage_rows = result[result["stage"] != "TOTAL"]
        expected = round(
            (stage_rows["input_tokens"].sum() * 1.0
             + stage_rows["output_tokens"].sum() * 2.0) / 1_000_000,
            4
        )
        assert total["estimated_cost_usd"] == pytest.approx(expected, abs=1e-4)

    def test_empty_dataframe_zero_tokens(self, causal_config):
        df_empty = pd.DataFrame(columns=["document_id", "sentence_id", "sentence"])
        result = estimate_cost(df_empty, causal_config)
        stage_rows = result[result["stage"] != "TOTAL"]
        assert (stage_rows["input_tokens"] == 0).all()
        assert (stage_rows["output_tokens"] == 0).all()
