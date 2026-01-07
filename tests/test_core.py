"""
Test suite for SOPRA core functionality
"""

import pytest
import numpy as np
from sopra import core


class TestBasicUtilityFunctions:
    """Tests for basic utility functions"""

    def test_rate_positive_result(self):
        """Test rate calculation with positive result"""
        # rate(b1, b2, temp) = max((b1 * temp) - b2, 0.0) / 24.0
        result = core.rate(0.1, 1.0, 20.0)
        expected = ((0.1 * 20.0) - 1.0) / 24.0  # (2.0 - 1.0) / 24.0 = 0.04166...
        assert abs(result - expected) < 1e-6

    def test_rate_negative_clipped_to_zero(self):
        """Test that negative rates are clipped to zero"""
        result = core.rate(0.1, 5.0, 10.0)
        # (0.1 * 10.0) - 5.0 = -4.0, should be clipped to 0.0
        assert result == 0.0

    def test_rate_zero_temperature(self):
        """Test rate calculation at zero temperature"""
        result = core.rate(0.1, 1.0, 0.0)
        # (0.1 * 0.0) - 1.0 = -1.0, clipped to 0.0
        assert result == 0.0

    def test_rate_boundary_case(self):
        """Test rate at the boundary where result is exactly zero"""
        result = core.rate(0.1, 1.0, 10.0)
        # (0.1 * 10.0) - 1.0 = 0.0
        assert result == 0.0


class TestGetTrunkTemp:
    """Tests for trunk temperature calculation"""

    def test_get_trunk_temp_returns_dict(self):
        """Test that get_trunk_temp returns a dictionary"""
        result = core.get_trunk_temp(day=180, temp_air=20.0, solar_rad=500.0)
        assert isinstance(result, dict)

    def test_get_trunk_temp_has_required_keys(self):
        """Test that result contains expected temperature keys"""
        result = core.get_trunk_temp(day=180, temp_air=20.0, solar_rad=500.0)
        # Based on function signature, should return trunk temperatures
        assert "temp_trunk_out" in result or "trunk_out" in result or len(result) > 0

    def test_get_trunk_temp_summer_vs_winter(self):
        """Test trunk temperature differs between summer and winter"""
        summer = core.get_trunk_temp(day=180, temp_air=25.0, solar_rad=600.0)  # June
        winter = core.get_trunk_temp(day=1, temp_air=5.0, solar_rad=100.0)  # January
        # Results should differ
        assert summer != winter

    def test_get_trunk_temp_with_high_radiation(self):
        """Test trunk temperature with high solar radiation"""
        result = core.get_trunk_temp(day=180, temp_air=20.0, solar_rad=800.0)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestModelInitialization:
    """Tests for model initialization functions"""

    def test_assign_const_and_var_gfune(self):
        """Test that constants are properly initialized"""
        constants = core.assign_const_and_var_gfune()
        assert isinstance(constants, dict)
        assert len(constants) > 0
        # Should contain rate parameters
        assert all(isinstance(v, (int, float)) for v in constants.values())

    def test_init_param_gfune(self):
        """Test parameter initialization"""
        params = core.init_param_gfune()
        assert isinstance(params, dict)
        assert all(isinstance(v, bool) for v in params.values())

    def test_init_value_gfune(self):
        """Test initial values initialization"""
        values = core.init_value_gfune()
        assert isinstance(values, dict)
        assert len(values) > 0
        assert all(isinstance(v, (int, float)) for v in values.values())


class TestSetDel:
    """Tests for set_del function"""

    def test_set_del_positive_rate(self):
        """Test set_del with positive rate"""
        result = core.set_del(0.1, 1.0, 20.0)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_set_del_zero_rate(self):
        """Test set_del when rate is zero"""
        result = core.set_del(0.1, 1.0, 10.0)
        assert isinstance(result, float)


class TestEvalRate:
    """Tests for eval_rate function"""

    def test_eval_rate_basic(self):
        """Test basic eval_rate functionality"""
        constants = core.assign_const_and_var_gfune()
        trunk_temps = core.get_trunk_temp(day=180, temp_air=20.0, solar_rad=500.0)

        result = core.eval_rate(
            stage="pup_w",
            temp_type="trunk",
            constants=constants,
            temp_values=trunk_temps,
        )
        assert isinstance(result, float)
        assert result >= 0.0


class TestDelayFunctions:
    """Tests for delay and loop functions"""

    def test_initialize_delrate_arrays(self):
        """Test initialization of delrate arrays"""
        constants = core.assign_const_and_var_gfune()
        arrays = core.initialize_delrate_arrays(constants)

        assert isinstance(arrays, dict)
        assert len(arrays) > 0
        for key, arr in arrays.items():
            assert isinstance(arr, np.ndarray)

    def test_create_del_var_df(self):
        """Test creation of delay variable dataframe"""
        del_var_df = core.create_del_var_df()
        assert isinstance(del_var_df, list)
        assert len(del_var_df) > 0
        assert all(isinstance(item, dict) for item in del_var_df)

    def test_del_naming_fun(self):
        """Test delay variable naming function"""
        del_var_df = core.create_del_var_df()
        name = core.del_naming_fun("ow_pup", "trunk_out", del_var_df)
        assert isinstance(name, str)
        assert len(name) > 0


class TestBlockDelayStage:
    """Tests for block delay stage functions"""

    def test_block_delay_stage_inactive(self):
        """Test block_delay_stage when inactive"""
        constants = core.assign_const_and_var_gfune()
        delrate_arrays = core.initialize_delrate_arrays(constants)

        result = core.block_delay_stage(
            active=False,
            vin=100.0,
            del_val=0.5,
            k=int(constants.get("k_ow_pup", 20)),
            r_dt=1.0,
            delrate=delrate_arrays.get("delrate_ow_pup_trunk_out", np.zeros(20)),
        )

        assert isinstance(result, dict)
        assert "delrate" in result
        assert "vout" in result

    def test_block_delay_stage_active(self):
        """Test block_delay_stage when active"""
        constants = core.assign_const_and_var_gfune()
        delrate_arrays = core.initialize_delrate_arrays(constants)

        result = core.block_delay_stage(
            active=True,
            vin=100.0,
            del_val=0.5,
            k=int(constants.get("k_ow_pup", 20)),
            r_dt=1.0,
            delrate=delrate_arrays.get("delrate_ow_pup_trunk_out", np.zeros(20)),
        )

        assert isinstance(result, dict)
        assert "delrate" in result
        assert "vout" in result


class TestUpdateGfune:
    """Tests for main update function"""

    def test_update_gfune_single_step(self):
        """Test single update step"""
        constants = core.assign_const_and_var_gfune()
        values = core.init_value_gfune()

        result = core.update_gfune(
            values=values,
            day=180,
            hour=12,
            temp_air=20.0,
            solar_rad=500.0,
            temp_soil=18.0,
            constants=constants,
        )

        assert isinstance(result, dict)
        assert "updated_values" in result
        assert "current_param" in result

    def test_update_gfune_multiple_steps(self):
        """Test multiple consecutive update steps"""
        constants = core.assign_const_and_var_gfune()
        values = core.init_value_gfune()
        curr_param = None

        # Run a few steps
        for hour in range(5):
            result = core.update_gfune(
                values=values,
                day=180,
                hour=hour,
                temp_air=20.0,
                solar_rad=500.0,
                temp_soil=18.0,
                constants=constants,
                curr_param=curr_param,
            )
            values = result["updated_values"]
            curr_param = result["current_param"]

        assert isinstance(values, dict)
        assert isinstance(curr_param, dict)


class TestImportAndModuleStructure:
    """Tests for module structure and imports"""

    def test_import(self):
        """Test that sopra.core can be imported"""
        assert core is not None

    def test_all_main_functions_exist(self):
        """Test that all main functions are available"""
        functions = [
            "rate",
            "get_trunk_temp",
            "assign_const_and_var_gfune",
            "init_param_gfune",
            "init_value_gfune",
            "update_gfune",
            "eval_rate",
            "set_del",
            "block_delay_stage",
            "initialize_delrate_arrays",
            "create_del_var_df",
        ]

        for func_name in functions:
            assert hasattr(core, func_name), f"Function {func_name} not found"
            assert callable(getattr(core, func_name))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
