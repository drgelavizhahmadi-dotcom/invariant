"""
Unit tests for IEEE 738 Physics Module
"""

import pytest
import torch
import numpy as np

from core.physics import IEEE738HeatBalance, physics_loss_fn


class TestIEEE738HeatBalance:
    """Tests for IEEE 738 heat balance calculations"""
    
    @pytest.fixture
    def physics(self):
        """Create physics instance"""
        return IEEE738HeatBalance()
    
    def test_resistance_increases_with_temperature(self, physics):
        """Resistance should increase with temperature"""
        R_25 = physics.resistance(torch.tensor([25.0]))
        R_75 = physics.resistance(torch.tensor([75.0]))
        
        assert R_75 > R_25, "Resistance should increase with temperature"
    
    def test_joule_heating_increases_with_current(self, physics):
        """Joule heating should scale with I²"""
        I1 = torch.tensor([500.0])
        I2 = torch.tensor([1000.0])
        T = torch.tensor([50.0])
        
        Q1 = physics.joule_heating(I1, T)
        Q2 = physics.joule_heating(I2, T)
        
        # Q should scale with I², so Q2 ≈ 4 * Q1
        ratio = Q2 / Q1
        assert 3.5 < ratio < 4.5, f"Expected ~4x increase, got {ratio.item():.2f}x"
    
    def test_convective_cooling_increases_with_wind(self, physics):
        """More wind = more cooling"""
        T_c = torch.tensor([75.0])
        T_a = torch.tensor([25.0])
        
        Q_low_wind = physics.convective_heat_loss(T_c, T_a, torch.tensor([1.0]))
        Q_high_wind = physics.convective_heat_loss(T_c, T_a, torch.tensor([10.0]))
        
        assert Q_high_wind > Q_low_wind, "Higher wind should increase cooling"

    def test_low_wind_prefers_natural_convection(self, physics):
        """When wind < 0.6 m/s, convective_heat_loss should use natural convection"""
        T_c = torch.tensor([80.0])
        T_a = torch.tensor([20.0])

        wind_very_low = torch.tensor([0.1])
        q_total = physics.convective_heat_loss(T_c, T_a, wind_very_low)

        q_natural = physics.convective_heat_loss_natural(T_c, T_a)
        q_forced = physics.convective_heat_loss_forced(T_c, T_a, wind_very_low)

        # At very low wind, total cooling should equal natural convection
        assert torch.allclose(q_total, q_natural, rtol=1e-4, atol=1e-6), (
            "Low-wind convective loss should equal natural convection"
        )

        # Forced convection at very low wind should not exceed natural in final selection
        assert q_forced <= q_natural + 1e-6, "Forced convective term should not dominate at V<0.6 m/s"    
    def test_radiative_cooling_requires_temperature_difference(self, physics):
        """No radiation if conductor = ambient"""
        T_same = torch.tensor([25.0])
        Q_rad = physics.radiative_heat_loss(T_same, T_same)
        
        assert abs(Q_rad.item()) < 1e-6, "No radiation when T_conductor = T_ambient"
    
    def test_heat_balance_near_zero_at_steady_state(self, physics):
        """Steady-state temperature should give ~0 residual"""
        T_amb = torch.tensor([25.0])
        wind = torch.tensor([5.0])
        solar = torch.tensor([800.0])
        current = torch.tensor([800.0])
        
        # Get steady-state temperature
        T_ss = physics.steady_state_temperature(current, T_amb, wind, solar)
        
        # Calculate residual
        residual = physics.heat_balance_residual(
            current, T_ss, T_amb, wind, solar
        )
        
        assert abs(residual.item()) < 1.0, f"Residual {residual.item():.2f} too high"
    
    def test_ampacity_increases_with_cooling(self, physics):
        """Better cooling = higher ampacity"""
        T_amb = torch.tensor([25.0])
        solar = torch.tensor([500.0])
        
        # Low wind
        I_low = physics.ampacity(75.0, T_amb, torch.tensor([1.0]), solar)
        
        # High wind
        I_high = physics.ampacity(75.0, T_amb, torch.tensor([10.0]), solar)
        
        assert I_high > I_low, "Higher wind should allow higher ampacity"
    
    def test_static_rating_reasonable_range(self, physics):
        """Static rating should be in reasonable range"""
        static = physics.static_rating()
        
        # For Drake ACSR, static rating typically 700-1000 A
        assert 500 < static < 1500, f"Static rating {static:.0f} A out of range"
    
    def test_dynamic_rating_exceeds_static_in_good_conditions(self, physics):
        """Good conditions should beat static rating"""
        static = physics.static_rating()
        
        # Favorable conditions: cool, windy, low sun
        T_amb = torch.tensor([10.0])
        wind = torch.tensor([10.0])
        solar = torch.tensor([200.0])
        
        dynamic = physics.ampacity(75.0, T_amb, wind, solar).item()
        
        assert dynamic > static, "Dynamic should exceed static in good conditions"
        
        # Expect significant gain
        gain = (dynamic - static) / static * 100
        assert gain > 20, f"Expected >20% gain, got {gain:.1f}%"


class TestPhysicsLoss:
    """Tests for physics-informed loss function"""
    
    @pytest.fixture
    def physics(self):
        return IEEE738HeatBalance()
    
    def test_physics_loss_penalizes_wrong_temperature(self, physics):
        """Wrong temperature should have high physics loss"""
        current = torch.tensor([800.0])
        T_amb = torch.tensor([25.0])
        wind = torch.tensor([5.0])
        solar = torch.tensor([800.0])
        
        # Correct temperature
        T_correct = physics.steady_state_temperature(current, T_amb, wind, solar)
        loss_correct = physics_loss_fn(physics, T_correct, current, T_amb, wind, solar)
        
        # Wrong temperature (way off)
        T_wrong = torch.tensor([20.0])  # Too low
        loss_wrong = physics_loss_fn(physics, T_wrong, current, T_amb, wind, solar)
        
        assert loss_wrong > loss_correct, "Wrong temp should have higher loss"
    
    def test_physics_loss_differentiable(self, physics):
        """Loss should be differentiable"""
        T_pred = torch.tensor([50.0], requires_grad=True)
        current = torch.tensor([800.0])
        T_amb = torch.tensor([25.0])
        wind = torch.tensor([5.0])
        solar = torch.tensor([800.0])
        
        loss = physics_loss_fn(physics, T_pred, current, T_amb, wind, solar)
        loss.backward()
        
        assert T_pred.grad is not None, "Gradient should exist"
        assert not torch.isnan(T_pred.grad), "Gradient should not be NaN"


class TestBatchProcessing:
    """Test batch processing capabilities"""
    
    @pytest.fixture
    def physics(self):
        return IEEE738HeatBalance()
    
    def test_batch_ampacity(self, physics):
        """Ampacity should work with batches"""
        T_amb = torch.tensor([10.0, 25.0, 40.0])
        wind = torch.tensor([5.0, 5.0, 5.0])
        solar = torch.tensor([500.0, 500.0, 500.0])
        
        ratings = physics.ampacity(75.0, T_amb, wind, solar)
        
        assert ratings.shape == (3,), f"Expected shape (3,), got {ratings.shape}"
        
        # Higher ambient = lower rating
        assert ratings[0] > ratings[1] > ratings[2], "Rating should decrease with temp"
    
    def test_batch_residual(self, physics):
        """Residual should work with batches"""
        batch_size = 10
        current = torch.rand(batch_size) * 800 + 400
        T_c = torch.rand(batch_size) * 50 + 30
        T_a = torch.rand(batch_size) * 30 + 10
        wind = torch.rand(batch_size) * 10 + 1
        solar = torch.rand(batch_size) * 1000
        
        residual = physics.heat_balance_residual(current, T_c, T_a, wind, solar)
        
        assert residual.shape == (batch_size,), f"Wrong shape: {residual.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
