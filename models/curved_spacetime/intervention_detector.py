"""
Market Intervention Detection System

Detects when markets move against natural energy gradients due to:
- Large institutional interventions
- Central bank actions
- Market manipulation
- Sudden news events
- Program trading cascades

Uses energy dynamics to identify unnatural market movements that
deviate from geodesic trajectories and natural energy evolution.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from market_hamiltonian import MarketHamiltonian, MarketState, analyze_market_energy
from curved_candle_geometry import CandleMetric, create_candle_metrics_from_ohlc


class InterventionType(Enum):
    """Types of market interventions."""
    BULLISH_PUMP = "bullish_pump"
    BEARISH_DUMP = "bearish_dump"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_SURGE = "volume_surge"
    PATTERN_ANOMALY = "pattern_anomaly"
    ENERGY_INJECTION = "energy_injection"
    REVERSAL_FORCE = "reversal_force"


@dataclass
class InterventionSignal:
    """A detected intervention event."""
    timestamp: int  # Index in the time series
    intervention_type: InterventionType
    magnitude: float  # Strength of the intervention (0-1)
    confidence: float  # Detection confidence (0-1)
    energy_delta: float  # Change in market energy
    gradient_anomaly: float  # Deviation from natural gradient
    duration: int  # How many periods the intervention lasts
    description: str  # Human-readable description
    evidence: Dict  # Supporting evidence


class MarketInterventionDetector:
    """
    Detects when markets move against natural energy gradients.
    
    The detector uses multiple signals:
    1. Energy gradient anomalies
    2. Statistical outliers in energy flow
    3. Geodesic path deviations
    4. Volume-energy mismatches
    5. Pattern space jumps
    """
    
    def __init__(self,
                 energy_threshold: float = 2.0,
                 gradient_threshold: float = 1.5,
                 volume_threshold: float = 2.5,
                 pattern_threshold: float = 0.3,
                 lookback_window: int = 10,
                 confidence_min: float = 0.6):
        """
        Initialize the intervention detector.
        
        Args:
            energy_threshold: Std deviations for energy anomaly detection
            gradient_threshold: Std deviations for gradient anomaly detection
            volume_threshold: Std deviations for volume anomaly detection
            pattern_threshold: Threshold for pattern space jumps
            lookback_window: Window size for statistical analysis
            confidence_min: Minimum confidence for intervention signal
        """
        self.energy_threshold = energy_threshold
        self.gradient_threshold = gradient_threshold
        self.volume_threshold = volume_threshold
        self.pattern_threshold = pattern_threshold
        self.lookback_window = lookback_window
        self.confidence_min = confidence_min
        
        # Initialize Hamiltonian for energy calculations
        self.hamiltonian = MarketHamiltonian()
    
    def detect_energy_injection(self, 
                              energy_series: np.ndarray,
                              gradient_series: np.ndarray) -> List[InterventionSignal]:
        """
        Detect sudden energy injections that deviate from natural evolution.
        
        Energy injections manifest as:
        - Sudden spikes in total energy
        - Misalignment between energy and natural gradient flow
        - Non-smooth energy evolution
        """
        interventions = []
        
        if len(energy_series) < self.lookback_window * 2:
            return interventions
        
        # Compute energy statistics
        energy_diff = np.diff(energy_series)
        energy_mean = np.mean(energy_diff)
        energy_std = np.std(energy_diff)
        
        # Compute gradient magnitudes
        gradient_mags = np.array([np.linalg.norm(g) if g is not None else 0 
                                 for g in gradient_series])
        
        for i in range(self.lookback_window, len(energy_series) - 1):
            # Energy change at this point
            energy_change = energy_diff[i-1]
            
            # Statistical significance of energy change
            if energy_std > 0:
                energy_z_score = abs(energy_change - energy_mean) / energy_std
            else:
                energy_z_score = 0
            
            # Check for energy injection
            if energy_z_score > self.energy_threshold:
                # Analyze the nature of the injection
                gradient_before = gradient_mags[i-1] if i > 0 else 0
                gradient_after = gradient_mags[i] if i < len(gradient_mags) else 0
                
                # Energy injection should create gradient anomalies
                gradient_anomaly = abs(gradient_after - gradient_before)
                
                # Calculate intervention magnitude
                magnitude = min(1.0, energy_z_score / (self.energy_threshold * 2))
                
                # Calculate confidence based on multiple factors
                confidence_factors = []
                
                # Factor 1: Energy change significance
                confidence_factors.append(min(1.0, energy_z_score / self.energy_threshold))
                
                # Factor 2: Gradient consistency
                if gradient_before > 0:
                    gradient_consistency = gradient_anomaly / gradient_before
                    confidence_factors.append(min(1.0, gradient_consistency))
                
                # Factor 3: Surrounding context
                window_start = max(0, i - self.lookback_window//2)
                window_end = min(len(energy_series), i + self.lookback_window//2)
                local_energies = energy_series[window_start:window_end]
                
                if len(local_energies) > 1:
                    local_mean = np.mean(local_energies)
                    local_std = np.std(local_energies)
                    if local_std > 0:
                        context_score = abs(energy_series[i] - local_mean) / local_std
                        confidence_factors.append(min(1.0, context_score / 2.0))
                
                confidence = np.mean(confidence_factors) if confidence_factors else 0.5
                
                if confidence >= self.confidence_min:
                    # Determine intervention type
                    if energy_change > 0:
                        intervention_type = InterventionType.ENERGY_INJECTION
                        description = f"Positive energy injection (+{energy_change:.2f})"
                    else:
                        intervention_type = InterventionType.REVERSAL_FORCE
                        description = f"Energy extraction/reversal ({energy_change:.2f})"
                    
                    # Estimate duration
                    duration = self._estimate_intervention_duration(
                        energy_series, i, energy_change
                    )
                    
                    intervention = InterventionSignal(
                        timestamp=i,
                        intervention_type=intervention_type,
                        magnitude=magnitude,
                        confidence=confidence,
                        energy_delta=energy_change,
                        gradient_anomaly=gradient_anomaly,
                        duration=duration,
                        description=description,
                        evidence={
                            'energy_z_score': energy_z_score,
                            'gradient_before': gradient_before,
                            'gradient_after': gradient_after,
                            'local_context': local_energies.tolist() if len(local_energies) > 0 else []
                        }
                    )
                    
                    interventions.append(intervention)
        
        return interventions
    
    def detect_volume_anomalies(self, 
                               states: List[MarketState]) -> List[InterventionSignal]:
        """
        Detect volume anomalies that don't match energy patterns.
        
        Natural markets have volume that correlates with energy changes.
        Interventions often show volume spikes without corresponding energy evolution.
        """
        interventions = []
        
        if len(states) < self.lookback_window:
            return interventions
        
        # Extract volume and energy series
        volumes = np.array([s.candle.volume for s in states])
        energies = np.array([self.hamiltonian.total_energy(s) for s in states])
        
        # Log transform volumes for better statistics
        log_volumes = np.log1p(volumes)
        volume_mean = np.mean(log_volumes)
        volume_std = np.std(log_volumes)
        
        for i in range(self.lookback_window, len(states)):
            volume_z_score = abs(log_volumes[i] - volume_mean) / volume_std if volume_std > 0 else 0
            
            if volume_z_score > self.volume_threshold:
                # Check if volume spike matches energy change
                energy_change = energies[i] - energies[i-1] if i > 0 else 0
                expected_volume_increase = abs(energy_change) * 0.5  # Heuristic correlation
                
                actual_volume_increase = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1
                
                # Volume-energy mismatch suggests intervention
                mismatch = abs(np.log(actual_volume_increase) - expected_volume_increase)
                
                if mismatch > 0.5:  # Threshold for significant mismatch
                    magnitude = min(1.0, volume_z_score / self.volume_threshold)
                    confidence = min(1.0, mismatch / 2.0)
                    
                    if confidence >= self.confidence_min:
                        intervention = InterventionSignal(
                            timestamp=i,
                            intervention_type=InterventionType.VOLUME_SURGE,
                            magnitude=magnitude,
                            confidence=confidence,
                            energy_delta=energy_change,
                            gradient_anomaly=0.0,  # Not applicable here
                            duration=1,  # Volume spikes are usually single-period
                            description=f"Volume surge: {volumes[i]:,.0f} (z={volume_z_score:.1f})",
                            evidence={
                                'volume_z_score': volume_z_score,
                                'volume_ratio': actual_volume_increase,
                                'energy_change': energy_change,
                                'mismatch': mismatch
                            }
                        )
                        
                        interventions.append(intervention)
        
        return interventions
    
    def detect_pattern_jumps(self, 
                           states: List[MarketState]) -> List[InterventionSignal]:
        """
        Detect sudden jumps in pattern space that violate smooth geodesic evolution.
        
        Natural market evolution follows smooth geodesics in pattern space.
        Interventions cause abrupt jumps that violate this smoothness.
        """
        interventions = []
        
        if len(states) < 3:
            return interventions
        
        # Extract pattern coordinates
        patterns = np.array([s.candle.pattern_coordinates for s in states])
        
        # Compute distances between consecutive patterns
        distances = []
        for i in range(1, len(patterns)):
            dist = np.linalg.norm(patterns[i] - patterns[i-1])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Statistical analysis of pattern distances
        if len(distances) < self.lookback_window:
            return interventions
        
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        
        for i in range(1, len(distances)):
            if dist_std > 0:
                dist_z_score = (distances[i] - dist_mean) / dist_std
            else:
                dist_z_score = 0
            
            # Check for pattern jump
            if distances[i] > self.pattern_threshold and dist_z_score > 2.0:
                # Verify this is not just natural volatility
                # Look at the geodesic prediction vs actual
                if i > 0 and i < len(states) - 1:
                    # Simple geodesic prediction: linear extrapolation
                    if i > 1:
                        predicted_pattern = patterns[i] + (patterns[i] - patterns[i-1])
                        actual_pattern = patterns[i+1]
                        prediction_error = np.linalg.norm(actual_pattern - predicted_pattern)
                        
                        if prediction_error > self.pattern_threshold:
                            magnitude = min(1.0, distances[i] / self.pattern_threshold)
                            confidence = min(1.0, prediction_error / self.pattern_threshold)
                            
                            if confidence >= self.confidence_min:
                                # Determine type based on direction
                                sentiment_change = patterns[i+1][0] - patterns[i][0]
                                
                                if sentiment_change > 0.1:
                                    intervention_type = InterventionType.BULLISH_PUMP
                                    description = "Bullish pattern jump"
                                elif sentiment_change < -0.1:
                                    intervention_type = InterventionType.BEARISH_DUMP
                                    description = "Bearish pattern jump"
                                else:
                                    intervention_type = InterventionType.PATTERN_ANOMALY
                                    description = "Pattern space anomaly"
                                
                                intervention = InterventionSignal(
                                    timestamp=i+1,
                                    intervention_type=intervention_type,
                                    magnitude=magnitude,
                                    confidence=confidence,
                                    energy_delta=0.0,  # Will be computed separately
                                    gradient_anomaly=prediction_error,
                                    duration=1,
                                    description=description,
                                    evidence={
                                        'pattern_distance': distances[i],
                                        'prediction_error': prediction_error,
                                        'sentiment_change': sentiment_change,
                                        'dist_z_score': dist_z_score
                                    }
                                )
                                
                                interventions.append(intervention)
        
        return interventions
    
    def detect_all_interventions(self, 
                               ohlc_data: pd.DataFrame) -> List[InterventionSignal]:
        """
        Run complete intervention detection on market data.
        
        Returns all detected interventions with timestamps.
        """
        # Perform energy analysis
        analysis = analyze_market_energy(ohlc_data, self.hamiltonian)
        
        energies = np.array(analysis['energies'])
        gradients = analysis['gradients']
        states = analysis['states']
        
        # Run all detection methods
        energy_interventions = self.detect_energy_injection(energies, gradients)
        volume_interventions = self.detect_volume_anomalies(states)
        pattern_interventions = self.detect_pattern_jumps(states)
        
        # Combine and sort by timestamp
        all_interventions = energy_interventions + volume_interventions + pattern_interventions
        all_interventions.sort(key=lambda x: x.timestamp)
        
        # Merge overlapping interventions
        merged_interventions = self._merge_overlapping_interventions(all_interventions)
        
        return merged_interventions
    
    def _estimate_intervention_duration(self, 
                                      energy_series: np.ndarray,
                                      intervention_index: int,
                                      energy_change: float) -> int:
        """
        Estimate how long an intervention effect lasts.
        """
        # Look for energy to return toward baseline
        baseline_energy = np.mean(energy_series[max(0, intervention_index-5):intervention_index])
        
        duration = 1
        for i in range(intervention_index + 1, min(len(energy_series), intervention_index + 10)):
            current_energy = energy_series[i]
            
            # If energy starts returning to baseline, intervention is fading
            if abs(current_energy - baseline_energy) < abs(energy_change) * 0.5:
                break
            
            duration += 1
        
        return duration
    
    def _merge_overlapping_interventions(self, 
                                       interventions: List[InterventionSignal]) -> List[InterventionSignal]:
        """
        Merge interventions that occur close together in time.
        """
        if len(interventions) < 2:
            return interventions
        
        merged = []
        current = interventions[0]
        
        for next_intervention in interventions[1:]:
            # If interventions are close in time, merge them
            if (next_intervention.timestamp - current.timestamp) <= max(current.duration, 2):
                # Create merged intervention
                merged_confidence = max(current.confidence, next_intervention.confidence)
                merged_magnitude = max(current.magnitude, next_intervention.magnitude)
                
                # Combine evidence
                combined_evidence = {**current.evidence, **next_intervention.evidence}
                
                current = InterventionSignal(
                    timestamp=current.timestamp,
                    intervention_type=current.intervention_type,  # Keep first type
                    magnitude=merged_magnitude,
                    confidence=merged_confidence,
                    energy_delta=current.energy_delta + next_intervention.energy_delta,
                    gradient_anomaly=max(current.gradient_anomaly, next_intervention.gradient_anomaly),
                    duration=next_intervention.timestamp - current.timestamp + next_intervention.duration,
                    description=f"Merged: {current.description} + {next_intervention.description}",
                    evidence=combined_evidence
                )
            else:
                # No overlap, add current to merged list
                merged.append(current)
                current = next_intervention
        
        # Add the last intervention
        merged.append(current)
        
        return merged


def analyze_intervention_patterns(interventions: List[InterventionSignal]) -> Dict:
    """
    Analyze patterns in detected interventions.
    
    Returns statistics about:
    - Intervention frequency
    - Type distribution
    - Magnitude patterns
    - Timing patterns
    """
    if not interventions:
        return {'total_interventions': 0}
    
    # Basic statistics
    total_interventions = len(interventions)
    
    # Type distribution
    type_counts = {}
    for intervention in interventions:
        intervention_type = intervention.intervention_type.value
        type_counts[intervention_type] = type_counts.get(intervention_type, 0) + 1
    
    # Magnitude statistics
    magnitudes = [i.magnitude for i in interventions]
    confidences = [i.confidence for i in interventions]
    durations = [i.duration for i in interventions]
    
    # Timing analysis
    timestamps = [i.timestamp for i in interventions]
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)
    else:
        avg_interval = 0
        interval_std = 0
    
    return {
        'total_interventions': total_interventions,
        'type_distribution': type_counts,
        'magnitude_stats': {
            'mean': np.mean(magnitudes),
            'std': np.std(magnitudes),
            'max': np.max(magnitudes),
            'min': np.min(magnitudes)
        },
        'confidence_stats': {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'max': np.max(confidences),
            'min': np.min(confidences)
        },
        'duration_stats': {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'max': np.max(durations),
            'min': np.min(durations)
        },
        'timing_stats': {
            'avg_interval': avg_interval,
            'interval_std': interval_std
        }
    }