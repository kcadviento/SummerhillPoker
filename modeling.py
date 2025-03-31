"""Module for opponent behavior modeling and profiling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
from collections import defaultdict

from summerhillpoker.state import Action, ActionType, BettingRound, PlayerState


class ProfileType(Enum):
    """Types of player profiles."""
    TIGHT_PASSIVE = auto()  # Plays few hands, rarely aggressive
    TIGHT_AGGRESSIVE = auto()  # Plays few hands, but aggressive when playing
    LOOSE_PASSIVE = auto()  # Plays many hands, but passive
    LOOSE_AGGRESSIVE = auto()  # Plays many hands, very aggressive
    MANIAC = auto()  # Extremely aggressive, plays almost any hand
    ROCK = auto()  # Extremely tight, only plays premium hands
    CALLING_STATION = auto()  # Calls a lot, rarely folds or raises
    AGGRESSIVE_FISH = auto()  # Bad player who plays aggressively
    PASSIVE_FISH = auto()  # Bad player who plays passively
    BALANCED = auto()  # GTO-style play, balanced strategy
    ADAPTIVE = auto()  # Adapts strategy based on opponents


@dataclass
class PlayerProfile:
    """Represents a player's behavioral profile."""
    profile_type: Union[str, ProfileType]  # Either a string label or a ProfileType enum
    stack: int = 1000
    
    # Base behavioral parameters
    # These define the core tendencies that drive decision-making
    vpip_range: Tuple[float, float] = (0.0, 1.0)  # Voluntarily Put $ In Pot range
    pfr_range: Tuple[float, float] = (0.0, 1.0)   # Pre-Flop Raise range
    aggression_factor_range: Tuple[float, float] = (0.0, 10.0)  # Bet/raise vs check/call ratio
    continuation_bet_freq: float = 0.5  # How often player continuation bets
    bluff_frequency: float = 0.2  # How often player bluffs in opportune situations
    slowplay_frequency: float = 0.3  # How often player slowplays strong hands
    fold_to_3bet: float = 0.6  # How often player folds to a 3-bet
    fold_to_cbet: float = 0.5  # How often player folds to continuation bet
    
    # Advanced metrics
    positional_awareness: float = 0.5  # How much position affects decisions (0-1)
    adaptability: float = 0.5  # How quickly player adapts to opponents (0-1)
    tilt_factor: float = 0.2  # Susceptibility to tilt (0-1)
    hand_reading_skill: float = 0.5  # Ability to read opponent ranges (0-1)
    risk_tolerance: float = 0.5  # Tolerance for variance (0-1)
    
    def __post_init__(self):
        """Initialize profile parameters based on profile type if it's a string."""
        if isinstance(self.profile_type, str):
            self.profile_type = self.profile_type.lower()
            
            # Set parameters based on profile string
            if self.profile_type == "aggressive":
                self._set_aggressive_profile()
            elif self.profile_type == "passive":
                self._set_passive_profile()
            elif self.profile_type == "tight":
                self._set_tight_profile()
            elif self.profile_type == "loose":
                self._set_loose_profile()
            elif self.profile_type == "balanced":
                self._set_balanced_profile()
            elif self.profile_type == "calling_station":
                self._set_calling_station_profile()
            elif self.profile_type == "maniac":
                self._set_maniac_profile()
            elif self.profile_type == "rock":
                self._set_rock_profile()
            elif self.profile_type == "fish":
                self._set_fish_profile()
            else:
                # Default to balanced
                self._set_balanced_profile()
        elif isinstance(self.profile_type, ProfileType):
            self._set_profile_from_enum(self.profile_type)
    
    def _set_aggressive_profile(self):
        """Set parameters for an aggressive player."""
        self.vpip_range = (0.25, 0.35)  # Plays 25-35% of hands
        self.pfr_range = (0.20, 0.30)   # Raises with 20-30% of hands
        self.aggression_factor_range = (3.0, 5.0)  # Significantly more bets/raises than calls
        self.continuation_bet_freq = 0.8  # Continuation bets frequently
        self.bluff_frequency = 0.3  # Bluffs more than average
        self.slowplay_frequency = 0.1  # Rarely slowplays (prefers aggression)
        self.fold_to_3bet = 0.4  # More likely to 4-bet or call a 3-bet
        self.fold_to_cbet = 0.3  # Less likely to fold to continuation bets
        self.risk_tolerance = 0.7  # Higher than average risk tolerance
    
    def _set_passive_profile(self):
        """Set parameters for a passive player."""
        self.vpip_range = (0.20, 0.30)  # Plays 20-30% of hands
        self.pfr_range = (0.05, 0.10)   # Rarely raises pre-flop
        self.aggression_factor_range = (0.5, 1.0)  # More calls than bets/raises
        self.continuation_bet_freq = 0.3  # Doesn't continuation bet often
        self.bluff_frequency = 0.1  # Rarely bluffs
        self.slowplay_frequency = 0.6  # Often slowplays strong hands
        self.fold_to_3bet = 0.7  # Usually folds to 3-bets
        self.fold_to_cbet = 0.6  # Often folds to continuation bets
        self.risk_tolerance = 0.3  # Lower than average risk tolerance
    
    def _set_tight_profile(self):
        """Set parameters for a tight player."""
        self.vpip_range = (0.12, 0.18)  # Plays only 12-18% of hands
        self.pfr_range = (0.10, 0.15)   # Raises with most hands they play
        self.aggression_factor_range = (2.0, 3.0)  # When playing, tends to be aggressive
        self.continuation_bet_freq = 0.7  # Frequently continuation bets
        self.bluff_frequency = 0.15  # Below average bluffing
        self.slowplay_frequency = 0.2  # Occasionally slowplays
        self.fold_to_3bet = 0.6  # Somewhat likely to fold to 3-bets
        self.fold_to_cbet = 0.4  # Less likely to fold to continuation bets
        self.hand_reading_skill = 0.7  # Above average hand reading
    
    def _set_loose_profile(self):
        """Set parameters for a loose player."""
        self.vpip_range = (0.35, 0.50)  # Plays 35-50% of hands
        self.pfr_range = (0.15, 0.25)   # Raises with a smaller subset
        self.aggression_factor_range = (1.5, 3.0)  # Moderately aggressive
        self.continuation_bet_freq = 0.6  # Average continuation betting
        self.bluff_frequency = 0.3  # Above average bluffing
        self.slowplay_frequency = 0.3  # Average slowplaying
        self.fold_to_3bet = 0.5  # Average folding to 3-bets
        self.fold_to_cbet = 0.4  # Less likely to fold to continuation bets
        self.risk_tolerance = 0.7  # Higher than average risk tolerance
    
    def _set_balanced_profile(self):
        """Set parameters for a balanced (GTO-style) player."""
        self.vpip_range = (0.20, 0.25)  # Plays about 20-25% of hands
        self.pfr_range = (0.18, 0.22)   # Raises with most hands they play
        self.aggression_factor_range = (2.0, 2.5)  # Balanced aggression
        self.continuation_bet_freq = 0.65  # Strategic continuation betting
        self.bluff_frequency = 0.2  # Balanced bluffing
        self.slowplay_frequency = 0.2  # Strategic slowplaying
        self.fold_to_3bet = 0.55  # Balanced response to 3-bets
        self.fold_to_cbet = 0.45  # Balanced response to continuation bets
        self.hand_reading_skill = 0.8  # Strong hand reading skills
        self.positional_awareness = 0.8  # High positional awareness
        self.adaptability = 0.7  # Adapts well to different opponents
    
    def _set_calling_station_profile(self):
        """Set parameters for a calling station player."""
        self.vpip_range = (0.35, 0.50)  # Plays many hands
        self.pfr_range = (0.05, 0.10)   # Rarely raises
        self.aggression_factor_range = (0.2, 0.5)  # Very passive
        self.continuation_bet_freq = 0.2  # Rarely continuation bets
        self.bluff_frequency = 0.05  # Almost never bluffs
        self.slowplay_frequency = 0.7  # Often slowplays
        self.fold_to_3bet = 0.3  # Often calls 3-bets
        self.fold_to_cbet = 0.2  # Rarely folds to continuation bets
        self.hand_reading_skill = 0.2  # Poor hand reading
    
    def _set_maniac_profile(self):
        """Set parameters for a maniac player."""
        self.vpip_range = (0.50, 0.70)  # Plays most hands
        self.pfr_range = (0.40, 0.60)   # Raises with most hands
        self.aggression_factor_range = (5.0, 8.0)  # Extremely aggressive
        self.continuation_bet_freq = 0.9  # Almost always continuation bets
        self.bluff_frequency = 0.5  # Bluffs frequently
        self.slowplay_frequency = 0.05  # Almost never slowplays
        self.fold_to_3bet = 0.2  # Rarely folds to 3-bets
        self.fold_to_cbet = 0.1  # Rarely folds to continuation bets
        self.risk_tolerance = 0.9  # Very high risk tolerance
        self.tilt_factor = 0.7  # Prone to tilt
    
    def _set_rock_profile(self):
        """Set parameters for a rock (extremely tight) player."""
        self.vpip_range = (0.05, 0.12)  # Plays very few hands
        self.pfr_range = (0.04, 0.10)   # Raises with most hands they play
        self.aggression_factor_range = (1.5, 2.5)  # Moderately aggressive when playing
        self.continuation_bet_freq = 0.8  # Frequently continuation bets
        self.bluff_frequency = 0.05  # Rarely bluffs
        self.slowplay_frequency = 0.3  # Sometimes slowplays
        self.fold_to_3bet = 0.8  # Usually folds to 3-bets
        self.fold_to_cbet = 0.7  # Usually folds to continuation bets
        self.risk_tolerance = 0.2  # Very low risk tolerance
    
    def _set_fish_profile(self):
        """Set parameters for a fish (weak) player."""
        self.vpip_range = (0.40, 0.60)  # Plays too many hands
        self.pfr_range = (0.10, 0.20)   # Doesn't raise enough
        self.aggression_factor_range = (0.8, 1.5)  # Not particularly aggressive
        self.continuation_bet_freq = 0.4  # Below average continuation betting
        self.bluff_frequency = 0.25  # Bluffs at bad times
        self.slowplay_frequency = 0.5  # Slowplays too much
        self.fold_to_3bet = 0.4  # Calls 3-bets too often
        self.fold_to_cbet = 0.3  # Calls continuation bets too often
        self.hand_reading_skill = 0.2  # Poor hand reading
        self.positional_awareness = 0.2  # Poor positional awareness
        self.tilt_factor = 0.7  # Prone to tilt
    
    def _set_profile_from_enum(self, profile_type: ProfileType):
        """Set parameters based on the ProfileType enum."""
        if profile_type == ProfileType.TIGHT_PASSIVE:
            self.vpip_range = (0.12, 0.18)
            self.pfr_range = (0.05, 0.10)
            self.aggression_factor_range = (0.5, 1.0)
        elif profile_type == ProfileType.TIGHT_AGGRESSIVE:
            self.vpip_range = (0.12, 0.18)
            self.pfr_range = (0.10, 0.15)
            self.aggression_factor_range = (2.0, 3.0)
        elif profile_type == ProfileType.LOOSE_PASSIVE:
            self.vpip_range = (0.35, 0.50)
            self.pfr_range = (0.10, 0.20)
            self.aggression_factor_range = (0.5, 1.0)
        elif profile_type == ProfileType.LOOSE_AGGRESSIVE:
            self.vpip_range = (0.35, 0.50)
            self.pfr_range = (0.20, 0.30)
            self.aggression_factor_range = (2.0, 4.0)
        elif profile_type == ProfileType.MANIAC:
            self._set_maniac_profile()
        elif profile_type == ProfileType.ROCK:
            self._set_rock_profile()
        elif profile_type == ProfileType.CALLING_STATION:
            self._set_calling_station_profile()
        elif profile_type == ProfileType.AGGRESSIVE_FISH:
            self.vpip_range = (0.40, 0.60)
            self.pfr_range = (0.20, 0.30)
            self.aggression_factor_range = (2.0, 3.0)
            self.hand_reading_skill = 0.2
        elif profile_type == ProfileType.PASSIVE_FISH:
            self.vpip_range = (0.40, 0.60)
            self.pfr_range = (0.05, 0.15)
            self.aggression_factor_range = (0.5, 1.0)
            self.hand_reading_skill = 0.2
        elif profile_type == ProfileType.BALANCED:
            self._set_balanced_profile()
        elif profile_type == ProfileType.ADAPTIVE:
            self.vpip_range = (0.20, 0.30)
            self.pfr_range = (0.15, 0.25)
            self.aggression_factor_range = (1.5, 3.0)
            self.adaptability = 0.9
            self.hand_reading_skill = 0.8


class BehaviorTracker:
    """Tracks and analyzes player behaviors over time."""
    
    def __init__(self):
        self.player_stats = defaultdict(lambda: {
            "hands_played": 0,
            "vpip_count": 0,  # Voluntarily put money in pot count
            "pfr_count": 0,   # Pre-flop raise count
            "aggression": {"bets": 0, "raises": 0, "calls": 0, "checks": 0},
            "round_stats": {
                BettingRound.PREFLOP: {"actions": []},
                BettingRound.FLOP: {"actions": []},
                BettingRound.TURN: {"actions": []},
                BettingRound.RIVER: {"actions": []},
            },
            "position_stats": {"early": [], "middle": [], "late": [], "blinds": []},
            "3bet_opportunities": 0,
            "3bet_count": 0,
            "fold_to_3bet_opportunities": 0,
            "fold_to_3bet_count": 0,
            "cbet_opportunities": 0,
            "cbet_count": 0,
            "fold_to_cbet_opportunities": 0,
            "fold_to_cbet_count": 0,
        })
    
    def track_action(self, player_id: int, action: Action, state: State) -> None:
        """Track a player action and update statistics."""
        stats = self.player_stats[player_id]
        
        # Update round-specific stats
        stats["round_stats"][state.betting_round]["actions"].append(action)
        
        # Update aggression metrics
        if action.action_type == ActionType.BET:
            stats["aggression"]["bets"] += 1
        elif action.action_type == ActionType.RAISE:
            stats["aggression"]["raises"] += 1
        elif action.action_type == ActionType.CALL:
            stats["aggression"]["calls"] += 1
        elif action.action_type == ActionType.CHECK:
            stats["aggression"]["checks"] += 1
        
        # Track pre-flop raises
        if state.betting_round == BettingRound.PREFLOP and action.action_type == ActionType.RAISE:
            stats["pfr_count"] += 1
        
        # Track VPIP (any voluntary money in pot pre-flop)
        if state.betting_round == BettingRound.PREFLOP and action.action_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE):
            stats["vpip_count"] += 1
        
        # Additional specialized tracking would go here
        # (3-bets, continuation bets, etc.)
    
    def start_hand(self, player_ids: List[int]) -> None:
        """Initialize tracking for a new hand."""
        for player_id in player_ids:
            self.player_stats[player_id]["hands_played"] += 1
    
    def get_player_profile(self, player_id: int) -> Dict[str, float]:
        """Calculate a player's behavioral profile from tracked statistics."""
        stats = self.player_stats[player_id]
        profile = {}
        
        # Calculate VPIP (Voluntarily Put $ In Pot)
        hands_played = stats["hands_played"]
        if hands_played > 0:
            profile["vpip"] = stats["vpip_count"] / hands_played
            profile["pfr"] = stats["pfr_count"] / hands_played
        else:
            profile["vpip"] = 0.0
            profile["pfr"] = 0.0
        
        # Calculate aggression factor
        aggressive_actions = stats["aggression"]["bets"] + stats["aggression"]["raises"]
        passive_actions = stats["aggression"]["calls"]
        
        if passive_actions > 0:
            profile["aggression_factor"] = aggressive_actions / passive_actions
        else:
            profile["aggression_factor"] = aggressive_actions if aggressive_actions > 0 else 0
        
        # Calculate 3-bet and fold to 3-bet percentages
        if stats["3bet_opportunities"] > 0:
            profile["three_bet_pct"] = stats["3bet_count"] / stats["3bet_opportunities"]
        else:
            profile["three_bet_pct"] = 0.0
            
        if stats["fold_to_3bet_opportunities"] > 0:
            profile["fold_to_3bet_pct"] = stats["fold_to_3bet_count"] / stats["fold_to_3bet_opportunities"]
        else:
            profile["fold_to_3bet_pct"] = 0.0
        
        # Calculate continuation bet percentages
        if stats["cbet_opportunities"] > 0:
            profile["cbet_pct"] = stats["cbet_count"] / stats["cbet_opportunities"]
        else:
            profile["cbet_pct"] = 0.0
            
        if stats["fold_to_cbet_opportunities"] > 0:
            profile["fold_to_cbet_pct"] = stats["fold_to_cbet_count"] / stats["fold_to_cbet_opportunities"]
        else:
            profile["fold_to_cbet_pct"] = 0.0
        
        return profile
    
    def guess_player_type(self, player_id: int) -> ProfileType:
        """Try to categorize a player based on their statistics."""
        profile = self.get_player_profile(player_id)
        
        # Simple classification logic
        vpip = profile.get("vpip", 0.0)
        pfr = profile.get("pfr", 0.0)
        af = profile.get("aggression_factor", 0.0)
        
        # Tight vs Loose
        is_tight = vpip < 0.25
        
        # Passive vs Aggressive
        is_aggressive = af > 2.0
        
        # Special cases
        if vpip > 0.5 and af > 4.0:
            return ProfileType.MANIAC
        elif vpip < 0.15:
            return ProfileType.ROCK
        elif vpip > 0.35 and af < 1.0:
            return ProfileType.CALLING_STATION
        elif vpip > 0.35 and pfr < 0.15 and af < 1.5:
            return ProfileType.PASSIVE_FISH
        elif vpip > 0.35 and pfr < 0.15 and af > 1.5:
            return ProfileType.AGGRESSIVE_FISH
        
        # Standard types
        if is_tight and is_aggressive:
            return ProfileType.TIGHT_AGGRESSIVE
        elif is_tight and not is_aggressive:
            return ProfileType.TIGHT_PASSIVE
        elif not is_tight and is_aggressive:
            return ProfileType.LOOSE_AGGRESSIVE
        else:
            return ProfileType.LOOSE_PASSIVE


class ProfileCluster:
    """Clusters players into profile groups based on behavior."""
    
    def __init__(self, num_clusters: int = 5):
        self.num_clusters = num_clusters
        self.clusters = {}
        self.player_cluster_map = {}
    
    def fit(self, player_profiles: Dict[int, Dict[str, float]]) -> None:
        """Cluster players based on their behavioral profiles."""
        if not player_profiles:
            return
            
        # Extract features for clustering
        feature_names = ["vpip", "pfr", "aggression_factor", "three_bet_pct", "fold_to_3bet_pct", 
                       "cbet_pct", "fold_to_cbet_pct"]
        
        # Prepare data
        player_ids = list(player_profiles.keys())
        X = []
        
        for player_id in player_ids:
            profile = player_profiles[player_id]
            # Extract features with defaults for missing values
            features = [profile.get(feature, 0.0) for feature in feature_names]
            X.append(features)
        
        # This would use scikit-learn in a real implementation
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=self.num_clusters)
        # cluster_labels = kmeans.fit_predict(X)
        
        # Simple placeholder implementation
        cluster_labels = self._simple_cluster(X)
        
        # Store cluster assignments
        for player_id, cluster_id in zip(player_ids, cluster_labels):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(player_id)
            self.player_cluster_map[player_id] = cluster_id
    
    def _simple_cluster(self, X: List[List[float]]) -> List[int]:
        """Simple clustering implementation (placeholder)."""
        # In a real implementation, this would use k-means or another algorithm
        # Just assign random clusters for demonstration
        import random
        return [random.randint(0, self.num_clusters-1) for _ in range(len(X))]
    
    def get_cluster(self, player_id: int) -> Optional[int]:
        """Get the cluster assignment for a player."""
        return self.player_cluster_map.get(player_id)
    
    def get_cluster_profiles(self) -> Dict[int, Dict[str, float]]:
        """Calculate the average profile for each cluster."""
        cluster_profiles = {}
        
        for cluster_id, player_ids in self.clusters.items():
            cluster_profiles[cluster_id] = self._calculate_average_profile(player_ids)
        
        return cluster_profiles
    
    def _calculate_average_profile(self, player_ids: List[int]) -> Dict[str, float]:
        """Calculate the average profile for a group of players."""
        # In a real implementation, this would average the features of all players
        # Just a placeholder
        return {
            "vpip": 0.25,
            "pfr": 0.18,
            "aggression_factor": 2.0,
            "three_bet_pct": 0.08,
            "fold_to_3bet_pct": 0.6,
            "cbet_pct": 0.7,
            "fold_to_cbet_pct": 0.5,
        }
