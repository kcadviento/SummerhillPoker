"""Implementation of Deep Q-Network (DQN) for poker."""

from __future__ import annotations

import random
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from summerhillpoker.state import State, Action, ActionType, BettingRound, PlayerState
from summerhillpoker.games import Game
from summerhillpoker.modeling import PlayerProfile
from summerhillpoker.hands import Card, Rank, Suit


class PokerStateEncoder:
    """Encodes poker states into numerical features for neural networks."""
    
    def __init__(self, max_players: int = 6):
        self.max_players = max_players
        self.feature_size = self._calculate_feature_size()
    
    def _calculate_feature_size(self) -> int:
        """Calculate the size of the feature vector."""
        # Features:
        # - Player cards (2 cards x 52 one-hot encoding)
        # - Board cards (5 cards x 52 one-hot encoding)
        # - Player stacks (max_players)
        # - Player positions (max_players)
        # - Current bets (max_players)
        # - Pot size (1)
        # - Betting round (4 - one-hot)
        # - Last actions (5 actions x max_players possible actors x 6 action types)
        return (2 * 52) + (5 * 52) + (3 * self.max_players) + 1 + 4 + (5 * self.max_players * 6)
    
    def encode(self, state: State, player_idx: int) -> torch.Tensor:
        """Encode a poker state into a feature vector for the specified player."""
        features = []
        
        # Encode player's cards (one-hot)
        for card in state.players[player_idx].hole_cards:
            card_idx = (card.rank.value_int - 2) * 4 + list(Suit).index(card.suit)
            card_vec = [0] * 52
            if card_idx < 52:  # Safeguard against index errors
                card_vec[card_idx] = 1
            features.extend(card_vec)
        
        # Pad if player has fewer than 2 cards
        for _ in range(2 - len(state.players[player_idx].hole_cards)):
            features.extend([0] * 52)
        
        # Encode board cards (one-hot)
        for card in state.board:
            card_idx = (card.rank.value_int - 2) * 4 + list(Suit).index(card.suit)
            card_vec = [0] * 52
            if card_idx < 52:  # Safeguard against index errors
                card_vec[card_idx] = 1
            features.extend(card_vec)
        
        # Pad if board has fewer than 5 cards
        for _ in range(5 - len(state.board)):
            features.extend([0] * 52)
        
        # Encode player stacks (normalized)
        max_stack = max(p.stack for p in state.players) if state.players else 1
        for i in range(self.max_players):
            if i < len(state.players):
                features.append(state.players[i].stack / max_stack if max_stack > 0 else 0)
            else:
                features.append(0)  # Padding for missing players
        
        # Encode player positions
        for i in range(self.max_players):
            if i < len(state.players):
                features.append(state.players[i].position / len(state.players))
            else:
                features.append(0)  # Padding for missing players
        
        # Encode current bets (normalized)
        max_bet = max(p.current_bet for p in state.players) if state.players else 1
        for i in range(self.max_players):
            if i < len(state.players):
                features.append(state.players[i].current_bet / max_bet if max_bet > 0 else 0)
            else:
                features.append(0)  # Padding for missing players
        
        # Encode pot size (normalized)
        total_chips = sum(p.stack + p.total_bet for p in state.players)
        features.append(state.pot / total_chips if total_chips > 0 else 0)
        
        # Encode betting round (one-hot)
        round_vec = [0] * 4
        round_idx = state.betting_round.value - 1
        if 0 <= round_idx < 4:
            round_vec[round_idx] = 1
        features.extend(round_vec)
        
        # Encode last actions (simplified)
        # For each of the last 5 actions, encode: player and action type
        action_history = state.actions[-5:] if state.actions else []
        for _ in range(5 - len(action_history)):
            # Padding for missing actions
            for _ in range(self.max_players):
                features.extend([0] * 6)  # 6 action types
        
        for action in action_history:
            for i in range(self.max_players):
                if i == action.player_id:
                    action_vec = [0] * 6
                    action_idx = action.action_type.value - 1
                    if 0 <= action_idx < 6:
                        action_vec[action_idx] = 1
                    features.extend(action_vec)
                else:
                    features.extend([0] * 6)
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32)


class DQNNetwork(nn.Module):
    """Neural network for DQN."""
    
    def __init__(self, input_size: int, output_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN agent for poker."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        
        # Q networks
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Tracking
        self.training_steps = 0
    
    def act(self, state: torch.Tensor, legal_actions: List[int]) -> int:
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        state = state.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state).squeeze(0)
        
        # Mask illegal actions
        masked_q_values = torch.full_like(q_values, float('-inf'))
        for action in legal_actions:
            masked_q_values[action] = q_values[action]
        
        return masked_q_values.argmax().item()
    
    def remember(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer."""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """Update Q network from experiences in replay buffer."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Update Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']


class DQNTrainer:
    """Trainer for DQN agents in poker environments."""
    
    def __init__(self, game: Game):
        self.game = game
        self.state_encoder = PokerStateEncoder()
        
        # Map action types to indices
        self.action_to_idx = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.BET: 3,
            ActionType.RAISE: 4,
            ActionType.ALL_IN: 5
        }
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        
        # Create agent
        self.agent = DQNAgent(
            state_size=self.state_encoder.feature_size,
            action_size=len(self.action_to_idx),
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=64,
            target_update=10
        )
    
    def train(self, episodes: int = 1000, opponent_profiles: List[PlayerProfile] = None) -> DQNAgent:
        """Train a DQN agent against opponents."""
        if opponent_profiles is None:
            # Default opponent profiles
            opponent_profiles = [PlayerProfile("balanced"), PlayerProfile("aggressive")]
        
        # Training loop
        for episode in range(episodes):
            # Create new game state
            num_players = len(opponent_profiles) + 1  # +1 for DQN agent
            state = self.game.create_state(num_players, 1000)
            self.game.deal_cards(state)
            
            # DQN agent always has player_idx 0
            player_idx = 0
            
            # Play until hand is over
            done = False
            episode_reward = 0
            
            while not done:
                # Get current state encoding
                current_state_tensor = self.state_encoder.encode(state, player_idx)
                
                # Get legal actions
                if state.current_player_idx == player_idx:
                    legal_actions = state.get_legal_actions(player_idx)
                    legal_action_indices = [self.action_to_idx[action] for action in legal_actions]
                    
                    # Select action
                    action_idx = self.agent.act(current_state_tensor, legal_action_indices)
                    action_type = self.idx_to_action[action_idx]
                    
                    # Create action
                    amount = 0
                    if action_type == ActionType.CALL:
                        amount = state.current_bet - state.players[player_idx].current_bet
                    elif action_type == ActionType.BET:
                        amount = state.pot // 2  # Bet half pot
                    elif action_type == ActionType.RAISE:
                        amount = max(state.current_bet * 2, state.big_blind)
                    elif action_type == ActionType.ALL_IN:
                        amount = state.players[player_idx].stack
                    
                    # Apply action
                    action = Action(action_type, amount, player_idx)
                    old_stack = state.players[player_idx].stack
                    state.apply_action(action)
                    
                    # Check if hand is over
                    if state.is_terminal():
                        done = True
                        # Calculate reward based on winnings
                        winnings = self.game.evaluate_hands(state)
                        reward = winnings.get(player_idx, 0) - (old_stack - state.players[player_idx].stack)
                        episode_reward += reward
                        
                        # Get next state encoding
                        next_state_tensor = self.state_encoder.encode(state, player_idx)
                        
                        # Store experience
                        self.agent.remember(
                            current_state_tensor,
                            action_idx,
                            reward,
                            next_state_tensor,
                            done
                        )
                    else:
                        # Immediate reward is negative of amount bet
                        immediate_reward = -(old_stack - state.players[player_idx].stack)
                        episode_reward += immediate_reward
                        
                        # Wait until agent's turn again to store experience
                        # (this is a simplification, in reality we'd want to model the state after opponent actions)
                else:
                    # Opponent's turn - simulate their action
                    opponent_idx = state.current_player_idx
                    profile = opponent_profiles[opponent_idx - 1]  # -1 because agent is player 0
                    
                    # Simulate the opponent's decision based on their profile
                    legal_actions = state.get_legal_actions(opponent_idx)
                    
                    # Very simple opponent logic based on profile aggressiveness
                    # In a real implementation, this would use the actual opponent models
                    aggression = (profile.aggression_factor_range[0] + profile.aggression_factor_range[1]) / 2
                    
                    if ActionType.CHECK in legal_actions and random.random() > aggression / 5:
                        action_type = ActionType.CHECK
                        amount = 0
                    elif ActionType.CALL in legal_actions and random.random() > aggression / 3:
                        action_type = ActionType.CALL
                        amount = state.current_bet - state.players[opponent_idx].current_bet
                    elif ActionType.FOLD in legal_actions and random.random() > aggression:
                        action_type = ActionType.FOLD
                        amount = 0
                    elif ActionType.RAISE in legal_actions:
                        action_type = ActionType.RAISE
                        amount = max(state.current_bet * 2, state.big_blind)
                    elif ActionType.BET in legal_actions:
                        action_type = ActionType.BET
                        amount = state.pot // 2
                    else:
                        # Default action
                        action_type = legal_actions[0]
                        amount = 0
                        if action_type == ActionType.CALL:
                            amount = state.current_bet - state.players[opponent_idx].current_bet
                    
                    # Apply opponent action
                    action = Action(action_type, amount, opponent_idx)
                    state.apply_action(action)
                    
                    # Check if hand is over
                    if state.is_terminal():
                        done = True
                        # Calculate reward based on winnings
                        winnings = self.game.evaluate_hands(state)
                        reward = winnings.get(player_idx, 0) - (old_stack - state.players[player_idx].stack)
                        episode_reward += reward
                        
                        # Get next state encoding
                        next_state_tensor = self.state_encoder.encode(state, player_idx)
                        
                        # Store experience
                        self.agent.remember(
                            current_state_tensor,
                            action_idx if 'action_idx' in locals() else 0,  # Default if agent didn't act
                            reward,
                            next_state_tensor,
                            done
                        )
            
            # Learn from experiences
            loss = self.agent.learn()
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Epsilon: {self.agent.epsilon:.2f}")
        
        return self.agent
    
    def save_agent(self, path: str):
        """Save the trained agent."""
        self.agent.save(path)
    
    def load_agent(self, path: str):
        """Load a trained agent."""
        self.agent.load(path)
        
    def evaluate(self, episodes: int = 100, opponent_profiles: List[PlayerProfile] = None) -> Dict[str, float]:
        """Evaluate the agent against opponents."""
        if opponent_profiles is None:
            opponent_profiles = [PlayerProfile("balanced"), PlayerProfile("aggressive")]
        
        total_reward = 0
        wins = 0
        
        for episode in range(episodes):
            # Create new game state
            num_players = len(opponent_profiles) + 1
            state = self.game.create_state(num_players, 1000)
            self.game.deal_cards(state)
            
            # DQN agent always has player_idx 0
            player_idx = 0
            initial_stack = state.players[player_idx].stack
            
            # Play until hand is over
            done = False
            
            while not done:
                # Get current state encoding
                current_state_tensor = self.state_encoder.encode(state, player_idx)
                
                # Get legal actions
                if state.current_player_idx == player_idx:
                    legal_actions = state.get_legal_actions(player_idx)
                    legal_action_indices = [self.action_to_idx[action] for action in legal_actions]
                    
                    # Select action (no exploration during evaluation)
                    self.agent.epsilon = 0
                    action_idx = self.agent.act(current_state_tensor, legal_action_indices)
                    action_type = self.idx_to_action[action_idx]
                    
                    # Create action
                    amount = 0
                    if action_type == ActionType.CALL:
                        amount = state.current_bet - state.players[player_idx].current_bet
                    elif action_type == ActionType.BET:
                        amount = state.pot // 2
                    elif action_type == ActionType.RAISE:
                        amount = max(state.current_bet * 2, state.big_blind)
                    elif action_type == ActionType.ALL_IN:
                        amount = state.players[player_idx].stack
                    
                    # Apply action
                    action = Action(action_type, amount, player_idx)
                    state.apply_action(action)
                else:
                    # Opponent's turn
                    opponent_idx = state.current_player_idx
                    profile = opponent_profiles[opponent_idx - 1]
                    
                    # Simple opponent logic based on profile
                    legal_actions = state.get_legal_actions(opponent_idx)
                    aggression = (profile.aggression_factor_range[0] + profile.aggression_factor_range[1]) / 2
                    
                    if ActionType.CHECK in legal_actions and random.random() > aggression / 5:
                        action_type = ActionType.CHECK
                        amount = 0
                    elif ActionType.CALL in legal_actions and random.random() > aggression / 3:
                        action_type = ActionType.CALL
                        amount = state.current_bet - state.players[opponent_idx].current_bet
                    elif ActionType.FOLD in legal_actions and random.random() > aggression:
                        action_type = ActionType.FOLD
                        amount = 0
                    elif ActionType.RAISE in legal_actions:
                        action_type = ActionType.RAISE
                        amount = max(state.current_bet * 2, state.big_blind)
                    elif ActionType.BET in legal_actions:
                        action_type = ActionType.BET
                        amount = state.pot // 2
                    else:
                        # Default action
                        action_type = legal_actions[0]
                        amount = 0
                        if action_type == ActionType.CALL:
                            amount = state.current_bet - state.players[opponent_idx].current_bet
                    
                    # Apply opponent action
                    action = Action(action_type, amount, opponent_idx)
                    state.apply_action(action)
                
                # Check if hand is over
                if state.is_terminal():
                    done = True
                    # Calculate reward based on winnings
                    winnings = self.game.evaluate_hands(state)
                    if player_idx in winnings and winnings[player_idx] > 0:
                        wins += 1
                    
                    final_stack = state.players[player_idx].stack
                    episode_reward = final_stack - initial_stack
                    total_reward += episode_reward
            
            if (episode + 1) % 10 == 0:
                print(f"Evaluation Episode {episode + 1}/{episodes}, Reward: {episode_reward}")
        
        # Calculate metrics
        avg_reward = total_reward / episodes
        win_rate = wins / episodes
        
        return {
            "average_reward": avg_reward,
            "win_rate": win_rate
        }