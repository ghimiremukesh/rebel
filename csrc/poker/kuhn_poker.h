// Copyright (c) Facebook, Inc. and its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// Kuhn Poker -- simplified version of Poker : 
// From: https://poker.cs.ualberta.ca/publications/AAAI05.pdf

// Kuhn poker is a very simple, two-player
// game (P1 - Player 1, P2 - Player 2). The deck consists
// of three cards (J - Jack, Q - Queen, and K - King). There
// are two actions available: bet and pass. The value of
// each bet is 1. In the event of a showdown (players have
// matched bets), the player with the higher card wins the
// pot (the King is highest and the Jack is lowest). 

// Rules: 

    // • Both players initially put an ante of 1 into the pot.
    // • Each player is dealt a single card and the remaining
    // card is unseen by either player.
    // • After the deal, P1 has the opportunity to bet or pass.
    // – If P1 bets in round one, then in round two P2 can:
    // ∗ bet (calling P1’s bet) and the game then ends in a
    // showdown, or
    // ∗ pass (folding) and forfeit the pot to P1.
    // – If P1 passes in round one, then in round two P2
    // can:
    // ∗ bet (in which case there is a third action where P1
    // can bet and go to showdown, or pass and forfeit
    // to P2), or
    // ∗ pass (game proceeds to a showdown).


    // For Reference :: https://github.com/Danielhp95/gym-kuhn-poker 


#pragma once

#include <assert.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace kuhn_poker
{

  // All possible actions the agent can play
  // In Kuhn Poker player can only bet or pass
  enum Action {kPass = 0, kBet = 1}; 


  //struct Card
  //{
  //  int number; // card numbers: J (12), Q (13), K (14)
  //}

  // Public state of the game without tracking the history of the game.
  struct PartialPublicState
  {
    // Previous call.
      Action last_action;
    // Player to make move next.
    //int player_id;

    // for poker, pps consists of: (page #18 of Rebel Paper) 

      // board cards
      std::vector<int> board_cards; // vector of cards in the table; for kuhn poker there is only one card left on the board

      // common pot size divided by stack size
      double relative_pot;

      // acting player
      int player_id;
    

    bool operator==(const PartialPublicState &state) const
    {
      return last_action == state.last_action && player_id == state.player_id;
    }
  };
  class Game
  {
  public: // Need to think about this, is this enough? 
    const int deck_size; // may be it should be the card
    // const int card_number; // card numbers: J (12), Q (13), K (14)
    const std::pair<int, int> community_pot; // (p1, p2) contribution of each player to the pot. ie. Player's antes

    // TODO: Need to change
    Game(int deck_size, std::pair<int, int> community_pot)
        : deck_size(deck_size),
          community_pot(community_pot),
          num_actions_(2), // always 2 (bet/pass)
          num_hands_(deck_size){}
          
    Action num_actions() const { return num_actions_; }
    // Number of distrinct game states at the beginning of the game. In other
    // words, number of different realization of the chance nodes.
    int num_hands() const { return num_hands_; }

    // Get range of possible actions in the state as [min_action, max_action).
    
    //Not needed?
    std::pair<Action, Action> get_bid_range(
        const PartialPublicState &state) const
    {
      return (0, 1); // only two actions throughout the game 
    }

    PartialPublicState get_initial_state() const
    {
      PartialPublicState state;
      state.last_bid = kInitialAction;
      state.player_id = 0;
      return state;
    }

    PartialPublicState act(const PartialPublicState &state, Action action) const
    {
      const auto range = get_bid_range(state)
      //assert(action >= range.first);
      //assert(action <= range.second);
      PartialPublicState new_state;
      new_state.last_bid = action;
      new_state.player_id = 1 - state.player_id;
      return new_state;
    }

    std::string action_to_string(Action action) const;
    std::string state_to_string(const PartialPublicState &state) const;
    std::string action_to_string_short(Action action) const;
    std::string state_to_string_short(const PartialPublicState &state) const;

  private:
    static constexpr int kInitialAction = -1;
    const Action num_actions_;
  };

} // namespace kuhn_poker