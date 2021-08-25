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

#pragma once

#include <assert.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace liars_dice
{

  // All possible actions the agent can play
  enum class Action
  {
    FOLD,
    CHECK,
    CALL,
    // Raise by 2 x big blind
    RAISE_2BB,
    RAISE_3BB,
    ALL_IN
  };

  enum class Suit
  {
    SPADES,
    HEARTS,
    DIAMONDS,
    CLUBS
  };

  struct Card
  {
    Suit suit, int number;
  }

  // Public state of the game without tracking the history of the game.
  struct PartialPublicState
  {
    // Previous call.
    Action last_action;
    // Player to make move next.
    int player_id;

    bool operator==(const PartialPublicState &state) const
    {
      return last_action == state.last_action && player_id == state.player_id;
    }
  };
  class Game
  {
  public:
    const std::vector<Card> cards_on_table;
    const int community_pot;
    // Player ids for small and big blind
    const int small_blind;
    const int big_blind;
    const int deck_size;

    // TODO: Need to change
    Game(int deck_size)
        : deck_size(deck_size),
          total_num_dice_(num_dice * 2),
          num_actions_(1 + total_num_dice_ * num_faces),
          num_hands_(int_pow(num_faces, num_dice)),
          liar_call_(num_actions_ - 1),
          wild_face_(num_faces - 1) {}

    std::string action_to_string(Action action) const;
    std::string state_to_string(const PartialPublicState &state) const;
    std::string action_to_string_short(Action action) const;
    std::string state_to_string_short(const PartialPublicState &state) const;

  private:
    static constexpr int kInitialAction = -1;
    const Action num_actions_;
    const int num_hands_;
    const Action liar_call_;
    const int wild_face_;
  };

} // namespace liars_dice