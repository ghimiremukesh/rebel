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

#include <stdio.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include "real_net.h"
#include "recursive_solving.h"
#include "stats.h"

#include "rela/context.h"
#include "rela/data_loop.h"
#include "rela/prioritized_replay.h"
#include "rela/thread_loop.h"

namespace py = pybind11;
using namespace rela;

namespace {

std::shared_ptr<ThreadLoop> create_cfr_thread(
    std::shared_ptr<ModelLocker> modelLocker,
    std::shared_ptr<ValuePrioritizedReplay> replayBuffer,
    const kuhn_poker::RecursiveSolvingParams& cfg, int seed) {
  auto connector =
      std::make_shared<CVNetBufferConnector>(modelLocker, replayBuffer);
  return std::make_shared<DataThreadLoop>(std::move(connector), cfg, seed);
}

float compute_exploitability(kuhn_poker::RecursiveSolvingParams params,
                             const std::string& model_path) {
  py::gil_scoped_release release;
  kuhn_poker::Game game(params.deck_size, params.community_pot);
  std::shared_ptr<IValueNet> net =
      kuhn_poker::create_torchscript_net(model_path);
  const auto tree_strategy =
      compute_strategy_recursive(game, params.subgame_params, net);
  kuhn_poker::print_strategy(game, unroll_tree(game), tree_strategy);
  return kuhn_poker::compute_exploitability(game, tree_strategy);
}

auto compute_stats_with_net(kuhn_poker::RecursiveSolvingParams params,
                            const std::string& model_path) {
  py::gil_scoped_release release;
  kuhn_poker::Game game(params.deck_size, params.community_pot);
  std::shared_ptr<IValueNet> net =
      kuhn_poker::create_torchscript_net(model_path);
  const auto net_strategy =
      compute_strategy_recursive_to_leaf(game, params.subgame_params, net);
  kuhn_poker::print_strategy(game, unroll_tree(game), net_strategy);
  const float explotability =
      kuhn_poker::compute_exploitability(game, net_strategy);

  auto full_params = params.subgame_params;
  full_params.max_depth = 100000;
  auto fp = build_solver(game, full_params);
  fp->multistep();
  const auto& full_strategy = fp->get_strategy();

  const float mse_net_traverse = eval_net(
      game, net_strategy, full_strategy, params.subgame_params.max_depth,
      params.subgame_params.num_iters, net, /*traverse_by_net=*/true,
      /*verbose=*/true);
  const float mse_full_traverse = eval_net(
      game, net_strategy, full_strategy, params.subgame_params.max_depth,
      params.subgame_params.num_iters, net, /*traverse_by_net=*/false,
      /*verbose=*/true);
  return std::make_tuple(explotability, mse_net_traverse, mse_full_traverse);
}

float compute_exploitability_no_net(kuhn_poker::RecursiveSolvingParams params) {
  py::gil_scoped_release release;
  kuhn_poker::Game game(params.deck_size, params.community_pot);
  auto fp = kuhn_poker::build_solver(game, game.get_initial_state(),
                                     kuhn_poker::get_initial_beliefs(game),
                                     params.subgame_params, /*net=*/nullptr);
  float values[2] = {0.0};
  for (int iter = 0; iter < params.subgame_params.num_iters; ++iter) {
    if (((iter + 1) & iter) == 0 ||
        iter + 1 == params.subgame_params.num_iters) {
      auto values = compute_exploitability2(game, fp->get_strategy());
      printf("Iter=%8d exploitabilities=(%.3e, %.3e) sum=%.3e\n", iter + 1,
             values[0], values[1], (values[0] + values[1]) / 2.);
    }
    // Check for Ctrl-C.
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
  }
  kuhn_poker::print_strategy(game, unroll_tree(game), fp->get_strategy());
  return values[0] + values[1];
}

// std::shared_ptr<MyAgent> create_value_policy_agent(
//     std::shared_ptr<ModelLocker> modelLocker,
//     std::shared_ptr<ValuePrioritizedReplay> replayBuffer,
//     std::shared_ptr<ValuePrioritizedReplay> policyReplayBuffer,
//     bool compress_policy_values) {
//   return std::make_shared<MyAgent>(modelLocker, replayBuffer,
//                                    policyReplayBuffer,
//                                    compress_policy_values);
// }

}  // namespace

PYBIND11_MODULE(rela, m) {
  py::class_<ValueTransition, std::shared_ptr<ValueTransition>>(
      m, "ValueTransition")
      .def(py::init<>())
      .def_readwrite("query", &ValueTransition::query)
      .def_readwrite("values", &ValueTransition::values);

  py::class_<ValuePrioritizedReplay, std::shared_ptr<ValuePrioritizedReplay>>(
      m, "ValuePrioritizedReplay")
      .def(py::init<int,    // capacity,
                    int,    // seed,
                    float,  // alpha, priority exponent
                    float,  // beta, importance sampling exponent
                    int, bool, bool>(),
           py::arg("capacity"), py::arg("seed"), py::arg("alpha"),
           py::arg("beta"), py::arg("prefetch"), py::arg("use_priority"),
           py::arg("compressed_values"))
      .def("size", &ValuePrioritizedReplay::size)
      .def("num_add", &ValuePrioritizedReplay::numAdd)
      .def("sample", &ValuePrioritizedReplay::sample)
      .def("pop_until", &ValuePrioritizedReplay::popUntil)
      .def("load", &ValuePrioritizedReplay::load)
      .def("save", &ValuePrioritizedReplay::save)
      .def("extract", &ValuePrioritizedReplay::extract)
      .def("push", &ValuePrioritizedReplay::push,
           py::call_guard<py::gil_scoped_release>())
      .def("update_priority", &ValuePrioritizedReplay::updatePriority);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  py::class_<kuhn_poker::SubgameSolvingParams>(m, "SubgameSolvingParams")
      .def(py::init<>())
      .def_readwrite("num_iters", &kuhn_poker::SubgameSolvingParams::num_iters)
      .def_readwrite("max_depth", &kuhn_poker::SubgameSolvingParams::max_depth)
      .def_readwrite("linear_update",
                     &kuhn_poker::SubgameSolvingParams::linear_update)
      .def_readwrite("optimistic",
                     &kuhn_poker::SubgameSolvingParams::optimistic)
      .def_readwrite("use_cfr", &kuhn_poker::SubgameSolvingParams::use_cfr)
      .def_readwrite("dcfr", &kuhn_poker::SubgameSolvingParams::dcfr)
      .def_readwrite("dcfr_alpha",
                     &kuhn_poker::SubgameSolvingParams::dcfr_alpha)
      .def_readwrite("dcfr_beta", &kuhn_poker::SubgameSolvingParams::dcfr_beta)
      .def_readwrite("dcfr_gamma",
                     &kuhn_poker::SubgameSolvingParams::dcfr_gamma);

  py::class_<kuhn_poker::RecursiveSolvingParams>(m, "RecursiveSolvingParams")
      .def(py::init<>())
      .def_readwrite("deck_size", &kuhn_poker::RecursiveSolvingParams::deck_size)
      .def_readwrite("community_pot",
                     &kuhn_poker::RecursiveSolvingParams::community_pot)
      .def_readwrite("random_action_prob",
                     &kuhn_poker::RecursiveSolvingParams::random_action_prob)
      .def_readwrite("sample_leaf",
                     &kuhn_poker::RecursiveSolvingParams::sample_leaf)
      .def_readwrite("subgame_params",
                     &kuhn_poker::RecursiveSolvingParams::subgame_params);

  py::class_<DataThreadLoop, ThreadLoop, std::shared_ptr<DataThreadLoop>>(
      m, "DataThreadLoop")
      .def(py::init<std::shared_ptr<CVNetBufferConnector>,
                    const kuhn_poker::RecursiveSolvingParams&, int>(),
           py::arg("connector"), py::arg("params"), py::arg("thread_id"));

  py::class_<rela::Context>(m, "Context")
      .def(py::init<>())
      .def("push_env_thread", &rela::Context::pushThreadLoop,
           py::keep_alive<1, 2>())
      .def("start", &rela::Context::start)
      .def("pause", &rela::Context::pause)
      .def("resume", &rela::Context::resume)
      .def("terminate", &rela::Context::terminate)
      .def("terminated", &rela::Context::terminated);

  py::class_<ModelLocker, std::shared_ptr<ModelLocker>>(m, "ModelLocker")
      .def(py::init<std::vector<py::object>, const std::string&>())
      .def("update_model", &ModelLocker::updateModel);

  m.def("compute_exploitability_fp", &compute_exploitability_no_net,
        py::arg("params"));

  m.def("compute_exploitability_with_net", &compute_exploitability,
        py::arg("params"), py::arg("model_path"));

  m.def("compute_stats_with_net", &compute_stats_with_net, py::arg("params"),
        py::arg("model_path"));

  m.def("create_cfr_thread", &create_cfr_thread, py::arg("model_locker"),
        py::arg("replay"), py::arg("cfg"), py::arg("seed"));

  //   m.def("create_value_policy_agent", &create_value_policy_agent,
  //         py::arg("model_locker"), py::arg("replay"),
  //         py::arg("policy_replay"),
  //         py::arg("compress_policy_values"));
}
