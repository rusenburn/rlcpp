#include <deeplearning/deeplearning.h>
#include <deeplearning/deeplearning_constants.h>
#include <vector>
#include <string>
#include <utility>
#include <string_view>
#include <absl/log/log.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/string_view.h>

ABSL_FLAG(std::string, environment_name, AZ_ENVIRONMENT_NAME, "");
ABSL_FLAG(int, n_iterations, AZ_N_ITERATIONS, "");
ABSL_FLAG(int, n_episodes, AZ_N_EPISODES, "");
ABSL_FLAG(int, n_simulations, AZ_N_SIMULATIONS, "");
ABSL_FLAG(float, learning_rate, AZ_LEARNING_RATE, "");
ABSL_FLAG(float, critic_coef, AZ_CRITIC_COEF, "");
ABSL_FLAG(float, n_epochs, AZ_N_EPOCHS, "");
ABSL_FLAG(float, n_batches, AZ_N_BATCHES, "");
ABSL_FLAG(float, n_testing_episodes, AZ_N_TESTING_EPISODES, "");
ABSL_FLAG(std::string, network_load_name, AZ_NETWORK_LOAD_NAME, "");
ABSL_FLAG(std::string, network_save_name, AZ_NETWORK_SAVE_NAME, "");
ABSL_FLAG(int, n_testing_subtree_async_steps, AZ_N_TESTING_SUBTREE_ASYNC_STEPS, "");
ABSL_FLAG(float, n_visits, AZ_N_VISITS, "");
ABSL_FLAG(float, n_wins, AZ_N_WINS, "");
ABSL_FLAG(float, cpuct, AZ_CPUCT, "");
ABSL_FLAG(float, dirichlet_epsilon, AZ_DIRICHLET_EPSILON, "");
ABSL_FLAG(float, dirichlet_alpha, AZ_DIRICHLET_ALPHA, "");
ABSL_FLAG(int, n_subtrees, AZ_N_SUBTREES, "");
ABSL_FLAG(int, n_train_subtree_async_steps, AZ_N_TRAINING_SUBTREE_ASYNC_STEPS, "");
ABSL_FLAG(float, complete_to_end_ratio, AZ_COMPLETE_TO_END_RATIO, "");
ABSL_FLAG(float, no_resign_threshold, AZ_NO_RESIGN_THRESHOLD, "");
ABSL_FLAG(int, no_resign_steps, AZ_NO_RESIGN_STEPS, "");
int main(int argc, char** argv)
{
    absl::lts_20240722::ParseCommandLine(argc, argv);
    AzTrainParameters config{};
    config.environment_name = absl::GetFlag(FLAGS_environment_name).data();
    config.n_iterations = absl::GetFlag(FLAGS_n_iterations);
    config.n_episodes = absl::GetFlag(FLAGS_n_episodes);
    config.n_sims = absl::GetFlag(FLAGS_n_simulations);
    config.lr = absl::GetFlag(FLAGS_learning_rate);
    config.critic_coef = absl::GetFlag(FLAGS_critic_coef);
    config.n_epochs = absl::GetFlag(FLAGS_n_epochs);
    config.n_batches = absl::GetFlag(FLAGS_n_batches);
    config.n_testing_episodes = absl::GetFlag(FLAGS_n_testing_episodes);
    config.network_load_name = absl::GetFlag(FLAGS_network_load_name).data();
    config.network_save_name = absl::GetFlag(FLAGS_network_save_name).data();
    config.eval_async_steps = absl::GetFlag(FLAGS_n_testing_subtree_async_steps);
    config.n_visits = absl::GetFlag(FLAGS_n_visits);
    config.n_wins = absl::GetFlag(FLAGS_n_wins);
    config.cpuct = absl::GetFlag(FLAGS_cpuct);
    config.dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon);
    config.dirichlet_alpha = absl::GetFlag(FLAGS_dirichlet_alpha);
    config.n_subtrees = absl::GetFlag(FLAGS_n_subtrees);
    config.n_subtree_async_steps = absl::GetFlag(FLAGS_n_train_subtree_async_steps);
    config.complete_to_end_ratio = absl::GetFlag(FLAGS_complete_to_end_ratio);
    config.no_resign_threshold = absl::GetFlag(FLAGS_no_resign_threshold);
    config.no_resign_steps = absl::GetFlag(FLAGS_no_resign_steps);

    train_alphazero(config);

}
