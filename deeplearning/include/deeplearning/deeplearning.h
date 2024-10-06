#ifndef RL_DEEPLEARNING_DEEPLEARNING_H_
#define RL_DEEPLEARNING_DEEPLEARNING_H_

struct AzTrainParameters
{
    char* environment_name;
    // number of iterations to be run
    int n_iterations;
    // number of episodes to be collected per iteration
    int n_episodes;
    // number of monte carlo tree simulations to be perfromed per evironment step
    int n_sims;
    // learning rate
    float lr;
    // critic loss coefficient
    float critic_coef;
    // number of training epochs per iteration
    int n_epochs;
    // number of batches per 1 epoch which divide the data into smaller mini batches
    int n_batches;
    // number of testing episodes/sets to test the new network head to head vs the strongest network so far
    int n_testing_episodes;

    char* network_load_name;

    char* network_save_name;

    // Number of simulations to be performed concurrently before evaluating their states
    int eval_async_steps = 8;
    // default visit count to be added for all visited states along the path of a single simulation , before being decreased after evaluating the edge node
    float n_visits = 1.0f;
    // default score (win count) to be added for all visited states along the path of a single simulation , before being decreased after evaluating the edge node
    float n_wins = -1.0f;
    // MCTS CPUCT
    float cpuct = 2.5f;
    // policy dirichlet noise in range [0,1) , 0 to disable it
    float dirichlet_epsilon = 0.25f;
    // dirichlet noise alpha , if smaller than 0 then it be calculated automatically
    float dirichlet_alpha = -1.0f;
    // Number of games/subtrees that run asynchronously , each has its own tree but all share the same evaluator, used during data collection
    int n_subtrees = 128;
    // Number of states to be collected per sub tree before evaluation
    int n_subtree_async_steps = 1;

    float complete_to_end_ratio = 1.0f / 4.0f;
    // Players that reached below this score can resign IF THEY ARE NOT COMPLETE TO END PLAYERS
    float no_resign_threshold = -0.8f;
    // Players are not allowed to resign if the number of steps is below this number
    int no_resign_steps = 30;
};


extern "C" void train_alphazero(AzTrainParameters train_parameters);

#endif