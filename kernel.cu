#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// Constants
#define M_PI            3.14159265f
#define TRADING_DAYS    252.0

// Calibration
#define BUCKET_SIZE         0.25f // TODO: implement dynamic bucket size based on strike width or other factors
#define BUCKET_LOW          5.0f // TODO: sizing based on standard deviation (derived from monte carlo simulation?) and/or personal conviction
#define BUCKET_HIGH         200.0f
#define SIMULATIONS_COUNT   10000 // TODO: it breaks 100k, why?

// Code Stuff
#define BUCKET_ARRAY_LENGTH (int) ((BUCKET_HIGH - BUCKET_LOW)/BUCKET_SIZE)                  // TODO: use a collection sizeable at run time
                                                                                            // TODO: store macro formulas with constant value in tmp vars
#define RANDSTATETYPE       curandState*                                                    // TODO: might not be useful anymore
#define SIMS_SIZE(x)        (SIMULATIONS_COUNT   * sizeof(x))                               // TODO: store macro formulas with constant value in tmp vars
#define BUCKETS_SIZE(x)     (BUCKET_ARRAY_LENGTH * sizeof(x))                               // TODO: store macro formulas with constant value in tmp vars
#define THREADS_PER_BLOCK   1024
#define NUM_BLOCKS          (SIMULATIONS_COUNT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK // TODO: store macro formulas with constant value in tmp vars

// Option Stats
#define DTE          45.0  // TODO: support calendars and diagonals
#define STRIKE_PRICE 105.0 // TODO: support multiple option legs
#define PREMIUM      5.55
#define IS_PUT       true
#define IS_BUY       false

// Underlying Stats
#define UNDERLYING_PRICE 105.0
#define SIGMA            0.46

// Market/Economy Stats
#define RISK_FREE_RATE 0.05

// Formulas
#define YTE DTE/TRADING_DAYS

// Run GPU Code With Error Check
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void initRandomGenerator(RANDSTATETYPE state, unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void monteCarloSimulation(float* prices, float S, float T, float r, float sigma, RANDSTATETYPE state, int qty_simulations = SIMULATIONS_COUNT) // TODO: double or float
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double price = S; // TODO: better variable naming
    double dt = T / TRADING_DAYS; // TODO: this doesn't look right

    // Each thread performs its own simulation and places result into prices array
    for (int i = tid; i < qty_simulations; i += blockDim.x * gridDim.x) { // TODO: what is required to run >10000 simulations? More VRAM?
        // Generate a random price path for the underlying asset
        price = S;

        for (double t = 0; t < T; t += dt) {
            double2 randomNums;
            randomNums.x = curand_uniform(&state[tid]); // TODO: switch to curand_uniform_double2 (didn't work before)
            randomNums.y = curand_uniform(&state[tid]);
            double z;
            if (randomNums.x > 0) {
                z = sqrt(-2.0 * log(randomNums.x)) * cos(2.0 * M_PI * randomNums.y); // TODO: Validate this formula
            }
            else z = 0;
            double drift = (r - 0.5 * sigma * sigma) * dt;
            double diffusion = sigma * sqrt(dt) * z;
            price *= exp(drift + diffusion);
        }

        // Store the final price in the array
        prices[i] = static_cast<float>(price); // Convert back to float
    }
}

double option_pl_at_expiry(double underlying_price, double strike_price, double premium, bool is_put, bool is_buy) {
    if (is_buy) { // TODO: fix formatting to look better
        if (is_put) return fmax(strike_price - underlying_price, 0.0) - premium; // TODO: don't trust this until I sit down and give a real answer
        return fmax(underlying_price - strike_price, 0.0) - premium; // TODO: don't trust this until I sit down and give a real answer
    }
    if (is_put) {
        if (underlying_price >= strike_price) return premium;
        return premium - (strike_price - underlying_price);
    }
    return fmin(premium, strike_price - underlying_price); // TODO: don't trust this until I sit down and give a real answer
}

void option_pl_array_populate(double strike_price, double premium, bool is_put, bool is_buy, double *pl_array, int len = BUCKET_ARRAY_LENGTH) {
    double tmp_price = BUCKET_LOW;
    for (int i = 0; i < len; i++) {
        pl_array[i] = option_pl_at_expiry(tmp_price, strike_price, premium, is_put, is_buy);
        tmp_price += BUCKET_SIZE;
    }
}

int main()
{
    // Allocate memory on the host for price results
    float* prices = (float*)malloc(SIMS_SIZE(float));

    double* optionPL = (double*)malloc(BUCKETS_SIZE(double));

    // Allocate memory on the device for price results
    float* devPrices;
    gpuErrorCheck(cudaMalloc((void**)&devPrices, SIMS_SIZE(float)));

    // Allocate memory on the device for random number generation states
    RANDSTATETYPE devStates;
    gpuErrorCheck(cudaMalloc((void**)&devStates, SIMS_SIZE(RANDSTATETYPE)));

    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrorCheck(cudaSetDevice(0));

    // Launch the kernel
    int numBlocks = (SIMULATIONS_COUNT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    initRandomGenerator << <numBlocks, THREADS_PER_BLOCK >> > (devStates, time(NULL));
    monteCarloSimulation << <numBlocks, THREADS_PER_BLOCK >> > (devPrices, SIMULATIONS_COUNT, UNDERLYING_PRICE, yearsUntilExpiration, RISK_FREE_RATE, SIGMA, devStates);

    // Copy the results back to the host
    gpuErrorCheck(cudaMemcpy(prices, devPrices, SIMS_SIZE(float), cudaMemcpyDeviceToHost));

    //print generated prices
    /*for (int i = 0; i < SIMULATIONS_COUNT; i++) {
        printf("sim %i price: $%.2f", i, prices[i]);
    }*/

    float probabilities[BUCKET_ARRAY_SIZE];
    float current_price = BUCKET_LOW;
    float current_bucket_top;
    for (int i = 0; i < BUCKET_ARRAY_SIZE; i++) {
        current_bucket_top = current_price + BUCKET_SIZE;
        probabilities[i] = 0.0f;
        for (int j = 0; j < SIMULATIONS_COUNT; j++) {
            if (prices[j] >= current_price && prices[j] < current_bucket_top) {
                probabilities[i] += 1.0f;
            }
        }
        probabilities[i] /= SIMULATIONS_COUNT;
        current_price += BUCKET_SIZE;
    }

    current_price = BUCKET_LOW;

    option_pl_array_populate(STRIKEPRICE, PREMIUM, IS_PUT, IS_BUY, optionPL, BUCKET_ARRAY_SIZE);

    double total = 0.0;

    for (int i = 0; i < BUCKET_ARRAY_SIZE; i++) {
        printf("$%.2f to $%.2f expected return: $%.2f, probability: %.2f%%, factor: $%.2f\n", current_price, current_price + BUCKET_SIZE, optionPL[i], probabilities[i] * 100.0f, optionPL[i] * probabilities[i]);
        current_price += BUCKET_SIZE;
        total += optionPL[i] * probabilities[i];
    }

    std::cout << "total expected return: " << total << std::endl;

    // Free memory
    free(prices);
    free(optionPL);
    gpuErrorCheck(cudaFree(devPrices));
    gpuErrorCheck(cudaFree(devStates));

    std::cout << "Done." << std::endl;

    return 0;
}
