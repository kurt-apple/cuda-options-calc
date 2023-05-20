#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#define M_PI 3.14159265f
#define TRADING_DAYS 252
#define BUCKET_SIZE 0.25f
//TODO implement dynamic sizing based on standard deviation of monte carlo simulation
#define BUCKET_LOW 5.0f
#define BUCKET_HIGH 200.0f
#define BUCKET_ARRAY_SIZE (int) ((BUCKET_HIGH - BUCKET_LOW)/BUCKET_SIZE)
#define RANDSTATETYPE curandState*

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

__global__ void monteCarloSimulation(float* prices, int numSimulations, float S, float T, float r, float sigma, RANDSTATETYPE state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double price = S;
    double dt = T / TRADING_DAYS; // Assuming 252 trading days in a year

    // Each thread performs its own simulation
    for (int i = tid; i < numSimulations; i += blockDim.x * gridDim.x) {
        // Generate a random price path for the underlying asset
        price = S;

        for (double t = 0; t < T; t += dt) {
            double2 randomNums;
            randomNums.x = curand_uniform(&state[tid]);
            randomNums.y = curand_uniform(&state[tid]);
            double z;
            if (randomNums.x > 0) {
                z = sqrt(-2.0 * log(randomNums.x)) * cos(2.0 * M_PI * randomNums.y); // Standard normal random variable
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
    if (is_buy) {
        if (is_put) return fmax(strike_price - underlying_price, 0.0) - premium;
        return fmax(underlying_price - strike_price, 0.0) - premium;
    }
    if (is_put) {
        if (underlying_price >= strike_price) return premium;
        return premium - (strike_price - underlying_price);
    }
    return fmin(premium, strike_price - underlying_price);
}

void option_pl_array_populate(double strike_price, double premium, bool is_put, bool is_buy, double *pl_array, int len = BUCKET_ARRAY_SIZE) {
    double tmp_price = BUCKET_LOW;
    for (int i = 0; i < len; i++) {
        pl_array[i] = option_pl_at_expiry(tmp_price, strike_price, premium, is_put, is_buy);
        tmp_price += BUCKET_SIZE;
    }
}

int main()
{
    const double    STRIKEPRICE = 105.0;
    const double    PREMIUM = 5.55;
    const bool      IS_PUT = true;
    const bool      IS_BUY = false;
    int numSimulations = 10000;
    float initialstockPrice = 105.0f; // Initial stock price
    float yearsUntilExpiration = 42.0/252.0f;   // Time to expiration (in years)
    float riskFreeInterestRate = 0.05f;  // Risk-free interest rate
    float sigma = 0.46f; // Volatility

    // Allocate memory on the host for price results
    float* prices = (float*)malloc(numSimulations * sizeof(float));

    double* optionPL = (double*)malloc(BUCKET_ARRAY_SIZE * sizeof(double));

    // Allocate memory on the device for price results
    float* devPrices;
    gpuErrorCheck(cudaMalloc((void**)&devPrices, numSimulations * sizeof(float)));

    // Allocate memory on the device for random number generation states
    RANDSTATETYPE devStates;
    gpuErrorCheck(cudaMalloc((void**)&devStates, numSimulations * sizeof(RANDSTATETYPE)));

    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrorCheck(cudaSetDevice(0));

    // Launch the kernel
    int numThreadsPerBlock = 1024;
    int numBlocks = (numSimulations + numThreadsPerBlock - 1) / numThreadsPerBlock;
    initRandomGenerator << <numBlocks, numThreadsPerBlock >> > (devStates, time(NULL));
    monteCarloSimulation << <numBlocks, numThreadsPerBlock >> > (devPrices, numSimulations, initialstockPrice, yearsUntilExpiration, riskFreeInterestRate, sigma, devStates);

    // Copy the results back to the host
    gpuErrorCheck(cudaMemcpy(prices, devPrices, numSimulations * sizeof(float), cudaMemcpyDeviceToHost));

    //print generated prices
    /*for (int i = 0; i < numSimulations; i++) {
        printf("sim %i price: $%.2f", i, prices[i]);
    }*/

    float probabilities[BUCKET_ARRAY_SIZE];
    float current_price = BUCKET_LOW;
    float current_bucket_top;
    for (int i = 0; i < BUCKET_ARRAY_SIZE; i++) {
        current_bucket_top = current_price + BUCKET_SIZE;
        probabilities[i] = 0.0f;
        for (int j = 0; j < numSimulations; j++) {
            if (prices[j] >= current_price && prices[j] < current_bucket_top) {
                probabilities[i] += 1.0f;
            }
        }
        probabilities[i] /= numSimulations;
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
