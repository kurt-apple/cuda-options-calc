# cuda-options-calc
Using CUDA to explore options pricing and modeling

# Don't sue me
I am neither proficient with CUDA/C nor am I some sort of market wiz. In fact, this is my first time using CUDA. (I recommend it)

# Early Days
This is the early days of this project. My goal is to resolve some unanswered questions I have with options and options pricing.

# Which is better: Long or Short?
Over the long run, which is more lucrative? Is Theta/Vega Gang the way to be, or can Vega/Gamma pop the price of long options in some predictable fashion?

# Given volatility and deviation of price over time, what is the overall probability of profiting from a trade?
I don't just mean "how likely is it that it lands between two prices?" I mean this:
1. Collect the probability of landing on every possible price (down to dollar or penny) - monte carlo method
2. Multiply that probability by the P/L of the option if it were to land at that price
3. Net total all possible P/L weighted by their probability

# Is it more lucrative to manage options early, or let them expire?
I see no reason why I can't model the price of an option and the expected range of movement of the underlying given its volatility, and search through time to see if there is a particularly fruitful zone of time within that option's lifespan where I could plan to manage the winners or losers.

# Will this research affect my performance, trading mechanics, psychology/confidence as I trade live?
We'll see!
