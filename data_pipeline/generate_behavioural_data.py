"""
Synthtic behavioural data generator for phisihing susceptibility.

Simulates per-user interaction features (hover time, click speed, etc.)
calibrated to findings from the behavioural phishing literature, since
no public dataset with these features exists (see Methodology writeup).
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_USERS = 2000 #number of simulated user to generate

#Generating hidden susceptibility profile scores
def generate_profile_scores(n_users: int)-> np.ndarray:
    """
    Each simulated data gets hidden 'susceptibility profile' score 
    between 0 (cautions) and 1(susceptible).
    We use a Beta distribution because it's bounded to [0, 1] (unlike a
    normal distribution, which could give nonsensical negative scores)
    and its two shape parameters let us control the population's overall
    skew. 
    """
    return np.random.beta(a=2, b=3, size=n_users)

#Feature 1: Time-to-First-Click (or non-click decision)
def generate_time_to_first_click(profile_scores: np.ndarray)-> np.ndarray:
    """
    Seconds from opening the email to clicking the link (or deciding
    not to). Grounded in Lain et al.'s finding that susceptible users
    act fast, without pausing to evaluate.
    using lognormal distrubution because reaction times cannot be 
    negative and are typically right-skewed real data.
    """
    median_seconds = 35 - 30 * profile_scores
    mu = np.log(median_seconds)
    sigma = 0.5

    times = np.random.lognormal(mean=mu, sigma=sigma)
    return np.clip(times, 1, 120) #clip to plausible 1-120 second range

# Feature 2: Hover/dwell time on the link
def generate_hover_time(profile_scores:np.ndarray) -> np.ndarray:
    """
    Seconds the cursor rests on the link before a decision is made.
    Here Low profile score (cautations) means Longer hover.Cautious 
    users hover to check where the link really points; 
    susceptible users barely pause before clicking.
    """
    median_hover = 0.3 + 2.7 * (1 - profile_scores)
    mu = np.log(median_hover)
    sigma = 0.6

    hover_times = np.random.lognormal(mean=mu, sigma=sigma)
    return np.clip(times, 1, 120) #clip to plausible 1-120 second range
